import numpy as np
from typing import Callable, Dict, List, Union

import torch
import torch.nn.functional as F
from torch import Tensor
import math

from gsplat import quat_scale_to_covar_preci
from gsplat.relocation import compute_relocation
from gsplat.utils import normalized_quat_to_rotmat, xyz_to_polar


@torch.no_grad()
def _multinomial_sample(weights: Tensor, n: int, replacement: bool = True) -> Tensor:
    """Sample from a distribution using torch.multinomial or numpy.random.choice.

    This function adaptively chooses between `torch.multinomial` and `numpy.random.choice`
    based on the number of elements in `weights`. If the number of elements exceeds
    the torch.multinomial limit (2^24), it falls back to using `numpy.random.choice`.

    Args:
        weights (Tensor): A 1D tensor of weights for each element.
        n (int): The number of samples to draw.
        replacement (bool): Whether to sample with replacement. Default is True.

    Returns:
        Tensor: A 1D tensor of sampled indices.
    """
    num_elements = weights.size(0)

    if num_elements <= 2**24:
        # Use torch.multinomial for elements within the limit
        return torch.multinomial(weights, n, replacement=replacement)
    else:
        # Fallback to numpy.random.choice for larger element spaces
        weights = weights / weights.sum()
        weights_np = weights.detach().cpu().numpy()
        sampled_idxs_np = np.random.choice(
            num_elements, size=n, p=weights_np, replace=replacement
        )
        sampled_idxs = torch.from_numpy(sampled_idxs_np)

        # Return the sampled indices on the original device
        return sampled_idxs.to(weights.device)


@torch.no_grad()
def _update_param_with_optimizer(
        param_fn: Callable[[str, Tensor], Tensor],
        optimizer_fn: Callable[[str, Tensor], Tensor],
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        names: Union[List[str], None] = None,
):
    """Update the parameters and the state in the optimizers with defined functions.

    Args:
        param_fn: A function that takes the name of the parameter and the parameter itself,
            and returns the new parameter.
        optimizer_fn: A function that takes the key of the optimizer state and the state value,
            and returns the new state value.
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        names: A list of key names to update. If None, update all. Default: None.
    """
    if names is None:
        # If names is not provided, update all parameters
        names = list(params.keys())

    for name in names:
        param = params[name]
        new_param = param_fn(name, param)
        params[name] = new_param
        if name not in optimizers:
            assert not param.requires_grad, (
                f"Optimizer for {name} is not found, but the parameter is trainable."
                f"Got requires_grad={param.requires_grad}"
            )
            continue
        optimizer = optimizers[name]
        for i in range(len(optimizer.param_groups)):
            param_state = optimizer.state[param]
            del optimizer.state[param]
            for key in param_state.keys():
                if key != "step":
                    v = param_state[key]
                    param_state[key] = optimizer_fn(key, v)
            optimizer.param_groups[i]["params"] = [new_param]
            optimizer.state[new_param] = param_state


@torch.no_grad()
def duplicate(
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Tensor],
        mask: Tensor,
):
    """Inplace duplicate the Gaussian with the given mask.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to duplicate the Gaussians.
    """
    device = mask.device
    sel = torch.where(mask)[0]

    def param_fn(name: str, p: Tensor) -> Tensor:
        return torch.nn.Parameter(torch.cat([p, p[sel]]), requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return torch.cat([v, torch.zeros((len(sel), *v.shape[1:]), device=device)])

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # update the extra running state
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = torch.cat((v, v[sel]))


@torch.no_grad()
def split(
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Tensor],
        mask: Tensor,
        revised_opacity: bool = False,
):
    device = mask.device
    sel = torch.where(mask)[0]   # Indices to split
    rest = torch.where(~mask)[0] # Indices to keep
    N_sel = len(sel)
    if N_sel == 0:
        return  # Nothing to do

    mother_means_hom = params["means"][sel]                 # shape [N_sel, 3]
    mother_w          = params["w"][sel]                    # shape [N_sel]
    mother_w_inv      = mother_w.exp().reciprocal_().unsqueeze(-1)  # shape [N_sel,1]

    # Means in "cartesian coords":
    mother_means_cart = mother_means_hom * mother_w_inv  # shape [N_sel,3]

    # Because scales is stored in log-space:
    mother_scales_log = params["scales"][sel]             # shape [N_sel, 3]
    mother_scales_lin = mother_scales_log.exp()           # shape [N_sel, 3]
    # Convert to cart coords: incorporate w^-1:
    mother_scales_cart = mother_scales_lin * mother_w_inv # shape [N_sel, 3]

    # Quaternions + rotation for local->world transform:
    quats   = F.normalize(params["quats"][sel], dim=-1)   # shape [N_sel, 4]
    rotmats = normalized_quat_to_rotmat(quats)            # shape [N_sel, 3, 3]

    max_indices = torch.argmax(mother_scales_cart, dim=1)  # shape [N_sel]
    # e.g. 1.5 times the largest axis
    offset_magnitudes = 1.5 * mother_scales_cart[torch.arange(N_sel), max_indices]  # [N_sel]

    # Build local offsets (2 children: +/− direction along that axis)
    signs = torch.tensor([1.0, -1.0], device=device).view(2, 1)  # shape [2,1]
    # offsets_local has shape [2, N_sel, 3], all zeros except the chosen axis
    offsets_local = torch.zeros(2, N_sel, 3, device=device)
    offsets_local[0, torch.arange(N_sel), max_indices] = signs[0] * offset_magnitudes
    offsets_local[1, torch.arange(N_sel), max_indices] = signs[1] * offset_magnitudes

    # Rotate local offsets => world coords, shape [2, N_sel, 3]
    offsets_world = torch.einsum("nij,bnj->bni", rotmats, offsets_local)

    # mother_means_cart shape [N_sel,3], we want 2 children => shape [2, N_sel, 3]
    mother_means_cart = mother_means_cart.unsqueeze(0)  # shape [1, N_sel, 3]
    new_means_cart    = mother_means_cart + offsets_world  # shape [2, N_sel, 3]
    new_means_cart    = new_means_cart.reshape(-1, 3)      # shape [2*N_sel, 3]

    # Convert back to polar => get new child w = log(1/r), child means = xyz * (1/r).
    _, new_w_lin, _r = xyz_to_polar(new_means_cart)            # each shape [2*N_sel]
    new_means_hom = new_means_cart * new_w_lin.unsqueeze(1)    # shape [2*N_sel,3]
    new_w_log     = new_w_lin.log()                            # shape [2*N_sel]

    log_0_5  = math.log(0.5)
    log_0_85 = math.log(0.85)
    adj = torch.full_like(mother_scales_cart, log_0_85)  # [N_sel,3]
    for i in range(N_sel):
        idx = max_indices[i].item()
        adj[i, idx] = log_0_5

    # mother_scales_cart is shape [N_sel,3] => log => add => => exponent => done
    mother_scales_cart_log = mother_scales_cart.log()   # shape [N_sel,3]
    new_scales_cart_log    = mother_scales_cart_log + adj  # shape [N_sel,3]
    # Now replicate for the 2 children => shape [2*N_sel, 3]
    new_scales_cart_log    = new_scales_cart_log.repeat(2, 1)

    new_scales_hom = new_scales_cart_log + new_w_log.unsqueeze(1)

    old_count = params["means"].shape[0]
    N_child   = 2 * N_sel

    def param_fn(name: str, p: Tensor) -> Tensor:
        """Assemble the father rows + new child rows for each parameter."""
        # shape check
        if p.shape[0] != old_count:
            # Not a per-Gaussian param. Just return it unmodified.
            return p

        # father part:
        father_vals = p[rest]

        if name == "means":
            # shape => [2*N_sel, 3]
            p_split = new_means_hom
        elif name == "w":
            # shape => [2*N_sel]
            p_split = new_w_log
        elif name == "scales":
            # shape => [2*N_sel, 3]
            p_split = new_scales_hom
        elif name == "opacities":
            mother_part = p[sel]
            opacity = torch.sigmoid(mother_part)
            if revised_opacity:
                # arXiv:2404.06109 => new_opacity = 1 - sqrt(1 - old_sigmoid)
                new_opacity = 1.0 - torch.sqrt(1.0 - opacity)
            else:
                new_opacity = 0.6 * opacity
            p_split = torch.logit(new_opacity).repeat(2, *[1]*(p.ndim-1))
        else:
            # For everything else, replicate mother row 2x
            mother_part = p[sel]
            repeats = [2] + [1]*(p.ndim - 1)
            p_split = mother_part.repeat(*repeats)

        # Concatenate father + splitted
        out = torch.cat([father_vals, p_split], dim=0)
        # Keep requires_grad the same as original
        out = torch.nn.Parameter(out, requires_grad=p.requires_grad)
        return out

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        if not isinstance(v, torch.Tensor):
            return v
        if v.shape[0] != old_count:
            return v

        father_vals = v[rest]
        # new children => shape [2*N_sel, ...], set to zero:
        zero_vals = torch.zeros((N_child,) + v.shape[1:], device=v.device, dtype=v.dtype)
        out = torch.cat([father_vals, zero_vals], dim=0)
        return out

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)

    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.shape[0] != old_count:
            continue
        father_vals = v[rest]
        mother_vals = v[sel]
        # By default, we can init the new child states to zeros or a copy:
        zero_vals = torch.zeros_like(mother_vals)
        splitted = torch.cat([zero_vals, zero_vals], dim=0)  # [2*N_sel, ...]
        state[k] = torch.cat([father_vals, splitted], dim=0)


@torch.no_grad()
def remove(
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Tensor],
        mask: Tensor,
):
    """Inplace remove the Gaussian with the given mask.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to remove the Gaussians.
    """
    sel = torch.where(~mask)[0]

    def param_fn(name: str, p: Tensor) -> Tensor:
        return torch.nn.Parameter(p[sel], requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return v[sel]

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # update the extra running state
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v[sel]


@torch.no_grad()
def reset_opa(
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Tensor],
        value: float,
):
    """Inplace reset the opacities to the given post-sigmoid value.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        value: The value to reset the opacities
    """

    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "opacities":
            opacities = torch.clamp(p, max=torch.logit(torch.tensor(value)).item())
            return torch.nn.Parameter(opacities, requires_grad=p.requires_grad)
        else:
            raise ValueError(f"Unexpected parameter name: {name}")

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return torch.zeros_like(v)

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(
        param_fn, optimizer_fn, params, optimizers, names=["opacities"]
    )


@torch.no_grad()
def relocate(
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Tensor],
        mask: Tensor,
        binoms: Tensor,
        min_opacity: float = 0.005,
):
    """Inplace relocate some dead Gaussians to the lives ones.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to indicates which Gaussians are dead.
    """
    opacities = torch.sigmoid(params["opacities"]).flatten()  # shape [N,]
    dead_indices = mask.nonzero(as_tuple=True)[0]
    alive_indices = (~mask).nonzero(as_tuple=True)[0]
    n = len(dead_indices)
    if n == 0:
        return  # nothing to do

    # sample from alive
    eps = torch.finfo(torch.float32).eps
    probs = opacities[alive_indices]
    sampled_idxs = _multinomial_sample(probs, n, replacement=True)
    sampled_idxs = alive_indices[sampled_idxs]
    w_inv = 1.0 / torch.exp(params["w"][sampled_idxs]).unsqueeze(1)

    # new opacities/scales
    new_opacities, new_scales = compute_relocation(
        opacities=opacities[sampled_idxs],
        scales=torch.exp(params["scales"])[sampled_idxs] * w_inv,
        ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
        binoms=binoms,
    )
    new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)

    def param_fn(name: str, p: Tensor) -> Tensor:
        # for the chosen "alive" indices => update them
        if name == "opacities":
            p[sampled_idxs] = torch.logit(new_opacities)
        elif name == "scales":
            p[sampled_idxs] = torch.log(new_scales * torch.exp(params["w"][sampled_idxs]).unsqueeze(1))
        else:
            # e.g. means, w, quats => no "random offset" in relocate;
            # we simply keep them as is or we might do something more advanced.
            # By default, let's keep the sampling approach consistent:
            #   dead => replaced by the chosen sample
            #   alive => unchanged except the ones in sampled_idxs might
            #            forcibly have their optimizer state reset below.
            pass

        # for the truly "dead" => set them to the newly updated indices
        # (the same location/orientation as the sampled one).
        p[dead_indices] = p[sampled_idxs]
        return torch.nn.Parameter(p, requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        # reset states for the ones we just relocated
        v[sampled_idxs] = 0
        return v

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # also reset any extra state
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            # similarly set "dead" from "sampled_idxs"
            v[sampled_idxs] = 0
            v[dead_indices] = v[sampled_idxs]


@torch.no_grad()
def sample_add(
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Tensor],
        n: int,
        binoms: Tensor,
        min_opacity: float = 0.005,
):
    opacities = torch.sigmoid(params["opacities"])

    eps = torch.finfo(torch.float32).eps
    probs = opacities.flatten()
    sampled_idxs = _multinomial_sample(probs, n, replacement=True)
    w = params["w"][sampled_idxs].exp().unsqueeze(-1)
    w_inv = 1 / w

    new_opacities, new_scales = compute_relocation(
        opacities=opacities[sampled_idxs],
        scales=torch.exp(params["scales"])[sampled_idxs] * w_inv,
        ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
        binoms=binoms,
    )
    new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)

    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "opacities":
            p[sampled_idxs] = torch.logit(new_opacities)
        elif name == "scales":
            p[sampled_idxs] = torch.log(new_scales  * w)
        # For means, w, quats, etc., we simply copy from the chosen source.
        p_new = torch.cat([p, p[sampled_idxs]], dim=0)
        return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        # The newly added items have fresh (zeroed) states
        v_new = torch.zeros((len(sampled_idxs), *v.shape[1:]), device=v.device)
        return torch.cat([v, v_new], dim=0)

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # expand any state vectors similarly
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            v_new = torch.zeros((len(sampled_idxs), *v.shape[1:]), device=v.device)
            state[k] = torch.cat((v, v_new), dim=0)


@torch.no_grad()
def inject_noise_to_position(
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Tensor],
        scaler: float,
        noise_stepness: int = 100,
):
    """Inject noise into the positions, respecting the 'homogeneous' representation.

    We have:
      means = (x*r, y*r, z*r),
      w = log(r).

    Procedure:
      1) Convert means to Cartesian => cart = means / exp(w).
      2) Compute a noise offset in that local tangent space. We can still
         scale noise by the covariance from quat & scale, but that is up to you.
      3) cart_new = cart + noise
      4) Convert cart_new => new radius => means/w.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers (not updated here, but we keep the signature).
        state: A dictionary of extra states (not used here, but we keep the signature).
        scaler: scalar factor for noise amplitude.
    """

    def op_sigmoid(x, k=noise_stepness, x0=0.995):
        return 1.0 / (1.0 + torch.exp(-k * (x - x0)))

    # Grab the existing data
    means = params["means"]       # shape [N, 3]
    w_log = params["w"]          # shape [N]
    opacities = torch.sigmoid(params["opacities"].flatten())  # shape [N]
    scales = torch.exp(params["scales"])                      # shape [N]

    # Convert to Cartesian
    cart = means / torch.exp(w_log).unsqueeze(-1)  # shape [N, 3]

    # Covariance-based noise scale
    # The function returns (covars, None), where covars is [N,3,3] if triu=False.
    covars, _ = quat_scale_to_covar_preci(
        params["quats"], scales / torch.exp(w_log).unsqueeze(-1), compute_covar=True, compute_preci=False, triu=False
    )

    # The factor (op_sigmoid(1 - opacities)) is near 1 for low opacities and near 0 for high.
    noise_factor = op_sigmoid(1.0 - opacities).unsqueeze(-1)  # shape [N,1]
    raw_noise = torch.randn_like(means)  # shape [N,3]

    # transform noise by the covariance
    # noise_out = covars @ raw_noise, i.e. shape [N,3]
    noise = torch.einsum("nij,nj->ni", covars, raw_noise) * noise_factor * scaler

    # Add to Cartesian
    cart_new = cart + noise

    # Convert back to homogeneous
    #   r_new = radius of cart_new
    #   means_new = cart_new * r_new
    #   w_new = log(r_new)
    _, r_new, _ = xyz_to_polar(cart_new)
    means_new = cart_new * r_new.unsqueeze(1)

    # Store
    params["means"].data = means_new
    params["w"].data = torch.log(r_new.clamp_min(1e-8))
