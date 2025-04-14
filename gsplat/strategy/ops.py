import numpy as np
from typing import Callable, Dict, List, Union

import torch
import torch.nn.functional as F
from torch import Tensor

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
        params: Dict[str, torch.nn.Parameter],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        revised_opacity: bool = False,
        alpha_t: float = 1.0,
        alpha_g: float = 0.2,
):
    """
    Split each selected Gaussian (where `mask[i] == True`) into 2 children.

    This version:
      • Uses *homogeneous coordinates* by referencing a parameter "w".
      • Recomputes new means from polar coords: xyz -> (r, w, phi), etc.
      • Logs and reassigns "w" so that new children get a consistent homogeneous coordinate.
      • Partially inherits optimizer states for "exp_avg" and "exp_avg_sq", etc.
      • Leaves ephemeral `state[...]` buffers with 2 new zero rows for each splitted Gaussian.

    Args:
      params: Dictionary of all model parameters, including "means", "scales", "quats", "opacities", and "w".
      optimizers: Dictionary of corresponding optimizers (e.g. Adam).
      state: Extra running states (like "grad2d", "count", etc.).
      mask: Boolean mask, shape [N], indicating which Gaussians to split.
      revised_opacity: If True, apply revised opacity from arXiv:2404.06109.
      alpha_t, alpha_g: partial-inheritance factors for the optimizer's Adam states.
    """

    device = mask.device
    sel = torch.where(mask)[0]  # indices of the "mother" Gaussians
    rest = torch.where(~mask)[0]  # indices of the remaining (father) Gaussians
    N_sel = len(sel)
    if N_sel == 0:
        return  # nothing to split

    old_count = len(rest) + len(sel)  # total before splitting
    new_count = len(rest) + 2 * N_sel  # total after splitting

    # 1) Compute new child "means" using homogeneous coords
    #    (like your original snippet).
    mother_w = params["w"][sel]  # shape [N_sel]
    w_inv = 1.0 / torch.exp(mother_w).unsqueeze(1)  # shape [N_sel, 1]

    mother_means = params["means"][sel]  # shape [N_sel, 3]
    # scales = exp(params["scales"]) * w_inv * ||means||
    # the old code does this:
    r_means = torch.norm(mother_means, dim=1).unsqueeze(1)  # shape [N_sel, 1]
    mother_scales = torch.exp(params["scales"][sel]) * w_inv * r_means
    # Rotation from quats:
    mother_quats = F.normalize(params["quats"][sel], dim=-1)  # shape [N_sel, 4]
    rotmats = normalized_quat_to_rotmat(mother_quats)  # shape [N_sel, 3,3]

    # sample 2 random offsets per mother, shape [2, N_sel, 3]
    rand_samples = torch.randn(2, N_sel, 3, device=device)
    # local_shifts = R * scales * random
    local_shifts = torch.einsum("nij,nj,bnj->bni", rotmats, mother_scales, rand_samples)

    # new_means in "Cartesian coords" = (mother_means * w_inv + local_shifts)
    # shape => [2, N_sel, 3] => flatten => [2*N_sel, 3]
    child_means_cart = (mother_means * w_inv + local_shifts).reshape(-1, 3)

    # Then convert back to polar => w, etc.
    # We'll define a helper function (like your original xyz_to_polar).
    # If you already have xyz_to_polar in your code, use that.  We'll show a small mock version below:
    # (r, w, phi) = xyz_to_polar(cart)
    # then child_means = cart * w
    new_r, new_w, new_phi = xyz_to_polar(child_means_cart)  # each shape [2*N_sel]
    # final child means in homogeneous coords => cart * w
    child_means = child_means_cart * new_w.unsqueeze(1)

    ####################################################
    # 2) Build param_fn for all param keys
    ####################################################
    def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
        """
        For each param name, produce the new [new_count, ...] parameter data,
        reusing the "father" rows from `rest` plus new splitted rows for `sel`.
        """
        if p.shape[0] != old_count:
            # not a per-Gaussian param, skip
            return p

        # father part
        father_part = p[rest]
        mother_part = p[sel]

        if name == "means":
            # We just computed 'child_means' above, shape [2*N_sel, 3]
            splitted = child_means
            new_p = torch.cat([father_part, splitted], dim=0)
            return torch.nn.Parameter(new_p, requires_grad=p.requires_grad)

        elif name == "scales":
            # replicate the old logic: new scales = log(scales/1.6)
            splitted = torch.log(torch.exp(params["scales"][sel]) / 1.6).repeat(2, 1)
            new_p = torch.cat([father_part, splitted], dim=0)
            return torch.nn.Parameter(new_p, requires_grad=p.requires_grad)

        elif name == "quats":
            # replicate mother quats 2x
            splitted = mother_part.repeat(2, 1)
            new_p = torch.cat([father_part, splitted], dim=0)
            return torch.nn.Parameter(new_p, requires_grad=p.requires_grad)

        elif name == "opacities":
            if revised_opacity:
                # revised => 1 - sqrt(1 - sigmoid(...))
                sigm = torch.sigmoid(mother_part)
                new_sigm = 1.0 - torch.sqrt(1.0 - sigm)
                splitted = torch.logit(new_sigm).repeat(2)
            else:
                # normal => replicate mother
                splitted = mother_part.repeat(2)
            new_p = torch.cat([father_part, splitted], dim=0)
            return torch.nn.Parameter(new_p, requires_grad=p.requires_grad)

        elif name == "w":
            # we do: new w = log(new_w)
            # new_w is shape [2*N_sel]. => no .repeat needed, it’s already 2*N_sel
            splitted = torch.log(new_w)
            new_p = torch.cat([father_part, splitted], dim=0)
            return torch.nn.Parameter(new_p, requires_grad=p.requires_grad)

        else:
            # default: replicate mother row 2x
            # shape: [old_count, ...]
            splitted = mother_part.repeat(2, *[1] * (p.ndim - 1))
            new_p = torch.cat([father_part, splitted], dim=0)
            return torch.nn.Parameter(new_p, requires_grad=p.requires_grad)

    ####################################################
    # 3) Build optimizer_fn for partial inheritance
    ####################################################
    def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
        """
        For each key in the optimizer state (e.g. 'exp_avg', 'exp_avg_sq', etc.),
        produce the new per-Gaussian array with partial inheritance.
        """
        if not isinstance(v, torch.Tensor):
            return v
        if v.dim() == 0 or v.shape[0] != old_count:
            return v

        father_vals = v[rest]
        mother_vals = v[sel]

        if key == "exp_avg":
            c1 = alpha_g * mother_vals
            c2 = alpha_g * mother_vals
        elif key == "exp_avg_sq":
            c1 = (alpha_g ** 2) * mother_vals
            c2 = (alpha_g ** 2) * mother_vals
        else:
            # e.g. zero them out or do alpha_t if you have 'lifespan'
            c1 = torch.zeros_like(mother_vals)
            c2 = torch.zeros_like(mother_vals)

        splitted = torch.cat([c1, c2], dim=0)
        return torch.cat([father_vals, splitted], dim=0)

    ####################################################
    # 4) Actually update the params + optimizer states
    ####################################################
    def _update_param_with_optimizer(
            param_fn: Callable[[str, torch.Tensor], torch.Tensor],
            optimizer_fn: Callable[[str, torch.Tensor], torch.Tensor],
            params: Dict[str, torch.nn.Parameter],
            optimizers: Dict[str, torch.optim.Optimizer],
            names: Union[List[str], None] = None,
    ):
        if names is None:
            names = list(params.keys())
        for name in names:
            old_p = params[name]
            new_p = param_fn(name, old_p)
            params[name] = new_p

            if name in optimizers:
                opt = optimizers[name]
                # Typically 1 param group
                for group in opt.param_groups:
                    if old_p in group["params"]:
                        group["params"].remove(old_p)
                    if new_p not in group["params"]:
                        group["params"].append(new_p)

                if old_p in opt.state:
                    old_state = opt.state.pop(old_p)
                    new_state = {}
                    for k_, v_ in old_state.items():
                        new_state[k_] = optimizer_fn(k_, v_)
                    opt.state[new_p] = new_state

    # Call the update
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers, names=None)

    ####################################################
    # 5) Update ephemeral states in `state`
    ####################################################
    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.shape[0] != old_count:
            continue  # mismatch => skip

        father_part = v[rest]
        mother_part = v[sel]
        zero_part = torch.zeros_like(mother_part)
        splitted = torch.cat([zero_part, zero_part], dim=0)  # shape [2*N_sel, ...]
        state[k] = torch.cat([father_part, splitted], dim=0)


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

    # new opacities/scales
    new_opacities, new_scales = compute_relocation(
        opacities=opacities[sampled_idxs],
        scales=torch.exp(params["scales"])[sampled_idxs],
        ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
        binoms=binoms,
    )
    new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)

    def param_fn(name: str, p: Tensor) -> Tensor:
        # for the chosen "alive" indices => update them
        if name == "opacities":
            p[sampled_idxs] = torch.logit(new_opacities)
        elif name == "scales":
            p[sampled_idxs] = torch.log(new_scales)
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

    new_opacities, new_scales = compute_relocation(
        opacities=opacities[sampled_idxs],
        scales=torch.exp(params["scales"])[sampled_idxs],
        ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
        binoms=binoms,
    )
    new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)

    def param_fn(name: str, p: Tensor) -> Tensor:
        # shape: oldN = p.shape[0]
        # we will append new entries = p[sampled_idxs], possibly with modifications
        if name == "opacities":
            p[sampled_idxs] = torch.logit(new_opacities)
        elif name == "scales":
            p[sampled_idxs] = torch.log(new_scales)
        # For means, w, quats, etc., we simply copy from the chosen source.
        # If you prefer a random offset (like in 'split'), you can do so here.
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
):
    """Inject noise into the positions, respecting the 'homogeneous' representation.

    We have:
      means = (x*r, y*r, z*r),
      w = log(r).

    Procedure:
      1) Convert means to Cartesian => cart = means / exp(w).
      2) Compute a noise offset in that local tangent space.
      3) cart_new = cart + noise
      4) Convert cart_new => new radius => means/w.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers (not updated here, but we keep the signature).
        state: A dictionary of extra states (not used here, but we keep the signature).
        scaler: scalar factor for noise amplitude.
    """

    # Slightly sharpened control for how quickly noise goes to 0 as opacity -> 1
    def op_sigmoid(x, k=100, x0=0.995):
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
        params["quats"], scales, compute_covar=True, compute_preci=False, triu=False
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