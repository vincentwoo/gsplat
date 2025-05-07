from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union, Optional

import torch
import math
from typing_extensions import Literal

from .base import Strategy
from .ops import duplicate, remove, relocate, split, inject_noise_to_position


@dataclass
class DefaultStrategy(Strategy):
    """A default strategy that follows the original 3DGS paper:

    `3D Gaussian Splatting for Real-Time Radiance Field Rendering <https://arxiv.org/abs/2308.04079>`_

    The strategy will:

    - Periodically duplicate GSs with high image plane gradients and small scales.
    - Periodically split GSs with high image plane gradients and large scales.
    - Periodically prune GSs with low opacity.
    - Periodically reset GSs to a lower opacity.

    If `absgrad=True`, it will use the absolute gradients instead of average gradients
    for GS duplicating & splitting, following the AbsGS paper:

    `AbsGS: Recovering Fine Details for 3D Gaussian Splatting <https://arxiv.org/abs/2404.10484>`_

    Which typically leads to better results but requires to set the `grow_grad2d` to a
    higher value, e.g., 0.0008. Also, the :func:`rasterization` function should be called
    with `absgrad=True` as well so that the absolute gradients are computed.

    Args:
        prune_opa (float): GSs with opacity below this value will be pruned. Default is 0.005.
        grow_grad2d (float): GSs with image plane gradient above this value will be
          split/duplicated. Default is 0.0002.
        grow_scale3d (float): GSs with 3d scale (normalized by scene_scale) below this
          value will be duplicated. Above will be split. Default is 0.01.
        grow_scale2d (float): GSs with 2d scale (normalized by image resolution) above
          this value will be split. Default is 0.05.
        prune_scale3d (float): GSs with 3d scale (normalized by scene_scale) above this
          value will be pruned. Default is 0.1.
        prune_scale2d (float): GSs with 2d scale (normalized by image resolution) above
          this value will be pruned. Default is 0.15.
        refine_scale2d_stop_iter (int): Stop refining GSs based on 2d scale after this
          iteration. Default is 0. Set to a positive value to enable this feature.
        refine_start_iter (int): Start refining GSs after this iteration. Default is 500.
        refine_stop_iter (int): Stop refining GSs after this iteration. Default is 15_000.
        reset_every (int): Reset opacities every this steps. Default is 6000.
        refine_every (int): Refine GSs every this steps. Default is 200.
        pause_refine_after_reset (int): Pause refining GSs until this number of steps after
          reset, Default is 0 (no pause at all) and one might want to set this number to the
          number of images in training set.
        absgrad (bool): Use absolute gradients for GS splitting. Default is False.
        revised_opacity (bool): Whether to use revised opacity heuristic from
          arXiv:2404.06109 (experimental). Default is False.
        verbose (bool): Whether to print verbose information. Default is False.
        key_for_gradient (str): Which variable uses for densification strategy.
          3DGS uses "means2d" gradient and 2DGS uses a similar gradient which stores
          in variable "gradient_2dgs".

    Examples:

        >>> from gsplat import DefaultStrategy, rasterization
        >>> params: Dict[str, torch.nn.Parameter] | torch.nn.ParameterDict = ...
        >>> optimizers: Dict[str, torch.optim.Optimizer] = ...
        >>> strategy = DefaultStrategy()
        >>> strategy.check_sanity(params, optimizers)
        >>> strategy_state = strategy.initialize_state()
        >>> for step in range(1000):
        ...     render_image, render_alpha, info = rasterization(...)
        ...     strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        ...     loss = ...
        ...     loss.backward()
        ...     strategy.step_post_backward(params, optimizers, strategy_state, step, info)

    """

    prune_opa: float = 0.005
    grow_grad2d: float = 0.00012
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    noise_lr: float = 5e5
    noise_stepness: int = 100
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 0
    refine_start_iter: int = 1_500
    refine_stop_iter: int = 55_000
    max_budget: int = 10_000_000 # set -1 to disable.
    reset_every: int = 6000
    refine_every: int = 200
    pause_refine_after_reset: int = 0
    absgrad: bool = False
    revised_opacity: bool = False
    verbose: bool = False
    key_for_gradient: Literal["means2d", "gradient_2dgs"] = "means2d"
    p_init: int = 0
    p_fin: int = 0
    binoms: Optional[torch.Tensor] = None


    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy.

        The returned state should be passed to the `step_pre_backward()` and
        `step_post_backward()` functions.
        """
        # Postpone the initialization of the state to the first step so that we can
        # put them on the correct device.
        # - grad2d: running accum of the norm of the image plane gradients for each GS.
        # - count: running accum of how many time each GS is visible.
        # - radii: the radii of the GSs (normalized by the image resolution).
        state = {"grad2d": None, "count": None, "scene_scale": scene_scale, "importance": None}
        if self.refine_scale2d_stop_iter > 0:
            state["radii"] = None
        return state

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Sanity check for the parameters and optimizers.

        Check if:
            * `params` and `optimizers` have the same keys.
            * Each optimizer has exactly one param_group, corresponding to each parameter.
            * The following keys are present: {"means", "scales", "quats", "opacities"}.

        Raises:
            AssertionError: If any of the above conditions is not met.

        .. note::
            It is not required but highly recommended for the user to call this function
            after initializing the strategy to ensure the convention of the parameters
            and optimizers is as expected.
        """

        super().check_sanity(params, optimizers)
        # The following keys are required for this strategy.
        for key in ["means", "scales", "quats", "opacities", "w"]:
            assert key in params, f"{key} is required in params but missing."

    def step_pre_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        """Callback function to be executed before the `loss.backward()` call."""
        assert (
            self.key_for_gradient in info
        ), "The 2D means of the Gaussians is required but missing."
        info[self.key_for_gradient].retain_grad()

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        factor: float,
        lr: float,
        info: Dict[str, Any],
        packed: bool = False,
    ):
        """Callback function to be executed after the `loss.backward()` call."""
        if step >= self.refine_stop_iter:
            return

        if self.binoms is None:
            n_max = 51
            self.binoms = torch.zeros((n_max, n_max))
            for n in range(n_max):
                for k in range(n + 1):
                    self.binoms[n, k] = math.comb(n, k)

            self.binoms = self.binoms.to(params["means"].device)

        self._update_state(params, state, info, packed=packed)

        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            and step % self.reset_every >= self.pause_refine_after_reset
        ):
            #n_prune = self._prune_gs(params, optimizers, state, step)
            n_prune = 0
            n_relocated_gs = self._relocate_gs(params, optimizers, state, n_prune, self.binoms)
            n_dupli, n_split = self._grow_gs(params, optimizers, state, step, factor)
            if self.verbose:
                total_growth = n_dupli + n_split - n_prune
                print(
                    f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split, {n_prune} pruned, {n_relocated_gs} relocated. Total growth {total_growth}"
                )

            # reset running stats
            state["grad2d"].zero_()
            state["count"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            torch.cuda.empty_cache()

        if step % self.reset_every == 0 and self.refine_start_iter <= step < self.refine_stop_iter:
            # Apply logit to current opacities
            # Mask for pruning in logit space
            threshold = 2.0 * self.prune_opa
            mask = state["importance"] < threshold
            n_prune = mask.sum().item()

            # Prune
            if n_prune > 0:
                self.prune_mask(params, optimizers, state, mask)

            state["importance"].zero_()
            print(f"Pruning {n_prune} GSs with opacity below {threshold:.2f}.")

        if (self.refine_start_iter < step < self.refine_stop_iter):
            inject_noise_to_position(
                params=params,
                optimizers=optimizers,
                state=state,
                scaler=lr * self.noise_lr,
                noise_stepness=self.noise_stepness,
            )


    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ):
        for key in [
            "width",
            "height",
            "n_cameras",
            "pixels",
            "radii",
            "gaussian_ids",
            self.key_for_gradient,
        ]:
            assert key in info, f"{key} is required but missing."

        # normalize grads to [-1, 1] screen space
        if self.absgrad:
            grads = info[self.key_for_gradient].absgrad.clone()
        else:
            grads = info[self.key_for_gradient].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        # initialize state on the first run
        n_gaussian = len(list(params.values())[0])

        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
            self.p_init = params["means"].shape[0]
            if self.max_budget > 0:
                self.p_fin = self.max_budget
            else:
                self.p_fin = 5 * self.p_init
        if state["count"] is None:
            state["count"] = torch.zeros(n_gaussian, device=grads.device)
        if self.refine_scale2d_stop_iter > 0 and state["radii"] is None:
            assert "radii" in info, "radii is required but missing."
            state["radii"] = torch.zeros(n_gaussian, device=grads.device)

        # update the running state
        if packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz]
            radii = info["radii"].max(dim=-1).values  # [nnz]
            pixels = torch.ones_like(radii)
        else:
            # grads is [C, N, 2]
            sel = (info["radii"] > 0.0).all(dim=-1)  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            grads = grads[sel]  # [nnz, 2]
            radii = info["radii"][sel].max(dim=-1).values  # [nnz]
            pixels = info["pixels"][sel]
        state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1) * pixels)
        state["count"].index_add_(
            0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32) * pixels
        )
        if self.refine_scale2d_stop_iter > 0:
            # Should be ideally using scatter max
            state["radii"][gs_ids] = torch.maximum(
                state["radii"][gs_ids],
                # normalize radii to [0, 1] screen space
                radii / float(max(info["width"], info["height"])),
            )

    @torch.no_grad()
    def _grow_gs(
            self,
            params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
            optimizers: Dict[str, torch.optim.Optimizer],
            state: Dict[str, Any],
            step: int,
            factor: float
    ) -> Tuple[int, int]:
        """
        A version of `_grow_gs` that implements a *budgeted* approach for
        deciding how many duplications vs. splits can happen, based on largest
        gradient norms first. Optimized for performance.
        """
        if params["means"].shape[0] >= self.max_budget and self.max_budget != -1:
            return 0, 0
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        device = grads.device

        # Identify the typical thresholds/flags, same as before:
        is_grad_high = grads > self.grow_grad2d

        # Calculate scales in one operation to avoid redundant computations
        w_inv = 1.0 / torch.exp(params["w"]).unsqueeze(1)
        scales = torch.exp(params["scales"]) * w_inv
        max_scales = scales.max(dim=-1).values
        is_small = max_scales <= self.grow_scale3d * state["scene_scale"]
        is_large = ~is_small

        # Create masks for duplication and splitting
        is_dupli = is_grad_high & is_small
        is_split = is_grad_high & is_large

        # Calculate costs for each point (2 for split, 1 for duplication)
        # This allows us to prioritize by gradient magnitude while accounting for different costs
        candidate_mask = is_dupli | is_split

        # Quick exit if no candidates
        n_candidates = candidate_mask.sum().item()
        if n_candidates == 0:
            return 0, 0  # no candidates => nothing to do

        # Compute total budget
        current_count = params["means"].shape[0]
        budget_left = int((self.p_init + (self.p_fin - self.p_init) * factor) - current_count)

        # Quick exit if no budget
        if budget_left <= 0:
            return 0, 0

        # Prepare costs - 2 for split, 1 for duplication
        candidate_idxs = candidate_mask.nonzero(as_tuple=True)[0]
        costs = torch.ones_like(candidate_idxs, dtype=torch.int)
        costs[is_split[candidate_idxs]] = 2

        # Get the gradient values for these candidates
        candidate_grads = grads[candidate_idxs]

        # If budget is very limited compared to candidates, use topk instead of full sort
        # This is faster when we have many candidates but limited budget
        if budget_left * 3 < n_candidates:  # Heuristic: if budget can handle less than 1/3 of candidates
            k = min(n_candidates, budget_left * 3)
            _, top_indices = torch.topk(candidate_grads, k=k, largest=True)
            sorted_candidate_idxs = candidate_idxs[top_indices]
            sorted_costs = costs[top_indices]
        else:
            # If we might use most candidates, do a full sort
            _, sort_indices = torch.sort(candidate_grads, descending=True)
            sorted_candidate_idxs = candidate_idxs[sort_indices]
            sorted_costs = costs[sort_indices]

        # Select candidates within budget - greedy approach
        cum_costs = torch.cumsum(sorted_costs, dim=0)
        valid_mask = cum_costs <= budget_left

        # If no valid candidates due to budget constraints
        if not valid_mask.any():
            # If we can't even afford the first item
            return 0, 0

        # Get the indices we can afford
        affordable_indices = sorted_candidate_idxs[valid_mask]

        # Separate into duplication and split operations
        dupli_indices = affordable_indices[~is_split[affordable_indices]]
        split_indices = affordable_indices[is_split[affordable_indices]]

        n_dupli = len(dupli_indices)
        n_split = len(split_indices)

        if n_dupli == 0 and n_split == 0:
            return 0, 0

        # Create final masks for operations
        if n_dupli > 0:
            dupli_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
            dupli_mask[dupli_indices] = True
            duplicate(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=dupli_mask,
            )

        if n_split > 0:
            split_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
            split_mask[split_indices] = True

            # Extend mask if needed after duplication
            if n_dupli > 0:
                split_mask = torch.cat([
                    split_mask,
                    torch.zeros(n_dupli, dtype=torch.bool, device=device)
                ], dim=0)

            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=split_mask,
                revised_opacity=self.revised_opacity,
            )

        return n_dupli, n_split

    def prune_mask(
            self,
            params: Dict[str, torch.nn.Parameter],
            optimizers: Dict[str, torch.optim.Optimizer],
            state: Dict[str, Any],
            mask,
    ):
        remove(params=params, optimizers=optimizers, state=state, mask=mask)

    @torch.no_grad()
    def _prune_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> int:
        is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return n_prune

    @torch.no_grad()
    def _relocate_gs(
            self,
            params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
            optimizers: Dict[str, torch.optim.Optimizer],
            state: Dict[str, Any],
            N: int,
            binoms: torch.Tensor,
    ) -> int:
        importance: torch.Tensor = state["importance"]

        # Elements eligible for relocation: 0 < importance < self.prune_opa
        candidate_mask = (importance > 0) & (importance < self.prune_opa)
        candidate_indices = torch.nonzero(candidate_mask, as_tuple=True)[0]

        if candidate_indices.numel() == 0:
            return 0

        # Sort by ascending importance so we relocate the *least* important first
        sorted_indices = candidate_indices[torch.argsort(importance[candidate_mask])]

        opacities = torch.sigmoid(params["opacities"].flatten())
        dead_mask = torch.zeros_like(opacities, dtype=torch.bool)
        dead_mask[sorted_indices] = True

        min_opacity = opacities[sorted_indices].max().item()

        relocate(
            params=params,
            optimizers=optimizers,
            state=state,
            mask=dead_mask,
            binoms=binoms,
            min_opacity=min_opacity,
        )

        return sorted_indices.numel()
