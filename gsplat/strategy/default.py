from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch
from typing_extensions import Literal

from .base import Strategy
from .ops import duplicate, remove, reset_opa, split


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
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 0
    refine_start_iter: int = 1_500
    refine_stop_iter: int = 55_000
    max_budget: int = 4_000_000 # set -1 to disable.
    reset_every: int = 6000
    refine_every: int = 200
    pause_refine_after_reset: int = 0
    absgrad: bool = False
    revised_opacity: bool = False
    verbose: bool = False
    key_for_gradient: Literal["means2d", "gradient_2dgs"] = "means2d"
    p_init: int = 0
    last_p_fin: int = 0
    alpha_t: float = 1.0
    alpha_g: float = 0.2


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
        info: Dict[str, Any],
        packed: bool = False,
    ):
        """Callback function to be executed after the `loss.backward()` call."""
        if step >= self.refine_stop_iter:
            return

        self._update_state(params, state, info, packed=packed)

        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            and step % self.reset_every >= self.pause_refine_after_reset
        ):
            # grow GSs
            n_dupli, n_split = self._grow_gs(params, optimizers, state, step, factor)
            if self.verbose:
                print(
                    f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                    f"Now having {len(params['means'])} GSs."
                )

            # prune GSs
            n_prune = self._prune_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step}: {n_prune} GSs pruned. "
                    f"Now having {len(params['means'])} GSs."
                )

            # reset running stats
            state["grad2d"].zero_()
            state["count"].zero_()
            # Note: We don't zero importance!
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            torch.cuda.empty_cache()

        if step % self.reset_every == 0 and self.refine_start_iter <= step < self.refine_stop_iter:
            # Apply logit to current opacities
            opacities = torch.sigmoid(params["opacities"].detach())

            # Compute the quantile threshold in logit space
            threshold = torch.quantile(opacities, 2.0 * self.prune_opa)

            # Mask for pruning in logit space
            mask = opacities < threshold
            n_prune = mask.sum().item()

            # Prune
            if n_prune > 0:
                self.prune_mask(params, optimizers, state, mask)

            print(f"Pruning {n_prune} GSs with opacity below {threshold:.2f}.")
            #reset_opa(
            #        params=params,
            #        optimizers=optimizers,
            #        state=state,
            #        value=self.prune_opa * 2.0,
            #    )

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
            self.p_init = min(5 * n_gaussian, 4_000_000)
            self.last_p_fin = self.p_init
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
        gradient norms first.
        """
        if params["means"].shape[0] >= self.max_budget and self.max_budget != -1:
            return 0, 0  # No more GSs to grow.

        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        device = grads.device

        # Identify the typical thresholds/flags, same as before:
        is_grad_high = grads > self.grow_grad2d

        w_inv = 1.0 / torch.exp(params["w"]).unsqueeze(1)
        scales = torch.exp(params["scales"]) * w_inv * torch.norm(params["means"], dim=1).unsqueeze(1)
        is_small = (
                scales.max(dim=-1).values
                <= self.grow_scale3d * state["scene_scale"]
        )
        # "Large" => splitting candidate
        is_large = ~is_small

        # initial masks for duplication vs split
        is_dupli = is_grad_high & is_small
        is_split = is_grad_high & is_large

        # If we haven't yet stopped scale2d refining, also force-split those with bigger radii
        if step < self.refine_scale2d_stop_iter:
            is_split |= state["radii"] > self.grow_scale2d

        # Combine them to get "candidate_mask" for ANY type of growth
        candidate_mask = is_dupli | is_split
        candidate_idxs = candidate_mask.nonzero(as_tuple=False).flatten()

        if candidate_idxs.numel() == 0:
            return 0, 0  # no candidates => nothing to do

        # Sort candidates by descending gradient magnitude
        candidate_norms = grads[candidate_idxs]
        sorted_norms, sorted_rel_idx = torch.sort(candidate_norms, descending=True)
        sorted_candidate_idxs = candidate_idxs[sorted_rel_idx]

        # ------------------------------------------------------------------------
        #  BUDGET LOGIC
        # ------------------------------------------------------------------------
        p_fin = max(self.last_p_fin, 0.98 * self.last_p_fin + candidate_idxs.numel())
        self.last_p_fin = p_fin
        # (Your factor, p_init, etc. come from your original approach)
        current_count = params["means"].shape[0]  # e.g. how many GS we have
        budget_left = (self.p_init + (p_fin - self.p_init) / factor) - current_count

        split_cost = 2
        clone_cost = 1

        # We'll figure out which indices we choose to dupli vs. split
        chosen_dupli = []
        chosen_split = []

        for idx in sorted_candidate_idxs:
            if is_split[idx]:
                cost = split_cost
            else:
                cost = clone_cost

            # If we still have enough budget, choose this idx for that operation
            if budget_left >= cost:
                if is_split[idx]:
                    chosen_split.append(idx.item())
                else:
                    chosen_dupli.append(idx.item())
                budget_left -= cost
            else:
                # no more budget => break from loop
                break

        n_dupli = len(chosen_dupli)
        n_split = len(chosen_split)

        if n_dupli == 0 and n_split == 0:
            return 0, 0  # Not enough budget to do anything.

        # Build final boolean masks from the chosen idx lists
        final_dupli_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
        final_split_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
        if n_dupli > 0:
            final_dupli_mask[chosen_dupli] = True
        if n_split > 0:
            final_split_mask[chosen_split] = True

        if n_dupli > 0:
            duplicate(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=final_dupli_mask,
            )

        # Newly duplicated points are appended at the end, so we need to extend
        # final_split_mask with zeros for the new rows added by duplication:
        if n_dupli > 0:
            final_split_mask = torch.cat([
                final_split_mask,
                torch.zeros(n_dupli, dtype=torch.bool, device=device)
            ], dim=0)

        if n_split > 0:
            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=final_split_mask,
                revised_opacity=self.revised_opacity,
                alpha_t=self.alpha_t,
                alpha_g=self.alpha_g,
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
        #if step > self.reset_every:
        #    w_inv = 1.0 / torch.exp(params["w"]).unsqueeze(1)
        #    scales = torch.exp(params["scales"]) * w_inv * torch.norm(params["means"], dim=1).unsqueeze(1)
        #    is_too_big = (
        #            scales.max(dim=-1).values
        #            > self.prune_scale3d * state["scene_scale"]
        #    )
        #    # The official code also implements sreen-size pruning but
        #    # it's actually not being used due to a bug:
        #    # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
        #    # We implement it here for completeness but set `refine_scale2d_stop_iter`
        #    # to 0 by default to disable it.
        #    if step < self.refine_scale2d_stop_iter:
        #        is_too_big |= state["radii"] > self.prune_scale2d

        #    is_prune = is_prune | is_too_big

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return n_prune
