"""
Scalable DRL in the non-stationary SCLSP — AMBS-based rollout policy (Algorithm 1).

Paper: Van Hezewijk, Dellaert, Van Jaarsveld (2025), IJPE 284:109601.
Implements §4.2 / Algorithm 1: a sub-decision policy derived from the aggregate
modified base-stock (AMBS) heuristic. It serves two roles:
  1. the BENCHMARK heuristic the DCL policy must beat (§5);
  2. the GENERATION-1 rollout policy inside DCL (§4.1 — "DCL can benefit
     significantly from selecting an appropriate initial rollout policy").

  Educational note — what AMBS does:
  AMBS is a base-stock ("order-up-to") rule adapted to a shared-capacity multi-item
  machine. Each product has a reorder level (when to set up) and an order-up-to level
  (how much to make). The decomposed version (Algorithm 1) re-expresses the full-period
  rule as: pick the product furthest below its reorder level, make it up to its
  order-up-to level, repeat until a setup quota Z_max is hit or nothing needs producing.

  --------------------------------------------------------------------------------
  [PARTIALLY_SPECIFIED] The order-up-to / continue-condition lines of Algorithm 1 were
  extracted from a table image (s2orc) with reordered cells. The reconstructed quantity
  target uses an EOQ-style cycle stock + safety stock:
        target = μ_ω + sqrt(2·μ_ω·k_ω / h_ω) + H_max·B_min·σ_ω
  Verify against the original Algorithm 1 image before trusting reproduction numbers.
  [UNSPECIFIED] B_min, H_max, Z_max — "found by searching a grid" (§4.2); not reported.
  --------------------------------------------------------------------------------
"""
from __future__ import annotations

import math

import numpy as np

from env import SCLSPEnv, Stage
from utils import ActionSpace


class AMBSRolloutPolicy:
    """Stateless w.r.t. its own memory — it reads everything from the env state."""

    def __init__(self, config: dict, action_space: ActionSpace):
        r = config["rollout"]
        self.B_min: float = r["B_min"]   # safety-stock multiplier (reorder level)
        self.H_max: float = r["H_max"]   # order-up-to horizon multiplier
        self.Z_max: int = int(r["Z_max"])  # max new setups per period
        self.aspace = action_space

    # ------------------------------------------------------------------ helpers
    def _reorder_level(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Reorder level R_i = μ_i + B_min·σ_i (Algorithm 1, scaled-gap line)."""
        return mu + self.B_min * sigma

    # --------------------------------------------------------------- act
    def act(self, env: SCLSPEnv) -> int:
        """Return a flat action index for the current sub-decision (§4.2)."""
        s = env.state
        if s.stage == Stage.FIRST:
            return self._first_stage(env)
        return self._second_stage(env)

    def _first_stage(self, env: SCLSPEnv) -> int:
        s = env.state
        reorder = self._reorder_level(s.mu, s.sigma)
        # Scaled gap to the reorder level (Algorithm 1): gs_i = max(R_i − I_i, 0).
        gs = np.maximum(reorder - s.I, 0.0)
        gs[s.produced_mask] = 0.0  # §3.3 — a product already made this period is ineligible

        # 1) If any product has a positive gap and we may still set up: produce the
        #    one with the largest gap (Algorithm 1, first if-branch).
        if gs.max() > 0:
            i_star = int(np.argmax(gs))
            new_setup = (i_star != s.omega)
            if (not new_setup) or (s.setups_this_period < self.Z_max):
                return self.aspace.product_index(i_star)

        # 2) Else, if the currently set-up product is below its reorder level and not
        #    yet produced this period, continue it (no new setup needed).
        if (
            0 <= s.omega < env.K
            and not s.produced_mask[s.omega]
            and s.I[s.omega] < reorder[s.omega]
            and (env.C - s.tau) >= 1
        ):
            return self.aspace.product_index(s.omega)

        # 3) Otherwise stop production and idle the machine (Algorithm 1, else → p0).
        return self.aspace.stop_index()

    def _second_stage(self, env: SCLSPEnv) -> int:
        """Order-up-to quantity for the pending product, snapped to the restricted grid."""
        s = env.state
        i = s.pending_product
        mu_i, sigma_i = s.mu[i], s.sigma[i]
        # EOQ-style cycle stock sqrt(2·μ·k/h) + safety stock H_max·B_min·σ (reconstructed).
        cycle = math.sqrt(max(2.0 * mu_i * env.k / max(env.h, 1e-9), 0.0))
        target = mu_i + cycle + self.H_max * self.B_min * sigma_i
        desired = max(int(math.ceil(target - s.I[i])), 1)  # produce up to target
        remaining = env.C - s.tau
        desired = min(desired, remaining)

        # Snap to the largest restricted grid quantity ≤ desired (Eq.6).
        grid = env.quantities
        valid = grid[grid <= max(desired, 1)]
        if valid.size == 0:
            slot = 0  # smallest grid quantity; step() will clip to capacity
        else:
            slot = int(np.where(grid == valid.max())[0][0])
        return self.aspace.quantity_index(slot)


def rollout_action(policy, env: SCLSPEnv) -> int:
    """Tiny adapter so DCL can treat any policy with `.act(env)` uniformly."""
    return policy.act(env)
