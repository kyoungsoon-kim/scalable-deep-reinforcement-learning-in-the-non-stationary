"""
Scalable DRL in the non-stationary SCLSP — the decomposed MDP environment.

Paper: Van Hezewijk, Dellaert, Van Jaarsveld (2025), IJPE 284:109601.
Implements the paper's CORE CONTRIBUTION: the new MDP formulation that decomposes
the full-period production decision into a sequence of sub-decisions (§3), turning an
action space that grows EXPONENTIALLY in the number of products into one that grows
linearly (§1, third challenge; §4.4).

  Educational note — the decomposition idea (§3.1):
  A period t is split into small time units. At each sub-step we make a *first-stage*
  sub-decision (which product to set up / produce next, or stop) and then a
  *second-stage* sub-decision (how much of it to produce). We repeat until capacity is
  used up or we choose to stop. Within a period transitions are DETERMINISTIC (§3.4);
  the only stochasticity is the demand observed at the END of the period.

Section map:
  §3.2 Eq.1 — state vector  s_τ = {I_{i,t}, μ_{i,t}, σ_{i,t}, ω_i, τ, q_{i,t}} ∀i∈K
  §3.3 Eq.2 — first-stage action  A1_τ ∈ {p0, p1, …, pK}   (p0 = stop / go idle)
  §3.3 Eq.3 — second-stage action A2_τ ∈ {q1, …, q_{C−τ}}
  §3.4      — transitions; setup cost charged in the period the setup is STARTED
  §3.5 Eq.4 — period cost = Σ_i (h_i·[I_i]^+ + b_i·[I_i]^- + k_i·z_{i,t})  [reconstructed]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from utils import ActionSpace


class Stage(IntEnum):
    FIRST = 0   # choose which product to produce next (or stop)
    SECOND = 1  # choose the quantity for the product just selected


@dataclass
class State:
    """MDP state (§3.2, Eq.1) plus bookkeeping the decomposed env needs.

    The paper's Eq.1 state is {I, μ, σ, ω, τ, q}. We additionally track which
    products were already produced this period (to forbid producing the same product
    twice, §3.3) and the remaining-periods counter used only in the finite
    non-stationary MDP (§4.3).
    """
    I: np.ndarray          # (K,) inventory position; negative = backorders
    mu: np.ndarray         # (K,) forecast mean demand μ_{i,t}
    sigma: np.ndarray      # (K,) forecast std σ_{i,t}
    omega: int             # index of product the machine is set up for (-1 = none)
    tau: int               # capacity (time units) already used this period
    q: np.ndarray          # (K,) units already produced this period
    produced_mask: np.ndarray  # (K,) bool — product already chosen this period
    stage: Stage
    pending_product: int   # product chosen at first stage, awaiting its quantity
    setups_this_period: int  # new setups z performed this period (for Z_max rule, Eq.4)
    t: int                 # period index
    remaining_periods: int


class SCLSPEnv:
    """Decomposed stochastic capacitated lot-sizing MDP (§3).

    One *environment step* = one sub-decision (first- OR second-stage), NOT one
    period. A period closes when the agent chooses p0 (stop) or capacity runs out;
    demand is then observed and holding/backorder costs are charged.

    Rewards are returned as NEGATIVE costs (RL maximizes reward = minimizes cost):
      - first-stage setup of a new product i:  reward -= k_i           (§3.4)
      - period close:                          reward -= Σ_i(h·[I]^+ + b·[I]^-)  (Eq.4)
    """

    def __init__(self, config: dict, action_space: ActionSpace):
        p = config["problem"]
        self.K: int = p["K"]
        self.h: float = p["holding_cost"]
        self.b: float = p["backorder_cost"]
        self.k: float = p["setup_cost"]
        self.theta: int = int(p["setup_time"])
        self.horizon: int = int(p["horizon"])
        self.aspace = action_space
        self.quantities = np.array(action_space.quantities)

        # Capacity C = f_c · Σ_i μ_{i,0}  (§5.2.2). Computed at reset once the initial
        # forecast is known, since μ_{i,0} comes from the demand process.
        self.capacity_factor: float = p["capacity_factor"]
        self.C: int = 0  # set in reset()

        self._demand = None  # injected DemandProcess (set via attach_demand)
        self._initial_inventory_rng_done = False

    def attach_demand(self, demand_process) -> None:
        """Inject the DGP (src/demand.py). Kept separate so the same env can be run
        against stationary or non-stationary demand."""
        self._demand = demand_process

    # ------------------------------------------------------------------ reset
    def reset(self, rng: np.random.Generator) -> State:
        assert self._demand is not None, "call attach_demand() before reset()"
        forecast = self._demand.reset(rng)
        # Capacity from the INITIAL mean demand (§5.2.2): C = f_c · Σ_i μ_{i,0}.
        self.C = int(round(self.capacity_factor * float(np.sum(forecast.mu))))
        self.C = max(self.C, 1)
        self.state = State(
            I=np.zeros(self.K),                # [UNSPECIFIED] initial inventory not stated; start at 0
            mu=forecast.mu.copy(),
            sigma=forecast.sigma.copy(),
            omega=-1,                          # machine not yet set up for any product
            tau=0,
            q=np.zeros(self.K),
            produced_mask=np.zeros(self.K, dtype=bool),
            stage=Stage.FIRST,
            pending_product=-1,
            setups_this_period=0,
            t=0,
            remaining_periods=self.horizon,
        )
        return self.state

    # ------------------------------------------------------------ legal actions
    def legal_action_mask(self) -> np.ndarray:
        """Boolean mask over the unified action space (§4.4 — "part of action space
        is masked out" depending on stage, remaining capacity C−τ, and set-up product).
        """
        s = self.state
        mask = np.zeros(self.aspace.size, dtype=bool)
        remaining = self.C - s.tau

        if s.stage == Stage.FIRST:
            # p0 (stop) is always available — the agent may idle the machine (§3.3).
            mask[self.aspace.stop_index()] = True
            if remaining > 0:
                for i in range(self.K):
                    if s.produced_mask[i]:
                        continue  # §3.3 — cannot select the same product twice in a period
                    # need room for a setup if switching; if i == omega no setup needed
                    setup_units = 0 if i == s.omega else self.theta
                    if remaining - setup_units >= 1:  # room to produce ≥1 unit after setup
                        mask[self.aspace.product_index(i)] = True
        else:  # SECOND stage — pick a quantity ≤ remaining capacity (Eq.3)
            for slot, qv in enumerate(self.aspace.quantities):
                if qv <= remaining:
                    mask[self.aspace.quantity_index(slot)] = True
            if not mask.any():
                # No quantity fits (capacity consumed by the setup). Allow the smallest
                # so the sub-decision can complete; step() clips to remaining.
                mask[self.aspace.quantity_index(0)] = True
        return mask

    # ------------------------------------------------------------------- step
    def step(self, action_idx: int, rng: np.random.Generator) -> tuple[State, float, bool, dict]:
        """Apply one sub-decision. `rng` drives the end-of-period demand draw and is
        passed explicitly to support Common Random Numbers in DCL (§4.1)."""
        s = self.state
        reward = 0.0
        info: dict = {}

        if s.stage == Stage.FIRST:
            if action_idx == self.aspace.stop_index():
                # p0 — stop production; close the period now.
                reward += self._close_period(rng, info)
            else:
                i = action_idx - 1  # product index (Eq.2 ordering p1..pK -> 0..K-1)
                assert 0 <= i < self.K, f"illegal first-stage action {action_idx}"
                if i != s.omega:
                    # New setup: charge k_i (§3.4) and consume θ_i time units now.
                    reward -= self.k
                    s.omega = i
                    s.tau += self.theta
                    s.setups_this_period += 1
                    # z_{i,t}=1 for this period (used by Eq.4 setup term + Z_max rule).
                    s.produced_mask[i] = s.produced_mask[i]  # no-op clarity
                    info["setup"] = i
                    if s.tau > self.C:
                        # Incomplete setup: setup time overruns the period. §3.4 — cost is
                        # still incurred in the period the setup starts. We carry the
                        # overflow into next period's τ and close now (no production fits).
                        # [SIMPLIFICATION] carry-over of partial setup time; flagged.
                        carry = s.tau - self.C
                        reward += self._close_period(rng, info, carry_tau=carry)
                        return self.state, reward, self._done(), info
                s.pending_product = i
                s.stage = Stage.SECOND

        else:  # SECOND stage — produce a quantity of pending_product
            slot = action_idx - self.aspace.n_first
            qty = int(self.quantities[slot])
            qty = min(qty, self.C - s.tau)  # clip to remaining capacity (Eq.3)
            i = s.pending_product
            if qty > 0:
                s.q[i] += qty
                s.I[i] += qty            # produced units go into inventory immediately
                s.tau += qty             # 1 unit consumes 1 time unit of capacity (§3.1)
                s.produced_mask[i] = True
            else:
                # Could not produce anything; mark product done to avoid relooping.
                s.produced_mask[i] = True
            s.pending_product = -1
            s.stage = Stage.FIRST
            # If capacity is now exhausted, the period closes automatically (§3.1).
            if self.C - s.tau < 1 or s.produced_mask.all():
                reward += self._close_period(rng, info)

        return self.state, reward, self._done(), info

    # --------------------------------------------------------- period closing
    def _close_period(self, rng: np.random.Generator, info: dict, carry_tau: int = 0) -> float:
        """Observe demand, net it against inventory, charge holding/backorder costs
        (Eq.4 holding+backorder terms), then advance the forecast (§3.4)."""
        s = self.state
        demand, next_forecast = self._demand.step(rng)
        s.I = s.I - demand  # net demand from inventory; negative => backorders

        # Eq.4 (holding + backorder part); setup term was charged at setup time.
        holding = self.h * np.maximum(s.I, 0.0).sum()
        backorder = self.b * np.maximum(-s.I, 0.0).sum()
        period_cost = holding + backorder
        info["period_cost"] = period_cost + 0.0
        info["demand"] = demand

        # advance to next period
        s.mu = next_forecast.mu.copy()
        s.sigma = next_forecast.sigma.copy()
        s.tau = int(carry_tau)             # carry-over partial setup time, else 0
        s.q = np.zeros(self.K)
        s.produced_mask = np.zeros(self.K, dtype=bool)
        s.setups_this_period = 0
        s.stage = Stage.FIRST
        s.pending_product = -1
        s.t += 1
        s.remaining_periods = max(self.horizon - s.t, 0)
        # ω (which product is set up) PERSISTS across the period boundary (§3.3):
        # "the machine can continue producing the product produced last ... without a setup".
        return -period_cost

    def _done(self) -> bool:
        # Finite horizon for the non-stationary MDP (§4.3). For stationary (infinite
        # horizon) the caller simply runs a fixed number of periods.
        return self.state.t >= self.horizon
