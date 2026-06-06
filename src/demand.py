"""
Scalable DRL in the non-stationary SCLSP — demand generating process (DGP).

Paper: Van Hezewijk, Dellaert, Van Jaarsveld (2025), IJPE 284:109601.
Implements: the demand processes that drive the stochastic period-to-period
transitions of the MDP (§3.4) and supply the forecast (μ_{i,t}, σ_{i,t}) that the
state carries (§3.2).

  Educational note — why the DGP matters:
  In a simulation-trained RL method, the demand model IS the data. The whole point
  of this paper (§1, third challenge handling) is that the policy is trained on a
  *realistic* auto-regressive demand model so it copes with forecast error and
  auto-correlation, instead of i.i.d. draws with known parameters.

  --------------------------------------------------------------------------------
  [PARTIALLY_SPECIFIED] The exact non-stationary DGP is van Hezewijk et al. (2023a),
  "A new discrete non-stationary demand process with applications in inventory
  control" — it is NOT defined in this paper. This paper only describes it (§1, §6):
    "auto-regressive ... discrete, non-negative ... otherwise very closely related
     to the celebrated ARIMA(0,1,1) process."
  and parameterizes it by initial mean μ_0, initial std σ_0, smoothing α (§5.2).

  We therefore implement an IMA(0,1,1)-style approximation below and FLAG it. Swap
  in the exact 2023a process for faithful reproduction. See REPRODUCTION_NOTES.md.
  --------------------------------------------------------------------------------
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Forecast:
    """Per-product forecast exposed to the MDP state (§3.2): mean and std of demand."""
    mu: np.ndarray   # shape (K,)
    sigma: np.ndarray  # shape (K,)


class DemandProcess:
    """Base interface: a per-period generator of (observed demand, next forecast)."""

    def reset(self, rng: np.random.Generator) -> Forecast:
        raise NotImplementedError

    def step(self, rng: np.random.Generator) -> tuple[np.ndarray, Forecast]:
        """Draw this period's demand and advance to the next period's forecast.

        Returns (demand_t shape (K,), forecast for t+1).
        `rng` is passed in EXPLICITLY so Deep Controlled Learning can share random
        numbers across the actions evaluated from one state (Common Random Numbers,
        §4.1) — two rollouts seeded identically see identical demand sequences.
        """
        raise NotImplementedError


class StationaryUniformDemand(DemandProcess):
    """§5.1 — stationary discrete uniform demand U{low, high} (e.g. U{3,5}, U{0,8}).

    Used for the PPO-comparison experiments. Forecast is constant: the mean and std
    of the uniform distribution.
    """

    def __init__(self, K: int, low: int, high: int):
        self.K = K
        self.low = low
        self.high = high
        # Mean and std of a discrete uniform on {low,...,high} (inclusive).
        n = high - low + 1
        self._mu = (low + high) / 2.0
        self._sigma = np.sqrt((n * n - 1) / 12.0)

    def _forecast(self) -> Forecast:
        return Forecast(
            mu=np.full(self.K, self._mu),
            sigma=np.full(self.K, self._sigma),
        )

    def reset(self, rng: np.random.Generator) -> Forecast:
        return self._forecast()

    def step(self, rng: np.random.Generator) -> tuple[np.ndarray, Forecast]:
        d = rng.integers(self.low, self.high + 1, size=self.K)  # inclusive high
        return d.astype(float), self._forecast()


class NonStationaryIMADemand(DemandProcess):
    """IMA(0,1,1)-style discrete non-negative non-stationary DGP — APPROXIMATION.

    [PARTIALLY_SPECIFIED] Stands in for van Hezewijk et al. (2023a). Parameters from
    §5.2.2 / Table 4: μ_0 = 2, COV ∈ {0.5,1.0} (so σ_0 = COV·μ_0), α ∈ {0,0.025,0.05}.

    Model (one of several IMA(0,1,1) parameterizations; flagged):
      level_{i,t+1} = max(level_{i,t} + α · ε_{i,t}, 0)      # smoothed random walk of the mean
      d_{i,t}       = round( max( level_{i,t} + ε_{i,t}, 0 ) )  # integer, non-negative observation
      ε_{i,t} ~ Normal(0, σ_0)
    α = 0 collapses to a stationary mean (pure noise around μ_0), matching §5.2.2
    ("Degree of non-stationarity α"). Forecast at t = (level_{i,t}, σ_0).

      Why this shape: ARIMA(0,1,1) = "integrated MA(1)"; its level performs a random
      walk whose increment is a fraction (here α) of the previous shock. Truncating
      at 0 and rounding yields the "discrete, non-negative" property the paper needs.
    """

    def __init__(self, K: int, mu0: float, cov0: float, alpha: float):
        self.K = K
        self.mu0 = mu0
        self.sigma0 = cov0 * mu0  # §5.2.2 — COV = σ/μ  ⇒  σ_0 = COV·μ_0
        self.alpha = alpha
        self.level = np.full(K, mu0, dtype=float)
        self._last_eps = np.zeros(K)

    def _forecast(self) -> Forecast:
        # Forecast mean = current level; forecast std held at σ_0 (initial COV basis).
        # [UNSPECIFIED] how σ_{i,t} evolves over time in 2023a — we hold it constant.
        return Forecast(mu=self.level.copy(), sigma=np.full(self.K, self.sigma0))

    def reset(self, rng: np.random.Generator) -> Forecast:
        self.level = np.full(self.K, self.mu0, dtype=float)
        self._last_eps = np.zeros(self.K)
        return self._forecast()

    def step(self, rng: np.random.Generator) -> tuple[np.ndarray, Forecast]:
        eps = rng.normal(0.0, self.sigma0, size=self.K)
        d = np.round(np.maximum(self.level + eps, 0.0))
        # advance the level (integrated MA(1) with smoothing α)
        self.level = np.maximum(self.level + self.alpha * eps, 0.0)
        self._last_eps = eps
        return d.astype(float), self._forecast()


def build_demand_process(config: dict) -> DemandProcess:
    """Factory selecting the DGP from config['demand']['mode']."""
    d = config["demand"]
    K = config["problem"]["K"]
    if d["mode"] == "stationary":
        return StationaryUniformDemand(K, d["uniform_low"], d["uniform_high"])
    elif d["mode"] == "nonstationary":
        return NonStationaryIMADemand(K, d["mu0"], d["cov0"], d["alpha"])
    raise ValueError(f"Unknown demand mode: {d['mode']!r}")
