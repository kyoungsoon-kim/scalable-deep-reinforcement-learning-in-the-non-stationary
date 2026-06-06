"""
Scalable DRL in the non-stationary SCLSP — evaluation / simulation harness.

Paper: Van Hezewijk, Dellaert, Van Jaarsveld (2025), IJPE 284:109601.
Implements the evaluation protocol of §5.1:
  "The performance ... is determined by using the best trained network as a policy in
   10,000 simulation runs with a length of 100 periods per run. This is compared with
   the performance of the AMBS heuristic and rollout policy. All policies are
   encountering the same demand sequences ... to enable a fair comparison."
and the reported metric: average cost per period, with Δ = (policy − AMBS) / AMBS
(Tables 3 & 5; negative Δ means the policy beats AMBS).

  Common demand sequences across policies are realized by re-seeding the RNG with the
  same per-run seed for every policy: demand is drawn once per period close, and every
  policy closes exactly `run_length` periods, so the draws line up (a coarse form of
  Common Random Numbers at evaluation time, §5.1).
"""
from __future__ import annotations

import argparse

import numpy as np

from demand import build_demand_process
from env import SCLSPEnv
from rollout import AMBSRolloutPolicy
from utils import build_action_space, load_config, set_global_seed


def simulate_policy(env: SCLSPEnv, demand_proc, policy,
                    n_runs: int, run_length: int, base_seed: int) -> float:
    """Average cost per period of `policy` over `n_runs` independent runs (§5.1)."""
    total_cost = 0.0
    for run in range(n_runs):
        # Same seed per run index across policies => same demand sequence (fair compare).
        rng = np.random.default_rng(base_seed + run)
        env.attach_demand(demand_proc)
        env.reset(rng)
        done = False
        while not done and env.state.t < run_length:
            a = policy.act(env)
            _, r, done, _ = env.step(a, rng)
            total_cost += -r  # reward = −cost
    return total_cost / (n_runs * run_length)


def delta_vs_ambs(policy_cost: float, ambs_cost: float) -> float:
    """Δ as reported in Tables 3/5: relative gap to the AMBS heuristic."""
    return (policy_cost - ambs_cost) / ambs_cost


def evaluate(config: dict, policies: dict, demo: bool = True) -> dict:
    """Evaluate a dict {name: policy} and return {name: (avg_cost, Δ_vs_AMBS)}."""
    e = config["evaluation"]
    n_runs = e["n_runs_demo"] if demo else e["n_runs"]   # §5.1 — 10,000 at paper scale
    run_length = e["run_length"]                          # §5.1 — 100 periods
    aspace = build_action_space(config)

    results = {}
    costs = {}
    for name, policy in policies.items():
        env = SCLSPEnv(config, aspace)
        demand_proc = build_demand_process(config)
        costs[name] = simulate_policy(env, demand_proc, policy, n_runs, run_length,
                                      base_seed=config["seed"] + 10_000)
    ambs_cost = costs.get("AMBS", next(iter(costs.values())))
    for name, c in costs.items():
        results[name] = (c, delta_vs_ambs(c, ambs_cost))
    return results


def main():
    ap = argparse.ArgumentParser(description="Evaluate AMBS (and optionally a DCL policy).")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--train-dcl", action="store_true",
                    help="train a DCL policy (demo scale) and compare it to AMBS")
    args = ap.parse_args()

    config = load_config(args.config)
    set_global_seed(config["seed"])
    aspace = build_action_space(config)

    policies = {"AMBS": AMBSRolloutPolicy(config, aspace)}

    if args.train_dcl:
        # Imported here so the AMBS-only path needs no torch.
        from dcl import train_dcl
        from policy import NeuralPolicy
        env = SCLSPEnv(config, aspace)
        env.attach_demand(build_demand_process(config))
        net = train_dcl(config, env, aspace)
        policies["DCL"] = NeuralPolicy(net, aspace)

    results = evaluate(config, policies, demo=not config["dcl"]["use_paper_scale"])
    print("\n=== Average cost per period (lower is better) ===")
    for name, (cost, delta) in results.items():
        tag = "" if name == "AMBS" else f"   Δ_vs_AMBS = {delta:+.1%}"
        print(f"  {name:6s}: {cost:8.3f}{tag}")
    print("\nPaper targets (non-stationary, Table 5): DCL beats AMBS by ~3–14% "
          "(Δ negative). Demo-scale numbers will be noisier; see REPRODUCTION_NOTES.")


if __name__ == "__main__":
    main()
