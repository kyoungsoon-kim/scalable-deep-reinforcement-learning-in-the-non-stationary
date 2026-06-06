"""
Scalable DRL in the non-stationary SCLSP — Deep Controlled Learning (DCL) training.

Paper: Van Hezewijk, Dellaert, Van Jaarsveld (2025), IJPE 284:109601.
Implements §4.1: the DCL approximate-policy-improvement loop. DCL learns by
SUPERVISED CLASSIFICATION of the best action, not by policy gradient:

  For each of N sampled states, roll out every candidate action M times for H periods
  to estimate its action-value (expected cost), label the state with the argmin-cost
  action, then train a network to classify it. Repeat for several GENERATIONS; from
  generation 2 on, the trained network becomes the rollout policy (§4.1).

  Two variance-control devices from §4.1 are the whole reason DCL works in a stochastic
  environment:
    • Common Random Numbers (CRN): the SAME demand draws evaluate the different actions
      of a state, so their cost differences are signal, not noise.
    • Sequential Halving (SH): the per-state rollout budget M·|A_s| is spent adaptively —
      after r rollouts the worst half of actions is dropped, concentrating budget on
      promising actions.

  --------------------------------------------------------------------------------
  [REFERENCE] DCL internals (network training schedule, exact state-sampling
  distribution) follow Temizöz et al. (2023) and the DynaPlex library (Akkerman et al.
  2023), which are NOT reproduced in this paper. The loop below is a faithful
  standalone implementation of the prose; NN-training hyperparameters (lr, batch size,
  epochs) are [UNSPECIFIED] and flagged in the config.
  --------------------------------------------------------------------------------
"""
from __future__ import annotations

import copy
import math
from typing import List, Tuple

import numpy as np

from env import SCLSPEnv
from policy import NeuralPolicy, build_policy_net, encode_state
from rollout import AMBSRolloutPolicy
from utils import ActionSpace


# ----------------------------------------------------------------- rollout cost
def _rollout_cost(env: SCLSPEnv, first_action: int, policy, H: int,
                  rng: np.random.Generator) -> float:
    """Apply `first_action`, then follow `policy` for H periods; return total cost.

    Cost = −Σ reward (env returns reward = −cost). `rng` supplies the demand draws;
    seeding it identically across actions realizes Common Random Numbers (§4.1).
    """
    total = 0.0
    _, r, done, _ = env.step(first_action, rng)
    total += -r
    start_t = env.state.t
    while not done and (env.state.t - start_t) < H:
        a = policy.act(env)
        _, r, done, _ = env.step(a, rng)
        total += -r
    return total


# ------------------------------------------------------- per-state best action
def estimate_best_action(
    env_snapshot: SCLSPEnv,
    valid_actions: List[int],
    policy,
    M: int,
    H: int,
    crn_seed: int,
    use_sequential_halving: bool,
    use_crn: bool,
) -> int:
    """Estimate action-values for one state and return the argmin-cost action (§4.1)."""
    if len(valid_actions) == 1:
        return valid_actions[0]

    def eval_actions(actions: List[int], r: int, seed_offset: int) -> dict[int, float]:
        """Mean cost of `r` rollouts for each action. With CRN, rollout m of every
        action shares the same demand seed (crn_seed + seed_offset + m)."""
        means = {}
        for a in actions:
            costs = []
            for m in range(r):
                # CRN: identical seed across actions for the same rollout index m.
                seed = crn_seed + seed_offset + m if use_crn else None
                rng = np.random.default_rng(seed)
                env_copy = copy.deepcopy(env_snapshot)
                costs.append(_rollout_cost(env_copy, a, policy, H, rng))
            means[a] = float(np.mean(costs))
        return means

    if not use_sequential_halving:
        # Uniform allocation: M rollouts per action, pick the cheapest.
        means = eval_actions(valid_actions, M, 0)
        return min(means, key=means.get)

    # Sequential Halving (§4.1). Total budget = M·|A_s|, split over ceil(log2 n) rounds.
    actions = list(valid_actions)
    total_budget = M * len(actions)
    n_rounds = max(1, math.ceil(math.log2(len(actions))))
    per_round_budget = total_budget / n_rounds
    seed_offset = 0
    while len(actions) > 1:
        r = max(1, int(per_round_budget // len(actions)))
        means = eval_actions(actions, r, seed_offset)
        seed_offset += r
        # keep the best (lowest-cost) half
        ranked = sorted(actions, key=means.get)
        keep = max(1, len(actions) // 2)
        actions = ranked[:keep]
    return actions[0]


# ----------------------------------------------------------- state sampling
def sample_states(env: SCLSPEnv, policy, n_states: int,
                  rng: np.random.Generator) -> List[SCLSPEnv]:
    """Roll the current rollout policy through episodes, snapshotting env states at
    each sub-decision until `n_states` are collected (§4.1 — "constructed ... by
    following the rollout policy")."""
    snapshots: List[SCLSPEnv] = []
    while len(snapshots) < n_states:
        env.reset(rng)
        done = False
        while not done and len(snapshots) < n_states:
            snapshots.append(copy.deepcopy(env))  # snapshot BEFORE acting
            a = policy.act(env)
            _, _, done, _ = env.step(a, rng)
    return snapshots


# ------------------------------------------------------------- supervised train
def train_network(net, dataset: List[Tuple[np.ndarray, np.ndarray, int]], config: dict):
    """Train the classifier on (state_vec, action_mask, best_action) triples (§4.1)."""
    import torch
    import torch.nn as nn

    d = config["dcl"]
    opt = torch.optim.Adam(net.parameters(), lr=d["lr"])  # [PARTIALLY_SPECIFIED] §4.1
    loss_fn = nn.CrossEntropyLoss()  # §4.1 — classify the best action (supervised)

    X = torch.tensor(np.stack([s for s, _, _ in dataset]), dtype=torch.float32)
    Masks = torch.tensor(np.stack([m for _, m, _ in dataset]), dtype=torch.bool)
    Y = torch.tensor([a for _, _, a in dataset], dtype=torch.long)

    n = len(dataset)
    bs = d["batch_size"]
    for _ in range(d["epochs_per_generation"]):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            logits = net.masked_logits(X[idx], Masks[idx])  # (b, |A|) invalid -> -inf
            loss = loss_fn(logits, Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net


# --------------------------------------------------------------- DCL driver
def train_dcl(config: dict, env: SCLSPEnv, action_space: ActionSpace, verbose: bool = True):
    """Full DCL loop over generations (§4.1, Table 2). Returns the trained net."""
    d = config["dcl"]
    if d["use_paper_scale"]:
        N, M, H = d["N"], d["M"], d["H"]            # Table 2 (requires a cluster)
    else:
        N, M, H = d["N_demo"], d["M_demo"], d["H_demo"]  # laptop-runnable

    rng = np.random.default_rng(config["seed"])
    net = build_policy_net(config, action_space)

    # Generation 1 uses the AMBS-based rollout policy (§4.2); later generations use the net.
    rollout_policy = AMBSRolloutPolicy(config, action_space)

    for gen in range(d["generations"]):
        if verbose:
            print(f"[DCL] generation {gen + 1}/{d['generations']} — sampling {N} states")
        snapshots = sample_states(env, rollout_policy, N, rng)

        dataset = []
        for j, snap in enumerate(snapshots):
            valid = [a for a, ok in enumerate(snap.legal_action_mask()) if ok]
            best = estimate_best_action(
                snap, valid, rollout_policy, M, H,
                crn_seed=int(rng.integers(0, 2**31)),
                use_sequential_halving=d["sequential_halving"],
                use_crn=d["common_random_numbers"],
            )
            dataset.append((encode_state(snap.state, snap),
                            snap.legal_action_mask().astype(np.float32),
                            best))
        if verbose:
            print(f"[DCL]   labeled {len(dataset)} states; training network")
        net = train_network(net, dataset, config)
        # From generation 2 on, the trained network becomes the rollout policy (§4.1).
        rollout_policy = NeuralPolicy(net, action_space)

    return net
