# Paper Guide — section by section

A learning-oriented walk through *Scalable deep reinforcement learning in the
non-stationary capacitated lot sizing problem* (Van Hezewijk et al., 2025), tying each
section to the code in `src/`. Read alongside `notebooks/walkthrough.ipynb`.

---

## The problem in one picture (§1, §3.1)

One machine, limited time capacity `C` per period, makes `K` products. Each period you
decide how much of each product to produce; switching products costs a **setup** (time
`θ_i` + money `k_i`). At period end demand `d_{i,t}` arrives; leftover stock costs `h_i`
per unit (**holding**), unmet demand costs `b_i` per unit (**backorder**). Demand is
**stochastic and non-stationary** (its mean drifts over time). Goal: a production policy
that minimizes long-run average cost.

Why it is hard (the paper's three challenges, §1):
1. **Data** — you cannot learn from real history; you must simulate. → needs a realistic
   demand model (`src/demand.py`).
2. **Uncertainty** — high variance makes good vs near-good actions hard to tell apart.
   → DCL's variance control (`src/dcl.py`).
3. **Action explosion** — choosing quantities for `K` products at once is exponential.
   → the MDP decomposition (`src/env.py`). **This is the headline contribution.**

---

## §3 — The decomposed MDP  (`src/env.py`)

### Background: what is an MDP?
A Markov Decision Process is `(state, action, transition, reward)` where the future
depends only on the current state. RL searches for a policy `π(state) → action` that
minimizes expected cost.

### The key trick (§3.1)
Instead of one giant per-period action `(q_1,…,q_K)` (exponential), split a period into
small time units and make a sequence of **two-stage sub-decisions**:
- **1st stage** `A1` (Eq.2): which product to set up/produce next, or `p0` = stop.
- **2nd stage** `A2` (Eq.3): how many units of that product (≤ remaining capacity).

Repeat until you stop or run out of capacity. Now each decision is small and the action
space grows **linearly** in `K`. Within a period everything is deterministic; the only
randomness is the end-of-period demand (§3.4).

### State (§3.2, Eq.1)
`s_τ = {I_{i,t}, μ_{i,t}, σ_{i,t}, ω_i, τ, q_{i,t}}` — inventory, demand forecast
(mean+std), which product is currently set up (`ω`), capacity used so far (`τ`), and
units already produced this period. Carrying the **forecast** in the state is what lets
one trained policy keep working as demand changes — no retraining (§4.3, §6).

### Reward (§3.5, Eq.4)
Period cost `= Σ_i ( h_i·[I_i]^+ + b_i·[I_i]^- + k_i·z_{i,t} )`. In code, holding+backorder
are charged at `_close_period`, setup at the moment of setup. *(Eq.4 was reconstructed —
see REPRODUCTION_NOTES.)*

---

## §4.2 — The AMBS rollout policy  (`src/rollout.py`, Algorithm 1)

### Background: base-stock / order-up-to
A classic inventory rule: keep a **reorder level** (produce when stock drops below it)
and an **order-up-to level** (how high to refill). AMBS adapts this to a shared-capacity
multi-item machine.

### Algorithm 1 in words
Each sub-step: compute every product's gap to its reorder level
`gs_i = max(μ_i + B_min·σ_i − I_i, 0)`. Produce the largest-gap product (if a setup is
still allowed, `Z_max`); else keep refilling the currently set-up product if it is below
its reorder level; else stop. Quantity = order-up-to target snapped to the restricted grid.

This policy is both the **benchmark** to beat and DCL's **generation-1 rollout policy**.

---

## §4.3–4.4 — The policy network  (`src/policy.py`)

A plain **dense MLP** (Table 2: `[128,128]` for K=5). One output per action over the
**unified action space** `A = A1 ∪ A2*` (Eq.7); invalid actions are **masked** to −∞
depending on stage/capacity/setup (§4.4). One representation trick (§4.3): put the
**currently set-up product's features first** in the input vector, giving the net a
consistent slot for the most decision-relevant product. The restricted quantity grid
(Eq.6) keeps fine resolution near 0 and coarsens for large quantities.

---

## §4.1 — Deep Controlled Learning  (`src/dcl.py`)

### Background: why not PPO?
§4.1 shows PPO fails here: with one env-step per sub-decision, the agent learns to stall
the period (7.4 actions/period, 3.6× worse than AMBS) because costs only land at period
end. So the authors use **DCL** instead.

### How DCL learns (supervised, not policy-gradient)
1. **Sample** `N` states by following the current rollout policy.
2. For each state, **roll out** every candidate action `M` times for `H` periods and
   estimate its expected cost; **label** the state with the cheapest action (argmin).
3. **Train** the MLP to *classify* that best action (cross-entropy).
4. Repeat for several **generations**; from gen 2 the trained net is the rollout policy.

### The two variance tricks (the whole point, §4.1)
- **Common Random Numbers**: evaluate all actions of a state against the *same* demand
  draws, so cost differences are signal not noise. (`use_crn`, shared RNG seeds.)
- **Sequential Halving**: spend the budget `M·|A_s|` adaptively — after `r` rollouts drop
  the worst half of actions, recurse. (`use_sequential_halving`.)

---

## §5 — Evaluation  (`src/evaluate.py`)

Simulate each policy over 10,000 runs × 100 periods on **common demand sequences**,
report **average cost per period**, and `Δ = (policy − AMBS)/AMBS`. Headline result:
DCL beats AMBS for up to 15 products (Δ negative, Tables 3/5); for ≥20 products it only
ties AMBS — the dense net struggles to scale (§5.1, §6).

---

## Further reading
- Temizöz et al. (2023) — *Deep Controlled Learning for inventory control* (the DCL method).
- van Hezewijk et al. (2023a) — the non-stationary demand process used here.
- van Hezewijk et al. (2023b) — the prior PPO / full-period AMBS formulation.
- Boute et al. (2021) — *DRL for inventory control: a roadmap* (gentle survey intro).
