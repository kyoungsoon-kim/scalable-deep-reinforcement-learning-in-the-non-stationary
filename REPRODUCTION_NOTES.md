# Reproduction Notes

Implementation of **"Scalable deep reinforcement learning in the non-stationary
capacitated lot sizing problem"** — Van Hezewijk, Dellaert, Van Jaarsveld (2025),
*International Journal of Production Economics* 284:109601.

This is a **faithful, citation-anchored scaffold**, not a bit-exact reproduction. The
paper is not on arxiv, releases no official code for itself (it contributes example
code to the **DynaPlex** library, Akkerman et al. 2023, not reproduced here), and a few
formulas were recovered from table images. Every guess is flagged below and in code via
`[UNSPECIFIED]`, `[PARTIALLY_SPECIFIED]`, `[REFERENCE]`, `[SIMPLIFICATION]` tags.

---

## What is implemented (in scope)

| Component | File | Paper |
|---|---|---|
| Decomposed SCLSP MDP environment | `src/env.py` | §3.1–3.5, Eq.1–4 |
| Non-stationary demand process (DGP) | `src/demand.py` | §1/§6, §5.2 (approximation) |
| AMBS-based rollout / benchmark policy | `src/rollout.py` | §4.2, Algorithm 1 |
| MLP policy + structured state/action encoding | `src/policy.py` | §4.3, §4.4, Eq.6–7 |
| DCL training (CRN + Sequential Halving + supervised classification) | `src/dcl.py` | §4.1, Table 2 |
| Simulation + Δ-vs-AMBS metric | `src/evaluate.py` | §5.1, Tables 3/5 |

## Intentionally excluded (out of scope)

- **PPO benchmark** (van Hezewijk 2023b) and the **full-period AMBS** heuristic — comparison
  baselines, not the contribution (§5.1, scope rules).
- **Capacity-shortage analysis** figures (§5.2.1, Figs 3–4) — diagnostic, not the method.
- **Multi-machine extension** (§6 future work).
- **DynaPlex parallel/cluster infrastructure** (Table 6 compute) — engineering, not method.

---

## Critical unspecified / reconstructed details

### 1. Eq.4 — period cost (reward)  `[PARTIALLY_SPECIFIED]`
The PDF equation image did not extract. Reconstructed from Table 1 notation and §3.5 prose as
`C_t = Σ_i ( h_i·[I_i]^+ + b_i·[I_i]^- + k_i·z_{i,t} )`. Holding + backorder charged at period
close; setup term charged when the setup is started (§3.4). **Verify against the original Eq.4.**

### 2. Demand generating process  `[PARTIALLY_SPECIFIED]`
The exact DGP is **van Hezewijk et al. (2023a)** — not defined in this paper. We implement an
IMA(0,1,1)-style level random walk with integer, non-negative truncation (`src/demand.py`),
parameterized by μ₀=2, σ₀=COV·μ₀, α∈{0,0.025,0.05} (§5.2.2 / Table 4). **Swap in the exact
2023a process for faithful numbers.** σ_{i,t} evolution over time is `[UNSPECIFIED]` — held at σ₀.

### 3. Algorithm 1 quantity / continue lines  `[PARTIALLY_SPECIFIED]`
Recovered from a reordered s2orc table. Reorder level `R_i = μ_i + B_min·σ_i`; order-up-to
target `μ_ω + sqrt(2·μ_ω·k_ω/h_ω) + H_max·B_min·σ_ω`. **Verify against the Algorithm 1 image.**

### 4. Rollout parameters B_min, H_max, Z_max  `[UNSPECIFIED]`
"Found by searching a grid" (§4.2); values not reported. Defaults `B_min=2.0, H_max=1.0,
Z_max=1` (Z_max=1 from §4.1 "AMBS would allow only one new setup per period"). Tune per instance.

### 5. DCL network-training hyperparameters  `[UNSPECIFIED]`
§4.1 defers to Temizöz et al. (2023) / DynaPlex. lr=1e-3, batch=256, 10 epochs/gen, Adam,
cross-entropy — all our defaults. The structural choices (N, M, H, generations, CRN, Sequential
Halving) ARE specified (Table 2) and implemented as such.

### 6. Activation, init, input normalization  `[UNSPECIFIED]`
§4.3 says only "dense network". Defaults: ReLU, PyTorch init, no input normalization.

### 7. Discount factor γ  `[UNSPECIFIED]`
Not stated. γ=1.0 (undiscounted), consistent with the "average cost per period" objective (§5.1).

### 8. Partial setup carry-over  `[SIMPLIFICATION]`
When a setup's θ time units overrun the period, we charge the cost (§3.4) and carry the overflow
into next period's τ, closing the current period. The paper notes incomplete setups exist but does
not fully specify their dynamics.

---

## Conflicts found in the paper

- **Cost parameters differ by experiment set.** §5.1 stationary replication uses `k_i=200, θ_i=0`;
  Table 4 non-stationary study uses `k_i=100, θ_i=1`. `configs/base.yaml` ships the non-stationary
  preset; switch for §5.1 replication.

---

## Scale: paper vs demo

`configs/base.yaml → dcl.use_paper_scale` toggles:

| | N (states) | M (rollouts/action) | H | eval runs |
|---|---|---|---|---|
| **paper** (Table 2/6, needs a cluster) | 150k–300k | 1000 | 30 | 10,000 |
| **demo** (laptop CPU, default) | 2,000 | 32 | 20 | 200 |

Demo numbers are **noisy** and will not match Tables 3/5 quantitatively — they exist to make the
full pipeline runnable and to verify the *mechanics* (action masking, decomposition, CRN, SH,
supervised improvement). For quantitative reproduction, set `use_paper_scale: true` on adequate
hardware AND replace the DGP (item 2) and verify Eq.4 / Algorithm 1 (items 1, 3).

## Sanity targets (from the paper, in `Scalable_DRL_nonstationary_parsed.md` appendix)

- Stationary, K=5, U{3,5}, f_c=1.1 (Table 3): AMBS 36.63, DCL 33.22 → Δ_DCL = −9.3%.
- Non-stationary (Table 5): DCL beats AMBS by ~3–14% (Δ negative) across the grid.

---

## Would need for full reproduction

1. van Hezewijk et al. (2023a) — exact DGP.
2. Original PDF Eq.4 and Algorithm 1 (un-garbled).
3. DynaPlex / Temizöz et al. (2023) — DCL network-training schedule.
4. Cluster compute for paper-scale N·M·H (Table 6: up to ~8 h for K=15).
