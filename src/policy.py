"""
Scalable DRL in the non-stationary SCLSP — policy network + state/action encoding.

Paper: Van Hezewijk, Dellaert, Van Jaarsveld (2025), IJPE 284:109601.
Implements §4.3 (state representation + dense network) and §4.4 (unified, masked
action space, Eq.6–7). In DCL the network is a CLASSIFIER: it outputs one logit per
action and is trained by supervised learning to predict the best action (§4.1) — there
is no policy-gradient and no value head.

  Educational note — why a plain MLP (not a fancy architecture):
  The contribution of this paper is the MDP decomposition + the DCL training, NOT the
  network. §4.3: "we adopt a dense (fully connected) neural network ... one output for
  every possible action." The only architectural trick is ordering the state vector so
  the currently set-up product appears FIRST, giving the net a consistent slot for the
  most decision-relevant product (§4.3).
"""
from __future__ import annotations

from typing import List

import numpy as np

from env import State, Stage, SCLSPEnv
from utils import ActionSpace

# Number of per-product features in the encoded state vector. See encode_state().
PER_PRODUCT_FEATURES = 7


def _product_order(state: State, K: int) -> List[int]:
    """§4.3 — "the representation for the product that is currently set up is presented
    first in the vector". Returns a product ordering with ω first, others after."""
    if 0 <= state.omega < K:
        rest = [i for i in range(K) if i != state.omega]
        return [state.omega] + rest
    return list(range(K))


def encode_state(state: State, env: SCLSPEnv) -> np.ndarray:
    """Encode an MDP state (§3.2 Eq.1) into a flat float vector for the MLP (§4.3).

    Global features (3): τ/C, stage flag, remaining_periods/horizon (§4.3 finite MDP).
    Per-product features (7, products ordered with ω first):
        I_i, μ_i, σ_i, q_i, is_setup(ω), is_pending(2nd-stage target), produced_flag.
    """
    K = env.K
    C = max(env.C, 1)
    order = _product_order(state, K)

    glob = np.array([
        state.tau / C,                                 # capacity used fraction (τ, Eq.1)
        float(state.stage == Stage.SECOND),            # which sub-decision we are at (§3.1)
        state.remaining_periods / max(env.horizon, 1),  # §4.3 — only meaningful for finite MDP
    ], dtype=np.float32)

    feats = []
    for i in order:
        feats.extend([
            state.I[i],                                # inventory position I_i (Eq.1)
            state.mu[i],                               # forecast mean μ_i (Eq.1)
            state.sigma[i],                            # forecast std σ_i (Eq.1)
            state.q[i],                                # produced-so-far this period q_i (Eq.1)
            float(i == state.omega),                   # ω_i — machine set up for i (Eq.1)
            float(i == state.pending_product),         # 2nd-stage target product
            float(state.produced_mask[i]),             # already produced this period (§3.3)
        ])
    return np.concatenate([glob, np.array(feats, dtype=np.float32)])


def state_dim(K: int) -> int:
    return 3 + K * PER_PRODUCT_FEATURES


# ---------------------------------------------------------------------------------
# The network itself (torch). Imported lazily so env/rollout/demand work without torch.
# ---------------------------------------------------------------------------------
def build_policy_net(config: dict, action_space: ActionSpace):
    import torch
    import torch.nn as nn

    K = config["problem"]["K"]
    in_dim = state_dim(K)
    out_dim = action_space.size
    hidden = config["policy"]["hidden_layers"]  # Table 2 — [128,128] (K=5) / [256,256]
    act_name = config["policy"]["activation"]

    # [UNSPECIFIED] §4.3 does not state the activation; ReLU is the DynaPlex/MLP default.
    act_layer = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[act_name]

    class PolicyNet(nn.Module):
        """§4.3 — dense classifier; output dim = |A1 ∪ A2*| (Eq.7)."""

        def __init__(self):
            super().__init__()
            dims = [in_dim] + list(hidden)
            layers = []
            for a, b in zip(dims[:-1], dims[1:]):
                layers += [nn.Linear(a, b), act_layer()]  # (.., a) -> (.., b)
            layers += [nn.Linear(dims[-1], out_dim)]       # (.., hidden[-1]) -> (.., |A|)
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            # x: (batch, in_dim) -> logits: (batch, out_dim)
            return self.net(x)

        def masked_logits(self, x, mask):
            """Apply the action mask (§4.4) by setting invalid logits to -inf."""
            logits = self.forward(x)                       # (batch, |A|)
            neg_inf = torch.finfo(logits.dtype).min
            return logits.masked_fill(~mask.bool(), neg_inf)

    return PolicyNet()


class NeuralPolicy:
    """Wraps a trained PolicyNet to expose the same `.act(env)` interface as the
    AMBS rollout policy, so DCL/evaluation can use either interchangeably."""

    def __init__(self, net, action_space: ActionSpace):
        self.net = net
        self.aspace = action_space

    def act(self, env: SCLSPEnv, greedy: bool = True) -> int:
        import torch
        x = torch.from_numpy(encode_state(env.state, env)).float().unsqueeze(0)  # (1, in_dim)
        mask = torch.from_numpy(env.legal_action_mask()).unsqueeze(0)            # (1, |A|)
        with torch.no_grad():
            logits = self.net.masked_logits(x, mask)                            # (1, |A|)
            if greedy:
                return int(logits.argmax(dim=-1).item())
            probs = torch.softmax(logits, dim=-1)
            return int(torch.multinomial(probs, 1).item())
