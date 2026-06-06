"""
Scalable DRL in the non-stationary SCLSP — shared utilities.

Paper: Van Hezewijk, Dellaert, Van Jaarsveld (2025), IJPE 284:109601.
Implements: config loading + the unified/restricted action space (§4.4, Eq.6–7).

Only helpers shared across multiple modules live here (per scope rules).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import yaml


def load_config(path: str | Path) -> dict:
    """Load the YAML config (configs/base.yaml)."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_grid_spec(spec: str) -> List[int]:
    """Parse the restricted second-stage quantity grid from a compact spec string.

    §4.4, Eq.6 — the restricted action space A2* keeps every integer near 0 and
    grows coarser for large quantities, because "it is not that interesting to
    analyze the difference between quantity 91 and 92, while the difference
    between quantity 1 and 2 is quite relevant".

    Spec format: comma-separated "lo-hi:step" blocks, e.g. "1-20:1, 22-40:2".
    Returns the sorted list of allowed quantities (the runtime upper bound is
    later clipped to remaining capacity C-τ, Eq.3).
    """
    quantities: set[int] = set()
    for block in spec.split(","):
        block = block.strip()
        m = re.match(r"(\d+)-(\d+):(\d+)", block)
        if not m:
            raise ValueError(f"Bad grid block: {block!r}")
        lo, hi, step = int(m.group(1)), int(m.group(2)), int(m.group(3))
        quantities.update(range(lo, hi + 1, step))
    return sorted(quantities)


@dataclass
class ActionSpace:
    """Unified action space A = A1 ∪ A2* (§4.4, Eq.7).

    The DCL network has ONE output per action and masks out the part that is not
    valid given the current stage / remaining capacity / set-up product (§4.4).

    Layout of the flat action index:
      [0]            -> p0  (stop production / go idle), §3.3
      [1 .. K]       -> p1..pK  (first-stage: set up & switch to product i), Eq.2
      [K+1 .. end]   -> restricted quantities q in A2*  (second-stage), Eq.3/Eq.6
    """
    K: int
    quantities: List[int]

    @property
    def n_first(self) -> int:
        return self.K + 1  # p0 + p1..pK

    @property
    def n_second(self) -> int:
        return len(self.quantities)

    @property
    def size(self) -> int:
        return self.n_first + self.n_second

    # --- index helpers ---
    def stop_index(self) -> int:
        return 0

    def product_index(self, i: int) -> int:
        """Flat index of first-stage action 'produce product i' (0-based i)."""
        return 1 + i

    def quantity_index(self, slot: int) -> int:
        """Flat index of the `slot`-th restricted quantity."""
        return self.n_first + slot

    def is_first_stage(self, idx: int) -> bool:
        return idx < self.n_first


def build_action_space(config: dict) -> ActionSpace:
    grid = parse_grid_spec(config["action"]["restricted_grid_spec"])
    return ActionSpace(K=config["problem"]["K"], quantities=grid)


def set_global_seed(seed: int) -> None:
    """Seed numpy + torch (torch only if available — env/rollout need only numpy)."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
