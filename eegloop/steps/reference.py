"""Re-referencing, block by block. Exact, instantaneous, and rank-reducing.

A reference is a matrix multiplication of the samples at one instant, so it adds no delay: the row
it contributes to the budget is zero, exactly. What it costs is rank -- an average reference over
``n`` channels leaves ``n - 1`` independent signals, which at four channels is a quarter of what
you had (lesson L7.5: rank-based analyses are the first to go). The step says which reference was
applied so a session log can state it, because "the signal at TP9" is not a definition until the
reference is named.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ..block import Block

__all__ = ["Reference"]


class Reference:
    kind = "reference"

    def __init__(self, mode: str, ch_names: Iterable[str], *, channels: Iterable[str] = (), name: str = "reference") -> None:
        self.name = name
        self.mode = str(mode)
        self.ch_names = tuple(str(c) for c in ch_names)
        n = len(self.ch_names)
        if self.mode == "none":
            self.matrix = np.eye(n)
            self.channels: tuple[str, ...] = ()
        elif self.mode == "average":
            self.matrix = np.eye(n) - np.full((n, n), 1.0 / n)
            self.channels = self.ch_names
        elif self.mode == "channels":
            picks = [self.ch_names.index(c) for c in channels]
            if not picks:
                raise ValueError("mode='channels' needs at least one reference channel")
            ref = np.zeros((1, n))
            ref[0, picks] = 1.0 / len(picks)
            self.matrix = np.eye(n) - np.repeat(ref, n, axis=0)
            self.channels = tuple(self.ch_names[i] for i in picks)
        else:
            raise ValueError(f"mode must be 'none', 'average' or 'channels', got {mode!r}")

    def process(self, block: Block) -> Block:
        if block.n_channels != len(self.ch_names):
            raise ValueError(f"{self.name}: expected {len(self.ch_names)} channels, got {block.n_channels}")
        if self.mode == "none":
            return block
        return block.replace(data=self.matrix @ block.data)

    def reset(self) -> None:
        pass

    @property
    def latency_samples(self) -> float:
        return 0.0

    @property
    def latency_note(self) -> str:
        return "re-referencing is a matrix multiplication at one instant: zero delay (exact); it costs rank, not time"

    @property
    def rank_cost(self) -> int:
        return 1 if self.mode == "average" else 0

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "mode": self.mode, "reference_channels": list(self.channels), "rank_cost": self.rank_cost}
