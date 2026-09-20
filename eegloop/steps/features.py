"""One feature vector per block, computed from values already in hand. Zero delay, exactly.

A feature step is a pure function of the block and its flags: log-variance per channel from the
data, a ratio of two band powers, or the band powers themselves. It adds no window and no memory,
which is why its delay is zero and exact -- the windows that produced the inputs have already
declared themselves. It writes ``block.flags['features']``, the vector a decoder reads.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..block import Block

__all__ = ["FeatureExtractor"]


class FeatureExtractor:
    kind = "features"

    def __init__(self, method: str, *, bands: tuple[str, ...] = (), name: str = "features") -> None:
        self.name = name
        self.method = str(method)
        self.bands = tuple(str(b) for b in bands)
        if self.method not in ("log-var", "band-power", "band-ratio"):
            raise ValueError("method must be 'log-var', 'band-power' or 'band-ratio'")
        if self.method == "band-ratio" and len(self.bands) != 2:
            raise ValueError("band-ratio needs exactly two band names (numerator, denominator)")
        if self.method == "band-power" and not self.bands:
            raise ValueError("band-power needs the band names to read")

    def process(self, block: Block) -> Block:
        if self.method == "log-var":
            v = np.log(np.var(block.data, axis=1) + 1e-12)
        else:
            bp = block.flags.get("bandpower")
            if not isinstance(bp, dict):
                raise ValueError(f"{self.name}: needs a 'bandpower' flag from a BandPower step")
            if self.method == "band-power":
                v = np.concatenate([np.asarray(bp[b], dtype=float) for b in self.bands])
            else:
                num, den = (np.asarray(bp[b], dtype=float) for b in self.bands)
                v = num / (den + 1e-12)
        return block.with_flag("features", v)

    def reset(self) -> None:
        pass

    @property
    def latency_samples(self) -> float:
        return 0.0

    @property
    def latency_note(self) -> str:
        return "a pure function of values already computed: zero delay (exact)"

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "method": self.method, "bands": list(self.bands)}
