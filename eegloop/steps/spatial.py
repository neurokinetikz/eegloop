"""A frozen spatial filter: weights fitted once, applied block by block, never adapted.

Lesson L7.3 on artifacts you cannot look at: ICA can run online only with a frozen unmixing matrix
estimated during a calibration block; SSP projectors are a fixed matrix multiplication with no
latency and a cost that is countable in rank. CSP and xDAWN filters for a decoder are the same shape.
All of them are one ``W`` of ``(n_out, n_in)`` and one matrix product per block -- exact, zero
delay -- and this step exists so that the fit and the application are visibly different things: the
weights come in from outside, and the step cannot change them.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ..block import Block

__all__ = ["SpatialFilter"]


class SpatialFilter:
    kind = "spatial"

    def __init__(self, weights: np.ndarray, in_names: Iterable[str], out_names: Iterable[str] | None = None,
                 *, kind_label: str = "spatial", name: str = "spatial") -> None:
        self.name = name
        self.kind_label = str(kind_label)
        self.W = np.atleast_2d(np.asarray(weights, dtype=np.float64))
        self.in_names = tuple(str(c) for c in in_names)
        if self.W.shape[1] != len(self.in_names):
            raise ValueError(f"weights are {self.W.shape}, but {len(self.in_names)} input channels were named")
        self.out_names = (tuple(str(c) for c in out_names) if out_names is not None
                          else tuple(f"{self.kind_label}{i + 1}" for i in range(self.W.shape[0])))
        if len(self.out_names) != self.W.shape[0]:
            raise ValueError(f"weights produce {self.W.shape[0]} outputs, but {len(self.out_names)} were named")

    @classmethod
    def from_projector(cls, projector: np.ndarray, ch_names: Iterable[str], **kw: Any) -> "SpatialFilter":
        """An SSP projector ``P`` (``n × n``): output channels keep their names."""
        names = tuple(ch_names)
        return cls(projector, names, names, kind_label=kw.pop("kind_label", "ssp"), **kw)

    @classmethod
    def from_ica(cls, unmixing: np.ndarray, mixing: np.ndarray, exclude: Iterable[int],
                 ch_names: Iterable[str], **kw: Any) -> "SpatialFilter":
        """``mixing[:, keep] @ unmixing[keep, :]`` -- the frozen 'remove these components' matrix."""
        unmixing = np.asarray(unmixing, dtype=np.float64)
        mixing = np.asarray(mixing, dtype=np.float64)
        keep = [i for i in range(unmixing.shape[0]) if i not in set(int(e) for e in exclude)]
        W = mixing[:, keep] @ unmixing[keep, :]
        names = tuple(ch_names)
        return cls(W, names, names, kind_label=kw.pop("kind_label", "ica-frozen"), **kw)

    def process(self, block: Block) -> Block:
        if block.n_channels != len(self.in_names):
            raise ValueError(f"{self.name}: expected {len(self.in_names)} channels, got {block.n_channels}")
        return block.replace(data=self.W @ block.data).with_flag("ch_names", self.out_names)

    def reset(self) -> None:
        pass

    @property
    def latency_samples(self) -> float:
        return 0.0

    @property
    def latency_note(self) -> str:
        return f"frozen {self.kind_label} weights, one matrix product per block: zero delay (exact)"

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "kind_label": self.kind_label, "shape": list(self.W.shape),
                "in_names": list(self.in_names), "out_names": list(self.out_names),
                "rank": int(np.linalg.matrix_rank(self.W))}
