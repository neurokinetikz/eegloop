"""Decision 3 of L7.3: what the baseline is, and whether it moves.

"An adaptive baseline means the participant is always near threshold by construction, which makes
the feedback feel responsive and makes 'improvement' much harder to define." Both choices are here;
each carries a ``note`` saying what it does to a learning claim, because the choice is part of the
protocol and belongs in the log.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Sequence

import numpy as np

__all__ = ["FixedBaseline", "AdaptiveBaseline", "robust_spread"]


def robust_spread(values: np.ndarray) -> float:
    med = np.median(values)
    return float(np.median(np.abs(values - med)) * 1.4826)


class FixedBaseline:
    """A centre and spread fixed from calibration values; the threshold never moves."""

    def __init__(self, centre: float, spread: float, *, centre_kind: str = "median", spread_kind: str = "mad", n: int = 0) -> None:
        self.centre, self.spread = float(centre), max(float(spread), 1e-12)
        self.centre_kind, self.spread_kind, self.n = centre_kind, spread_kind, int(n)

    @classmethod
    def from_values(cls, values: Sequence[float], *, centre: str = "median", spread: str = "mad") -> "FixedBaseline":
        v = np.asarray(values, dtype=float)
        if v.size < 2:
            raise ValueError("a fixed baseline needs at least two calibration values")
        c = float(np.median(v)) if centre == "median" else float(np.mean(v))
        s = robust_spread(v) if spread == "mad" else float(np.std(v))
        return cls(c, s, centre_kind=centre, spread_kind=spread, n=v.size)

    def update(self, v: float) -> None:
        pass  # fixed means fixed

    def z(self, v: float) -> float:
        return (float(v) - self.centre) / self.spread

    @property
    def note(self) -> str:
        return (f"fixed from {self.n} calibration values ({self.centre_kind} {self.centre:.3g}, {self.spread_kind} "
                f"{self.spread:.3g}): the threshold does not move, so a change in the signal is a change relative "
                f"to the calibration and 'improvement' has a definition")

    def describe(self) -> dict[str, Any]:
        return {"mode": "fixed", "centre": self.centre, "spread": self.spread, "n": self.n,
                "centre_kind": self.centre_kind, "spread_kind": self.spread_kind, "note": self.note}


class AdaptiveBaseline:
    """A centre and spread tracked over the last ``window_s``; the threshold follows the participant."""

    def __init__(self, window_s: float, fs: float, block_samples: int, *, percentile: float = 50.0,
                 spread: str = "mad") -> None:
        self.window_s, self.percentile, self.spread_kind = float(window_s), float(percentile), spread
        n = max(2, int(round(window_s / (block_samples / fs))))
        self._hist: deque = deque(maxlen=n)
        self._n = n

    def update(self, v: float) -> None:
        self._hist.append(float(v))

    @property
    def centre(self) -> float | None:
        return None if len(self._hist) < 2 else float(np.percentile(np.asarray(self._hist), self.percentile))

    @property
    def spread(self) -> float | None:
        if len(self._hist) < 2:
            return None
        h = np.asarray(self._hist)
        s = robust_spread(h) if self.spread_kind == "mad" else float(np.std(h))
        return max(s, 1e-12)

    def z(self, v: float) -> float:
        c, s = self.centre, self.spread
        return float("nan") if c is None or s is None else (float(v) - c) / s

    @property
    def note(self) -> str:
        return (f"adaptive: the {self.percentile:g}th percentile of the last {self.window_s:g} s ({self._n} blocks). "
                f"The participant is always near threshold by construction, which makes the feedback feel responsive "
                f"and makes 'improvement' much harder to define; the raw value is logged beside the threshold so the "
                f"outcome can still be pre-specified on the signal")

    def describe(self) -> dict[str, Any]:
        return {"mode": "adaptive", "window_s": self.window_s, "percentile": self.percentile,
                "spread_kind": self.spread_kind, "note": self.note}
