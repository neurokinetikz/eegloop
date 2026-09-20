"""A steady-state detector: canonical correlation between a window and sine/cosine references.

No fitting, no labels: for each candidate frequency the reference is ``sin`` and ``cos`` at that
frequency and its harmonics, and the score is the largest canonical correlation between the window
and the reference. The frequency with the largest score wins if it wins by a margin. That is the
whole of the classical SSVEP-CCA detector, and it is numpy and one QR.

One caveat is in the numbers before it is in the text. A standing alpha rhythm correlates with a
10 Hz reference in *every* window, flicker or not (the synthetic ``ssvep`` scenario shows r ≈ 0.5 at
10 Hz throughout). A stimulus frequency inside the alpha band is therefore a bad choice, and a
detector that does not compare against a no-stimulus baseline will report alpha as a response.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..block import Block
from ..ring import RingBuffer

__all__ = ["SSVEP_NOTE", "cca_correlation", "ssvep_cca", "ssvep_decide", "SSVEPDetector"]

SSVEP_NOTE = ("a standing alpha rhythm correlates with a 10 Hz reference in every window, flicker or not; choose "
              "stimulus frequencies outside 8–12 Hz, and compare each score against a no-stimulus baseline before "
              "calling it a response")


def _references(n: int, fs: float, freq_hz: float, n_harmonics: int) -> np.ndarray:
    t = np.arange(n) / float(fs)
    cols = []
    for h in range(1, int(n_harmonics) + 1):
        w = 2 * np.pi * float(freq_hz) * h * t
        cols += [np.sin(w), np.cos(w)]
    return np.column_stack(cols)


def cca_correlation(window: np.ndarray, fs: float, freq_hz: float, *, n_harmonics: int = 2) -> float:
    """Largest canonical correlation between ``window`` (n_channels × n_samples) and the reference."""
    X = np.atleast_2d(np.asarray(window, dtype=float)).T
    Y = _references(X.shape[0], fs, freq_hz, n_harmonics)
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    Qx, _ = np.linalg.qr(Xc)
    Qy, _ = np.linalg.qr(Yc)
    s = np.linalg.svd(Qx.T @ Qy, compute_uv=False)
    return float(s[0]) if s.size else 0.0


def ssvep_cca(window: np.ndarray, fs: float, freqs_hz: Sequence[float], *, n_harmonics: int = 2) -> dict[float, float]:
    return {float(f): cca_correlation(window, fs, f, n_harmonics=n_harmonics) for f in freqs_hz}


def ssvep_decide(window: np.ndarray, fs: float, freqs_hz: Sequence[float], *, n_harmonics: int = 2,
                 margin: float = 0.1, baseline: dict[float, float] | None = None) -> tuple[float | None, dict[float, float]]:
    """The winning frequency, or ``None`` when no candidate leads the runner-up by ``margin``.
    With ``baseline`` (scores from a no-stimulus window) each score is taken relative to it first."""
    scores = ssvep_cca(window, fs, freqs_hz, n_harmonics=n_harmonics)
    rel = {f: s - (baseline.get(f, 0.0) if baseline else 0.0) for f, s in scores.items()}
    order = sorted(rel.items(), key=lambda kv: kv[1], reverse=True)
    if not order:
        return None, scores
    if len(order) == 1 or order[0][1] - order[1][1] >= margin:
        return order[0][0], scores
    return None, scores


class SSVEPDetector:
    """The detector over a stream: every ``step_samples``, the latest ``window_s`` through :func:`ssvep_decide`."""

    def __init__(self, fs: float, n_channels: int, freqs_hz: Sequence[float], *, window_s: float = 4.0,
                 step_samples: int = 32, n_harmonics: int = 2, margin: float = 0.1) -> None:
        self.fs, self.freqs = float(fs), tuple(float(f) for f in freqs_hz)
        self.n = int(round(window_s * fs))
        self.step, self.n_harmonics, self.margin = int(step_samples), int(n_harmonics), float(margin)
        self.ring = RingBuffer(int(n_channels), self.n)
        self._since = 0
        self.baseline: dict[float, float] | None = None

    def set_baseline(self, window: np.ndarray) -> dict[float, float]:
        self.baseline = ssvep_cca(window, self.fs, self.freqs, n_harmonics=self.n_harmonics)
        return self.baseline

    def update(self, block: Block) -> tuple[float | None, dict[float, float]] | None:
        self.ring.push(block)
        self._since += block.n_samples
        if self.ring.filled < self.n or self._since < self.step:
            return None
        self._since = 0
        return ssvep_decide(self.ring.latest(self.n), self.fs, self.freqs, n_harmonics=self.n_harmonics,
                            margin=self.margin, baseline=self.baseline)

    @property
    def latency_samples(self) -> float:
        return float(self.n)

    @property
    def latency_note(self) -> str:
        return f"the {self.n / self.fs:g}-s window is the delay; the step sets the update rate; {SSVEP_NOTE}"

    def describe(self) -> dict[str, Any]:
        return {"freqs_hz": list(self.freqs), "window_s": self.n / self.fs, "step_samples": self.step,
                "n_harmonics": self.n_harmonics, "margin": self.margin, "baseline": self.baseline, "note": SSVEP_NOTE}
