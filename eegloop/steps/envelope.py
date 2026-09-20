"""From a filtered block to a number per channel -- and what each way of doing it costs in time.

Three envelope steps, three different honesty statements about latency:

* :class:`BlockRMS` -- the RMS of *this block*. Its delay is the block itself, which is already the
  buffer row of the budget; the step adds nothing on top and says ``0`` (exact). The loopback probe's
  expectation of ``(B + 1) / 2`` samples for a per-block value comes from the block, not the step.
* :class:`Smoother` -- a one-pole exponential on the per-block values, with a stated time constant.
  It buys a steadier number and costs latency, and there is **no single number** for that cost: the
  impulse response never ends, so its "delay" depends on the criterion. ``latency_samples`` is
  ``None`` and the probe measures it (``helpers_l7.alpha_feedback``: "the notebook measures how much").
* :class:`BandPower` -- Welch over the last ``window_s`` from a ring. The window's centroid sits at
  roughly half its length; again a criterion, not a fact, so it is measured too.

Each writes ``block.flags['envelope']`` (a value per channel) and leaves ``data`` untouched, so a
chain can carry the filtered signal onward while the loop reads the value.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..block import Block
from ..ring import RingBuffer

__all__ = ["BlockRMS", "Smoother", "BandPower"]


class BlockRMS:
    kind = "envelope"

    def __init__(self, *, name: str = "rms") -> None:
        self.name = name

    def process(self, block: Block) -> Block:
        return block.with_flag("envelope", np.sqrt(np.mean(block.data ** 2, axis=1)))

    def reset(self) -> None:
        pass

    @property
    def latency_samples(self) -> float:
        return 0.0

    @property
    def latency_note(self) -> str:
        return "RMS of this block: the delay is the block itself, already counted in the buffer row (exact: 0 on top)"

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "method": "rms-per-block"}


class Smoother:
    kind = "envelope"

    def __init__(self, tau_s: float, fs: float, block_samples: int, *, name: str = "smoother") -> None:
        self.name = name
        self.tau_s, self.fs, self.block_samples = float(tau_s), float(fs), int(block_samples)
        self.alpha = 1.0 if self.tau_s <= 0 else float(1.0 - np.exp(-(self.block_samples / self.fs) / self.tau_s))
        self._state: np.ndarray | None = None

    def process(self, block: Block) -> Block:
        v = np.asarray(block.flags.get("envelope"), dtype=float)
        if v.ndim == 0 or v.size == 0:
            raise ValueError(f"{self.name}: needs an 'envelope' flag from an earlier envelope step")
        self._state = v.copy() if self._state is None else self._state + self.alpha * (v - self._state)
        return block.with_flag("envelope", self._state.copy())

    def reset(self) -> None:
        self._state = None

    @property
    def latency_samples(self) -> None:
        return None

    @property
    def latency_note(self) -> str:
        return (f"one-pole smoother, tau = {self.tau_s:g} s: no single delay (the impulse response never ends); "
                f"measured with the loopback probe, never summed")

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "tau_s": self.tau_s, "alpha_per_block": self.alpha}


class BandPower:
    kind = "envelope"

    def __init__(self, fs: float, n_channels: int, bands: dict[str, tuple[float, float]], *,
                 window_s: float = 2.0, name: str = "bandpower") -> None:
        self.name = name
        self.fs, self.n_channels = float(fs), int(n_channels)
        self.bands = {str(k): (float(lo), float(hi)) for k, (lo, hi) in bands.items()}
        self.window_s = float(window_s)
        self.window = max(8, int(round(self.window_s * self.fs)))
        self._ring = RingBuffer(self.n_channels, self.window, fs=self.fs)

    def process(self, block: Block) -> Block:
        from scipy.signal import welch

        self._ring.push(block.data)
        win = self._ring.latest(self.window)
        fr, p = welch(win, fs=self.fs, nperseg=min(win.shape[1], int(self.fs)), axis=-1)
        power = {}
        for name, (lo, hi) in self.bands.items():
            m = (fr >= lo) & (fr <= hi)
            power[name] = p[:, m].mean(axis=1) if m.any() else np.full(self.n_channels, np.nan)
        first = next(iter(power.values()))
        return block.with_flag("bandpower", power).with_flag("envelope", np.sqrt(first))

    def reset(self) -> None:
        self._ring = RingBuffer(self.n_channels, self.window, fs=self.fs)

    @property
    def latency_samples(self) -> None:
        return None

    @property
    def latency_note(self) -> str:
        return (f"Welch over the last {self.window_s:g} s: the window's centroid sits near half its length, "
                f"a criterion rather than a fact; measured, never summed")

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "window_s": self.window_s, "bands": {k: list(v) for k, v in self.bands.items()}}
