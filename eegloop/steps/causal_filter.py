"""Causal filters run block by block, carrying their state across blocks.

``scipy.signal.lfilter`` (or ``sosfilt``) with an explicit ``zi`` is exactly what a real-time system
does, and the result is identical to filtering the whole recording at once -- the *timing* changes,
the *signal* does not. ``tests/test_steps.py`` asserts that, because the alternative (filtering each
block independently, with the filter's memory reset at every boundary) is ``pf-filter-across-
boundaries`` arriving at the block rate, and it does not.

The filter designs are ported from ``notebooks/_shared/helpers_l7.py`` (``fir_taps``, ``butter_sos``,
2026-09-20) with one change: the sampling rate is always an argument. The notebook helpers default
to the pipeline lesson L7.3 quotes; a library that defaulted to 160 Hz would be wrong on every headset
this course connects to.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..block import Block
from ..latency import fir_group_delay_samples, iir_group_delay_samples

__all__ = ["fir_taps", "butter_sos", "CausalFIR", "CausalSOS", "Notch"]


def fir_taps(n_taps: int, band: tuple[float, float], fs: float, *, window: str = "hamming") -> np.ndarray:
    """A linear-phase windowed-sinc band-pass -- the filter whose delay is the same at every frequency."""
    from scipy.signal import firwin

    return firwin(int(n_taps), list(band), pass_zero=False, fs=float(fs), window=window)


def butter_sos(order: int, band: tuple[float, float], fs: float) -> np.ndarray:
    """A Butterworth band-pass as second-order sections."""
    from scipy.signal import butter

    return butter(int(order), list(band), btype="bandpass", fs=float(fs), output="sos")


class CausalFIR:
    """A linear-phase FIR with its ``lfilter`` state carried across blocks.

    ``latency_samples`` is ``(N - 1) / 2``, exact and the same at every frequency, which is what
    makes a linear-phase FIR the reference case for a latency budget: length and latency are the
    same dial.
    """

    kind = "causal_filter"

    def __init__(self, taps: np.ndarray, n_channels: int, *, name: str = "fir") -> None:
        from scipy.signal import lfilter_zi

        self.name = name
        self.taps = np.asarray(taps, dtype=np.float64)
        if self.taps.ndim != 1 or self.taps.size < 2:
            raise ValueError("taps must be a 1-D array of at least two coefficients")
        self.n_channels = int(n_channels)
        # Zero initial conditions: the same assumption `lfilter` makes on a whole recording, which
        # is why the block-wise result equals the whole-signal one.
        self.zi = np.tile(lfilter_zi(self.taps, [1.0]), (self.n_channels, 1)) * 0.0

    def process(self, block: Block) -> Block:
        from scipy.signal import lfilter

        if block.n_channels != self.n_channels:
            raise ValueError(f"{self.name}: expected {self.n_channels} channels, got {block.n_channels}")
        y, self.zi = lfilter(self.taps, [1.0], block.data, axis=-1, zi=self.zi)
        return block.replace(data=y)

    def reset(self) -> None:
        self.zi[:] = 0.0

    @property
    def latency_samples(self) -> float:
        return fir_group_delay_samples(self.taps.size)

    @property
    def latency_note(self) -> str:
        return f"linear-phase FIR, {self.taps.size} taps: (N - 1) / 2 samples at every frequency (exact)"

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "n_taps": int(self.taps.size), "n_channels": self.n_channels}


class CausalSOS:
    """An IIR in second-order sections with its ``sosfilt`` state carried across blocks.

    An IIR's delay is not one number: it varies across the pass band. ``latency_samples`` is the group
    delay at ``ref_hz`` -- summed over the sections, because expanding the cascade into one transfer
    function is numerically unstable at high order (see :func:`eegloop.latency.iir_group_delay_samples`)
    -- and the note says which frequency it was read at, so a budget that quotes it cannot pretend
    the number is flat.
    """

    kind = "causal_filter"

    def __init__(self, sos: np.ndarray, n_channels: int, *, fs: float, ref_hz: float,
                 name: str = "sos") -> None:
        from scipy.signal import sosfilt_zi

        self.name = name
        self.sos = np.atleast_2d(np.asarray(sos, dtype=np.float64))
        if self.sos.shape[1] != 6:
            raise ValueError("sos must be (n_sections, 6)")
        self.n_channels = int(n_channels)
        self.fs = float(fs)
        self.ref_hz = float(ref_hz)
        self.zi = np.repeat(sosfilt_zi(self.sos)[:, np.newaxis, :], self.n_channels, axis=1) * 0.0

    def process(self, block: Block) -> Block:
        from scipy.signal import sosfilt

        if block.n_channels != self.n_channels:
            raise ValueError(f"{self.name}: expected {self.n_channels} channels, got {block.n_channels}")
        y, self.zi = sosfilt(self.sos, block.data, axis=-1, zi=self.zi)
        return block.replace(data=y)

    def reset(self) -> None:
        self.zi[:] = 0.0

    @property
    def latency_samples(self) -> float:
        return float(iir_group_delay_samples(self.sos, self.ref_hz, self.fs))

    @property
    def latency_note(self) -> str:
        return (f"IIR, {self.sos.shape[0]} sections: group delay quoted at {self.ref_hz:g} Hz "
                f"(exact at that frequency; it varies across the pass band)")

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "n_sections": int(self.sos.shape[0]), "n_channels": self.n_channels,
                "fs": self.fs, "ref_hz": self.ref_hz}


class Notch(CausalSOS):
    """A stated-width IIR notch at the mains frequency, delay quoted at ``ref_hz``.

    A notch applied inside the loop removes the line -- and, as lesson L7.5 says of devices that
    notch before the file exists, it also removes the channel-quality diagnostic that the line noise
    was. That is why the quality gate runs before this step, on the raw block (``pf-notch-hole-in-band``).
    """

    def __init__(self, fs: float, mains_hz: float, n_channels: int, *, ref_hz: float, q: float = 30.0,
                 name: str = "notch") -> None:
        from scipy.signal import iirnotch, tf2sos

        b, a = iirnotch(float(mains_hz), float(q), fs=float(fs))
        super().__init__(tf2sos(b, a), n_channels, fs=fs, ref_hz=ref_hz, name=name)
        self.mains_hz = float(mains_hz)
        self.q = float(q)

    def describe(self) -> dict[str, Any]:
        return dict(super().describe(), mains_hz=self.mains_hz, q=self.q)
