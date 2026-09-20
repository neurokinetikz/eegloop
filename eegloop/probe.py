"""Measure a loop's delay with a loopback probe instead of trusting the arithmetic.

Ported and generalised from ``notebooks/_shared/helpers_l7.py`` (``energy_centroid``,
``measure_loop_delay``, the per-block value of ``alpha_feedback``, 2026-09-20): the probe here runs
through *any* chain, on any number of channels, and its expectations come from the chain's own
declared delays.

One Hann-windowed packet at a stated frequency is injected into silence and timed by its **energy
centroid**, which is what you would do on a real rig. Two lags come back:

* ``filter_ms`` -- the centroid of the continuously filtered stream minus the centroid of the input.
  For a linear-phase FIR this equals ``(N - 1) / 2`` samples exactly.
* ``value_ms`` -- the centroid of the per-block feedback values, timestamped at the instant each
  block completes, minus the centroid of the input. Its expectation is the exact delay plus
  ``(B + 1) / 2`` samples: the block's own centre of mass sits at ``(B - 1) / 2`` and the value
  does not exist until the block completes at ``B``.

Cross-correlating the output against an offline reference does **not** work here and fails quietly:
both signals are narrow-band, so the correlation peaks at every multiple of the carrier period. The
centroid does not have that failure mode. The probe is repeated at every phase of the block grid, so
the spread over ``alignments`` reports how much of the total depends on where an event happens to
fall inside a block.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .block import Block
from .steps import Chain

__all__ = ["make_probe", "energy_centroid", "run_offline", "measure_loop_delay"]


def make_probe(fs: float, *, centre_hz: float, cycles: int = 12, onset_s: float = 8.0,
               duration_s: float = 20.0, offset_samples: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``(t, x)``: silence with one Hann-windowed packet of ``cycles`` at ``centre_hz`` starting at
    ``onset_s`` (shifted by ``offset_samples`` to walk the block grid)."""
    fs = float(fs)
    n = int(duration_s * fs)
    n_burst = int(round(cycles / centre_hz * fs))
    w = np.hanning(n_burst) * np.sin(2 * np.pi * centre_hz * np.arange(n_burst) / fs)
    t = np.arange(n) / fs
    x = np.zeros(n)
    i0 = int(round(onset_s * fs)) + int(offset_samples)
    x[i0:i0 + n_burst] = w
    return t, x


def energy_centroid(t: np.ndarray, w: np.ndarray) -> float:
    """Time at the centre of mass of ``w**2`` -- how a loopback probe is timed on a real rig."""
    w = np.maximum(np.asarray(w, dtype=float) ** 2, 0.0)
    return float(np.sum(np.asarray(t, dtype=float) * w) / np.sum(w))


def run_offline(chain: Chain, x: np.ndarray, *, fs: float, block_samples: int,
                smooth_s: float = 0.0) -> dict[str, Any]:
    """Replay a recording through a chain in fixed blocks and return what a participant would see.

    The feedback value for a block is the **root mean square of that block's chain output** over
    every channel, so the loop adds no smoothing beyond the block itself and its measured delay can
    be compared directly with :func:`eegloop.latency.latency_budget_for`. ``smooth_s > 0`` adds a
    one-pole exponential smoother whose time constant is stated; it buys a steadier number and
    **costs more latency**, and the probe measures how much rather than this function estimating it.

    Returns the per-block values, the held sample-wise feedback (the value for a block is shown
    during the *following* block), the continuously filtered stream, and the block times.
    """
    fs = float(fs)
    B = int(block_samples)
    x = np.atleast_2d(np.asarray(x, dtype=float))
    n = x.shape[1] - x.shape[1] % B
    alpha = 1.0 if smooth_s <= 0 else float(1.0 - np.exp(-(B / fs) / smooth_s))
    filtered = np.zeros_like(x)
    values, times, state = [], [], None
    for k, i in enumerate(range(0, n, B)):
        block = Block(x[:, i:i + B], fs, seq=k, sample_index=i, t_start_s=i / fs)
        y = chain.process(block).data
        filtered[:, i:i + B] = y
        v = float(np.sqrt(np.mean(y ** 2)))
        state = v if state is None else state + alpha * (v - state)
        values.append(state)
        times.append((i + B) / fs)          # the instant the block is complete and the value exists
    values_a, times_a = np.asarray(values), np.asarray(times)
    held = np.zeros(x.shape[1])
    for k, i in enumerate(range(0, n, B)):
        held[i + B: i + 2 * B] = values_a[k]  # shown during the NEXT block
    return {
        "values": values_a, "times_s": times_a, "held": held, "filtered": filtered,
        "block_samples": B, "sfreq": fs, "smooth_s": smooth_s,
        "n_blocks": len(values), "update_rate_hz": fs / B,
        "note": "the value for a block is shown during the following block, which is why the held "
                "trace is one block behind the block times.",
    }


def measure_loop_delay(chain_factory: Callable[[], Chain], *, fs: float, block_samples: int,
                       centre_hz: float, cycles: int = 12, onset_s: float = 8.0,
                       duration_s: float = 20.0, alignments: int | None = None,
                       smooth_s: float = 0.0) -> dict[str, Any]:
    """Time a probe through a fresh chain at every alignment of the block grid.

    ``chain_factory`` builds a new, unfiltered chain per alignment so no state leaks between runs.
    ``expected_filter_ms`` and ``expected_value_ms`` come from the chain's exact delays; a chain with
    inexact steps still gets measured -- that is the point -- and the gap between measured and
    expected is then the cost of those steps.
    """
    fs = float(fs)
    B = int(block_samples)
    alignments = B if alignments is None else int(alignments)
    filt_lags, val_lags = [], []
    probe: dict[str, Any] | None = None
    exact = chain_factory().exact_latency_samples
    for off in range(alignments):
        t, x = make_probe(fs, centre_hz=centre_hz, cycles=cycles, onset_s=onset_s,
                          duration_s=duration_s, offset_samples=off)
        fb = run_offline(chain_factory(), x, fs=fs, block_samples=B, smooth_s=smooth_s)
        t_in = energy_centroid(t, x)
        filt_lags.append((energy_centroid(t, fb["filtered"][0]) - t_in) * 1000.0)
        val_lags.append((energy_centroid(fb["times_s"], fb["values"]) - t_in) * 1000.0)
        if off == 0:
            probe = {"signal": x, "feedback": fb, "times_s": t,
                     "onset_s": (int(round(onset_s * fs)) + off) / fs}
    fl, vl = np.asarray(filt_lags), np.asarray(val_lags)
    out: dict[str, Any] = dict(probe or {})
    out.update({
        "filter_ms": float(fl.mean()), "filter_sd_ms": float(fl.std()),
        "filter_samples": float(fl.mean() * fs / 1000.0),
        "value_ms": float(vl.mean()), "value_sd_ms": float(vl.std()),
        "value_min_ms": float(vl.min()), "value_max_ms": float(vl.max()),
        "value_samples": float(vl.mean() * fs / 1000.0),
        "expected_filter_ms": exact / fs * 1000.0,
        "expected_value_ms": (exact + (B + 1) / 2.0) / fs * 1000.0,
        "alignments": alignments, "burst_hz": float(centre_hz), "burst_cycles": int(cycles),
        "smooth_s": smooth_s, "block_samples": B, "sfreq": fs,
        "criterion": "energy centroid of a Hann-windowed packet at the stated frequency",
        "delay_ms": float(vl.mean()),
    })
    return out
