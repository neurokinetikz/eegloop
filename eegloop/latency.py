"""The end-to-end latency of a real-time loop, decomposed into what causes it.

Ported from ``notebooks/_shared/helpers_l7.py`` (``BUFFER_CONVENTIONS``, the group-delay functions and
``latency_budget``, 2026-09-20). The arithmetic and the dictionary keys are unchanged so that the
notebooks' ``print_budget`` and ``plot_latency_budget`` render a budget from here without knowing
which package produced it, and so that ``tests/test_parity_helpers_l7.py`` can hold the two to
agreement. What changed: every sampling rate is an argument (the helpers default to lesson L7.3's
pipeline; a library must not), and :func:`latency_budget_for` computes the filter row from a chain's
own declared delays instead of from a tap count typed beside it.

**End-to-end latency is the time between an event in the signal and the feedback reflecting it.**
An event at sample ``n`` appears in the filter output at ``n + D`` where ``D`` is the group delay;
that output sample then waits for its block to complete; then the stated processing delay::

    worst case = D/fs + buffer + P
    mean       = D/fs + buffer/2 + P

Three causes, three rows, three different remedies -- which is why the total alone is not the answer.
The buffer term has two defensible conventions and this module prints both; the reason one is the
default is written in :data:`BUFFER_CONVENTIONS`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .steps import Chain

__all__ = [
    "BUFFER_CONVENTIONS",
    "fir_group_delay_samples",
    "iir_group_delay_samples",
    "iir_group_delay_by_phase",
    "zero_phase_lookahead_samples",
    "latency_budget",
    "latency_budget_for",
    "print_budget", "with_probe"]

#: The readings of the buffer term, and why the course teaches one of them.
BUFFER_CONVENTIONS: dict[str, str] = {
    "block-period":
        "B / fs — the whole acquisition period of the block.  A sample is not available until its "
        "acquisition period completes, so a block of B samples is handed over at the end of the last "
        "sample's period and the oldest sample in it has waited a full B / fs.  This is the "
        "convention w-latency-budget defaults to, the one its shipped asset defines, and the one "
        "site/notes/integration-phase4.md settles on.",
    "first-to-last":
        "(B - 1) / fs — the difference between the arrival instants of the first and the last sample "
        "of the block.  A real quantity, and not the wait: it is what 'how long did this sample "
        "wait' means if a sample is treated as an instant rather than as an interval.  One sample "
        "smaller, which is 6.25 ms at 160 Hz for any block size.",
    "plus-hold":
        "B / fs again on top, for the time the displayed value stays on screen before the next "
        "update.  That answers 'how stale is the number in front of me right now', which is a "
        "different question from 'how late does an event appear', and it is not budgeted here.",
}


def fir_group_delay_samples(n_taps: int) -> float:
    """``(N - 1) / 2``: a linear-phase FIR delays every frequency by the same amount."""
    return (int(n_taps) - 1) / 2.0


def iir_group_delay_samples(sos: np.ndarray, f_hz: float | np.ndarray, fs: float) -> float | np.ndarray:
    """Group delay of an SOS filter at one frequency, **summed over the sections**.

    Expanding the sections into one transfer function and calling ``scipy.signal.group_delay`` on
    it is numerically unstable at high order -- at order 8 the expansion returns a *smaller* delay
    than at order 6, which would teach that a steeper filter is faster. Group delay is additive
    over cascaded sections, so summing per section is both stable and exact.
    :func:`iir_group_delay_by_phase` is the independent check.
    """
    from scipy.signal import group_delay

    w = np.atleast_1d(2 * np.pi * np.asarray(f_hz, dtype=float) / float(fs))
    total = np.zeros(len(w))
    for section in np.atleast_2d(sos):
        _, g = group_delay((section[:3], section[3:]), w=w)
        total += g
    return float(total[0]) if np.ndim(f_hz) == 0 else total


def iir_group_delay_by_phase(sos: np.ndarray, f_hz: float, fs: float, *, half_width_hz: float = 0.1,
                             n: int = 2001) -> float:
    """Group delay from ``-d(phase)/d(omega)`` of the full cascade -- the independent check."""
    from scipy.signal import sosfreqz

    fs = float(fs)
    w = np.linspace(2 * np.pi * (f_hz - half_width_hz) / fs, 2 * np.pi * (f_hz + half_width_hz) / fs, n)
    _, H = sosfreqz(sos, worN=w)
    return float(-np.gradient(np.unwrap(np.angle(H)), w)[n // 2])


def zero_phase_lookahead_samples(kernel_or_sos: np.ndarray, *, fir: bool, tol: float = 0.01,
                                 n: int = 4000) -> float:
    """How far into the **future** a zero-phase filter must see before it can answer.

    For an FIR of ``N`` taps run forwards and backwards the answer is exact: ``N - 1`` samples, the
    support of the kernel. For an IIR there is no exact answer, only a truncation criterion -- how
    far before an impulse the forward-backward impulse response still exceeds ``tol`` of its peak.
    A tighter tolerance demands a longer wait, so the criterion is part of the number.
    """
    from scipy.signal import sosfiltfilt

    if fir:
        return float(len(np.asarray(kernel_or_sos)) - 1)
    imp = np.zeros(int(n))
    imp[n // 2] = 1.0
    y = sosfiltfilt(kernel_or_sos, imp)
    peak = float(np.max(np.abs(y)))
    pre = np.where(np.abs(y[: n // 2]) > tol * peak)[0]
    return float(n // 2 - int(pre[0])) if len(pre) else 0.0


def _rows(filt_samples: float, filt_label: str, *, fs: float, block_samples: int, processing_ms: float,
          convention: str) -> dict[str, Any]:
    if convention not in BUFFER_CONVENTIONS:
        raise ValueError(f"convention must be one of {sorted(BUFFER_CONVENTIONS)}")
    B, P = int(block_samples), float(processing_ms)
    buf_samples = float(B) if convention == "block-period" else float(B - 1)
    rows = [
        {"row": "filter", "detail": filt_label, "samples": filt_samples, "ms": filt_samples / fs * 1000.0},
        {"row": "buffer", "detail": f"{B} samples, {convention}", "samples": buf_samples,
         "ms": buf_samples / fs * 1000.0},
        {"row": "processing", "detail": "stated, not measured", "samples": P * fs / 1000.0, "ms": P},
    ]
    total = sum(r["ms"] for r in rows)
    other = "first-to-last" if convention == "block-period" else "block-period"
    other_buf = float(B - 1) if other == "first-to-last" else float(B)
    return {
        "fs_hz": fs, "block_samples": B, "processing_ms": P,
        "convention": convention, "convention_note": BUFFER_CONVENTIONS[convention],
        "rows": rows, "total_ms": total,
        "mean_ms": total - rows[1]["ms"] / 2.0,
        "total_other_convention_ms": total - rows[1]["ms"] + other_buf / fs * 1000.0,
        "other_convention": other,
        "update_rate_hz": fs / B,
    }


def latency_budget(*, fs: float, block_samples: int, processing_ms: float, n_taps: int | None = None,
                   sos: np.ndarray | None = None, ref_hz: float | None = None,
                   convention: str = "block-period", phase: str = "causal") -> dict[str, Any]:
    """The budget for a stated filter: an FIR by tap count, an IIR by its sections, or the FIR run
    zero-phase live (``phase='zero-phase-live'``), where the filter row becomes the look-ahead.

    Same keys as ``helpers_l7.latency_budget``. Every row is arithmetic and the total is their sum.
    """
    fs = float(fs)
    if sos is not None:
        if ref_hz is None:
            raise ValueError("an IIR budget needs ref_hz: its group delay is quoted at one frequency")
        filt = float(iir_group_delay_samples(sos, float(ref_hz), fs))
        label = f"IIR group delay at {float(ref_hz):g} Hz"
    elif n_taps is None:
        raise ValueError("give n_taps for an FIR or sos for an IIR")
    elif phase == "zero-phase-live":
        filt = float(int(n_taps) - 1)
        label = f"zero-phase FIR look-ahead (N - 1), {int(n_taps)} taps"
    else:
        filt = fir_group_delay_samples(int(n_taps))
        label = f"FIR group delay (N - 1) / 2, {int(n_taps)} taps"
    out = _rows(filt, label, fs=fs, block_samples=block_samples, processing_ms=processing_ms,
                convention=convention)
    out.update({"n_taps": None if n_taps is None else int(n_taps), "phase": phase})
    return out


def latency_budget_for(chain: "Chain", *, fs: float, block_samples: int, processing_ms: float,
                       convention: str = "block-period") -> dict[str, Any]:
    """The budget computed from a chain's own declared delays.

    The filter row is the sum of every step whose delay is exact (see
    :class:`eegloop.steps.OnlineStep`), with each contributor named in ``exact_steps``. Steps whose
    delay is not one number are listed under ``measured`` with their note and are **not** in
    ``total_ms``: that row is filled by the loopback probe (:func:`eegloop.probe.measure_loop_delay`;
    :func:`with_probe` writes its result back). ``decision`` lists the steps that delay a *decision*
    rather than the signal -- the quality gate's window and hold here; the BCI runner appends the
    decoder's window and the dwell -- each with its seconds, and none of them in ``total_ms`` either,
    because a held display or a late verdict is not a late sample.
    """
    fs = float(fs)
    exact = chain.exact_steps
    filt = chain.exact_latency_samples
    label = (" + ".join(f"{s.name} {s.latency_samples:g}" for s in exact) + " samples"
             if exact else "no exact delay in the chain")
    out = _rows(filt, label, fs=fs, block_samples=block_samples, processing_ms=processing_ms,
                convention=convention)
    out.update({
        "n_taps": None, "phase": "causal",
        "exact_steps": [{"name": s.name, "samples": float(s.latency_samples), "note": s.latency_note}
                        for s in exact],
        "measured": [], "decision": [],
    })
    for step in chain.steps:
        if step.latency_samples is not None:
            continue
        entry: dict[str, Any] = {"name": step.name, "note": step.latency_note}
        if getattr(step, "kind", None) == "quality":
            d = step.describe()
            window = d.get("window_s")
            entry.update({"window_s": window, "hold_s": d.get("hold_s"),
                          "ms": None if window is None else float(window) * 1000.0})
            out["decision"].append(entry)
        else:
            out["measured"].append(entry)
    return out


def with_probe(budget: dict[str, Any], probe: dict[str, Any]) -> dict[str, Any]:
    """Write a loopback-probe result (:func:`eegloop.probe.measure_loop_delay`) into a budget.

    The exact rows stay what the arithmetic says; ``probe`` records what was measured -- the filter
    delay and the value delay, each beside its expectation -- and ``probe_within_one_block`` says
    whether the measured value delay is within one block period of the arithmetic, the criterion
    lesson L7.16's rubric applies. The budget is modified in place and returned.
    """
    block_ms = float(budget["block_samples"]) / float(budget["fs_hz"]) * 1000.0
    budget["probe"] = {
        "filter_ms": float(probe["filter_ms"]), "expected_filter_ms": float(probe["expected_filter_ms"]),
        "value_ms": float(probe["value_ms"]), "expected_value_ms": float(probe["expected_value_ms"]),
        "value_range_ms": [float(probe["value_min_ms"]), float(probe["value_max_ms"])],
        "alignments": int(probe["alignments"]), "burst_hz": float(probe["burst_hz"]),
        "note": "measured by the loopback probe on this chain; the gap between value_ms and expected_value_ms is the cost of the steps listed under measured",
    }
    budget["probe_within_one_block"] = abs(budget["probe"]["value_ms"] - budget["probe"]["expected_value_ms"]) <= block_ms
    return budget


def print_budget(budget: dict[str, Any], *, title: str = "") -> None:
    """The budget as a table, with the other convention and the mean printed beside the total."""
    if title:
        print(title)
    print(f"{'row':12s} {'detail':46s} {'samples':>9s} {'ms':>9s}")
    print("-" * 80)
    for r in budget["rows"]:
        print(f"{r['row']:12s} {r['detail'][:46]:46s} {r['samples']:9.2f} {r['ms']:9.2f}")
    print("-" * 80)
    print(f"{'TOTAL':12s} {'worst case, ' + budget['convention']:46s} "
          f"{budget['total_ms'] * budget['fs_hz'] / 1000:9.2f} {budget['total_ms']:9.2f}")
    print(f"{'':12s} {'worst case, ' + budget['other_convention']:46s} "
          f"{'':>9s} {budget['total_other_convention_ms']:9.2f}")
    print(f"{'':12s} {'mean over block position':46s} {'':>9s} {budget['mean_ms']:9.2f}")
    for m in budget.get("measured", []):
        print(f"{'measured':12s} {m['name']}: {m['note']}")
    for m in budget.get("decision", []):
        print(f"{'decision':12s} {m['name']}: {m['note']}")
    if budget.get("probe"):
        pr = budget["probe"]
        print(f"{'probe':12s} value delay measured {pr['value_ms']:.2f} ms, expected {pr['expected_value_ms']:.2f} ms"
              + (" — within one block" if budget.get("probe_within_one_block") else " — NOT within one block"))
    print(f"\nloop update rate {budget['update_rate_hz']:.2f} Hz "
          f"({budget['block_samples']} samples at {budget['fs_hz']:.0f} Hz)")
