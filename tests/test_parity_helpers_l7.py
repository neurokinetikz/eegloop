"""eegloop against notebooks/_shared/helpers_l7.py, the code it was ported from.

The notebook helpers are not edited (a published notebook executes them weekly in CI) and this
package does not import them (they are not a package). So the same arithmetic exists twice on
purpose, and this file is what makes that a checked duplication rather than a silent one. Under CI
the import is hard: a missing dependency fails the build rather than skipping the one test that
guards the copy. Locally, a machine without mne skips it and says so.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from eegloop import CausalFIR, Chain, RingBuffer, butter_sos, fir_taps, latency_budget, measure_loop_delay, run_offline, zero_phase_lookahead_samples

pytestmark = pytest.mark.parity

REPO = Path(__file__).resolve().parents[2]
SHARED = REPO / "notebooks" / "_shared"


def _helpers_l7():
    if not (SHARED / "helpers_l7.py").exists():
        pytest.skip("notebooks/_shared/helpers_l7.py not present")
    if str(SHARED) not in sys.path:
        sys.path.insert(0, str(SHARED))
    if os.environ.get("CI", "").lower() in ("1", "true"):
        import helpers_l7  # a hard import: in CI a skipped guard is no guard
    else:
        pytest.importorskip("mne", reason="helpers_l7 imports mne; install the dev extra to run the parity test")
        import helpers_l7
    return helpers_l7


@pytest.fixture(scope="module")
def L7():
    return _helpers_l7()


def test_spec_is_the_one_helpers_pin(L7, spec):
    s = L7.LATENCY_SPEC
    assert (s["sfreq_hz"], tuple(s["band_hz"]), s["fir_taps"], s["block_samples"], s["processing_ms"], s["ref_hz"]) == \
        (spec["fs"], spec["band"], spec["n_taps"], spec["block_samples"], spec["processing_ms"], spec["ref_hz"])


def test_ring_buffer_parity(L7, rng):
    ours, theirs = RingBuffer(2, 50), L7.RingBuffer(2, 50)
    for _ in range(9):
        block = rng.standard_normal((2, int(rng.integers(1, 40))))
        ours.push(block)
        theirs.push(block)
    for n in (1, 10, 50, 80):
        assert np.array_equal(ours.latest(n), theirs.latest(n))


def test_causal_fir_parity(L7, signal, spec):
    taps = fir_taps(spec["n_taps"], spec["band"], spec["fs"])
    assert np.allclose(taps, L7.fir_taps())
    ours, theirs = CausalFIR(taps, 3), L7.OnlineFIR(taps, n_channels=3)
    B = spec["block_samples"]
    for i in range(0, signal.shape[1] - B, B):
        from eegloop import Block
        a = ours.process(Block(signal[:, i:i + B], spec["fs"])).data
        b = theirs.process(signal[:, i:i + B])
        assert np.max(np.abs(a - b)) < 1e-12


def test_latency_budget_parity(L7, spec):
    kw = dict(fs=spec["fs"], n_taps=spec["n_taps"], block_samples=spec["block_samples"], processing_ms=spec["processing_ms"])
    for conv in ("block-period", "first-to-last"):
        ours, theirs = latency_budget(convention=conv, **kw), L7.latency_budget(convention=conv, **kw)
        for key in ("total_ms", "mean_ms", "total_other_convention_ms", "update_rate_hz", "other_convention"):
            assert ours[key] == pytest.approx(theirs[key]) if isinstance(theirs[key], float) else ours[key] == theirs[key], key
        assert [r["ms"] for r in ours["rows"]] == pytest.approx([r["ms"] for r in theirs["rows"]])
    ours = latency_budget(phase="zero-phase-live", **kw)
    theirs = L7.latency_budget(phase="zero-phase-live", **kw)
    assert ours["total_ms"] == pytest.approx(theirs["total_ms"])
    sos = butter_sos(4, spec["band"], spec["fs"])
    ours = latency_budget(fs=spec["fs"], sos=sos, ref_hz=spec["ref_hz"], block_samples=32, processing_ms=10.0)
    theirs = L7.latency_budget(sos=sos, ref_hz=spec["ref_hz"])
    assert ours["rows"][0]["samples"] == pytest.approx(theirs["rows"][0]["samples"], abs=1e-9)


def test_zero_phase_lookahead_parity(L7, spec):
    for order in (2, 4, 6, 8):
        sos = butter_sos(order, spec["band"], spec["fs"])
        assert zero_phase_lookahead_samples(sos, fir=False) == L7.zero_phase_lookahead_samples(sos, fir=False)


def test_offline_loop_parity(L7, signal, spec):
    taps = fir_taps(spec["n_taps"], spec["band"], spec["fs"])
    x = signal[:1]
    for smooth in (0.0, 0.5):
        ours = run_offline(Chain([CausalFIR(taps, 1)]), x, fs=spec["fs"], block_samples=spec["block_samples"], smooth_s=smooth)
        theirs = L7.alpha_feedback(x[0], fs=spec["fs"], taps=taps, block_samples=spec["block_samples"], smooth_s=smooth)
        assert np.max(np.abs(ours["values"] - theirs["values"])) < 1e-12
        assert np.max(np.abs(ours["held"] - theirs["held"])) < 1e-12
        assert np.array_equal(ours["times_s"], theirs["times_s"])
        assert np.max(np.abs(ours["filtered"][0] - theirs["filtered"])) < 1e-12


def test_loop_delay_probe_parity(L7, spec):
    taps = fir_taps(spec["n_taps"], spec["band"], spec["fs"])
    ours = measure_loop_delay(lambda: Chain([CausalFIR(taps, 1)]), fs=spec["fs"], block_samples=spec["block_samples"],
                              centre_hz=float(np.mean(spec["band"])), alignments=8)
    theirs = L7.measure_loop_delay(fs=spec["fs"], taps=taps, block_samples=spec["block_samples"], alignments=8)
    for key in ("filter_ms", "value_ms", "expected_filter_ms", "expected_value_ms", "value_min_ms", "value_max_ms"):
        assert ours[key] == pytest.approx(theirs[key], abs=1e-9), key
    # and both agree with the arithmetic the lesson quotes — to the probe's own accuracy. The energy
    # centroid times a causal FIR to within 5e-4 samples (site/notes/widget-latency.md §2.2), which is
    # 3.1e-3 ms at 160 Hz; a tighter tolerance would be asking the measurement for more than it has.
    assert ours["filter_ms"] == pytest.approx(400.0, abs=5e-4 / spec["fs"] * 1000.0)
