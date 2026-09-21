"""The latency budget against the shipped SciPy reference, and the lesson's own worked example.

Every number here is one the course already quotes: the reference designs in
assets/traces.json (the site's shipped SciPy reference, mirrored into this package), and the
610.00 ms pipeline of lesson L7.3. This package is the third implementation of the same quantities;
the house pattern is that independent implementations agree on published numbers.
"""
from __future__ import annotations

import numpy as np
import pytest

from eegloop import (
    BUFFER_CONVENTIONS,
    CausalFIR,
    CausalSOS,
    Chain,
    butter_sos,
    fir_group_delay_samples,
    fir_taps,
    iir_group_delay_by_phase,
    iir_group_delay_samples,
    latency_budget,
    latency_budget_for,
    print_budget,
    zero_phase_lookahead_samples,
)


def test_reference_fir_designs(reference):
    fs = reference["sfreq"]
    for d in (x for x in reference["designs"] if x["family"] == "fir"):
        gd = fir_group_delay_samples(d["numtaps"])
        assert gd == d["group_delay_samples"], d["id"]
        assert gd / fs * 1000.0 == pytest.approx(d["group_delay_ms"]), d["id"]
        # the kernel really is symmetric, which is what makes the delay flat across frequency
        taps = fir_taps(d["numtaps"], reference["band_hz"], fs)
        assert np.allclose(taps, taps[::-1])


def test_reference_iir_designs_section_summed(reference):
    fs, band, at = reference["sfreq"], reference["band_hz"], reference["group_delay_at_hz"]
    for d in (x for x in reference["designs"] if x["family"] == "iir-butter"):
        sos = butter_sos(d["order"], band, fs)
        assert sos.shape[0] == d["n_sections"], d["id"]
        gd = iir_group_delay_samples(sos, at, fs)
        assert gd == pytest.approx(d["group_delay_samples"], abs=5e-3), d["id"]
        # the reference's own criterion: the phase derivative agrees to < 0.05 samples
        assert abs(iir_group_delay_by_phase(sos, at, fs) - gd) < 0.05, d["id"]


def test_iir_delay_grows_with_order(reference):
    """The instability the per-section sum avoids: an expanded cascade reports order 8 FASTER than 6."""
    fs, band, at = reference["sfreq"], reference["band_hz"], reference["group_delay_at_hz"]
    delays = [iir_group_delay_samples(butter_sos(k, band, fs), at, fs) for k in (2, 4, 6, 8)]
    assert all(b > a for a, b in zip(delays, delays[1:])), delays


def test_reference_zero_phase_lookaheads(reference):
    fs, band = reference["sfreq"], reference["band_hz"]
    for d in (x for x in reference["designs"] if x["family"] == "zero-phase"):
        sos = butter_sos(d["order"], band, fs)
        la = zero_phase_lookahead_samples(sos, fir=False, tol=0.01)
        assert la == d["lookahead_samples"], d["id"]
        assert la / fs * 1000.0 == pytest.approx(d["lookahead_ms"]), d["id"]
    fir = fir_taps(129, band, fs)
    assert zero_phase_lookahead_samples(fir, fir=True) == 128.0


def test_the_l73_pipeline_is_610_ms(spec):
    b = latency_budget(fs=spec["fs"], n_taps=spec["n_taps"], block_samples=spec["block_samples"],
                       processing_ms=spec["processing_ms"])
    assert [r["ms"] for r in b["rows"]] == pytest.approx([400.0, 200.0, 10.0])
    assert b["total_ms"] == pytest.approx(610.0)
    assert b["mean_ms"] == pytest.approx(510.0)
    assert b["total_other_convention_ms"] == pytest.approx(603.75)
    assert b["other_convention"] == "first-to-last"
    assert b["update_rate_hz"] == pytest.approx(5.0)
    other = latency_budget(fs=spec["fs"], n_taps=spec["n_taps"], block_samples=spec["block_samples"],
                           processing_ms=spec["processing_ms"], convention="first-to-last")
    assert other["total_ms"] == pytest.approx(603.75)
    assert other["total_other_convention_ms"] == pytest.approx(610.0)


def test_zero_phase_live_costs_1010_ms(spec):
    b = latency_budget(fs=spec["fs"], n_taps=spec["n_taps"], block_samples=spec["block_samples"],
                       processing_ms=spec["processing_ms"], phase="zero-phase-live")
    assert b["rows"][0]["samples"] == 128.0
    assert b["total_ms"] == pytest.approx(1010.0)


def test_iir_budget_quotes_a_frequency(spec):
    sos = butter_sos(4, spec["band"], spec["fs"])
    b = latency_budget(fs=spec["fs"], sos=sos, ref_hz=spec["ref_hz"], block_samples=spec["block_samples"],
                       processing_ms=spec["processing_ms"])
    assert b["rows"][0]["ms"] == pytest.approx(204.5446, abs=5e-2)
    with pytest.raises(ValueError):
        latency_budget(fs=spec["fs"], sos=sos, block_samples=32, processing_ms=10.0)


def test_budget_for_a_chain_matches_the_typed_budget(spec, capsys):
    taps = fir_taps(spec["n_taps"], spec["band"], spec["fs"])
    chain = Chain([CausalFIR(taps, 1, name="fir")])
    b = latency_budget_for(chain, fs=spec["fs"], block_samples=spec["block_samples"],
                           processing_ms=spec["processing_ms"])
    typed = latency_budget(fs=spec["fs"], n_taps=spec["n_taps"], block_samples=spec["block_samples"],
                           processing_ms=spec["processing_ms"])
    for key in ("total_ms", "mean_ms", "total_other_convention_ms", "update_rate_hz"):
        assert b[key] == pytest.approx(typed[key]), key
    assert b["exact_steps"][0]["samples"] == 64.0 and b["measured"] == []
    print_budget(b, title="chain")
    assert "610.00" in capsys.readouterr().out


def test_budget_for_a_chain_keeps_inexact_steps_out_of_the_total(spec):
    class Smoother:
        kind, name = "envelope", "smoother"
        latency_samples, latency_note = None, "one-pole smoother: measured with the probe, not summed"

        def process(self, b):
            return b

        def reset(self):
            pass

        def describe(self):
            return {"name": self.name}

    fir = CausalFIR(fir_taps(spec["n_taps"], spec["band"], spec["fs"]), 1, name="fir")
    sos = CausalSOS(butter_sos(2, spec["band"], spec["fs"]), 1, fs=spec["fs"], ref_hz=spec["ref_hz"], name="iir")
    chain = Chain([fir, sos, Smoother()])
    b = latency_budget_for(chain, fs=spec["fs"], block_samples=32, processing_ms=10.0)
    assert b["rows"][0]["samples"] == pytest.approx(64.0 + sos.latency_samples)
    assert [m["name"] for m in b["measured"]] == ["smoother"]
    assert b["total_ms"] == pytest.approx((64.0 + sos.latency_samples) / spec["fs"] * 1000 + 200.0 + 10.0)


def test_unknown_convention_is_refused(spec):
    with pytest.raises(ValueError):
        latency_budget(fs=spec["fs"], n_taps=129, block_samples=32, processing_ms=10.0, convention="hold")
    assert set(BUFFER_CONVENTIONS) == {"block-period", "first-to-last", "plus-hold"}
