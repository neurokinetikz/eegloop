"""Causal filters with carried state, and the chain's two rules: order is argued, gaps reset."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import lfilter, sosfilt

from eegloop import CANONICAL_ONLINE_ORDER, Block, CausalFIR, CausalSOS, Chain, Notch, RingBuffer, butter_sos, fir_taps


def _blocks(x: np.ndarray, fs: float, B: int, **kw):
    n = x.shape[1] - x.shape[1] % B
    for k, i in enumerate(range(0, n, B)):
        yield Block(x[:, i:i + B], fs, seq=k, sample_index=i, t_start_s=i / fs, **kw)


@pytest.mark.parametrize("B", [1, 7, 32, 1000])
def test_blockwise_fir_equals_whole_signal(signal, spec, B):
    taps = fir_taps(spec["n_taps"], spec["band"], spec["fs"])
    step = CausalFIR(taps, n_channels=3)
    out = np.concatenate([step.process(b).data for b in _blocks(signal, spec["fs"], B)], axis=1)
    whole = lfilter(taps, [1.0], signal, axis=-1)[:, : out.shape[1]]
    # identical arithmetic in identical order: the timing changed, the signal did not
    assert np.max(np.abs(out - whole)) < 1e-9


@pytest.mark.parametrize("B", [1, 7, 32, 1000])
def test_blockwise_sos_equals_whole_signal(signal, spec, B):
    sos = butter_sos(4, spec["band"], spec["fs"])
    step = CausalSOS(sos, n_channels=3, fs=spec["fs"], ref_hz=spec["ref_hz"])
    out = np.concatenate([step.process(b).data for b in _blocks(signal, spec["fs"], B)], axis=1)
    whole = sosfilt(sos, signal, axis=-1)[:, : out.shape[1]]
    assert np.max(np.abs(out - whole)) < 1e-9


def test_per_block_filtering_without_state_is_the_pitfall(signal, spec):
    """Filtering each block independently resets the filter's memory at every boundary — that is
    pf-filter-across-boundaries at the block rate, and it must NOT equal the whole-signal result."""
    taps = fir_taps(spec["n_taps"], spec["band"], spec["fs"])
    B = spec["block_samples"]
    out = np.concatenate([CausalFIR(taps, 3).process(b).data for b in _blocks(signal, spec["fs"], B)], axis=1)
    whole = lfilter(taps, [1.0], signal, axis=-1)[:, : out.shape[1]]
    rms = float(np.sqrt(np.mean(whole ** 2)))
    assert np.max(np.abs(out - whole)) > 0.1 * rms


def test_fir_declares_its_exact_delay(spec):
    step = CausalFIR(fir_taps(spec["n_taps"], spec["band"], spec["fs"]), 1)
    assert step.latency_samples == (spec["n_taps"] - 1) / 2
    assert "exact" in step.latency_note
    assert step.describe()["n_taps"] == spec["n_taps"]


def test_sos_declares_delay_at_the_reference_frequency(spec):
    step = CausalSOS(butter_sos(4, spec["band"], spec["fs"]), 1, fs=spec["fs"], ref_hz=spec["ref_hz"])
    assert step.latency_samples == pytest.approx(32.727137, abs=5e-3)   # the shipped reference value
    assert "10 Hz" in step.latency_note and "varies" in step.latency_note


def test_notch_is_a_causal_sos_with_its_mains_recorded(spec):
    step = Notch(spec["fs"], 60.0, 2, ref_hz=spec["ref_hz"])
    b = step.process(Block(np.zeros((2, 64)), spec["fs"]))
    assert b.n_samples == 64 and step.describe()["mains_hz"] == 60.0
    assert step.latency_samples >= 0.0


def test_filters_reject_the_wrong_channel_count(spec):
    step = CausalFIR(fir_taps(65, spec["band"], spec["fs"]), n_channels=2)
    with pytest.raises(ValueError):
        step.process(Block(np.zeros((3, 32)), spec["fs"]))


def test_chain_resets_state_across_a_long_gap_and_not_a_short_one(signal, spec):
    taps = fir_taps(spec["n_taps"], spec["band"], spec["fs"])
    fir = CausalFIR(taps, 3)
    chain = Chain([fir], reset_on_gap_samples=64)
    chain.process(Block(signal[:, :32], spec["fs"]))
    assert np.any(fir.zi != 0.0)
    out = chain.process(Block(signal[:, 32:64], spec["fs"], dropped_before=10))
    assert "chain_reset" not in out.flags and chain.n_resets == 0
    out = chain.process(Block(signal[:, 64:96], spec["fs"], dropped_before=100))
    assert out.flags["chain_reset"] == 100 and chain.n_resets == 1
    # after the reset the state is what a fresh filter would have after ONE block
    fresh = CausalFIR(taps, 3)
    fresh.process(Block(signal[:, 64:96], spec["fs"]))
    assert np.allclose(fir.zi, fresh.zi)


class _Stub:
    """A step of any kind, for order tests."""

    def __init__(self, kind, name="stub"):
        self.kind, self.name = kind, name

    def process(self, block):
        return block

    def reset(self):
        pass

    latency_samples = None
    latency_note = "stub"

    def describe(self):
        return {"name": self.name}


def test_chain_enforces_the_canonical_order():
    Chain([_Stub("quality"), _Stub("causal_filter"), _Stub("envelope")])
    with pytest.raises(ValueError, match="cannot follow"):
        Chain([_Stub("envelope"), _Stub("causal_filter")])
    with pytest.raises(ValueError, match="expected one of"):
        Chain([_Stub("cleaning")])
    assert CANONICAL_ONLINE_ORDER[0] == "quality", "the gate must see the raw block"


def test_chain_separates_exact_from_inexact_latency(spec):
    fir = CausalFIR(fir_taps(spec["n_taps"], spec["band"], spec["fs"]), 1, name="fir")
    chain = Chain([_Stub("quality", "gate"), fir, _Stub("envelope", "smoother")])
    assert chain.exact_latency_samples == 64.0
    assert [s.name for s in chain.exact_steps] == ["fir"]
    assert [n for n, _ in chain.inexact] == ["gate", "smoother"]
    d = chain.describe()
    assert d[1]["latency_samples"] == 64.0 and d[2]["latency_samples"] is None


def test_ring_buffer_returns_newest_in_order(rng):
    ring = RingBuffer(2, capacity=100, fs=256.0)
    x = rng.standard_normal((2, 250))
    for i in range(0, 250, 37):
        ring.push(x[:, i:i + 37])
    assert ring.n_pushed == 250 and ring.filled == 100
    assert np.array_equal(ring.latest(100), x[:, 150:250])
    assert np.array_equal(ring.latest(10), x[:, 240:250])
    assert ring.latest_seconds(0.25).shape == (2, 64)
    assert ring.t_latest_s == pytest.approx(250 / 256.0)
    with pytest.raises(ValueError):
        ring.push(np.zeros((3, 4)))
