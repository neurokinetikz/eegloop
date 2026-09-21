"""Reference, spatial, envelope and feature steps: exact where they say so, honest where they cannot be."""
from __future__ import annotations

import numpy as np
import pytest

from eegloop import Block, CausalFIR, Chain, fir_taps, latency_budget_for
from eegloop.steps.envelope import BandPower, BlockRMS, Smoother
from eegloop.steps.features import FeatureExtractor
from eegloop.steps.reference import Reference
from eegloop.steps.spatial import SpatialFilter

NAMES = ("TP9", "AF7", "AF8", "TP10")


def _block(rng, n=64, fs=256.0):
    return Block(rng.standard_normal((4, n)) * 10.0, fs)


def test_average_reference_subtracts_the_mean_and_costs_one_rank(rng):
    b = _block(rng)
    out = Reference("average", NAMES).process(b)
    assert np.allclose(out.data.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(out.data, b.data - b.data.mean(axis=0, keepdims=True))
    step = Reference("average", NAMES)
    assert step.latency_samples == 0.0 and step.rank_cost == 1


def test_channel_reference_subtracts_the_named_channels(rng):
    b = _block(rng)
    out = Reference("channels", NAMES, channels=("TP9", "TP10")).process(b)
    ref = (b.data[0] + b.data[3]) / 2
    assert np.allclose(out.data, b.data - ref[None, :])
    assert Reference("none", NAMES).process(b) is b
    with pytest.raises(ValueError):
        Reference("channels", NAMES)
    with pytest.raises(ValueError):
        Reference("mastoid", NAMES)


def test_spatial_filter_applies_frozen_weights(rng):
    b = _block(rng)
    W = rng.standard_normal((2, 4))
    step = SpatialFilter(W, NAMES, ("c1", "c2"), kind_label="csp")
    out = step.process(b)
    assert np.allclose(out.data, W @ b.data) and out.flags["ch_names"] == ("c1", "c2")
    assert step.latency_samples == 0.0 and step.describe()["rank"] == 2
    with pytest.raises(ValueError):
        SpatialFilter(W, NAMES[:3])


def test_frozen_ica_removes_exactly_the_excluded_component(rng):
    # an orthogonal mixing matrix makes the unmixing its transpose, so the arithmetic is exact
    Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    mixing, unmixing = Q, Q.T
    sources = rng.standard_normal((4, 500))
    x = mixing @ sources
    step = SpatialFilter.from_ica(unmixing, mixing, exclude=[1], ch_names=NAMES)
    cleaned = step.process(Block(x, 256.0)).data
    expected = mixing[:, [0, 2, 3]] @ sources[[0, 2, 3]]
    assert np.allclose(cleaned, expected, atol=1e-10)
    assert step.describe()["rank"] == 3
    P = np.eye(4) - np.outer(np.ones(4), np.ones(4)) / 4
    ssp = SpatialFilter.from_projector(P, NAMES)
    assert ssp.out_names == NAMES and ssp.kind_label == "ssp"


def test_block_rms_is_per_channel_and_exact(rng):
    b = _block(rng)
    out = BlockRMS().process(b)
    assert np.allclose(out.flags["envelope"], np.sqrt(np.mean(b.data ** 2, axis=1)))
    assert out.data is b.data and BlockRMS().latency_samples == 0.0


def test_smoother_declares_no_single_delay_and_tracks_a_step(rng):
    fs, B = 256.0, 32
    sm = Smoother(0.5, fs, B)
    assert sm.latency_samples is None and "measured" in sm.latency_note
    vals = []
    for k in range(80):
        v = np.array([0.0, 0.0]) if k < 40 else np.array([10.0, 10.0])
        out = sm.process(Block(np.zeros((2, B)), fs).with_flag("envelope", v))
        vals.append(out.flags["envelope"][0])
    # after one time constant (0.5 s = 4 blocks) the response sits near 63 % of the step
    assert vals[39] == 0.0 and vals[43] == pytest.approx(10.0 * (1 - np.exp(-1)), rel=0.05)
    with pytest.raises(ValueError):
        Smoother(0.5, fs, B).process(Block(np.zeros((2, B)), fs))


def test_band_power_finds_the_ten_hertz_line():
    fs = 256.0
    t = np.arange(int(4 * fs)) / fs
    x = np.vstack([20.0 * np.sin(2 * np.pi * 10.0 * t), 2.0 * np.sin(2 * np.pi * 25.0 * t)])
    bp = BandPower(fs, 2, {"alpha": (8.0, 12.0), "beta": (20.0, 30.0)}, window_s=2.0)
    last = None
    for i in range(0, x.shape[1], 64):
        last = bp.process(Block(x[:, i:i + 64], fs))
    p = last.flags["bandpower"]
    assert p["alpha"][0] > 20 * p["beta"][0] and p["beta"][1] > 20 * p["alpha"][1]
    assert bp.latency_samples is None and np.allclose(last.flags["envelope"], np.sqrt(p["alpha"]))


def test_feature_extractor_variants(rng):
    b = _block(rng)
    lv = FeatureExtractor("log-var").process(b).flags["features"]
    assert np.allclose(lv, np.log(np.var(b.data, axis=1) + 1e-12))
    b2 = b.with_flag("bandpower", {"alpha": np.array([4.0, 2.0, 1.0, 8.0]), "beta": np.array([2.0, 2.0, 2.0, 2.0])})
    ratio = FeatureExtractor("band-ratio", bands=("alpha", "beta")).process(b2).flags["features"]
    assert np.allclose(ratio, [2.0, 1.0, 0.5, 4.0])
    both = FeatureExtractor("band-power", bands=("alpha", "beta")).process(b2).flags["features"]
    assert both.shape == (8,)
    with pytest.raises(ValueError):
        FeatureExtractor("band-ratio", bands=("alpha",))
    with pytest.raises(ValueError):
        FeatureExtractor("band-power").process(b)


def test_a_full_chain_budgets_only_the_exact_steps(rng):
    fs, B = 256.0, 64
    from eegloop.steps.quality import QualityGate

    chain = Chain([
        QualityGate(fs, NAMES, window_s=1.0),
        Reference("average", NAMES),
        CausalFIR(fir_taps(129, (8.0, 12.0), fs), 4, name="fir"),
        BlockRMS(),
        Smoother(0.25, fs, B),
    ])
    b = latency_budget_for(chain, fs=fs, block_samples=B, processing_ms=10.0)
    assert b["rows"][0]["samples"] == 64.0                      # only the FIR contributes
    assert [m["name"] for m in b["measured"]] == ["smoother"] and [m["name"] for m in b["decision"]] == ["quality"]
    assert b["decision"][0]["window_s"] == 1.0 and b["decision"][0]["ms"] == 1000.0
    assert b["total_ms"] == pytest.approx(64 / fs * 1000 + B / fs * 1000 + 10.0)
    out = chain.process(_block(rng, B))
    assert "quality" in out.flags and "envelope" in out.flags
