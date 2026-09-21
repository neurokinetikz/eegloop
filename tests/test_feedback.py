"""The five decisions as objects: each one honest about what it does to a learning claim."""
from __future__ import annotations

import numpy as np
import pytest

from eegloop import StreamInfo
from eegloop.feedback import (
    CREDNF_ITEMS, SHAM_MODES, AdaptiveBaseline, ContinuousMapping, FixedBaseline, ShamPolicy, SignalSpec,
    ThresholdReward, build_chain,
)


def test_fixed_baseline_is_fixed(rng):
    vals = rng.normal(10.0, 2.0, 200)
    b = FixedBaseline.from_values(vals)
    assert b.centre == pytest.approx(np.median(vals)) and b.n == 200
    z0 = b.z(b.centre + b.spread)
    b.update(1000.0)
    assert b.z(b.centre + b.spread) == z0 == pytest.approx(1.0)
    assert "does not move" in b.note and b.describe()["mode"] == "fixed"
    with pytest.raises(ValueError):
        FixedBaseline.from_values([1.0])


def test_adaptive_baseline_follows_the_participant():
    b = AdaptiveBaseline(2.0, 256.0, 32)
    assert b.z(5.0) != b.z(5.0)  # nan before there is a history
    for v in [1.0, 1.0, 1.0, 1.0]:
        b.update(v)
    z_low = b.z(2.0)
    for v in [10.0] * 40:
        b.update(v)
    assert b.centre == pytest.approx(10.0) and b.z(10.0) == pytest.approx(0.0)
    assert z_low > 0 and "near threshold by construction" in b.note


def test_continuous_mapping_clips():
    m = ContinuousMapping(-1.0, 2.0)
    assert m.map(-1.0) == 0.0 and m.map(2.0) == 1.0 and m.map(0.5) == pytest.approx(0.5)
    assert m.map(5.0) == 1.0 and m.map(float("nan")) != m.map(float("nan"))
    assert ContinuousMapping(-1.0, 2.0, clip=False).map(5.0) == 2.0
    with pytest.raises(ValueError):
        ContinuousMapping(2.0, 1.0)


def test_threshold_reward_needs_dwell_and_respects_refractory():
    r = ThresholdReward(1.0, dwell_s=0.5, refractory_s=2.0, on_miss="zero")
    t = 0.0
    events = []
    for k in range(40):
        z = 1.5 if k >= 4 else 0.0     # above threshold from 0.5 s on
        e = r.update(z, t)
        if e:
            events.append(e)
        t += 0.125
    assert len(events) == 2                       # one at dwell, one after the refractory period
    assert events[0].t_s == pytest.approx(1.0) and events[0].dwell_s == pytest.approx(0.5)
    assert events[1].t_s - events[0].t_s >= 2.0
    assert r.map(1.5) == 1.0
    r.update(0.0, t)
    assert r.map(0.0) == 0.0 and r.n_rewards == 2


def test_sham_policy_modes_seal_and_unblind():
    donor = np.arange(10.0)
    assert set(SHAM_MODES) == {"veridical", "sham-yoked", "sham-band", "sham-inverted"} and len(CREDNF_ITEMS) == 6
    y = ShamPolicy("sham-yoked", donor=donor, seed=7)
    assert y.value(99.0, 3, centre=5.0) == 3.0 and y.value(99.0, 13, centre=5.0) == 3.0  # replayed on the schedule
    inv = ShamPolicy("sham-inverted", seed=7)
    assert inv.value(7.0, 0, centre=5.0) == 3.0 and inv.value(7.0, 0, centre=None) == 7.0
    assert "calibration baseline" in inv.note and "divergence" in inv.note
    band = ShamPolicy("sham-band", control_band=(16.0, 20.0))
    assert band.adapt_spec(SignalSpec()).band == (16.0, 20.0)
    assert ShamPolicy("veridical").adapt_spec(SignalSpec()).band == (8.0, 12.0)
    token = inv.seal()
    assert len(token) == 16 and ShamPolicy.unblind(token, 7) == "sham-inverted"
    with pytest.raises(ValueError):
        ShamPolicy.unblind(token, 8)
    with pytest.raises(ValueError):
        ShamPolicy("sham-yoked")
    with pytest.raises(ValueError):
        ShamPolicy("placebo")


def test_signal_spec_describes_itself_in_full():
    spec = SignalSpec(band=(8.0, 12.0), channels=("TP9", "TP10"), n_taps=129, smooth_s=0.25)
    s = spec.describe(256.0, 64)
    for piece in ("8–12 Hz", "TP9, TP10", "129 taps", "250.0 ms", "root-mean-square", "250 ms", "4.0 Hz", "0.25 s"):
        assert piece in s, piece
    assert spec.picks(("TP9", "AF7", "AF8", "TP10")) == [0, 3]
    with pytest.raises(ValueError, match="not in the stream"):
        spec.picks(("O1", "O2"))


def test_build_chain_is_quality_first_and_lists_the_smoother_as_measured():
    from eegloop.session import QualityConfig

    info = StreamInfo(256.0, ("TP9", "AF7", "AF8", "TP10"), kind="synthetic")
    spec = SignalSpec(channels=("TP9",), reference="average", smooth_s=0.5)
    chain = build_chain(spec, info, 32, quality=QualityConfig(mains_hz=60.0))
    kinds = [s.kind for s in chain.steps]
    assert kinds == ["quality", "reference", "causal_filter", "envelope", "envelope"]
    assert [n for n, _ in chain.inexact] == ["quality", "smoother"]
    assert chain.exact_latency_samples == 64.0 and chain.reset_on_gap_samples == 129
    plain = build_chain(spec, info, 32, quality=None, include_smoother=False)
    assert [s.kind for s in plain.steps] == ["reference", "causal_filter", "envelope"]


def test_smoother_shifts_the_measured_probe_delay_and_the_exact_rows_do_not_move():
    """A smoother has no single delay, so the budget lists it as measured; the probe is what measures it.
    Without one the value delay equals the arithmetic; with one it is later by an amount of the order of the
    time constant, and the expected (exact) value delay is the same number both times."""
    from eegloop import CausalFIR, Chain, fir_taps, measure_loop_delay

    fs, B = 256.0, 64
    factory = lambda: Chain([CausalFIR(fir_taps(129, (8.0, 12.0), fs), 1)])  # noqa: E731
    plain = measure_loop_delay(factory, fs=fs, block_samples=B, centre_hz=10.0, alignments=4)
    smoothed = measure_loop_delay(factory, fs=fs, block_samples=B, centre_hz=10.0, alignments=4, smooth_s=0.5)
    block_ms = B / fs * 1000.0
    assert abs(plain["value_ms"] - plain["expected_value_ms"]) <= block_ms
    assert smoothed["expected_value_ms"] == pytest.approx(plain["expected_value_ms"])
    # the energy centroid of a 1.2-s burst through a one-pole smoother with tau = 0.5 s arrives about half a tau
    # later (measured: ~240 ms); the cost is real, of the order of tau, and NOT in the exact rows
    extra = smoothed["value_ms"] - plain["value_ms"]
    assert 0.25 * 500.0 <= extra <= 1.5 * 500.0, extra


def test_threshold_rewards_land_within_the_budget_of_the_planted_onsets(tmp_path):
    """Proposal §8 C gate: threshold crossings land within budget ± one block of the planted onsets. The
    alpha-schedule scenario plants two bursts; a threshold protocol with no dwell must reward each within
    the chain's declared delay of its onset, and never reward in the twenty seconds before the first."""
    from eegloop.session import SessionReader, run_protocol, validate_protocol
    from eegloop.sources import SyntheticSource

    proto = {
        "name": "thr", "seed": 20260920, "block_samples": 32, "processing_ms": 10.0,
        "source": {"kind": "synthetic", "scenario": "alpha-schedule", "pace": "fast"},
        "signal": {"band": [8, 12], "channels": ["TP9", "TP10"], "n_taps": 129},
        "quality": {"enabled": False},
        "baseline": {"mode": "fixed"},
        "reward": {"mode": "threshold", "threshold_z": 3.0, "dwell_s": 0.0, "refractory_s": 5.0, "on_miss": "zero"},
        "phases": [{"kind": "calibrate", "duration_s": 20}, {"kind": "train", "duration_s": 100}],
    }
    p = validate_protocol(proto, source="mem")
    result = run_protocol(p, out_dir=tmp_path / "thr")
    assert result.status == "ok"
    onsets = [b["onset_s"] for b in SyntheticSource("alpha-schedule", seed=p.seed, duration_s=121.0).truth["alpha_bursts"]]
    rewards = [e["t_s"] for e in SessionReader(tmp_path / "thr").events("reward")]
    total_s, block_s = result.budget["total_ms"] / 1000.0, 32 / 256.0
    assert onsets == [30.0, 75.0] and rewards, rewards
    assert not [r for r in rewards if r < onsets[0]], rewards
    for onset in onsets:
        first = min((r for r in rewards if r >= onset), default=None)
        assert first is not None and onset + total_s - 2 * block_s <= first <= onset + total_s + block_s, (onset, first, total_s, rewards)
