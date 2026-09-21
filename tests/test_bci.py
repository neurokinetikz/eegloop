"""The BCI loop's parts: epochs equal to the sample, a frozen replay equal to the pipeline, a chance band
equal to the notebooks', decisions that find the planted answers."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import lfilter

from eegloop.bci import (
    BCILoop, Calibrator, DwellDecision, EpochCutter, FrozenDecoder, SSVEPDetector, StreamingPosterior, chance_interval,
    cut_epochs, fit_frozen, ssvep_cca, ssvep_decide, temporal_split,
)
from eegloop.bci.decoder import oas_covariance, tangent_space
from eegloop.sources import SyntheticSource
from eegloop.sources.base import blocks
from eegloop.sources.synthetic import SCENARIOS, make_synthetic_stream
from eegloop.steps import Chain
from eegloop.steps.causal_filter import CausalFIR, fir_taps

SEED = 20260920
REPO = Path(__file__).resolve().parents[2]
SHARED = REPO / "notebooks" / "_shared"


def _decode():
    if os.environ.get("CI", "").lower() in ("1", "true"):
        import pyriemann  # noqa: F401  a hard import: in CI a skipped guard is no guard
        import sklearn  # noqa: F401
    else:
        pytest.importorskip("sklearn", reason="install the decode extra")
        pytest.importorskip("pyriemann", reason="install the decode extra")


def _mi_offline(band=(8.0, 30.0), tmin=0.5, tmax=3.5):
    data, ts, cues, truth = make_synthetic_stream(SEED, duration_s=120.0, **SCENARIOS["mi-2class"])
    fs = truth["fs"]
    xf = lfilter(fir_taps(129, band, fs), [1.0], data, axis=1)
    X, y, t = cut_epochs(xf, fs, cues, tmin_s=tmin, tmax_s=tmax, classes=("left", "right"))
    return X, y, t, fs, tuple(truth["ch_names"]), truth


def _chain(fs, n_ch, band=(8.0, 30.0), reset=True):
    return Chain([CausalFIR(fir_taps(129, band, fs), n_ch)], reset_on_gap_samples=129 if reset else None)


# --------------------------------------------------------------------------------------------------- #
def test_frozen_arithmetic_matches_the_libraries(rng):
    _decode()
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.covariance import oas

    X = rng.standard_normal((12, 4, 300))
    C = oas_covariance(X)
    assert np.allclose(C, np.array([oas(x.T)[0] for x in X]), atol=1e-12)
    assert np.allclose(C, Covariances(estimator="oas").transform(X), atol=1e-12)
    ts = TangentSpace(metric="riemann").fit(C)
    assert np.allclose(tangent_space(C, ts.reference_), ts.transform(C), atol=1e-8)


def test_online_epochs_equal_the_offline_cut_to_the_sample():
    X_off, y_off, t_off, fs, names, _ = _mi_offline()
    src = SyntheticSource("mi-2class", seed=SEED, duration_s=120.0, pace="fast")
    chain = _chain(fs, len(names))
    cutter = EpochCutter(src.info, tmin_s=0.5, tmax_s=3.5)
    markers = src.marker_source()
    got = []
    for block in blocks(src, 32):
        got += cutter.push(chain.process(block), markers.read())
    assert len(got) == len(y_off) == 20 and cutter.n_cut == 20
    X_on = np.asarray([x for _, _, x in got])
    assert np.allclose(X_on, X_off, atol=1e-9)
    assert [lab for _, lab, _ in got] == [("left", "right")[k] for k in y_off]
    assert np.allclose([t for t, _, _ in got], t_off)
    assert cutter.clock == "sample"
    # cutting by the timestamps instead: the synthetic clock runs 200 ppm fast, so the late epochs land
    # samples early -- L7.11's alignment trap, measurable here
    src2 = SyntheticSource("mi-2class", seed=SEED, duration_s=120.0, pace="fast")
    chain2, ts_cutter, m2 = _chain(fs, len(names)), EpochCutter(src2.info, tmin_s=0.5, tmax_s=3.5, clock="timestamps"), src2.marker_source()
    got2 = []
    for block in blocks(src2, 32):
        got2 += ts_cutter.push(chain2.process(block), m2.read())
    assert len(got2) == 20 and src2.truth["drift_ppm"] == 200.0
    last = got2[-1][2]  # the cue at 105 s: the fast timestamps put its window ≈ 5 samples early
    shifts = [k for k in range(1, 12) if np.allclose(last[:, k:], X_off[-1][:, :-k], atol=1e-9)]
    assert shifts and shifts[0] >= 4, shifts


def test_cutter_drops_epochs_over_gaps_and_gated_stretches_and_keeps_the_rest_exact():
    X_off, y_off, t_off, fs, names, _ = _mi_offline()
    src = SyntheticSource("mi-2class", seed=SEED, duration_s=120.0, pace="fast", gap=(31.0, 64))
    assert src.truth["gap"]["n_missing"] == 64
    chain = _chain(fs, len(names), reset=False)  # a finite memory: 129 samples after the gap the filter has forgotten it
    cal = Calibrator(src.info, tmin_s=0.5, tmax_s=3.5, classes=("left", "right"), markers=src.marker_source())
    cal.mark_bad(40.0, 41.0)  # as the gate would: the cue at 40 s has a window [40.5, 43.5) that overlaps it
    for block in blocks(src, 32):
        cal.push(chain.process(block))
    X, y, t = cal.harvest()
    d = cal.describe()
    assert d["n_dropped"] == {"gap": 1, "gated": 1, "late": 0} and len(y) == 18
    assert 30.0 not in t and 40.0 not in t
    keep = ~np.isin(t_off, [30.0, 40.0])
    assert np.allclose(X, X_off[keep], atol=1e-9) and (y == y_off[keep]).all()


def test_fit_frozen_clears_chance_and_replays_the_pipeline_exactly(tmp_path):
    _decode()
    X, y, t, fs, names, _ = _mi_offline()
    for kind in ("csp_lda", "riemann_ts"):
        dec = fit_frozen(kind, X, y, classes=("left", "right"), fs=fs, ch_names=names, tmin_s=0.5, tmax_s=3.5,
                         folds=5, seed=SEED, band=(8.0, 30.0), filter_delay_samples=64.0)
        r = dec.report
        assert r["n_epochs"] == 20 and r["n_per_class"] == {"left": 10, "right": 10} and r["folds"] == 5
        assert r["cv_accuracy"] > r["chance"]["band_95"][1] and r["clears_chance"] and r["cv_auc"] > 0.9
        assert r["frozen_replay_max_abs_err"] < 1e-6 and "never shuffled" in r["cv_note"]
        assert dec.epoch_samples == 768 and dec.latency_samples == 768.0 and "64 samples" in dec.latency_note
        p = dec.predict_proba_batch(X)
        assert p.shape == (20, 2) and np.allclose(p.sum(axis=1), 1.0)
        assert (np.argmax(p, axis=1) == y).mean() == 1.0
        f = dec.save(tmp_path / f"{kind}.npz")
        back = FrozenDecoder.load(f)
        assert back.kind == kind and back.classes == ("left", "right") and back.report["cv_accuracy"] == r["cv_accuracy"]
        assert np.allclose(back.predict_proba_batch(X), p, atol=1e-12)
        assert back.predict(X[0]) in ("left", "right") and back.describe()["params"]["coef"] == [1, 4 if kind == "csp_lda" else 10]
    with pytest.raises(ValueError, match="never saw"):
        fit_frozen("csp_lda", X[y == 0], y[y == 0], classes=("left", "right"), fs=fs, ch_names=names, tmin_s=0.5, tmax_s=3.5)
    with pytest.raises(ValueError, match="not an eegloop decoder"):
        np.savez(tmp_path / "x.npz", a=1)
        FrozenDecoder.load(tmp_path / "x.npz")


def test_temporal_split_is_disjoint_and_non_adjacent():
    fit, apply = temporal_split(20, apply_frac=0.4, gap=1)
    assert fit.tolist() == list(range(11)) and apply.tolist() == list(range(12, 20))
    assert not set(fit) & set(apply) and apply[0] - fit[-1] == 2
    with pytest.raises(ValueError):
        temporal_split(3)


def test_chance_interval_is_helpers_l6_chance_band():
    if not (SHARED / "helpers_l6.py").exists():
        pytest.skip("notebooks/_shared/helpers_l6.py not present")
    if str(SHARED) not in sys.path:
        sys.path.insert(0, str(SHARED))
    if os.environ.get("CI", "").lower() in ("1", "true"):
        import helpers_l6  # a hard import in CI
    else:
        pytest.importorskip("mne")
        import helpers_l6
    for n, p in ((20, 0.5), (60, 0.25), (8, 0.5), (100, 0.5)):
        assert chance_interval(n, p) == helpers_l6.chance_band(n, p)


def test_dwell_decision_needs_dwell_and_respects_refractory():
    d = DwellDecision(("a", "b"), threshold=0.7, dwell_s=0.5, refractory_s=2.0)
    out = []
    t = 0.0
    for k in range(40):
        p = np.array([0.2, 0.8]) if k >= 4 else np.array([0.6, 0.4])
        r = d.update(p, t)
        if r:
            out.append(r)
        t += 0.125
    assert len(out) == 2 and out[0].label == "b" and out[0].t_s == pytest.approx(1.0) and out[0].dwell_s == pytest.approx(0.5)
    assert out[1].t_s - out[0].t_s >= 2.0 and out[0].mode == "sliding"
    assert d.update(np.array([0.9, 0.1]), t) is None  # a new class restarts the dwell
    with pytest.raises(ValueError):
        DwellDecision(("a", "b"), threshold=1.5)


def test_sliding_loop_decides_the_planted_imagery_after_a_temporal_split():
    _decode()
    X, y, t, fs, names, _ = _mi_offline(tmin=0.5, tmax=2.0)  # a window shorter than the imagery
    fit_idx = t < 65.0  # the first twelve cues; the loop is scored on the eight after 70 s
    dec = fit_frozen("csp_lda", X[fit_idx], y[fit_idx], classes=("left", "right"), fs=fs, ch_names=names,
                     tmin_s=0.5, tmax_s=2.0, folds=5, seed=SEED, filter_delay_samples=64.0)
    src = SyntheticSource("mi-2class", seed=SEED, duration_s=120.0, pace="fast")
    loop = BCILoop(src, _chain(fs, len(names)), block_samples=32, decoder=dec, mode="sliding", step_samples=32,
                   smooth_s=0.5, threshold=0.7, dwell_s=0.5, refractory_s=1.0, markers=src.marker_source(), grace_s=1.5)
    stats = loop.run([("test", 120.0)])
    held_out = [tr for tr in stats.trials if tr["t_cue_s"] >= 70.0]
    assert len(held_out) == 8
    decided = [tr for tr in held_out if tr["decided"]]
    assert len(decided) >= 7 and np.mean([tr["correct"] for tr in decided]) >= 0.85
    d = stats.describe(("left", "right"))
    assert d["n_posteriors"] > 800 and d["n_decisions"] >= 8 and d["online"]["chance_band_95"] is not None
    sp = StreamingPosterior(dec, step_samples=32, smooth_s=0.5)
    assert "no single delay" in sp.latency_note and sp.latency_samples == 384.0


def test_cue_locked_p300_classifies_each_cue_once():
    _decode()
    data, ts, cues, truth = make_synthetic_stream(SEED, duration_s=120.0, **SCENARIOS["p300"])
    fs, names = truth["fs"], tuple(truth["ch_names"])
    xf = lfilter(fir_taps(129, (1.0, 20.0), fs), [1.0], data, axis=1)
    X, y, t = cut_epochs(xf, fs, cues, tmin_s=0.0, tmax_s=0.8, classes=("standard", "target"))
    fit_idx = t < 65.0
    dec = fit_frozen("xdawn_lda", X[fit_idx], y[fit_idx], classes=("standard", "target"), fs=fs, ch_names=names,
                     tmin_s=0.0, tmax_s=0.8, folds=5, seed=SEED, n_components=2, filter_delay_samples=64.0)
    assert dec.report["cv_auc"] is not None and dec.report["frozen_replay_max_abs_err"] < 1e-6
    src = SyntheticSource("p300", seed=SEED, duration_s=120.0, pace="fast")
    loop = BCILoop(src, _chain(fs, len(names), band=(1.0, 20.0)), block_samples=32, decoder=dec, mode="cue-locked",
                   markers=src.marker_source())
    stats = loop.run([("test", 120.0)])
    n_cues = sum(1 for tc, _ in cues if tc + 0.8 <= 120.0)
    assert len(stats.decisions) == n_cues and all(tr["decided"] for tr in stats.trials) and len(stats.trials) == n_cues
    assert all(d["mode"] == "cue-locked" for d in stats.decisions)
    held = [tr for tr in stats.trials if tr["t_cue_s"] >= 65.0]
    assert np.mean([tr["correct"] for tr in held]) > 0.5


def test_ssvep_cca_finds_the_planted_flicker_and_the_alpha_caveat_is_real():
    data, ts, cues, truth = make_synthetic_stream(SEED, duration_s=90.0, **SCENARIOS["ssvep"])
    fs = truth["fs"]
    assert [f["freq_hz"] for f in truth["flicker"]] == [12.0, 15.0]
    win = lambda t0: data[:, int(t0 * fs):int((t0 + 4) * fs)]  # noqa: E731
    assert ssvep_decide(win(25.0), fs, (12.0, 15.0))[0] == 12.0
    assert ssvep_decide(win(65.0), fs, (12.0, 15.0))[0] == 15.0
    assert ssvep_decide(win(5.0), fs, (12.0, 15.0))[0] is None
    # the standing alpha correlates with a 10 Hz reference whether or not anything flickers
    assert all(ssvep_cca(win(t0), fs, (10.0,))[10.0] > 0.35 for t0 in (5.0, 25.0, 65.0))
    src = SyntheticSource("ssvep", seed=SEED, duration_s=90.0, pace="fast")
    det = SSVEPDetector(fs, src.info.n_channels, (12.0, 15.0), window_s=4.0, step_samples=64)
    seen = {}
    for block in blocks(src, 32):
        r = det.update(block)
        if r is not None:
            seen[round(block.t_end_s, 2)] = r[0]
    at = lambda t0: seen[min(seen, key=lambda k: abs(k - t0))]  # noqa: E731
    assert at(30.0) == 12.0 and at(70.0) == 15.0 and at(50.0) is None
    assert "alpha" in det.latency_note and det.latency_samples == 4 * fs


def test_chance_interval_agrees_with_the_csp_widget_to_the_normal_approximation():
    """w-csp-explorer's csp.ts chanceInterval is the normal approximation with z = 1.96, clipped to [0, 1];
    the library's (helpers_l6's) is the exact binomial interval. They must agree to within one trial's
    worth of accuracy for every n a page reports, or a learner would see two chance bands."""
    import math

    for n in (20, 40, 45, 60, 100, 144, 509, 4000):
        for k in (2, 4):
            chance = 1.0 / k
            se = math.sqrt(chance * (1 - chance) / n)
            lo_w, hi_w = max(0.0, chance - 1.96 * se), min(1.0, chance + 1.96 * se)
            lo, hi = chance_interval(n, chance)
            assert abs(lo - lo_w) <= 1.0 / n + 1e-12 and abs(hi - hi_w) <= 1.0 / n + 1e-12, (n, k, (lo, hi), (lo_w, hi_w))


def test_decision_latency_is_the_window_within_one_block_and_the_filter_is_in_the_content():
    """Proposal §8 D gate, measured rather than assumed. Cue-locked: the decision is computed at the end of
    the block that completes the window, so its WALL latency after the cue is tmax within one block. The
    causal filter's 64 samples are not a wait on top of that: offline and online windows are both cut on
    the filtered stream's clock, so the filter delay is inside the window's content -- stated, exact, in
    the decoder's latency note -- and the information the decision uses ends 64 samples before the
    window does. Sliding: the wall latency is a distribution -- never under the dwell, its median at or
    under tmax + dwell + one block, its tail bounded only by the grace (the posterior may cross the
    threshold late in the window and the dwell starts there); a sliding decoder that scores well can also
    decide before the window lies wholly inside the imagery. That is why the runner states the window as
    the delay and the dwell as a choice, not an identity."""
    _decode()
    X, y, t, fs, names, _ = _mi_offline(tmin=0.5, tmax=2.0)
    fit_idx = t < 65.0
    dec = fit_frozen("csp_lda", X[fit_idx], y[fit_idx], classes=("left", "right"), fs=fs, ch_names=names,
                     tmin_s=0.5, tmax_s=2.0, folds=5, seed=SEED, filter_delay_samples=64.0)
    assert dec.filter_delay_samples == 64.0 and "64 samples" in dec.latency_note
    B, dwell, block_s = 32, 0.5, 32 / fs
    # cue-locked: wall latency = the end of the block that emitted the decision, minus the cue
    src = SyntheticSource("mi-2class", seed=SEED, duration_s=120.0, pace="fast")
    loop = BCILoop(src, _chain(fs, len(names)), block_samples=B, decoder=dec, mode="cue-locked",
                   markers=src.marker_source(), classes=("left", "right"), grace_s=1.5)
    emitted = []
    for b in blocks(src, B):
        if b.t_end_s > 120.0:
            break
        rec = loop.step(b, phase="test")
        if rec["decision"] is not None:
            emitted.append((b.t_end_s, rec["decision"].t_s - dec.tmax_s))   # (wall time, cue time)
    late = [(t_wall, t_cue) for t_wall, t_cue in emitted if t_cue >= 70.0]
    assert len(late) >= 6
    for t_wall, t_cue in late:
        lag = t_wall - t_cue
        assert dec.tmax_s - 1e-9 <= lag <= dec.tmax_s + block_s + 1e-9, (lag, dec.tmax_s, block_s)
    # sliding: an upper bound, and never before the dwell has elapsed
    src = SyntheticSource("mi-2class", seed=SEED, duration_s=120.0, pace="fast")
    loop = BCILoop(src, _chain(fs, len(names)), block_samples=B, decoder=dec, mode="sliding", step_samples=B,
                   smooth_s=0.0, threshold=0.7, dwell_s=dwell, refractory_s=1.0, markers=src.marker_source(), grace_s=1.5)
    stats = loop.run([("test", 120.0)])
    held_out = [tr for tr in stats.trials if tr["t_cue_s"] >= 70.0 and tr["decided"]]
    assert len(held_out) >= 6
    lags = sorted(tr["t_decision_s"] - tr["t_cue_s"] for tr in held_out)
    assert all(dwell <= lag <= dec.tmax_s + 1.5 + 1e-9 for lag in lags), lags          # the scoring window: tmax to tmax + grace
    assert float(np.median(lags)) <= dec.tmax_s + dwell + block_s + 1e-9, lags
