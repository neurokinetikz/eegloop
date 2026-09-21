"""The gate finds what was planted, on the raw block, and never on a montage that cannot see it."""
from __future__ import annotations

import numpy as np
import pytest

from eegloop import Block, CausalFIR, Chain, fir_taps
from eegloop.sources import SyntheticSource, blocks
from eegloop.steps.quality import QualityGate


def _run(src, gate, B=32):
    out = []
    for b in blocks(src, B):
        q = gate.process(b).flags["quality"]
        out.append((b.t_start_s, b.t_end_s, q))
    return out


def _labelled(rows, label):
    return [(t0, t1) for t0, t1, q in rows if label in q["labels"]]


def test_blinks_trip_the_gate_on_a_frontal_montage_within_the_window():
    src = SyntheticSource("blinks", duration_s=100.0)
    gate = QualityGate(src.info.fs, src.info.ch_names, window_s=1.0, mains_hz=60.0)
    rows = _run(src, gate)
    hits = _labelled(rows, "blink")
    for onset in src.truth["blink_onsets_s"]:
        assert any(onset - 0.15 <= t1 <= onset + 1.0 + 0.15 for _, t1 in hits), f"blink at {onset}s not gated"
    # no false alarms away from the planted blinks
    stray = [t1 for _, t1 in hits if not any(abs(t1 - o) < 1.2 for o in src.truth["blink_onsets_s"])]
    assert stray == []
    # the verdict on a blink block is unusable, and the hold keeps it so for hold_s afterwards
    first = next(q for _, _, q in rows if "blink" in q["labels"])
    assert first["state"] == "unusable" and first["ok"] is False
    assert gate.describe()["frontal_channels"] == ["AF7", "AF8"]


def test_an_occipital_montage_cannot_see_a_blink_and_says_so():
    src = SyntheticSource("blinks", ch_names=("O1", "O2", "T7", "T8"), duration_s=60.0)
    gate = QualityGate(src.info.fs, src.info.ch_names, mains_hz=60.0)
    rows = _run(src, gate)
    assert _labelled(rows, "blink") == []
    assert gate.describe()["blink_detection"] == "no frontal channel in this montage"


def test_flat_channel_is_flagged_on_the_first_full_window():
    src = SyntheticSource("flat-channel", duration_s=10.0)
    gate = QualityGate(src.info.fs, src.info.ch_names, window_s=1.0)
    rows = _run(src, gate)
    ready = [q for _, _, q in rows if q["ready"]]
    assert ready and ready[0]["per_channel"]["TP10"]["labels"] == ["flat"]
    assert ready[0]["per_channel"]["TP10"]["status"] == "unusable" and ready[0]["ok"] is False
    assert all("flat" in q["per_channel"]["TP10"]["labels"] for q in ready)
    assert all("flat" not in q["per_channel"]["TP9"]["labels"] for q in ready)


def test_pops_are_flagged_on_the_channel_that_popped():
    src = SyntheticSource("pops", duration_s=60.0)
    gate = QualityGate(src.info.fs, src.info.ch_names, window_s=1.0)
    rows = _run(src, gate)
    for pop in src.truth["pops"]:
        near = [q for t0, t1, q in rows if pop["onset_s"] - 0.1 <= t1 <= pop["onset_s"] + 1.2]
        assert any("pop" in q["per_channel"][pop["channel"]]["labels"] for q in near), pop
        others = [c for c in src.info.ch_names if c != pop["channel"]]
        assert not any("pop" in q["per_channel"][c]["labels"] for q in near for c in others)


def test_strong_line_noise_is_suspect_not_unusable():
    src = SyntheticSource("line-noise", duration_s=20.0)
    gate = QualityGate(src.info.fs, src.info.ch_names, window_s=2.0, mains_hz=60.0)
    rows = _run(src, gate)
    ready = [q for _, _, q in rows if q["ready"]]
    assert all("line-noise" in q["labels"] for q in ready)
    assert all(q["per_channel"]["TP9"]["status"] == "suspect" for q in ready)
    assert all(q["per_channel"]["TP9"]["metrics"]["line_ratio"] > 10 for q in ready)


def test_a_clean_stream_is_good_once_warmed_up():
    src = SyntheticSource("clean", duration_s=40.0)
    gate = QualityGate(src.info.fs, src.info.ch_names, window_s=1.0, mains_hz=60.0)
    rows = _run(src, gate)
    ready = [q for _, _, q in rows if q["ready"]]
    ok_frac = np.mean([q["ok"] for q in ready])
    assert ok_frac > 0.95, ok_frac
    assert rows[0][2]["labels"] == ["warming-up"] and rows[0][2]["ok"] is False


def test_a_reported_loss_is_a_gap_label_and_a_gap_resets_the_chain():
    src = SyntheticSource("gaps", duration_s=90.0)
    gate = QualityGate(src.info.fs, src.info.ch_names, window_s=1.0)
    fir = CausalFIR(fir_taps(65, (8.0, 12.0), src.info.fs), src.info.n_channels)
    chain = Chain([gate, fir], reset_on_gap_samples=32)
    seen = []
    for b in blocks(src, 32):
        out = chain.process(b)
        if b.dropped_before:
            seen.append(out)
    assert len(seen) == 1
    q = seen[0].flags["quality"]
    assert "gap" in q["labels"] and q["state"] == "unusable"
    assert seen[0].flags["chain_reset"] == 64 and chain.n_resets == 1


def test_hold_keeps_the_gate_closed_after_a_violation():
    fs, names = 256.0, ("AF7", "AF8", "TP9", "TP10")
    gate = QualityGate(fs, names, window_s=0.5, hold_s=1.0, min_good_channels=1)
    rng = np.random.default_rng(3)
    t = 0.0
    states = []
    for k in range(int(6 * fs / 32)):
        x = rng.standard_normal((4, 32)) * 8.0
        if 30 <= k < 35:
            x[0] = 0.0  # five flat blocks on AF7: the 0.5-s window fills with zeros and the criterion fires
        out = gate.process(Block(x, fs, seq=k, t_start_s=t))
        states.append((out.t_end_s, out.flags["quality"]))
        t += 32 / fs
    flat = [t1 for t1, q in states if "flat" in q["labels"]]
    assert flat, "the flat window never fired"
    last_flat = flat[-1]
    # closed (not ok) for hold_s after the LAST violating block, then open again
    closed = [t1 for t1, q in states if not q["ok"] and t1 >= last_flat]
    reopened = [t1 for t1, q in states if q["ok"] and t1 > last_flat]
    assert closed and closed[-1] >= last_flat + 1.0 - 32 / fs - 1e-9
    assert reopened and reopened[0] >= last_flat + 1.0 - 1e-9


def test_gate_is_quality_first_and_declares_a_decision_delay():
    gate = QualityGate(256.0, ("AF7", "AF8"), window_s=1.0)
    assert gate.kind == "quality" and gate.latency_samples is None
    assert "decision" in gate.latency_note
    with pytest.raises(ValueError, match="cannot follow"):
        Chain([CausalFIR(fir_taps(9, (8.0, 12.0), 256.0), 2), gate])


def test_gate_labels_are_the_shared_artifact_labels_plus_the_two_online_states():
    """The gate ports data/scripts/detectors.py; its labels must be the notebook helpers' names, so a page
    that says 'blink' or 'line-noise' means the same thing on the live gate and in the offline atlas.
    ``gap`` and ``warming-up`` are online-only states with no offline counterpart, and are the only
    labels allowed to differ."""
    import ast
    from pathlib import Path

    from eegloop.steps.quality import QUALITY_LABELS

    helpers = Path(__file__).resolve().parents[2] / "notebooks" / "_shared" / "helpers.py"
    if not helpers.exists():
        pytest.skip("notebooks/_shared/helpers.py not present")
    line = next(ln for ln in helpers.read_text(encoding="utf-8").splitlines() if ln.startswith("ARTIFACT_LABELS = "))
    shared = set(ast.literal_eval(line.split("=", 1)[1].strip()))
    assert set(QUALITY_LABELS) - {"gap", "warming-up"} <= shared, set(QUALITY_LABELS) - shared
