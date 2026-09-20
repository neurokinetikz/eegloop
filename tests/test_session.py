"""A whole session: run, log, read back, replay as a donor, record and re-open as an asset."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eegloop.present import CallbackPresenter, ConsolePresenter
from eegloop.session import Recorder, SessionReader, export_fif, run_protocol, scrub_paths, validate_protocol
from eegloop.sources import ReplaySource, SyntheticSource

PROTO = {
    "name": "alpha-up", "seed": 20260920, "block_samples": 32, "processing_ms": 10.0,
    "source": {"kind": "synthetic", "scenario": "alpha-schedule", "pace": "fast"},
    "signal": {"band": [8, 12], "channels": ["TP9", "TP10"], "n_taps": 129},
    "quality": {"enabled": True, "window_s": 1.0, "hold_s": 0.5, "mains_hz": 60},
    "baseline": {"mode": "fixed"},
    "reward": {"mode": "continuous", "lo_z": -1.0, "hi_z": 2.0},
    "phases": [{"kind": "calibrate", "duration_s": 20}, {"kind": "train", "duration_s": 100}],
}


def _no_machine_paths(out: Path):
    for f in ("session.json", "events.jsonl", "protocol-resolved.json"):
        text = (out / f).read_text(encoding="utf-8")
        for prefix in ("/" + "Users/", "/" + "home/", "/tmp/", "/private/"):
            assert prefix not in text, (f, prefix)


def test_scrub_paths_keeps_two_segments():
    # the machine-shaped prefixes are assembled at runtime so this file itself carries none (§10.10)
    mac, linux = "/" + "Users" + "/someone", "/" + "home" + "/u"
    assert scrub_paths(f"wrote {mac}/x/y/sessions/alpha") == "wrote .../sessions/alpha"
    assert scrub_paths(f'  File "{linux}/proj/a.py", line 3') == '  File ".../proj/a.py", line 3'
    assert scrub_paths("relative/path is fine") == "relative/path is fine"


def test_run_writes_a_session_the_reader_opens_and_the_feedback_rises_on_the_planted_bursts(tmp_path):
    p = validate_protocol(PROTO, source="mem")
    seen = []
    result = run_protocol(p, presenter=CallbackPresenter(lambda shown, **kw: seen.append((kw["t_s"], shown, kw["phase"]))),
                          out_dir=tmp_path / "s1")
    assert result.status == "ok" and result.out_dir == "s1" and result.sealed_sham
    out = tmp_path / "s1"
    for f in ("session.json", "events.jsonl", "signal.npz", "protocol-resolved.json"):
        assert (out / f).exists(), f
    _no_machine_paths(out)
    r = SessionReader(out)
    t, v = r.feedback
    assert t.size == result.stats["n_blocks"] and np.isfinite(v).all()
    # calibration shows nothing; training shows a bar
    phases = r.signal["phase"]
    assert np.isnan(r.shown[phases == "calibrate"]).all()
    train = (phases == "train") & r.signal["gate_ok"]
    assert np.isfinite(r.shown[train]).mean() > 0.95
    # the feedback is higher during the planted alpha bursts than outside them
    src = SyntheticSource("alpha-schedule", seed=p.seed, duration_s=121.0)
    inside = np.zeros_like(t, dtype=bool)
    for b in src.truth["alpha_bursts"]:
        inside |= (t >= b["onset_s"] + 1.0) & (t <= b["onset_s"] + b["duration_s"])
    outside = train & ~inside & (t > 25.0)
    assert np.nanmean(r.shown[inside & train]) > np.nanmean(r.shown[outside]) + 0.3
    # the log carries the signal in full, the budget with a measured processing row, and the baseline
    s = r.session
    assert "8–12 Hz" in s["signal"] and s["budget"]["rows"][0]["samples"] == 64.0
    assert "processing_measured_ms" in s["budget"] and s["budget"]["processing_measured_ms"]["n"] == t.size
    assert s["baseline"]["mode"] == "fixed" and s["baseline"]["n"] > 100
    assert any(e["kind"] == "baseline" for e in r.events()) and [e["kind"] for e in r.events("phase")] == ["phase", "phase"]
    assert seen and seen[-1][2] == "train" and all(np.isnan(sh) for _, sh, ph in seen if ph == "calibrate")


def test_yoked_sham_replays_the_donor_and_inverted_reflects(tmp_path):
    p = validate_protocol(PROTO, source="mem")
    run_protocol(p, out_dir=tmp_path / "donor")
    donor = SessionReader(tmp_path / "donor")
    # the yoked participant is a different stream (a different source seed); the donor's values are what they see
    yoked = validate_protocol(dict(PROTO, name="yoked", source=dict(PROTO["source"], seed=7),
                                   sham={"mode": "sham-yoked", "donor": str(tmp_path / "donor")}), source="mem")
    run_protocol(yoked, out_dir=tmp_path / "yoked")
    y = SessionReader(tmp_path / "yoked")
    # what the yoked participant was shown came from the donor's values on the same schedule, block for block
    n = min(y.signal["shown_values"].size, donor.feedback[1].size)
    assert np.allclose(y.signal["shown_values"][:n], donor.feedback[1][:n])
    assert not np.allclose(y.signal["values"][:n], donor.feedback[1][:n])  # its own veridical values differ
    inv = validate_protocol(dict(PROTO, name="inv", sham={"mode": "sham-inverted"}), source="mem")
    run_protocol(inv, out_dir=tmp_path / "inv")
    i = SessionReader(tmp_path / "inv")
    train = i.signal["phase"] == "train"
    assert np.corrcoef(i.signal["values"][train], i.signal["shown_values"][train])[0, 1] < -0.99
    # the mode is sealed in the log and recoverable only with the seed
    from eegloop.feedback import ShamPolicy

    assert ShamPolicy.unblind(i.sealed_sham, p.seed) == "sham-inverted"
    _no_machine_paths(tmp_path / "yoked")


def test_gate_closures_are_logged_and_the_display_freezes(tmp_path):
    proto = dict(PROTO, name="blinky", source={"kind": "synthetic", "scenario": "blinks", "pace": "fast"},
                 quality={"enabled": True, "window_s": 1.0, "hold_s": 0.5, "mains_hz": 60, "policy": "freeze"})
    p = validate_protocol(proto, source="mem")
    result = run_protocol(p, out_dir=tmp_path / "b")
    r = SessionReader(tmp_path / "b")
    gates = r.events("gate")
    assert gates and result.stats["n_gated"] == len(gates)
    # every closure after warm-up sits within two seconds after a planted blink (the blink, then the hold)
    onsets = SyntheticSource("blinks", seed=p.seed, duration_s=121.0).truth["blink_onsets_s"]
    late = [g for g in gates if g["t_s"] > 2.0]
    assert late and all(any(-0.2 <= g["t_s"] - o <= 2.0 for o in onsets) for g in late)
    assert any("blink" in g["labels"] for g in late)
    # the un-notched mains line is 'suspect', not a closure: most training blocks are fed back and say so
    states, warm = r.signal["gate_state"], r.signal["times_s"] < p.quality.window_s  # the window is not yet full
    assert (states == "suspect").mean() > 0.9 and r.signal["gate_ok"][(states == "suspect") & ~warm].all()
    assert not r.signal["gate_ok"][warm].any()  # warming up is suspect too, and not yet fed back
    # while gated, the shown value is held at its previous value
    shown, ok, phases = r.shown, r.signal["gate_ok"], r.signal["phase"]
    for k in np.nonzero(~ok & (phases == "train"))[0]:
        if k > 0 and np.isfinite(shown[k - 1]):
            assert shown[k] == shown[k - 1]


def test_record_raw_writes_the_site_asset_format_that_replay_reads_back(tmp_path):
    proto = dict(PROTO, name="rec", output={"dir": "sessions", "record_raw": True},
                 phases=[{"kind": "calibrate", "duration_s": 3}, {"kind": "train", "duration_s": 3}])
    p = validate_protocol(proto, source="mem")
    run_protocol(p, out_dir=tmp_path / "rec")
    r = SessionReader(tmp_path / "rec")
    assert r.session["raw"] == {"bin": "raw.bin", "sidecar": "raw.json"}
    side = json.loads((tmp_path / "rec" / "raw.json").read_text())
    assert side["units"] == "uV" and side["dtype"] == "float32-le" and side["channels"] == ["TP9", "AF7", "AF8", "TP10"]
    assert side["label_source"] == "none" and side["license"].startswith("private")
    assert side["n_samples"] * 4 * 4 == side["bytes"] == (tmp_path / "rec" / "raw.bin").stat().st_size
    assert (tmp_path / "rec" / "raw.timestamps.npy").exists()
    # the recording opens exactly as a shipped asset does
    src = ReplaySource.from_asset(tmp_path / "rec" / "raw.bin", pace="fast")
    assert src.info.fs == 256.0 and src.n_samples == side["n_samples"]
    data, fs, names, ts = r.raw
    assert data.shape == (4, side["n_samples"]) and ts is not None and ts.size == side["n_samples"]
    mne = pytest.importorskip("mne")
    fif = export_fif(tmp_path / "rec" / "raw.bin")
    raw = mne.io.read_raw_fif(fif, preload=True, verbose="ERROR")
    assert raw.info["sfreq"] == 256.0 and raw.ch_names == ["TP9", "AF7", "AF8", "TP10"]
    assert np.allclose(raw.get_data()[:, :100] * 1e6, data[:, :100], atol=1e-3)


def test_recorder_counts_gaps_and_console_presenter_prints(tmp_path, capsys):
    src = SyntheticSource("gaps", duration_s=90.0, pace="fast")
    rec = Recorder(tmp_path, src.info, name="g")
    src.start()
    while (b := src.read()) is not None:
        rec.write(b)
    b_path, s_path = rec.close()
    side = json.loads(s_path.read_text())
    assert side["gaps"] == [{"sample_index": src.truth["gap"]["index"] + 64, "n_missing": 64}]
    assert side["n_samples"] == src.n_samples
    pres = ConsolePresenter(width=10)
    pres.start(src.info)
    pres.update(0.5, z=0.1, gated=False, t_s=1.0, phase="train")
    pres.update(0.5, z=0.1, gated=True, t_s=2.0, phase="train")
    pres.stop()
    out = capsys.readouterr().out
    assert "synthetic" in out and "GATE CLOSED" in out


def test_replay_protocol_over_the_shipped_asset():
    pytest.importorskip("yaml")
    from eegloop.session import load_protocol

    cfg = Path(__file__).resolve().parents[1] / "configs" / "alpha-up-replay.yaml"
    p = load_protocol(cfg)
    asset = (cfg.parent / p.source.path).resolve()
    if not asset.exists():
        pytest.skip("shipped asset not present")
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        result = run_protocol(p, out_dir=Path(d) / "r")
        assert result.status == "ok" and result.budget["total_ms"] == pytest.approx(610.0)
        assert result.stats["n_blocks"] == 1600 // 32
