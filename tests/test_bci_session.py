"""Whole BCI sessions from the shipped protocols: calibrate, fit, freeze, apply, log, read back."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from eegloop.cli import main
from eegloop.present import CallbackPresenter
from eegloop.session import ProtocolError, SessionReader, load_protocol, run_protocol, validate_protocol

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(autouse=True)
def _deps():
    pytest.importorskip("yaml")
    if os.environ.get("CI", "").lower() in ("1", "true"):
        import pyriemann  # noqa: F401
        import sklearn  # noqa: F401
    else:
        pytest.importorskip("sklearn")
        pytest.importorskip("pyriemann")


def _no_machine_paths(out: Path):
    for f in ("session.json", "events.jsonl", "protocol-resolved.json"):
        text = (out / f).read_text(encoding="utf-8")
        for prefix in ("/" + "Users/", "/" + "home/", "/tmp/", "/private/"):
            assert prefix not in text, (f, prefix)


def test_imagery_protocol_calibrates_fits_applies_and_scores(tmp_path):
    p = load_protocol(CONFIGS / "mi-2class-synthetic.yaml")
    decisions = []
    result = run_protocol(p, presenter=CallbackPresenter(on_decision=decisions.append), out_dir=tmp_path / "mi")
    assert result.status == "ok" and result.sealed_sham is None
    s = result.stats
    assert s["n_epochs_calibration"] == 12 and s["fit_ms"] is not None
    o = s["online"]
    assert o["n_trials"] == 8 and o["n_decided"] >= 7 and o["accuracy"] >= 0.85 and o["chance_band_95"] is not None
    assert len(decisions) == s["n_decisions"] >= 8
    r = SessionReader(tmp_path / "mi")
    assert r.session["decoder"] == "decoder.npz" and (tmp_path / "mi" / "decoder.npz").exists()
    dec = r.decoder()
    assert dec.kind == "csp_lda" and dec.report["clears_chance"] and dec.report["n_epochs"] == 12
    t_post, post = r.posteriors
    assert post.shape[1] == 2 and t_post.min() > 69.0 and np.allclose(post.sum(axis=1), 1.0)
    kinds = [e["kind"] for e in r.events()]
    assert "decoder" in kinds and "decision" in kinds and "cue" in kinds and kinds.count("phase") == 2
    assert r.budget["decoder"]["latency_samples"] == 384.0 and "processing_measured_ms" in r.budget
    per = r.session["bci"]["calibration"]["n_per_class"]
    assert r.session["bci"]["mode"] == "sliding" and sum(per.values()) == 12 and min(per.values()) >= 5
    _no_machine_paths(tmp_path / "mi")
    assert "8–30 Hz" in r.session["signal"]


def test_oddball_protocol_is_cue_locked(tmp_path):
    p = load_protocol(CONFIGS / "p300-synthetic.yaml")
    result = run_protocol(p, out_dir=tmp_path / "p3")
    assert result.status == "ok"
    s = result.stats
    assert s["online"]["n_trials"] == s["online"]["n_decided"] == s["n_decisions"] > 20
    r = SessionReader(tmp_path / "p3")
    dec = r.decoder()
    assert dec.kind == "xdawn_lda" and dec.report["cv_auc"] is not None and dec.epoch_samples == 205
    assert all(d["mode"] == "cue-locked" for d in r.events("decision"))
    _no_machine_paths(tmp_path / "p3")


def test_bci_block_problems_keep_their_paths():
    base = {"name": "b", "seed": 1, "source": {"kind": "synthetic", "scenario": "mi-2class"},
            "phases": [{"kind": "calibrate", "duration_s": 60}, {"kind": "test", "duration_s": 30}]}
    with pytest.raises(ProtocolError) as e:
        validate_protocol(dict(base, bci={"pipeline": "cspp_lda", "classes": ["left"], "tmin_s": 2.0, "tmax_s": 1.0, "mode": "slide"}), source="x.yaml")
    problems = e.value.problems
    assert any(p.startswith("bci.pipeline: 'cspp_lda' is not valid") and "did you mean 'csp_lda'" in p for p in problems)
    assert any(p.startswith("bci.classes:") for p in problems)
    assert any(p.startswith("bci.tmin_s/tmax_s:") for p in problems)
    assert any(p.startswith("bci.mode: 'slide' is not valid") for p in problems)
    assert "x.yaml: 4 problems" in str(e.value)
    with pytest.raises(ProtocolError) as e:
        validate_protocol(dict(base, bci={}, phases=[{"kind": "test", "duration_s": 30}]))
    assert any("needs a calibrate phase" in p for p in e.value.problems)
    ok = validate_protocol(dict(base, bci={}))
    assert ok.bci is not None and ok.bci.pipeline == "csp_lda" and ok.baseline.mode == "fixed"  # feedback blocks stay defaulted, unused


def test_cli_runs_a_bci_protocol(tmp_path, capsys):
    assert main(["run", "--protocol", str(CONFIGS / "mi-2class-synthetic.yaml"), "--out", str(tmp_path / "s"), "--quiet"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ok" and report["stats"]["online"]["n_trials"] == 8 and report["out_dir"] == "s"
    assert (tmp_path / "s" / "decoder.npz").exists()
