"""The example applications run end to end, and the rubric of lesson L7.16 is checkable from what they write.

Each example is run as a program, the way a learner runs it, on the synthetic stream and on the site's
shipped four-channel asset; nothing here needs a device or the network. The rubric items that a CI
run can judge are asserted below in the order the lesson lists them.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from eegloop.feedback import ShamPolicy
from eegloop.session import SessionReader, load_protocol, run_protocol
from eegloop.sources import SyntheticSource

LOOP = Path(__file__).resolve().parents[1]
EXAMPLES, CONFIGS = LOOP / "examples", LOOP / "configs"
OC4 = CONFIGS / "alpha-up-oc4-replay.yaml"


def _run(*argv: str, timeout: float = 300.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, *argv], cwd=LOOP, capture_output=True, text=True, timeout=timeout, check=False)


def _oc4_asset_present() -> bool:
    pytest.importorskip("yaml")
    p = load_protocol(OC4)
    return (OC4.parent / p.source.path).resolve().exists()


@pytest.fixture(autouse=True)
def _yaml():
    pytest.importorskip("yaml")


def _decode_deps():
    if os.environ.get("CI", "").lower() in ("1", "true"):
        import pyriemann  # noqa: F401
        import sklearn  # noqa: F401
    else:
        pytest.importorskip("sklearn")
        pytest.importorskip("pyriemann")


def test_eo_ec_switch_calibrates_freezes_applies_and_scores_the_planted_periods(tmp_path):
    _decode_deps()
    out = tmp_path / "eoec"
    r = _run(str(EXAMPLES / "eo_ec_switch.py"), "--out", str(out))
    assert r.returncode == 0, r.stderr[-2000:]
    session = json.loads((out / "session.json").read_text(encoding="utf-8"))
    assert session["status"] == "ok" and (out / "decoder.npz").exists() and (out / "events.jsonl").exists()
    online = session["stats"]["online"]
    assert online["n_trials"] == 7 and online["n_decided"] == 7 and online["n_correct"] == 7  # the planted answer
    assert online["chance_band_95"] is not None
    assert session["budget"]["decoder"]["latency_ms"] == pytest.approx(3000.0)  # the window is the delay
    assert "7/7 cued periods correct" in r.stdout and "decoder latency 3000 ms" in r.stdout
    kinds = {e["kind"] for e in SessionReader(out).events()}
    assert {"cue", "phase", "end"} <= kinds
    for prefix in ("/" + "Users/", "/" + "home/", "/tmp/", "/private/"):
        assert prefix not in (out / "session.json").read_text(encoding="utf-8")


def test_blink_switch_presses_once_per_planted_blink_and_never_otherwise():
    r = _run(str(EXAMPLES / "blink_switch.py"))
    assert r.returncode == 0, r.stderr[-2000:]
    last = r.stdout.strip().splitlines()[-1]
    assert "0 spurious" in last
    n_planted, n_press = int(last.split(" planted")[0]), int(last.split("presses")[0].split(",")[-1])
    assert n_planted == n_press == 3


@pytest.mark.skipif(not _oc4_asset_present(), reason="shipped four-channel asset not present")
def test_alpha_bar_on_the_shipped_four_channel_asset_writes_a_complete_session(tmp_path):
    out = tmp_path / "oc4"
    r = _run(str(EXAMPLES / "alpha_bar.py"), "--protocol", str(OC4), "--out", str(out))
    assert r.returncode == 0, r.stderr[-2000:]
    session = json.loads((out / "session.json").read_text(encoding="utf-8"))
    assert session["status"] == "ok" and (out / "events.jsonl").exists() and (out / "signal.npz").exists()
    # rubric: the budget's exact rows are present and the processing row is marked measured
    rows = session["budget"]["rows"]
    assert [row["row"] for row in rows] == ["filter", "buffer", "processing"]
    assert rows[2]["detail"].startswith("measured on this machine")
    assert session["budget"]["total_ms"] == pytest.approx(610.0)
    assert session["stats"]["n_blocks"] == 300
    # rubric: the sealed mode is in the log and only the seed unblinds it
    token = session["sealed_sham"]
    assert token and ShamPolicy.unblind(token, load_protocol(OC4).seed) == "veridical"


@pytest.mark.skipif(not _oc4_asset_present(), reason="shipped four-channel asset not present")
def test_all_four_sham_modes_run_on_the_asset_and_each_log_seals_its_own_mode(tmp_path):
    tokens = {}
    donor = tmp_path / "veridical"
    for mode in ("veridical", "sham-band", "sham-inverted", "sham-yoked"):
        overrides = {"sham.mode": mode}
        if mode == "sham-yoked":
            overrides["sham.donor"] = str(donor)
        p = load_protocol(OC4, overrides)
        result = run_protocol(p, out_dir=tmp_path / mode)
        assert result.status == "ok"
        tokens[mode] = SessionReader(tmp_path / mode).sealed_sham
        assert ShamPolicy.unblind(tokens[mode], p.seed) == mode
    assert len(set(tokens.values())) == 4


def test_the_gate_suspends_feedback_across_at_least_nine_in_ten_planted_artifact_windows(tmp_path):
    p = load_protocol(CONFIGS / "alpha-up-synthetic.yaml", {"source.scenario": "blinks"})
    run_protocol(p, out_dir=tmp_path / "blinks")
    r = SessionReader(tmp_path / "blinks")
    closures = [g["t_s"] for g in r.events("gate")]
    onsets = [o for o in SyntheticSource("blinks", seed=p.seed, duration_s=p.total_duration_s + 1.0).truth["blink_onsets_s"]
              if o > p.quality.window_s]
    covered = sum(any(-0.2 <= t - o <= 2.0 for t in closures) for o in onsets)
    assert onsets and covered / len(onsets) >= 0.9, (covered, len(onsets))
