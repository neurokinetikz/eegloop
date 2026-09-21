"""The command line: exit codes and outputs, on the shipped protocols, with no device."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eegloop.cli import main

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SYN = CONFIGS / "alpha-up-synthetic.yaml"


@pytest.fixture(autouse=True)
def _yaml():
    pytest.importorskip("yaml")


def test_validate(capsys):
    assert main(["validate", "--protocol", str(SYN)]) == 0
    out = capsys.readouterr().out
    assert "alpha-up-synthetic: valid" in out and "2 phase(s)" in out
    assert main(["validate", "--protocol", str(SYN), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["seed"] == 20260920


def test_a_broken_protocol_exits_2_and_lists_every_problem(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"name": "x", "seed": "one", "phases": [{"kind": "nap", "duration_s": 5}], "sham": {"mode": "placebo"}}), encoding="utf-8")
    assert main(["validate", "--protocol", str(bad)]) == 2
    err = capsys.readouterr().err
    assert "3 problems" in err and "seed: expected an integer" in err and "'nap' is not valid" in err and "'placebo' is not valid" in err
    assert main(["validate", "--protocol", str(tmp_path / "nope.yaml")]) == 2


def test_budget_and_probe(capsys):
    assert main(["budget", "--protocol", str(SYN)]) == 0
    out = capsys.readouterr().out
    assert "TOTAL" in out and "decision     quality" in out
    assert main(["budget", "--protocol", str(SYN), "--json"]) == 0
    b = json.loads(capsys.readouterr().out)
    assert b["rows"][0]["samples"] == 64.0 and b["total_ms"] == pytest.approx(64 / 256 * 1000 + 32 / 256 * 1000 + 10.0)
    assert main(["probe", "--protocol", str(SYN), "--alignments", "4"]) == 0
    out = capsys.readouterr().out
    assert "filter delay" in out and "expected   250.00 ms" in out


def test_run_writes_a_session_and_reports(tmp_path, capsys):
    assert main(["run", "--protocol", str(SYN), "--out", str(tmp_path / "s"), "--quiet"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ok" and report["out_dir"] == "s" and report["stats"]["n_blocks"] > 900
    assert (tmp_path / "s" / "session.json").exists()
    assert "/" not in report["out_dir"]


def test_versions(capsys):
    assert main(["versions"]) == 0
    assert "numpy" in capsys.readouterr().out
    assert main(["versions", "--json"]) == 0
    assert "scipy" in json.loads(capsys.readouterr().out)
