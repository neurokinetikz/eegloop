"""The acceptance report on streams whose facts are planted, and the CLI that prints it."""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from eegloop.check import bandwidth_edge_hz, line_ratio, render_check, run_check
from eegloop.cli import main
from eegloop.sources import SyntheticSource

from test_brainflow_source import KEY, LOSS_PACKETS, PACKET, FakeShim, _fake_source


def _rows(report):
    return {r["item"]: r for r in report["rows"]}


def test_a_clean_synthetic_stream_passes_and_the_report_carries_no_machine_path():
    src = SyntheticSource("clean", duration_s=20.0, pace="fast")
    report = run_check(src, seconds=15.0, mains_hz=60.0)
    rows = _rows(report)
    assert report["ok"] and rows["sampling rate"]["status"] == "ok" and rows["dropped samples"]["value"].startswith("0 in 0")
    assert rows["flat channels"]["status"] == "ok" and rows["bandwidth edge"]["value"] == "none below Nyquist"
    assert report["fs_measured_timestamps"] == pytest.approx(256.0, abs=0.05)
    text = render_check(report)
    assert "## Checks" in text and "## Per channel" in text and "| TP9 |" in text
    for prefix in ("/" + "Users/", "/" + "home/", "/tmp/"):
        assert prefix not in text


def test_planted_drift_gap_flat_channel_and_line_noise_each_show_in_their_row():
    gaps = _rows(run_check(SyntheticSource("gaps", duration_s=90.0, pace="fast"), seconds=80.0))
    assert gaps["dropped samples"]["value"].startswith("64 in 1") and gaps["sampling rate"]["status"] == "ok"
    flat = run_check(SyntheticSource("flat-channel", duration_s=15.0, pace="fast"), seconds=10.0)
    assert not flat["ok"] and _rows(flat)["flat channels"]["status"] == "fail" and "TP10" in _rows(flat)["flat channels"]["value"]
    line = _rows(run_check(SyntheticSource("line-noise", duration_s=15.0, pace="fast"), seconds=10.0, mains_hz=60.0))
    assert line["line noise"]["status"] == "warn" and "at 60 Hz" in line["line noise"]["value"]
    assert "line-noise" in line["quality gate"]["value"]


def test_bandwidth_edge_finds_an_on_device_lowpass(rng):
    fs = 256.0
    x = rng.standard_normal(int(60 * fs))
    assert bandwidth_edge_hz(x, fs) is None
    sos = butter(8, 40.0, fs=fs, output="sos")
    edge = bandwidth_edge_hz(sosfiltfilt(sos, x), fs)
    assert edge is not None and 40.0 < edge < 70.0
    t = np.arange(x.size) / fs
    assert line_ratio(x + 30 * np.sin(2 * np.pi * 60 * t), fs, 60.0) > 10 and line_ratio(x, fs, 60.0) < 5


def test_the_fake_board_report_sees_the_planted_loss_and_the_packet_structure():
    src = _fake_source()
    report = run_check(src, seconds=19.0, timeout_s=0.3)
    src.stop()
    rows = _rows(report)
    assert rows["dropped samples"]["value"].startswith(f"{LOSS_PACKETS * PACKET} in 1")
    assert "package counter" in rows and "1:" in rows["package counter"]["value"]
    assert "signature of arrival-time stamping" in rows["timestamps"]["note"]
    assert report["source"]["nominal"]["board"] == KEY and "serial" not in json.dumps(report).lower().replace("serial_number_note", "")
    text = render_check(report)
    assert "No serial number" in text and "driver: brainflow" in text


def test_cli_devices_and_check(tmp_path, capsys):
    assert main(["devices"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("| key |") and ("driver installed" in out or "driver not installed" in out)
    assert main(["devices", "--json"]) == 0
    assert [r["key"] for r in json.loads(capsys.readouterr().out)][0] == KEY
    assert main(["check", "--source", "synthetic:clean", "--seconds", "5", "--mains", "60", "--out", str(tmp_path / "r.md")]) == 0
    out = capsys.readouterr().out
    assert "## Checks" in out and (tmp_path / "r.md").read_text().startswith("# Device check")
    assert main(["check", "--source", "synthetic:flat-channel", "--seconds", "5", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
