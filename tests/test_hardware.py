"""A paired headset, when there is one. Never runs by default (``-m 'not hardware'`` in pyproject).

    EEGLOOP_HARDWARE=<board-key> [EEGLOOP_MAINS=50] python -m pytest loop -m hardware -rs

Pass the pairing details through ``EEGLOOP_MAC_ADDRESS`` / ``EEGLOOP_SERIAL_NUMBER``; nothing here
prints them, and the report the test writes is checked for their absence.
"""
from __future__ import annotations

import os

import pytest

from eegloop.check import render_check, run_check
from eegloop.sources import BOARDS, open_source

pytestmark = pytest.mark.hardware
KEY = os.environ.get("EEGLOOP_HARDWARE", "")


@pytest.mark.skipif(not KEY, reason="set EEGLOOP_HARDWARE=<board-key> with a paired headset to run")
def test_paired_headset_streams_and_the_report_is_clean(tmp_path):
    assert KEY in BOARDS, f"EEGLOOP_HARDWARE must be one of {', '.join(BOARDS)}"
    mains = float(os.environ.get("EEGLOOP_MAINS", "60"))
    src = open_source(f"brainflow:{KEY}")
    try:
        report = run_check(src, seconds=30.0, mains_hz=mains, timeout_s=10.0)
    finally:
        src.stop()
    assert "error" not in report, report.get("error")
    # the driver's descriptor must agree with the board table: channel names, count and rate
    assert src.info.nominal.get("descriptor_mismatch") in (None, []), src.info.nominal.get("descriptor_mismatch")
    assert tuple(src.info.ch_names) == tuple(BOARDS[KEY]["channels"]) and src.info.fs == BOARDS[KEY]["fs"]
    assert report["n_samples"] > 25 * src.info.fs
    # the measured rate within 2 % of the claimed one (the report's own 'warn' band is 1-3 %)
    measured = report["fs_measured_timestamps"] if report.get("fs_measured_timestamps") is not None else report.get("fs_measured_wall")
    assert measured is not None
    assert abs(measured - report["fs_claimed"]) / report["fs_claimed"] <= 0.02, (measured, report["fs_claimed"])
    rows = {r["item"]: r for r in report["rows"]}
    assert rows["sampling rate"]["status"] != "fail", rows["sampling rate"]
    assert rows["flat channels"]["status"] != "fail", rows["flat channels"]
    assert "dropped samples" in rows and rows["dropped samples"]["status"] != "fail", rows.get("dropped samples")
    assert report["drops"]["per_min"] < 60.0, report["drops"]
    text = render_check(report)
    for secret in (os.environ.get("EEGLOOP_MAC_ADDRESS", ""), os.environ.get("EEGLOOP_SERIAL_NUMBER", "")):
        assert not secret or secret not in text
    (tmp_path / f"device-{KEY}.md").write_text(text, encoding="utf-8")
