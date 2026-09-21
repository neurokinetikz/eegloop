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
        report = run_check(src, seconds=15.0, mains_hz=mains, timeout_s=10.0)
    finally:
        src.stop()
    assert "error" not in report, report.get("error")
    assert report["n_samples"] > 10 * src.info.fs
    rows = {r["item"]: r for r in report["rows"]}
    assert rows["sampling rate"]["status"] != "fail", rows["sampling rate"]
    assert rows["flat channels"]["status"] != "fail", rows["flat channels"]
    text = render_check(report)
    for secret in (os.environ.get("EEGLOOP_MAC_ADDRESS", ""), os.environ.get("EEGLOOP_SERIAL_NUMBER", "")):
        assert not secret or secret not in text
    (tmp_path / f"device-{KEY}.md").write_text(text, encoding="utf-8")
