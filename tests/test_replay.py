"""Replay: every format, pacing, loss, and the source contract every block must satisfy."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from eegloop import Block, Reblocker
from eegloop.sources import ReplaySource, Source, SyntheticSource, blocks, open_source, read_recording

REPO = Path(__file__).resolve().parents[2]
ASSET = REPO / "site" / "public" / "data" / "widgets" / "w-latency-budget" / "alpha-ec.bin"


def _contract(bs, n_channels):
    assert bs, "no blocks"
    for k, b in enumerate(bs):
        assert isinstance(b, Block) and b.data.dtype == np.float64 and b.n_channels == n_channels
        assert b.seq == k
    idx = [b.sample_index for b in bs]
    assert all(b > a for a, b in zip(idx, idx[1:]))


def test_array_replay_round_trips_and_keeps_the_contract(rng):
    x = rng.standard_normal((3, 1000))
    src = ReplaySource(x, fs=250.0, ch_names=["a", "b", "c"], pace="fast", block_samples=64)
    bs = list(blocks(src, 64))
    _contract(bs, 3)
    joined = np.concatenate([b.data for b in bs], axis=1)
    assert np.array_equal(joined, x[:, : joined.shape[1]]) and joined.shape[1] == 960  # remainder dropped
    assert src.done and src.info.kind == "replay" and src.info.clock == "sample"
    with pytest.raises(ValueError):
        ReplaySource(x, fs=250.0, ch_names=["a", "b"])


def test_npz_and_csv_readers(tmp_path, rng):
    x = rng.standard_normal((2, 500))
    ts = np.arange(500) / 128.0
    np.savez(tmp_path / "r.npz", data=x, fs=128.0, ch_names=np.array(["O1", "O2"]), timestamps_s=ts)
    d, fs, names, t, nominal = read_recording(tmp_path / "r.npz")
    assert fs == 128.0 and names == ("O1", "O2") and np.array_equal(d, x) and np.array_equal(t, ts)
    assert nominal["file"] == "r.npz" and "/" not in str(nominal["file"])
    csv = tmp_path / "r.csv"
    # nine decimals: at six, 1/128 s alternates between 0.007812 and 0.007813 and the inferred rate is 127.99 Hz
    csv.write_text("time,O1,O2\n" + "\n".join(f"{ts[i]:.9f},{x[0, i]:.6f},{x[1, i]:.6f}" for i in range(500)), encoding="utf-8")
    d2, fs2, names2, t2, nominal2 = read_recording(csv)
    assert names2 == ("O1", "O2") and fs2 == pytest.approx(128.0) and np.allclose(d2, x, atol=1e-6)
    assert nominal2["fs_inferred_from_time_column"] is True and np.allclose(t2, ts, atol=1e-6)
    with pytest.raises(ValueError):
        read_recording(x)  # an array needs fs


def test_site_asset_replays_with_its_sidecar():
    if not ASSET.exists():
        pytest.skip("shipped asset not present")
    src = ReplaySource.from_asset(ASSET, pace="fast", block_samples=32)
    meta = json.loads(ASSET.with_suffix(".json").read_text())
    assert src.info.fs == meta["sfreq"] and src.info.ch_names == tuple(meta["channels"])
    assert src.n_samples == meta["n_samples"] and src.info.nominal["dataset"] == meta["dataset"]
    bs = list(blocks(src, 32))
    _contract(bs, 1)
    assert sum(b.n_samples for b in bs) == meta["n_samples"] - meta["n_samples"] % 32
    x = np.concatenate([b.data[0] for b in bs])
    assert 1.0 < np.sqrt(np.mean(x ** 2)) < 100.0  # microvolts, as the sidecar says


def test_injected_gaps_are_never_straddled_and_are_counted(rng):
    x = rng.standard_normal((2, 2000))
    src = ReplaySource(x, fs=200.0, pace="fast", inject_gaps=[(2.0, 40), (5.0, 7)], block_samples=100)
    bs = list(blocks(src, 100))
    _contract(bs, 2)
    assert src.n_missing_total == 47 and src.n_samples == 2000 - 47
    assert src.duration_s == pytest.approx(10.0)
    assert src.info.clock == "device" and src.info.nominal["gaps"] == [(400, 40), (960, 7)]
    # the first loss falls on a block boundary: it is the next block's dropped_before, and the session
    # index skips the samples that are not there (400 in, 40 gone, next block starts at 440)
    before = [(b.sample_index, b.dropped_before) for b in bs if b.dropped_before]
    assert before == [(440, 40)]
    # the second falls 60 samples into a 100-sample block: the reblocker cannot make it a dropped_before
    # without lying about the block's first sample, so it records the seam inside the block instead
    inside = [(b.sample_index, b.flags["gaps_inside"]) for b in bs if "gaps_inside" in b.flags]
    assert inside == [(900 + 40, [(60, 7)])]
    # and the block after that one counts the seven in its session index
    after = next(b for b in bs if b.sample_index > 940)
    assert after.sample_index == 940 + 100 + 7


def test_wall_pace_releases_samples_on_the_clock(rng):
    x = rng.standard_normal((1, 500))
    src = ReplaySource(x, fs=1000.0, pace="wall", block_samples=100)
    src.start()
    assert src.read() is None or src.read().n_samples <= 100   # nothing much has elapsed yet
    t0 = time.perf_counter()
    bs = list(blocks(src, 100, timeout_s=2.0))
    elapsed = time.perf_counter() - t0
    assert len(bs) == 5 and 0.35 < elapsed < 1.5


def test_loop_and_crop(rng):
    x = rng.standard_normal((1, 400))
    src = ReplaySource(x, fs=100.0, pace="fast", loop=True, start_s=1.0, stop_s=3.0, block_samples=50)
    src.start()
    got = [src.read() for _ in range(6)]
    assert [b.sample_index for b in got] == [0, 50, 100, 150, 0 + 200, 50 + 200]
    assert got[4].t_start_s == pytest.approx(2.0)  # the second lap starts after one 2-s duration
    assert np.array_equal(got[0].data, x[:, 100:150]) and np.array_equal(got[4].data, x[:, 100:150])


def test_open_source_dispatch():
    assert isinstance(open_source("synthetic:clean", duration_s=4.0), SyntheticSource)
    assert isinstance(open_source({"kind": "synthetic", "scenario": "clean", "duration_s": 4.0}), Source)
    with pytest.raises(ValueError, match="unknown board key|binding decision"):
        open_source("brainflow:some-board")
    with pytest.raises(ValueError, match="unknown source kind"):
        open_source("telepathy")


def test_mne_formats_need_the_extra(tmp_path, rng):
    mne = pytest.importorskip("mne")
    x = rng.standard_normal((2, 300)) * 1e-6
    info = mne.create_info(["O1", "O2"], sfreq=100.0, ch_types="eeg")
    raw = mne.io.RawArray(x, info, verbose="ERROR")
    raw.save(tmp_path / "r_raw.fif", verbose="ERROR")
    d, fs, names, ts, nominal = read_recording(tmp_path / "r_raw.fif")
    assert fs == 100.0 and names == ("O1", "O2") and np.allclose(d, x * 1e6, atol=1e-6)
    assert ts is None and "highpass_hz_in_header" in nominal


def test_reblocker_and_blocks_agree(rng):
    x = rng.standard_normal((2, 333))
    src = ReplaySource(x, fs=100.0, pace="fast", block_samples=17)
    via_blocks = list(blocks(src, 32))
    rb, manual = Reblocker(32), []
    src2 = ReplaySource(x, fs=100.0, pace="fast", block_samples=17)
    src2.start()
    while (raw := src2.read()) is not None:
        manual.extend(rb.push(raw))
    assert len(via_blocks) == len(manual) == 333 // 32
    for a, b in zip(via_blocks, manual):
        assert np.array_equal(a.data, b.data) and a.sample_index == b.sample_index
