"""The block contract, gap detection and re-blocking -- the loss ledger every layer above relies on."""
from __future__ import annotations

import numpy as np
import pytest

from eegloop import Block, DropAccount, Reblocker, StreamInfo, detect_gaps


def test_block_coerces_shape_and_dtype():
    b = Block(np.arange(8, dtype=np.int32), 256.0)
    assert b.data.shape == (1, 8) and b.data.dtype == np.float64
    assert b.n_channels == 1 and b.n_samples == 8
    assert b.t_end_s == pytest.approx(8 / 256.0)


def test_block_rejects_bad_inputs():
    with pytest.raises(ValueError):
        Block(np.zeros((2, 3, 4)), 256.0)
    with pytest.raises(ValueError):
        Block(np.zeros((2, 4)), 0.0)
    with pytest.raises(ValueError):
        Block(np.zeros((2, 4)), 256.0, dropped_before=-1)
    with pytest.raises(ValueError):
        Block(np.zeros((2, 4)), 256.0, timestamps_s=np.zeros(3))


def test_block_is_frozen_and_replace_copies():
    b = Block(np.zeros((2, 4)), 256.0, flags={"a": 1})
    with pytest.raises(Exception):
        b.fs = 100.0  # type: ignore[misc]
    c = b.with_flag("quality", {"ok": True})
    assert "quality" in c.flags and "quality" not in b.flags
    d = b.replace(seq=7)
    assert d.seq == 7 and b.seq == 0 and d.data is b.data


def test_streaminfo_validates():
    info = StreamInfo(256.0, ["A", "B"], kind="synthetic")
    assert info.n_channels == 2 and info.ch_names == ("A", "B")
    with pytest.raises(ValueError):
        StreamInfo(0.0, ["A"], kind="synthetic")
    with pytest.raises(ValueError):
        StreamInfo(256.0, [], kind="synthetic")


def test_detect_gaps_finds_a_planted_hole_and_ignores_jitter(rng):
    fs, n = 256.0, 5000
    ts = np.arange(n) / fs + rng.uniform(-0.1, 0.1, n) / fs   # ±0.1 sample of jitter
    ts[1000:] += 64 / fs                                      # 64 samples never arrived
    gaps = detect_gaps(ts, fs)
    assert gaps == [(1000, 64)]
    assert detect_gaps(np.arange(n) / fs, fs) == []
    assert detect_gaps(np.array([0.0]), fs) == []


def test_drop_account():
    acc = DropAccount()
    acc.add(1.0, 0)
    acc.add(2.0, 5)
    acc.add(3.0, 7)
    assert acc.total == 12 and acc.events == [(2.0, 5), (3.0, 7)]
    assert acc.rate_per_min(60.0) == 12.0
    assert np.isnan(acc.rate_per_min(0.0))


def _raw_blocks(sizes, fs=256.0, n_ch=2, start_index=0, timestamps=True, drops=None):
    """Raw blocks of irregular size, contiguous unless `drops` says samples went missing before one."""
    drops = drops or {}
    idx, t = start_index, 0.0
    for k, n in enumerate(sizes):
        lost = drops.get(k, 0)
        idx += lost
        t += lost / fs
        data = np.full((n_ch, n), float(idx)) + np.arange(n)      # value = session index, so we can check
        ts = t + np.arange(n) / fs if timestamps else None
        yield Block(data, fs, seq=k, sample_index=idx, t_start_s=t, timestamps_s=ts, dropped_before=lost)
        idx += n
        t += n / fs


def test_reblocker_regroups_irregular_input_into_fixed_blocks():
    rb = Reblocker(32)
    out = []
    for raw in _raw_blocks([37, 27, 5, 100, 3]):
        out.extend(rb.push(raw))
    tail = rb.flush()
    assert [b.n_samples for b in out] == [32] * 5
    assert tail is not None and tail.n_samples == 12
    assert [b.seq for b in out] == [0, 1, 2, 3, 4]
    assert [b.sample_index for b in out] == [0, 32, 64, 96, 128]
    # every emitted sample is the session index it was planted with, in order
    joined = np.concatenate([b.data[0] for b in out] + [tail.data[0]])
    assert np.array_equal(joined, np.arange(172, dtype=float))
    # timestamps were sliced, not recomputed
    assert out[1].timestamps_s is not None and out[1].timestamps_s[0] == pytest.approx(32 / 256.0)
    assert out[1].t_start_s == pytest.approx(32 / 256.0)


def test_reblocker_attaches_a_loss_before_an_empty_buffer_to_the_next_block():
    rb = Reblocker(32)
    out = []
    for raw in _raw_blocks([32, 32], drops={1: 64}):   # 64 samples lost between the two raw blocks
        out.extend(rb.push(raw))
    assert out[0].dropped_before == 0 and out[1].dropped_before == 64
    assert out[1].sample_index == 32 + 64
    assert "gaps_inside" not in out[1].flags


def test_reblocker_records_a_loss_inside_a_block_instead_of_hiding_it():
    rb = Reblocker(32)
    out = []
    for raw in _raw_blocks([20, 12, 32], drops={1: 10}):   # the hole falls 20 samples into block 0
        out.extend(rb.push(raw))
    assert out[0].dropped_before == 0
    assert out[0].flags["gaps_inside"] == [(20, 10)]
    # the session count of the NEXT block accounts for the samples that never arrived
    assert out[1].sample_index == 32 + 10
    assert out[1].data[0, 0] == pytest.approx(42.0)


def test_reblocker_without_timestamps_advances_the_clock_itself():
    rb = Reblocker(16)
    out = []
    for raw in _raw_blocks([16, 16], timestamps=False):
        out.extend(rb.push(raw))
    assert out[0].timestamps_s is None
    assert out[1].t_start_s == pytest.approx(16 / 256.0)


def test_reblocker_refuses_a_changed_rate():
    rb = Reblocker(8)
    rb.push(Block(np.zeros((1, 8)), 256.0))
    with pytest.raises(ValueError):
        rb.push(Block(np.zeros((1, 8)), 250.0))
