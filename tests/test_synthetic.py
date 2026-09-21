"""The synthetic stream plants answers; the loop's primitives must find them."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import welch

from eegloop import detect_gaps
from eegloop.sources import SCENARIOS, SyntheticSource, blocks, make_synthetic_stream


def test_same_seed_same_stream():
    a = make_synthetic_stream(7, duration_s=10.0)
    b = make_synthetic_stream(7, duration_s=10.0)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[3] == b[3]


def test_planted_gap_is_the_only_gap_and_is_reported_exactly():
    src = SyntheticSource("gaps", duration_s=90.0)
    truth = src.truth
    assert truth["gap"] == {"onset_s": 60.0, "n_missing": 64, "index": int(60.0 * 256), "ms_lost": 250.0}
    # from the timestamps alone, with the 200 ppm drift present
    gaps = detect_gaps(src._ts, src.info.fs)
    assert gaps == [(truth["gap"]["index"], 64)]
    # and the stream reports it on exactly one block, whose sample_index skips the missing samples
    seen = [b for b in blocks(src, 32) if b.dropped_before]
    assert len(seen) == 1 and seen[0].dropped_before == 64
    assert seen[0].sample_index == truth["gap"]["index"] + 64
    assert "gaps_inside" not in seen[0].flags


def test_drift_is_recoverable_from_the_timestamps():
    src = SyntheticSource("gaps", gap=None, drift_ppm=200.0, duration_s=60.0)
    ts = src._ts
    ideal = np.arange(ts.size) / src.info.fs
    slope = np.polyfit(ideal, ts - ideal, 1)[0]
    assert slope * 1e6 == pytest.approx(200.0, abs=1.0)


def test_blinks_land_on_frontal_channels_at_the_planted_times():
    src = SyntheticSource("blinks", duration_s=100.0)
    data, fs, names = src._data, src.info.fs, src.info.ch_names
    af = [names.index(c) for c in ("AF7", "AF8")]
    tp = [names.index(c) for c in ("TP9", "TP10")]
    for onset in src.truth["blink_onsets_s"]:
        i = int(round(onset * fs))
        assert np.abs(data[af, i]).min() > 60.0
        assert np.abs(data[tp, i]).max() < np.abs(data[af, i]).min()


def test_alpha_bursts_raise_ten_hertz_power_when_planted():
    src = SyntheticSource("alpha-schedule", duration_s=100.0)
    fs, x = src.info.fs, src._data[src.info.ch_names.index("TP9")]

    def alpha(t0, t1):
        fr, p = welch(x[int(t0 * fs):int(t1 * fs)], fs=fs, nperseg=int(2 * fs))
        return p[(fr >= 8) & (fr <= 12)].mean()

    quiet, burst = alpha(10.0, 20.0), alpha(31.0, 37.0)
    assert burst > 5 * quiet


def test_flicker_peaks_at_the_planted_frequency():
    src = SyntheticSource("ssvep", duration_s=90.0)
    fs, x = src.info.fs, src._data[src.info.ch_names.index("TP10")]
    for f in src.truth["flicker"]:
        seg = x[int(f["onset_s"] * fs):int((f["onset_s"] + f["duration_s"]) * fs)]
        fr, p = welch(seg, fs=fs, nperseg=int(4 * fs))
        assert fr[np.argmax(p * (fr > 5))] == pytest.approx(f["freq_hz"], abs=0.3)


def test_mi_cues_reduce_contralateral_alpha():
    src = SyntheticSource("mi-2class", duration_s=120.0)
    fs, names = src.info.fs, src.info.ch_names
    c3, c4 = names.index("C3"), names.index("C4")

    def alpha(ch, t0, t1):
        fr, p = welch(src._data[ch, int(t0 * fs):int(t1 * fs)], fs=fs, nperseg=int(fs))
        return p[(fr >= 8) & (fr <= 12)].mean()

    lefts = [t for t, l in src.truth["cues"] if l == "left"][:5]
    rights = [t for t, l in src.truth["cues"] if l == "right"][:5]
    # a 'left' cue suppresses the RIGHT hemisphere (C4); a 'right' cue suppresses C3
    ratio_left = np.mean([alpha(c4, t + 0.5, t + 3.5) / alpha(c3, t + 0.5, t + 3.5) for t in lefts])
    ratio_right = np.mean([alpha(c4, t + 0.5, t + 3.5) / alpha(c3, t + 0.5, t + 3.5) for t in rights])
    assert ratio_left < 0.5 and ratio_right > 2.0


def test_cues_arrive_on_the_source_clock():
    src = SyntheticSource("p300", duration_s=30.0)
    markers = src.marker_source()
    got = []
    for _ in blocks(src, 64):
        got.extend(markers.read())
    expected = [c for c in src.truth["cues"] if c[0] <= src.duration_s]
    assert got == expected[: len(got)] and len(got) >= len(expected) - 1


def test_every_scenario_builds_and_names_its_channels():
    for name in SCENARIOS:
        src = SyntheticSource(name, duration_s=8.0)
        assert src.info.kind == "synthetic" and src.info.nominal["scenario"] == name
        assert src.truth["ch_names"] == list(src.info.ch_names)


def test_every_scenario_accounts_its_planted_drops_exactly():
    """The drop count a scenario's blocks carry equals what its truth says was removed — zero for every
    scenario but ``gaps`` — and the source's own total agrees; a scenario that planted a loss the blocks
    did not report would be the bug pf-dropped-samples describes, planted by the library itself."""
    for name in SCENARIOS:
        src = SyntheticSource(name, duration_s=90.0)
        counted = sum(b.dropped_before for b in blocks(src, 32))
        planted = src.truth["gap"]["n_missing"] if src.truth["gap"] else 0
        assert counted == planted == src.n_missing_total, (name, counted, planted)
    with pytest.raises(ValueError):
        SyntheticSource("not-a-scenario")
