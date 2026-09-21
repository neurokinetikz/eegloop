"""A stream with known answers planted in it, so a test can assert that the loop found what is there.

Not a model of EEG. Modelled on ``pipelines/eegpipe/synthetic.py``'s ``make_synthetic_raw``
(2026-09-20) so a planted blink means the same thing in both packages -- the same parameter names
(``blink_uv``, ``line_uv``, ``noise_uv``, ``alpha_uv``, ``pop_uv``, ``mains_hz``), the same crude
topographic weighting -- with what a *stream* needs on top: per-sample timestamps, a dropped-sample
gap that shows in them, clock drift, a flicker component, cues with an evoked response and a
lateralised power change, and a ``truth`` dictionary naming every one.

Every number comes from the seed; two calls with the same seed give identical data.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .replay import ReplaySource

__all__ = ["make_synthetic_stream", "SyntheticSource", "SCENARIOS", "DEFAULT_CHANNELS"]

#: A four-electrode frontotemporal headband, by geometry (the names are 10-10 sites).
DEFAULT_CHANNELS: tuple[str, ...] = ("TP9", "AF7", "AF8", "TP10")

# Weights by 10-20 letter group. Crude on purpose: enough structure that a frontal channel sees a
# blink and a posterior one sees alpha, no claim to be a head model.
_FRONTAL = {"FP": 1.0, "AF": 0.9, "F": 0.6, "FC": 0.35, "FT": 0.3, "T": 0.2, "C": 0.22, "CP": 0.12,
            "TP": 0.15, "P": 0.08, "PO": 0.06, "O": 0.05}
_POSTERIOR = {"O": 1.0, "PO": 0.9, "P": 0.75, "TP": 0.6, "CP": 0.45, "T": 0.35, "C": 0.3, "FC": 0.15,
              "FT": 0.2, "F": 0.12, "AF": 0.1, "FP": 0.08}
_PARIETAL = {"P": 1.0, "CP": 0.9, "C": 0.75, "PO": 0.6, "O": 0.5, "TP": 0.5, "T": 0.3, "FC": 0.4,
             "FT": 0.25, "F": 0.3, "AF": 0.2, "FP": 0.1}


def _site(name: str) -> tuple[str, str]:
    """``(letter group, hemisphere)`` from a 10-20 name: TP9 -> ('TP', 'L'), Cz -> ('C', 'z')."""
    n = name.upper()
    letters = "".join(c for c in n if c.isalpha() and c != "Z")
    digits = "".join(c for c in n if c.isdigit())
    if n.endswith("Z") or not digits:
        hemi = "z"
    else:
        hemi = "L" if int(digits[-1]) % 2 == 1 else "R"
    return letters, hemi


def _weights(names: Iterable[str], table: dict[str, float], default: float = 0.2) -> np.ndarray:
    return np.array([table.get(_site(n)[0], default) for n in names], dtype=float)


def _hemi(names: Iterable[str], which: str) -> np.ndarray:
    return np.array([1.0 if _site(n)[1] == which else 0.0 for n in names], dtype=float)


def _pink(rng: np.random.Generator, n_channels: int, n_times: int, exponent: float = 1.2) -> np.ndarray:
    white = rng.standard_normal((n_channels, n_times))
    spectrum = np.fft.rfft(white, axis=1)
    freqs = np.fft.rfftfreq(n_times, d=1.0)
    scale = np.ones_like(freqs)
    scale[1:] = freqs[1:] ** (-exponent / 2.0)
    scale[0] = scale[1]
    out = np.fft.irfft(spectrum * scale, n=n_times, axis=1)
    sd = out.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return out / sd


def _gaussian(t: np.ndarray, centre: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - centre) / width) ** 2)


def make_synthetic_stream(
    seed: int = 20260920,
    *,
    fs: float = 256.0,
    ch_names: Iterable[str] = DEFAULT_CHANNELS,
    duration_s: float = 120.0,
    noise_uv: float = 9.0,
    drift_uv: float = 25.0,
    alpha_uv: float = 6.0,
    alpha_bursts: Iterable[tuple[float, float, float]] = ((30.0, 8.0, 30.0), (75.0, 6.0, 25.0)),
    blink_uv: float = 130.0,
    blink_times_s: Iterable[float] = (5.0, 17.5, 42.0, 88.0),
    mains_hz: float | None = 60.0,
    line_uv: float = 2.5,
    gap: tuple[float, int] | None = (60.0, 64),
    drift_ppm: float = 200.0,
    flicker: Iterable[tuple[float, float, float, float]] = ((95.0, 10.0, 12.0, 6.0),),
    cues: Iterable[tuple[float, str]] = (),
    p300_uv: float = 9.0,
    mi_erd: float = 0.6,
    flat_channel: str | None = None,
    pops: Iterable[tuple[float, str, float]] = (),
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, str]], dict[str, Any]]:
    """``(data_uv, timestamps_s, cues, truth)``.

    ``alpha_bursts`` are ``(onset_s, duration_s, amplitude_uv)`` of 10 Hz on posterior-weighted
    channels above a small standing alpha; ``blink_times_s`` are frontal-weighted Gaussians of
    ``blink_uv``; ``gap`` is ``(onset_s, n_samples)`` **removed** from the stream so the timestamps
    jump; ``drift_ppm`` stretches the timestamps by that many parts per million; ``flicker`` entries
    are ``(onset_s, duration_s, freq_hz, amplitude_uv)`` on posterior channels (an SSVEP stand-in);
    ``cues`` are ``(t_s, label)`` -- a ``'target'`` gets a parietal P300-like deflection at +0.35 s,
    ``'left'``/``'right'`` reduce 8-12 Hz power on the contralateral hemisphere by ``mi_erd`` for 3 s;
    ``pops`` are ``(t_s, channel, amplitude_uv[, duration_s])`` -- an abrupt, *sustained* step on one
    channel, which is what ``data/scripts/detectors.py`` means by an electrode pop. ``truth`` names all
    of it.
    """
    rng = np.random.default_rng(int(seed))
    names = tuple(str(c) for c in ch_names)
    n_ch = len(names)
    n = int(round(duration_s * fs))
    t = np.arange(n) / fs

    def inside(t_s: float) -> bool:
        # an event outside the recording is not planted, and is not in truth either
        return 0.0 <= float(t_s) < duration_s

    data = noise_uv * _pink(rng, n_ch, n)
    data += drift_uv * np.sin(2 * np.pi * 0.07 * t[None, :] + rng.uniform(0, 2 * np.pi, (n_ch, 1)))

    # alpha: a standing 10 Hz on posterior channels, modulated by bursts and by MI cues
    post = _weights(names, _POSTERIOR)
    envelope = np.full(n, float(alpha_uv))
    bursts = []
    for onset, dur, amp in alpha_bursts:
        if not inside(onset):
            continue
        i0, i1 = int(round(onset * fs)), min(n, int(round((onset + dur) * fs)))
        envelope[i0:i1] += float(amp)
        bursts.append({"onset_s": float(onset), "duration_s": float(dur), "amplitude_uv": float(amp)})
    cue_list = sorted((float(a), str(b)) for a, b in cues if inside(a))
    alpha_gain = np.ones((n_ch, n))
    for t_c, label in cue_list:
        if label in ("left", "right"):
            i0, i1 = int(round((t_c + 0.5) * fs)), min(n, int(round((t_c + 3.5) * fs)))
            side = _hemi(names, "R" if label == "left" else "L")
            alpha_gain[:, i0:i1] *= (1.0 - mi_erd * side)[:, None]
    phase = rng.uniform(0, 2 * np.pi, (n_ch, 1))
    data += post[:, None] * alpha_gain * envelope[None, :] * np.sin(2 * np.pi * 10.0 * t[None, :] + phase)

    if mains_hz:
        lp = rng.uniform(0, 2 * np.pi, (n_ch, 1))
        data += line_uv * np.sin(2 * np.pi * mains_hz * t[None, :] + lp)
        data += 0.4 * line_uv * np.sin(2 * np.pi * 2 * mains_hz * t[None, :] + lp)

    front = _weights(names, _FRONTAL, default=0.1)
    blink_trace = np.zeros(n)
    blink_onsets = [float(b) for b in blink_times_s if inside(b)]
    for onset in blink_onsets:
        blink_trace += rng.uniform(0.8, 1.3) * _gaussian(t, onset, 0.05)
    data += (blink_uv * front)[:, None] * blink_trace[None, :]

    flick = []
    for onset, dur, f_hz, amp in flicker:
        if not inside(onset):
            continue
        i0, i1 = int(round(onset * fs)), min(n, int(round((onset + dur) * fs)))
        data[:, i0:i1] += (amp * post)[:, None] * np.sin(2 * np.pi * f_hz * t[i0:i1])[None, :]
        flick.append({"onset_s": float(onset), "duration_s": float(dur), "freq_hz": float(f_hz), "amplitude_uv": float(amp)})

    par = _weights(names, _PARIETAL, default=0.3)
    for t_c, label in cue_list:
        if label == "target":
            data += par[:, None] * (p300_uv * _gaussian(t, t_c + 0.35, 0.075))[None, :]
        elif label == "standard":
            data += par[:, None] * (0.25 * p300_uv * _gaussian(t, t_c + 0.35, 0.075))[None, :]

    # An electrode pop as data/scripts/detectors.py defines it: an abrupt, SUSTAINED step on one channel
    # (an 8 ms rise, held, then released). eegpipe.synthetic plants a 20 ms Gaussian transient under the
    # same name -- a different artifact sharing a word across the repository. Recorded here as a
    # divergence, not reconciled: the gate ports detectors.py, so this plants what that criterion means.
    pop_list = []
    for entry in pops:
        t_p, ch, amp = float(entry[0]), str(entry[1]), float(entry[2])
        dur = float(entry[3]) if len(entry) > 3 else 1.5
        if not inside(t_p):
            continue
        i = names.index(ch)
        i0, i1 = int(round(t_p * fs)), min(n, int(round((t_p + dur) * fs)))
        edge = max(2, int(round(0.008 * fs)))
        if i1 - i0 < 2 * edge:
            continue
        ramp = np.ones(i1 - i0)
        ramp[:edge] = np.linspace(0.0, 1.0, edge + 1)[1:]
        ramp[-edge:] = np.linspace(1.0, 0.0, edge + 1)[:-1]
        data[i, i0:i1] += amp * ramp
        pop_list.append({"onset_s": t_p, "duration_s": dur, "channel": ch, "amplitude_uv": amp})

    if flat_channel is not None:
        data[names.index(str(flat_channel))] = 0.05 * rng.standard_normal(n)

    ts = t * (1.0 + drift_ppm * 1e-6)
    gap_truth = None
    if gap is not None and inside(gap[0]):
        g_onset, g_n = float(gap[0]), int(gap[1])
        g0 = int(round(g_onset * fs))
        g_n = min(g_n, n - g0)
        data = np.delete(data, np.s_[g0:g0 + g_n], axis=1)
        ts = np.delete(ts, np.s_[g0:g0 + g_n])
        gap_truth = {"onset_s": g_onset, "n_missing": g_n, "index": g0, "ms_lost": g_n / fs * 1000.0}

    truth: dict[str, Any] = {
        "generator": "eegloop.sources.synthetic.make_synthetic_stream", "seed": int(seed),
        "fs": float(fs), "ch_names": list(names), "duration_s": float(duration_s),
        "alpha_bursts": bursts, "blink_onsets_s": blink_onsets, "gap": gap_truth,
        "drift_ppm": float(drift_ppm), "flicker": flick, "cues": cue_list, "pops": pop_list,
        "flat_channel": None if flat_channel is None else str(flat_channel),
        "mains_hz": None if not mains_hz else float(mains_hz),
        "line_uv": float(line_uv), "noise_uv": float(noise_uv), "blink_uv": float(blink_uv),
    }
    return data, ts, cue_list, truth


def _mi_cues(n: int = 20, start_s: float = 10.0, every_s: float = 5.0, seed: int = 1) -> list[tuple[float, str]]:
    rng = np.random.default_rng(seed)
    labels = ["left", "right"] * (n // 2 + 1)
    rng.shuffle(labels)
    return [(start_s + i * every_s, labels[i]) for i in range(n)]


def _oddball_cues(n: int = 60, start_s: float = 5.0, every_s: float = 1.8, target_every: int = 4) -> list[tuple[float, str]]:
    return [(start_s + i * every_s, "target" if i % target_every == 3 else "standard") for i in range(n)]


def _eo_ec_cues(n: int = 19, start_s: float = 4.0, period_s: float = 6.0) -> list[tuple[float, str]]:
    """Alternating eyes-closed / eyes-open periods, each announced by a cue at its start."""
    return [(start_s + i * period_s, "closed" if i % 2 == 0 else "open") for i in range(n)]


def _eo_ec_bursts(cues: list[tuple[float, str]], period_s: float = 6.0, amplitude_uv: float = 18.0) -> tuple[tuple[float, float, float], ...]:
    """An alpha burst covering every 'closed' period: the planted answer the eo-ec decoder is scored against."""
    return tuple((t, period_s, amplitude_uv) for t, label in cues if label == "closed")


_EO_EC_CUES = _eo_ec_cues()


#: Named scenarios: keyword overrides for :func:`make_synthetic_stream`. ``clean`` plants nothing.
SCENARIOS: dict[str, dict[str, Any]] = {
    "clean": {"alpha_bursts": (), "blink_times_s": (), "gap": None, "drift_ppm": 0.0, "flicker": (),
              "line_uv": 0.0, "drift_uv": 5.0},
    "alpha-schedule": {"blink_times_s": (), "gap": None, "flicker": ()},
    "blinks": {"alpha_bursts": (), "gap": None, "flicker": ()},
    "gaps": {"alpha_bursts": (), "blink_times_s": (), "flicker": (), "gap": (60.0, 64), "drift_ppm": 200.0},
    "flat-channel": {"alpha_bursts": (), "blink_times_s": (), "gap": None, "flicker": (), "flat_channel": "TP10"},
    "line-noise": {"alpha_bursts": (), "blink_times_s": (), "gap": None, "flicker": (), "line_uv": 40.0},
    "pops": {"alpha_bursts": (), "blink_times_s": (), "gap": None, "flicker": (),
             "pops": ((20.0, "TP9", 300.0), (50.0, "AF8", -280.0))},
    "ssvep": {"alpha_bursts": (), "blink_times_s": (), "gap": None,
              "flicker": ((20.0, 20.0, 12.0, 8.0), (60.0, 20.0, 15.0, 8.0))},
    "mi-2class": {"alpha_bursts": (), "blink_times_s": (), "gap": None, "flicker": (), "alpha_uv": 12.0,
                  "ch_names": ("C3", "Cz", "C4", "O1"), "cues": _mi_cues()},
    "p300": {"alpha_bursts": (), "blink_times_s": (), "gap": None, "flicker": (),
             "ch_names": ("Pz", "Cz", "O1", "O2"), "cues": _oddball_cues()},
    # eyes closed against eyes open on the default four channels: the one two-class state both montage
    # classes the course names can decode (posterior alpha reaches TP9/TP10 at 0.6 of its O1/O2 weight)
    "eo-ec": {"blink_times_s": (), "gap": None, "flicker": (), "alpha_uv": 4.0,
              "cues": _EO_EC_CUES, "alpha_bursts": _eo_ec_bursts(_EO_EC_CUES)},
}


class SyntheticSource(ReplaySource):
    """A :class:`~eegloop.sources.replay.ReplaySource` over :func:`make_synthetic_stream`.

    ``scenario`` picks a :data:`SCENARIOS` preset; keyword arguments override it. ``truth`` carries
    the planted answers and ``marker_source()`` the cues, on the same clock as the samples.
    """

    def __init__(self, scenario: str = "alpha-schedule", *, seed: int = 20260920, pace: str = "fast",
                 block_samples: int = 32, loop: bool = False, **overrides: Any) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; expected one of {', '.join(SCENARIOS)}")
        kw = dict(SCENARIOS[scenario])
        kw.update(overrides)
        data, ts, cues, truth = make_synthetic_stream(seed, **kw)
        self.scenario = scenario
        self.truth = truth
        super().__init__(data, fs=truth["fs"], ch_names=truth["ch_names"], timestamps_s=ts, pace=pace,
                         loop=loop, block_samples=block_samples, kind="synthetic",
                         nominal={"scenario": scenario, "seed": int(seed)}, markers=cues)
