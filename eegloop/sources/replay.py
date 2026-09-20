"""A recorded array or file, replayed as a stream -- at wall-clock pace or as fast as it will go.

Replay is how the course proves a loop without a headset: the same chain that runs on a device runs
on a public recording, and every number a lesson quotes comes from here or from the synthetic
generator. Two things make a replay honest rather than a shortcut:

* **Pacing.** ``pace='wall'`` releases samples on the wall clock, so a 60-s file takes 60 s and a
  loop that cannot keep up falls behind here exactly as it would on a device. ``pace='fast'`` is for
  tests and notebooks.
* **Loss.** A gap in the recording's timestamps -- or one injected with ``inject_gaps`` -- is never
  papered over: reads never straddle it, the block after it carries ``dropped_before``, and
  ``sample_index`` counts the samples that are not there (``pf-dropped-samples``).

Formats: an array; ``.npz`` with ``data``, ``fs``, ``ch_names`` (and optional ``timestamps_s``);
``.csv`` with one column per channel and an optional time column; the site's own §4.5 asset (a
Float32 little-endian ``.bin`` beside its JSON sidecar); and ``.fif`` / ``.edf`` / ``.bdf`` through
MNE when the ``mne`` extra is installed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..block import Block, StreamInfo
from ..stream import detect_gaps

__all__ = ["ReplaySource", "read_recording"]

_TIME_COLUMNS = ("time", "timestamp", "timestamps", "timestamps_s", "time_s", "t")


def read_recording(source: Any, *, fs: float | None = None, ch_names: Any = None,
                   timestamps_s: Any = None) -> tuple[np.ndarray, float, tuple[str, ...], np.ndarray | None, dict]:
    """``(data_uv, fs, ch_names, timestamps_s | None, nominal)`` from an array or a file path."""
    nominal: dict[str, Any] = {}
    ts = None if timestamps_s is None else np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        nominal["file"] = path.name  # the name only: a directory would be a machine-specific path
        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as z:
                data = np.asarray(z["data"], dtype=np.float64)
                fs = float(z["fs"]) if fs is None and "fs" in z else fs
                if ch_names is None and "ch_names" in z:
                    ch_names = [str(c) for c in z["ch_names"]]
                if ts is None and "timestamps_s" in z:
                    ts = np.asarray(z["timestamps_s"], dtype=np.float64).reshape(-1)
        elif suffix == ".csv":
            table = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64, encoding="utf-8")
            names = list(table.dtype.names or ())
            tcol = next((n for n in names if n.lower() in _TIME_COLUMNS), None)
            if tcol is not None and ts is None:
                ts = np.asarray(table[tcol], dtype=np.float64).reshape(-1)
            chans = [n for n in names if n != tcol]
            data = np.vstack([np.asarray(table[n], dtype=np.float64) for n in chans])
            ch_names = chans if ch_names is None else ch_names
            if fs is None and ts is not None and ts.size > 1:
                fs = float(1.0 / np.median(np.diff(ts)))
                nominal["fs_inferred_from_time_column"] = True
        elif suffix == ".bin":
            sidecar = path.with_suffix(".json")
            if not sidecar.exists():
                raise FileNotFoundError(f"{path.name} needs its sidecar {sidecar.name} beside it (§4.5)")
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            chans = [str(c) for c in meta["channels"]]
            data = np.fromfile(path, dtype="<f4").astype(np.float64).reshape(len(chans), -1)
            fs = float(meta["sfreq"]) if fs is None else fs
            ch_names = chans if ch_names is None else ch_names
            if str(meta.get("units", "uV")).lower() not in ("uv", "µv"):
                raise ValueError(f"asset units are {meta.get('units')!r}; eegloop blocks are µV")
            for k in ("dataset", "subject", "run", "reference", "filters_applied", "hardware_filters",
                      "mains_hz", "license", "source_doi", "t0_s", "condition"):
                if k in meta:
                    nominal[k] = meta[k]
        elif suffix in (".fif", ".edf", ".bdf"):
            try:
                import mne
            except ImportError as e:  # pragma: no cover - exercised only without the extra
                raise ImportError(f"reading {suffix} needs MNE: pip install 'eegloop[mne]'") from e
            reader = {".fif": mne.io.read_raw_fif, ".edf": mne.io.read_raw_edf, ".bdf": mne.io.read_raw_bdf}[suffix]
            raw = reader(path, preload=True, verbose="ERROR")
            raw.pick("eeg")
            data = raw.get_data() * 1e6
            fs = float(raw.info["sfreq"]) if fs is None else fs
            ch_names = list(raw.ch_names) if ch_names is None else ch_names
            nominal["highpass_hz_in_header"] = float(raw.info["highpass"])
            nominal["lowpass_hz_in_header"] = float(raw.info["lowpass"])
        else:
            raise ValueError(f"unsupported recording format {suffix!r}: use .npz, .csv, .bin (+.json), .fif, .edf or .bdf")
    else:
        data = np.asarray(source, dtype=np.float64)
        nominal["file"] = None
    data = np.atleast_2d(data)
    if fs is None:
        raise ValueError("fs is required unless the file records it")
    if ch_names is None:
        ch_names = tuple(f"ch{i + 1}" for i in range(data.shape[0]))
    ch_names = tuple(str(c) for c in ch_names)
    if len(ch_names) != data.shape[0]:
        raise ValueError(f"{len(ch_names)} channel names for {data.shape[0]} rows")
    if ts is not None and ts.shape[0] != data.shape[1]:
        raise ValueError(f"{ts.shape[0]} timestamps for {data.shape[1]} samples")
    nominal.update({"n_samples": int(data.shape[1]), "duration_s": data.shape[1] / float(fs)})
    return data, float(fs), ch_names, ts, nominal


class ReplaySource:
    """Replay an array or file as a stream (see the module docstring)."""

    def __init__(self, source: Any, *, fs: float | None = None, ch_names: Any = None,
                 timestamps_s: Any = None, pace: str = "wall", loop: bool = False,
                 inject_gaps: Any = (), start_s: float = 0.0, stop_s: float | None = None,
                 block_samples: int = 32, kind: str = "replay", nominal: dict | None = None,
                 markers: Any = ()) -> None:
        data, fs, names, ts, nom = read_recording(source, fs=fs, ch_names=ch_names, timestamps_s=timestamps_s)
        if pace not in ("wall", "fast"):
            raise ValueError("pace must be 'wall' or 'fast'")
        # crop on the ideal time axis
        i0 = int(round(start_s * fs))
        i1 = data.shape[1] if stop_s is None else min(data.shape[1], int(round(stop_s * fs)))
        if not 0 <= i0 < i1:
            raise ValueError("start_s/stop_s select no samples")
        data = data[:, i0:i1]
        ts = (ts[i0:i1] if ts is not None else None)
        # injected losses: samples removed, and the timestamps that remain show the jump
        gaps = [(float(t), int(n)) for t, n in inject_gaps]
        if gaps:
            if ts is None:
                ts = np.arange(data.shape[1]) / fs + (i0 / fs)
            for t_s, n in sorted(gaps, reverse=True):
                g0 = int(np.searchsorted(ts, t_s))
                data = np.delete(data, np.s_[g0:g0 + n], axis=1)
                ts = np.delete(ts, np.s_[g0:g0 + n])
        self._data, self._ts = data, ts
        self._gaps: dict[int, int] = dict(detect_gaps(ts, fs)) if ts is not None else {}
        self._gap_indices = sorted(self._gaps)
        nom = dict(nom, **(nominal or {}))
        nom.update({"pace": pace, "gaps": [(i, n) for i, n in sorted(self._gaps.items())]})
        self.info = StreamInfo(fs, names, kind=kind, clock="device" if ts is not None else "sample", nominal=nom)
        self.pace, self.loop, self.block_samples = pace, bool(loop), int(block_samples)
        self.markers = list(markers)
        self.done = False
        self._started = False
        self._pos = 0
        self._seq = 0
        self._lap = 0
        self._emitted = 0
        self._t0: float | None = None

    # -- the Source protocol ---------------------------------------------------------------------
    @property
    def n_samples(self) -> int:
        return int(self._data.shape[1])

    @property
    def n_missing_total(self) -> int:
        return int(sum(self._gaps.values()))

    @property
    def duration_s(self) -> float:
        """Session time the recording spans, dropped samples included."""
        return (self.n_samples + self.n_missing_total) / self.info.fs

    def start(self) -> None:
        self._started, self.done = True, False
        self._pos = self._seq = self._lap = self._emitted = 0
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        self.done = True

    def clock_now(self) -> float:
        """Session time of the newest sample handed out, on the recording's own axis."""
        return self._lap * self.duration_s + (self._time_at(self._pos) if self._pos < self.n_samples else self.duration_s)

    def read(self, max_samples: int | None = None) -> Block | None:
        if not self._started:
            self.start()
        if self.done:
            return None
        if self._pos >= self.n_samples:
            if not self.loop:
                self.done = True
                return None
            self._pos, self._lap = 0, self._lap + 1
        want = int(max_samples or self.block_samples)
        if self.pace == "wall":
            assert self._t0 is not None
            available = int((time.perf_counter() - self._t0) * self.info.fs) - self._emitted
            if available <= 0:
                return None
            want = min(want, available)
        next_gap = next((g for g in self._gap_indices if g > self._pos), self.n_samples)
        want = min(want, next_gap - self._pos, self.n_samples - self._pos)
        chunk = self._data[:, self._pos:self._pos + want]
        ts = self._ts[self._pos:self._pos + want] if self._ts is not None else None
        missing_before = sum(n for i, n in self._gaps.items() if i <= self._pos)
        block = Block(
            chunk, self.info.fs, seq=self._seq,
            sample_index=self._lap * (self.n_samples + self.n_missing_total) + self._pos + missing_before,
            t_start_s=self._lap * self.duration_s + self._time_at(self._pos),
            timestamps_s=None if ts is None else ts + self._lap * self.duration_s,
            dropped_before=self._gaps.get(self._pos, 0),
        )
        self._pos += want
        self._seq += 1
        self._emitted += want
        return block

    def __enter__(self) -> "ReplaySource":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # -- helpers ---------------------------------------------------------------------------------
    def _time_at(self, pos: int) -> float:
        if self._ts is not None and pos < self.n_samples:
            return float(self._ts[pos])
        return pos / self.info.fs

    def marker_source(self):
        """The recording's cues as a :class:`~eegloop.sources.markers.ListMarkers` on this clock."""
        from .markers import ListMarkers

        return ListMarkers(self.markers, self.clock_now)

    @classmethod
    def from_asset(cls, bin_path: str | Path, **kw: Any) -> "ReplaySource":
        """The site's own §4.5 asset: a Float32 LE ``.bin`` with its JSON sidecar beside it."""
        return cls(Path(bin_path), **kw)
