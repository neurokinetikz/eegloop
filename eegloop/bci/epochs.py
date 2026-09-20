"""Epochs cut from the stream, on one stated clock, with the same rounding online and offline.

A calibration needs windows around cue times. Online, the window is cut from a ring of the
*processed* stream when its last sample has arrived; offline, the same window is cut from the array.
:func:`cut_epochs` and :class:`EpochCutter` round the same way, and a test holds them equal to the
sample, so the features a decoder is fitted on are the features the loop computes.

**Which clock.** Lesson L7.11 puts it plainly: alignment between a stimulus stream and the EEG is a
property of the timestamps, not of the order packets arrived in. A block carries two clocks -- its
``sample_index`` (the source's own count, losses included) and its ``t_start_s``/``timestamps_s``
(what the source's clock said). They drift apart: the synthetic stream's timestamps run 200 ppm
fast, which is 24 ms, six samples, by the end of two minutes. Which clock the *cues* are on is a
property of the marker source, not of the EEG source, so the cutter is told: ``clock='sample'`` (cue
times are seconds on the sample clock, ``index = round(t × fs)``, exact integers -- the default, and
what a replayed or synthetic recording's own cues are) or ``clock='timestamps'`` (cue times are on
the clock the source's per-sample timestamps use, as they are when an application stamps its cues
from the wall clock and a device stamps its samples the same way). Choosing wrong misaligns every
epoch by the drift, and a test shows it.

Two things make an epoch unusable and are counted, never patched: a loss of samples inside it (the
source reported a gap) and a stretch the quality gate closed. *Gate rather than clean* applies to
calibration as much as to feedback -- a decoder fitted on a blink learns the blink.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Iterable, Sequence

import numpy as np

from ..block import Block, StreamInfo
from ..ring import RingBuffer
from ..sources.markers import MarkerSource

__all__ = ["EpochCutter", "Calibrator", "cut_epochs", "epoch_samples"]


def epoch_samples(tmin_s: float, tmax_s: float, fs: float) -> int:
    n = int(round((float(tmax_s) - float(tmin_s)) * float(fs)))
    if n < 2:
        raise ValueError("an epoch needs at least two samples: check tmin_s < tmax_s")
    return n


def cut_epochs(x: np.ndarray, fs: float, cues: Iterable[tuple[float, str]], *, tmin_s: float, tmax_s: float,
               classes: Sequence[str] | None = None, t0_s: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(X, y, t)`` from an array on the sample clock: ``X`` is ``(n_epochs, n_channels, n_samples)``,
    ``y`` the labels (class indices when ``classes`` is given, else the strings), ``t`` the cue times.
    Cues whose window runs off either end are skipped. ``t0_s`` is the time of ``x[:, 0]``."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    n = epoch_samples(tmin_s, tmax_s, fs)
    X, y, t = [], [], []
    for t_c, label in sorted((float(a), str(b)) for a, b in cues):
        if classes is not None and label not in classes:
            continue
        i0 = int(round((t_c + tmin_s - t0_s) * fs))
        if i0 < 0 or i0 + n > x.shape[1]:
            continue
        X.append(x[:, i0:i0 + n])
        y.append(classes.index(label) if classes is not None else label)
        t.append(t_c)
    Xa = np.asarray(X, dtype=float) if X else np.zeros((0, x.shape[0], n))
    return Xa, np.asarray(y), np.asarray(t, dtype=float)


class EpochCutter:
    """Cut ``[t_cue + tmin, t_cue + tmax)`` from a stream of blocks once its last sample has arrived."""

    def __init__(self, info: StreamInfo, *, tmin_s: float, tmax_s: float, history_s: float = 2.0,
                 clock: str = "sample") -> None:
        self.fs, self.n_channels = float(info.fs), info.n_channels
        self.tmin_s, self.tmax_s = float(tmin_s), float(tmax_s)
        self.n = epoch_samples(tmin_s, tmax_s, self.fs)
        # which clock the CUES are on -- a property of the marker source, not of the EEG source: a
        # replayed or synthetic recording's cues are on its sample clock even when its timestamps drift
        self.clock = clock
        if self.clock not in ("sample", "timestamps"):
            raise ValueError("clock must be 'sample' or 'timestamps'")
        self.capacity = self.n + int(round(max(float(history_s), 0.5) * self.fs)) + 2
        self.ring = RingBuffer(self.n_channels, self.capacity)
        self._ts = RingBuffer(1, self.capacity) if self.clock == "timestamps" else None
        self._idx_end: int | None = None        # sample clock: index after the newest sample, losses counted
        self._t_latest: float | None = None     # timestamps: the newest sample's own stamp
        self._pending: list[tuple[float, str]] = []
        self._gaps: deque = deque()             # (start, end, n_missing) in the cutter's clock units
        self._bad: deque = deque()              # (start, end) in the cutter's clock units
        self.n_dropped_gap = self.n_dropped_bad = self.n_dropped_late = 0
        self.n_cut = 0

    # -- the two clocks ----------------------------------------------------------------------------
    def _to_units(self, t_s: float) -> float:
        return float(round(t_s * self.fs)) if self.clock == "sample" else float(t_s)

    @property
    def newest(self) -> float | None:
        """The newest sample's position in the cutter's units (an index, or a timestamp)."""
        return float(self._idx_end) if self.clock == "sample" and self._idx_end is not None else self._t_latest

    def mark_bad(self, t_start_s: float, t_end_s: float) -> None:
        """Declare a stretch unusable, in seconds on the cutter's clock: an epoch overlapping it is dropped."""
        self._bad.append((self._to_units(t_start_s), self._to_units(t_end_s)))

    def mark_bad_block(self, block: Block) -> None:
        """Declare a block's own span unusable (the gate closed on it)."""
        if self.clock == "sample":
            self._bad.append((float(block.sample_index), float(block.sample_index + block.n_samples)))
        elif block.timestamps_s is not None:
            self._bad.append((float(block.timestamps_s[0]), float(block.timestamps_s[-1]) + 1.0 / self.fs))
        else:
            self._bad.append((float(block.t_start_s), float(block.t_end_s)))

    def _purge(self) -> None:
        if self.newest is None:
            return
        horizon = self.newest - (self.capacity + self.fs) * (1.0 if self.clock == "sample" else 1.0 / self.fs)
        while self._gaps and self._gaps[0][1] < horizon:
            self._gaps.popleft()
        while self._bad and self._bad[0][1] < horizon:
            self._bad.popleft()

    # -- feeding -----------------------------------------------------------------------------------
    def push(self, block: Block, cues: Iterable[tuple[float, str]] = ()) -> list[tuple[float, str, np.ndarray]]:
        """Push a block and any new cues; return every ``(t_cue, label, epoch)`` now complete."""
        if block.n_channels != self.n_channels:
            raise ValueError(f"cutter expects {self.n_channels} channels, got {block.n_channels}")
        if self.clock == "timestamps" and block.timestamps_s is None:
            raise ValueError("clock='timestamps' needs per-sample timestamps on every block; this source has none")
        if block.dropped_before:
            if self.clock == "sample":
                self._gaps.append((float(block.sample_index - block.dropped_before), float(block.sample_index), int(block.dropped_before)))
            else:
                self._gaps.append((float(block.t_start_s - block.dropped_before / self.fs), float(block.t_start_s), int(block.dropped_before)))
        self.ring.push(block)
        if self.clock == "sample":
            self._idx_end = int(block.sample_index + block.n_samples)
        else:
            assert self._ts is not None and block.timestamps_s is not None
            self._ts.push(np.asarray(block.timestamps_s, dtype=float)[None, :])
            self._t_latest = float(block.timestamps_s[-1])
        self._pending.extend((float(t), str(label)) for t, label in cues)
        self._purge()
        out, still = [], []
        for t_c, label in sorted(self._pending):
            got = self._cut_sample(t_c) if self.clock == "sample" else self._cut_timestamps(t_c)
            if got is None:
                still.append((t_c, label))
            elif got is not False:
                out.append((t_c, label, got))
                self.n_cut += 1
        self._pending = still
        return out

    def _overlaps(self, a0: float, a1: float) -> bool | None:
        """``True`` dropped for a gap, ``False`` dropped for a gated stretch, ``None`` clean."""
        if any(g0 < a1 and g1 > a0 for g0, g1, _ in self._gaps):
            return True
        if any(b0 < a1 and b1 > a0 for b0, b1 in self._bad):
            return False
        return None

    def _cut_sample(self, t_c: float) -> np.ndarray | None | bool:
        assert self._idx_end is not None
        i0 = int(round((t_c + self.tmin_s) * self.fs))
        i1 = i0 + self.n
        if i1 > self._idx_end:
            return None
        hit = self._overlaps(float(i0), float(i1))
        if hit is not None:
            self.n_dropped_gap += hit
            self.n_dropped_bad += not hit
            return False
        # samples delivered after the epoch's last one: the elapsed indices, less any loss in between
        end_off = self._idx_end - i1 - sum(n for g0, _, n in self._gaps if g0 >= i1)
        if end_off < 0 or end_off + self.n > self.ring.filled:
            self.n_dropped_late += 1
            return False
        return self.ring.latest(end_off + self.n)[:, :self.n].copy()

    def _cut_timestamps(self, t_c: float) -> np.ndarray | None | bool:
        assert self._ts is not None and self._t_latest is not None
        e0 = t_c + self.tmin_s
        if self._t_latest < e0 + (self.n - 0.5) / self.fs:
            return None
        ts = self._ts.latest(self._ts.filled)[0]
        pos = int(np.searchsorted(ts, e0 - 0.5 / self.fs))
        if pos + self.n > ts.size:
            return None
        e1 = float(ts[pos + self.n - 1]) + 1.0 / self.fs
        hit = self._overlaps(float(ts[pos]), e1)
        if hit is not None:
            self.n_dropped_gap += hit
            self.n_dropped_bad += not hit
            return False
        if pos < ts.size - self.ring.filled:
            self.n_dropped_late += 1
            return False
        return self.ring.latest(self.ring.filled)[:, pos:pos + self.n].copy()

    def describe(self) -> dict[str, Any]:
        return {"tmin_s": self.tmin_s, "tmax_s": self.tmax_s, "epoch_samples": self.n, "clock": self.clock, "n_cut": self.n_cut,
                "n_dropped": {"gap": self.n_dropped_gap, "gated": self.n_dropped_bad, "late": self.n_dropped_late}}


class Calibrator:
    """Harvest labelled epochs from the stream as cues arrive; ``harvest()`` gives ``(X, y, t)``."""

    def __init__(self, info: StreamInfo, *, tmin_s: float, tmax_s: float, classes: Sequence[str],
                 markers: MarkerSource | None = None, history_s: float = 2.0, clock: str = "sample") -> None:
        self.classes = tuple(str(c) for c in classes)
        if len(set(self.classes)) < 2:
            raise ValueError("a calibration needs at least two distinct classes")
        self.cutter = EpochCutter(info, tmin_s=tmin_s, tmax_s=tmax_s, history_s=history_s, clock=clock)
        self.markers = markers
        self._X: list[np.ndarray] = []
        self._y: list[int] = []
        self._t: list[float] = []
        self.n_ignored = 0

    def push(self, block: Block, cues: Iterable[tuple[float, str]] | None = None) -> int:
        """Push a block; cues come from ``markers`` unless given. Returns how many epochs completed."""
        cues = list(cues) if cues is not None else (self.markers.read() if self.markers is not None else [])
        keep = []
        for t_c, label in cues:
            if label in self.classes:
                keep.append((t_c, label))
            else:
                self.n_ignored += 1
        done = self.cutter.push(block, keep)
        for t_c, label, x in done:
            self._X.append(x)
            self._y.append(self.classes.index(label))
            self._t.append(t_c)
        return len(done)

    def mark_bad(self, t_start_s: float, t_end_s: float) -> None:
        self.cutter.mark_bad(t_start_s, t_end_s)

    def mark_bad_block(self, block: Block) -> None:
        self.cutter.mark_bad_block(block)

    @property
    def n_epochs(self) -> int:
        return len(self._y)

    def harvest(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.cutter.n
        X = np.asarray(self._X, dtype=float) if self._X else np.zeros((0, self.cutter.n_channels, n))
        return X, np.asarray(self._y, dtype=int), np.asarray(self._t, dtype=float)

    def describe(self) -> dict[str, Any]:
        d = self.cutter.describe()
        d.update({"classes": list(self.classes), "n_epochs": self.n_epochs, "n_ignored_cues": self.n_ignored,
                  "n_per_class": {c: int(sum(1 for k in self._y if k == i)) for i, c in enumerate(self.classes)}})
        return d
