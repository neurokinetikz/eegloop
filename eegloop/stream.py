"""Turning whatever a source delivers into fixed-size blocks, and keeping the loss ledger.

A driver hands over however many samples arrived since it was last asked; an analysis wants blocks
of a fixed length, because the block length is a term in the latency budget. :class:`Reblocker`
regroups the one into the other and carries the bookkeeping across the seam. :func:`detect_gaps`
reads a run of timestamps and reports where samples went missing, which is the only way a wireless
link's losses become visible (``pf-dropped-samples``); :class:`DropAccount` keeps the running total
a session log reports.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .block import Block

__all__ = ["detect_gaps", "DropAccount", "Reblocker"]


def detect_gaps(timestamps_s: np.ndarray, fs: float, *, tol_samples: float = 1.5) -> list[tuple[int, int]]:
    """Where a run of timestamps skips, and by how many samples.

    Returns ``(index, n_missing)`` for every interval longer than ``tol_samples`` sample periods:
    ``index`` is the position of the first sample *after* the gap and ``n_missing`` the number of
    sample periods that passed with nothing received, rounded. A tolerance of 1.5 periods means
    ordinary timestamp jitter never counts as a gap and a single lost sample does.
    """
    ts = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
    if ts.shape[0] < 2:
        return []
    periods = np.diff(ts) * float(fs)
    where = np.nonzero(periods > tol_samples)[0]
    return [(int(i + 1), int(round(periods[i])) - 1) for i in where]


class DropAccount:
    """The running total of samples a stream lost, with when each loss happened."""

    def __init__(self) -> None:
        self.total = 0
        self.events: list[tuple[float, int]] = []

    def add(self, t_s: float, n: int) -> None:
        if n > 0:
            self.total += int(n)
            self.events.append((float(t_s), int(n)))

    def rate_per_min(self, elapsed_s: float) -> float:
        """Dropped samples per minute of session; ``nan`` before any time has elapsed."""
        return float("nan") if elapsed_s <= 0 else self.total / (elapsed_s / 60.0)

    def describe(self) -> dict[str, Any]:
        return {"total": self.total, "n_events": len(self.events), "events": list(self.events)}


class Reblocker:
    """Fixed-size blocks out of whatever a source delivered.

    Samples are carried across calls, so a source that returns 37 samples and then 27 yields two
    32-sample blocks; ``seq`` counts emitted blocks, ``sample_index`` follows the session count the
    source reports (drops included), and a loss the source reported is attached to the next emitted
    block as ``dropped_before``. A loss that falls *inside* an emitted block -- the source reported
    it while samples from before it were still buffered -- cannot be a ``dropped_before`` and is
    recorded instead as ``flags['gaps_inside'] = [(position, n_missing), ...]`` so a downstream step
    can still see it.
    """

    def __init__(self, block_samples: int) -> None:
        if int(block_samples) < 1:
            raise ValueError("block_samples must be at least 1")
        self.block_samples = int(block_samples)
        self._fs: float | None = None
        self._data: np.ndarray | None = None
        self._ts: np.ndarray | None = None
        self._seq = 0
        self._index0 = 0          # session index of the first buffered sample
        self._t0 = 0.0            # source-clock time of the first buffered sample
        self._before = 0          # samples lost immediately before the first buffered sample
        self._gaps: list[tuple[int, int]] = []   # (position in buffer, n_missing) for losses inside it
        self._flags: dict[str, Any] = {}

    @property
    def buffered(self) -> int:
        return 0 if self._data is None else int(self._data.shape[1])

    def push(self, raw: Block) -> list[Block]:
        """Absorb a raw block; return every full block that can now be emitted, in order."""
        if self._fs is None:
            self._fs = raw.fs
        elif raw.fs != self._fs:
            raise ValueError(f"sampling rate changed mid-stream: {self._fs} -> {raw.fs}")
        if self.buffered == 0:
            self._index0 = raw.sample_index
            self._t0 = raw.t_start_s
            self._data = raw.data
            self._ts = raw.timestamps_s
            self._before += raw.dropped_before
        else:
            if raw.dropped_before > 0:
                self._gaps.append((self.buffered, raw.dropped_before))
            self._data = np.concatenate([self._data, raw.data], axis=1)
            if self._ts is not None and raw.timestamps_s is not None:
                self._ts = np.concatenate([self._ts, raw.timestamps_s])
            else:
                self._ts = None
        self._flags = dict(raw.flags) if raw.flags else {}
        out: list[Block] = []
        while self.buffered >= self.block_samples:
            out.append(self._emit(self.block_samples))
        return out

    def flush(self) -> Block | None:
        """The remainder as one short final block, or ``None`` when nothing is buffered."""
        return self._emit(self.buffered) if self.buffered else None

    def _emit(self, n: int) -> Block:
        assert self._data is not None and self._fs is not None
        chunk, self._data = self._data[:, :n], self._data[:, n:]
        ts = None
        if self._ts is not None:
            ts, self._ts = self._ts[:n], self._ts[n:]
        t0 = float(ts[0]) if ts is not None and ts.size else self._t0
        inside = [(p, k) for p, k in self._gaps if p < n]
        later = [(p - n, k) for p, k in self._gaps if p >= n]
        flags = dict(self._flags)
        if inside:
            flags["gaps_inside"] = inside
        block = Block(
            chunk, self._fs, seq=self._seq, sample_index=self._index0, t_start_s=t0,
            timestamps_s=ts, dropped_before=self._before, flags=flags,
        )
        lost_inside = sum(k for _, k in inside)
        self._seq += 1
        self._index0 += n + lost_inside
        self._t0 = t0 + (n + lost_inside) / self._fs
        self._before = 0
        self._gaps = later
        return block
