"""A fixed-capacity circular buffer of the most recent samples, as an online system keeps.

Ported from ``notebooks/_shared/helpers_l7.py`` (``RingBuffer``, 2026-09-20) with two additions:
it accepts a :class:`~eegloop.block.Block` as well as an array, and it can answer in seconds when it
knows the sampling rate. The reason for its existence is unchanged: ``push`` writes, ``latest``
reads the newest samples in time order, and the buffer never grows and never reallocates, because a
real-time loop must not allocate inside its callback.
"""
from __future__ import annotations

import numpy as np

from .block import Block

__all__ = ["RingBuffer"]


class RingBuffer:
    def __init__(self, n_channels: int, capacity: int, dtype=np.float64, *, fs: float | None = None):
        self.n_channels = int(n_channels)
        self.capacity = int(capacity)
        self.fs = None if fs is None else float(fs)
        if self.capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._buf = np.zeros((self.n_channels, self.capacity), dtype=dtype)
        self._write = 0
        self.n_pushed = 0

    def push(self, block: np.ndarray | Block) -> None:
        data = block.data if isinstance(block, Block) else np.atleast_2d(np.asarray(block))
        if data.shape[0] != self.n_channels:
            raise ValueError(f"expected {self.n_channels} channels, got {data.shape[0]}")
        n = data.shape[1]
        if n > self.capacity:
            data, n = data[:, -self.capacity:], self.capacity
        end = self._write + n
        if end <= self.capacity:
            self._buf[:, self._write:end] = data
        else:
            first = self.capacity - self._write
            self._buf[:, self._write:] = data[:, :first]
            self._buf[:, : n - first] = data[:, first:]
        self._write = end % self.capacity
        self.n_pushed += n

    def latest(self, n: int) -> np.ndarray:
        """The ``n`` newest samples, oldest first -- fewer if fewer have been pushed."""
        n = min(int(n), self.capacity, self.n_pushed)
        idx = (self._write - np.arange(n, 0, -1)) % self.capacity
        return self._buf[:, idx]

    def latest_seconds(self, seconds: float) -> np.ndarray:
        """The newest ``seconds`` of signal; needs ``fs`` to have been given."""
        if self.fs is None:
            raise ValueError("latest_seconds needs the sampling rate: RingBuffer(..., fs=...)")
        return self.latest(int(round(seconds * self.fs)))

    @property
    def filled(self) -> int:
        return min(self.n_pushed, self.capacity)

    @property
    def t_latest_s(self) -> float | None:
        """Session time of the newest sample, on the sample clock; ``None`` without ``fs``."""
        return None if self.fs is None else self.n_pushed / self.fs
