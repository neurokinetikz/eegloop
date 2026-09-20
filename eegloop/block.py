"""The unit of data every source emits and every step consumes.

A :class:`Block` is a few samples of every channel, in microvolts, with the bookkeeping a real-time
system cannot reconstruct afterwards: where it sits in the session's sample count, what the source's
clock said, per-sample timestamps when the source has them, and how many samples the source believes
it lost since the previous block. That last field lives on the block rather than in a side channel
because a filter has to see it -- carrying filter state across a 400 ms hole as if the signal were
continuous is silently wrong, and it is the online form of ``pf-filter-across-boundaries``.

Blocks are frozen. A step returns a new block (``block.replace(data=...)``) rather than mutating the
one it was given, so a chain can be replayed and two steps can never disagree about what they saw.
Lesson L7.3 lists what must be logged at acquisition because it cannot be recovered later: the
nominal versus effective sampling rate, and every dropped sample with its timestamp. The block is
where both of those become data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace as _replace
from typing import Any

import numpy as np

__all__ = ["Block", "StreamInfo"]


@dataclass(frozen=True)
class StreamInfo:
    """What a source knows about its stream before the first block arrives.

    ``nominal`` holds the facts as the driver or file reported them -- including any filtering the
    device applied before this package saw the signal. Lesson L7.5's point that a built-in notch
    deletes the channel-quality diagnostic as well as the noise means this record has to travel with
    every session, so an analysis can say what it could and could not have checked.
    """

    fs: float
    ch_names: tuple[str, ...]
    kind: str                       # 'synthetic' | 'replay' | 'brainflow' | 'lsl'
    clock: str = "sample"           # 'sample' | 'wall' | 'device' | 'lsl'
    units: str = "uV"
    nominal: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ch_names", tuple(str(c) for c in self.ch_names))
        if not self.fs > 0:
            raise ValueError(f"fs must be positive, got {self.fs!r}")
        if not self.ch_names:
            raise ValueError("a stream needs at least one channel name")

    @property
    def n_channels(self) -> int:
        return len(self.ch_names)


@dataclass(frozen=True, eq=False)
class Block:
    """``n_channels x n_samples`` of microvolts, and where those samples came from.

    ``sample_index`` counts every sample of the session *including the dropped ones*, so the
    difference between two blocks' indices is elapsed samples on the source's clock, not samples
    received. ``dropped_before`` is the count of samples the source believes were lost immediately
    before this block's first sample; it is zero on a healthy stream and it is what a chain reads to
    decide whether its filter state still describes the signal.
    """

    data: np.ndarray
    fs: float
    seq: int = 0
    sample_index: int = 0
    t_start_s: float = 0.0
    timestamps_s: np.ndarray | None = None
    dropped_before: int = 0
    flags: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        data = np.asarray(self.data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.ndim != 2:
            raise ValueError(f"data must be (n_channels, n_samples), got shape {data.shape}")
        object.__setattr__(self, "data", data)
        if not self.fs > 0:
            raise ValueError(f"fs must be positive, got {self.fs!r}")
        if self.dropped_before < 0:
            raise ValueError("dropped_before cannot be negative")
        if self.timestamps_s is not None:
            ts = np.asarray(self.timestamps_s, dtype=np.float64).reshape(-1)
            if ts.shape[0] != data.shape[1]:
                raise ValueError(
                    f"timestamps_s has {ts.shape[0]} entries for {data.shape[1]} samples"
                )
            object.__setattr__(self, "timestamps_s", ts)

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[1])

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs

    @property
    def t_end_s(self) -> float:
        """The instant this block is complete -- the earliest a value computed from it can exist."""
        return self.t_start_s + self.duration_s

    def replace(self, **changes: Any) -> "Block":
        """A copy with some fields changed; the original is untouched."""
        return _replace(self, **changes)

    def with_flag(self, key: str, value: Any) -> "Block":
        """A copy carrying one more entry in ``flags`` (the dict itself is copied, never shared)."""
        flags = dict(self.flags)
        flags[key] = value
        return _replace(self, flags=flags)
