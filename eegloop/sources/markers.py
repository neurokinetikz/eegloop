"""Cues, in the same clock as the EEG.

A BCI calibration needs to know when the cue was shown, on the axis the samples are on. Lesson L7.3
puts it plainly: alignment between a stimulus stream and the EEG is a property of the timestamps,
not of the order packets arrived in. So a marker source is anything that yields ``(t_s, label)``
pairs stamped with the EEG source's :meth:`~eegloop.sources.base.Source.clock_now`.

This phase ships :class:`ListMarkers` -- a fixed schedule, which is what a replayed session, a
synthetic scenario and a protocol file all have. Live markers over a network stream, or from the
board itself, are TODO(confirm) per device and arrive with the hardware sources.
"""
from __future__ import annotations

from typing import Callable, Iterable, Protocol, runtime_checkable

__all__ = ["MarkerSource", "ListMarkers"]


@runtime_checkable
class MarkerSource(Protocol):
    def read(self) -> list[tuple[float, str]]:
        """Every cue whose time has passed since the last call, each returned once."""
        ...


class ListMarkers:
    """A fixed schedule of cues, released as the source clock passes each one."""

    def __init__(self, cues: Iterable[tuple[float, str]], clock: Callable[[], float]) -> None:
        self.cues: list[tuple[float, str]] = sorted((float(t), str(label)) for t, label in cues)
        self._clock = clock
        self._next = 0

    def read(self) -> list[tuple[float, str]]:
        now = self._clock()
        out: list[tuple[float, str]] = []
        while self._next < len(self.cues) and self.cues[self._next][0] <= now:
            out.append(self.cues[self._next])
            self._next += 1
        return out

    def reset(self) -> None:
        self._next = 0

    @property
    def remaining(self) -> int:
        return len(self.cues) - self._next
