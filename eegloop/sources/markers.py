"""Cues, in the same clock as the EEG.

A BCI calibration needs to know when the cue was shown, on the axis the samples are on. Lesson L7.3
puts it plainly: alignment between a stimulus stream and the EEG is a property of the timestamps,
not of the order packets arrived in. So a marker source is anything that yields ``(t_s, label)``
pairs stamped with the EEG source's :meth:`~eegloop.sources.base.Source.clock_now`.

Two marker sources ship here: :class:`ListMarkers` -- a fixed schedule, which is what a replayed
session, a synthetic scenario and a protocol file all have -- and :class:`KeyboardMarkers`, a key
press stamped with the EEG clock at the moment it is read, for a self-paced session run by hand.
The driver source carries its own (:meth:`eegloop.sources.brainflow.BrainFlowSource.marker_source`,
the board's marker row); a network marker stream waits with the LSL binding.
"""
from __future__ import annotations

import sys
from typing import Any, Callable, Iterable, Mapping, Protocol, runtime_checkable

__all__ = ["MarkerSource", "ListMarkers", "KeyboardMarkers"]


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


def _stdin_reader() -> Callable[[], str]:
    """Keys pressed since the last call, without blocking: POSIX cbreak mode on a terminal, else nothing."""
    try:
        import select
        import termios
        import tty
    except ImportError:  # pragma: no cover - not POSIX
        return lambda: ""
    if not sys.stdin.isatty():
        return lambda: ""
    fd = sys.stdin.fileno()
    state: dict[str, Any] = {"old": None}

    def read() -> str:
        if state["old"] is None:
            state["old"] = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        keys = []
        while select.select([sys.stdin], [], [], 0)[0]:
            keys.append(sys.stdin.read(1))
        return "".join(keys)

    def restore() -> None:
        if state["old"] is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, state["old"])
            state["old"] = None

    read.restore = restore  # type: ignore[attr-defined]
    return read


class KeyboardMarkers:
    """Cues from key presses, each stamped with the EEG source's clock at the moment it is read.

    ``keymap`` maps a key (one character) to a label; ``clock`` is the source's ``clock_now``, so a
    cue lands on the axis the samples are on -- as late as the loop's own read of the keyboard,
    which is once per block, and that is the resolution stated in ``note``. ``reader`` is any
    zero-argument callable returning the keys pressed since the last call as a string (the default
    reads a POSIX terminal without blocking; tests pass their own). ``close()`` restores the
    terminal.
    """

    note = "a key press is stamped when the loop reads the keyboard, once per block: the cue's resolution is one block"

    def __init__(self, keymap: Mapping[str, str], clock: Callable[[], float], *, reader: Callable[[], str] | None = None) -> None:
        self.keymap = {str(k): str(v) for k, v in keymap.items()}
        if any(len(k) != 1 for k in self.keymap):
            raise ValueError("keymap keys are single characters")
        self._clock = clock
        self._reader = reader if reader is not None else _stdin_reader()
        self.n_read = 0

    def read(self) -> list[tuple[float, str]]:
        now = float(self._clock())
        out: list[tuple[float, str]] = []
        for ch in self._reader() or "":
            label = self.keymap.get(ch)
            if label is not None:
                out.append((now, label))
                self.n_read += 1
        return out

    def close(self) -> None:
        restore = getattr(self._reader, "restore", None)
        if callable(restore):
            restore()
