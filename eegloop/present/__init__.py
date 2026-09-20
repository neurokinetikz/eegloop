"""Where the learner's application begins.

A presenter receives what the participant is shown, once per block, and does something with it. The
package ships the least it can: a console bar that proves the loop is alive, and a callback that
hands every update to whatever the learner builds -- a game, a tone, a web page. Nothing here is a
feedback display worth using; that is the point of the course, not of the library.
"""
from __future__ import annotations

import sys
from typing import Any, Callable, Protocol, runtime_checkable

from ..block import StreamInfo

__all__ = ["Presenter", "NullPresenter", "ConsolePresenter", "CallbackPresenter"]


@runtime_checkable
class Presenter(Protocol):
    def start(self, info: StreamInfo) -> None: ...
    def update(self, shown: float, *, z: float, gated: bool, t_s: float, phase: str) -> None: ...
    def stop(self) -> None: ...


class NullPresenter:
    def start(self, info: StreamInfo) -> None:
        pass

    def update(self, shown: float, *, z: float, gated: bool, t_s: float, phase: str) -> None:
        pass

    def stop(self) -> None:
        pass


class ConsolePresenter:
    """One line, redrawn: a bar for the value, a marker when the gate is closed, the phase and the time.
    A BCI decision prints on its own line."""

    def __init__(self, width: int = 40, stream: Any = None, every_s: float = 0.1) -> None:
        self.width, self.stream, self.every_s = int(width), stream or sys.stdout, float(every_s)
        self._last = -1e9
        self.n_updates = 0

    def start(self, info: StreamInfo) -> None:
        print(f"[{info.kind}] {info.n_channels} ch at {info.fs:g} Hz — {', '.join(info.ch_names)}", file=self.stream)

    def update(self, shown: float, *, z: float, gated: bool, t_s: float, phase: str) -> None:
        self.n_updates += 1
        if t_s - self._last < self.every_s:
            return
        self._last = t_s
        if gated:
            bar = "░" * self.width + "  GATE CLOSED"
        elif shown != shown:
            bar = "·" * self.width + f"  {phase}"
        else:
            n = int(round(min(1.0, max(0.0, shown)) * self.width))
            bar = "█" * n + " " * (self.width - n) + f"  z {z:+.2f}"
        print(f"\r{t_s:7.1f} s  {bar}", end="", file=self.stream)

    def decision(self, d: Any) -> None:
        print(f"\n{d.t_s:7.1f} s  → {d.label}  (p = {d.posterior:.2f}, {d.mode})", file=self.stream)

    def stop(self) -> None:
        print(file=self.stream)


class CallbackPresenter:
    """Every update to ``fn(shown, z=..., gated=..., t_s=..., phase=...)`` and every decision to
    ``on_decision(decision)`` -- the seam for an application."""

    def __init__(self, fn: Callable[..., Any] | None = None, on_start: Callable[[StreamInfo], Any] | None = None,
                 on_stop: Callable[[], Any] | None = None, on_decision: Callable[[Any], Any] | None = None) -> None:
        self.fn, self.on_start, self.on_stop, self.on_decision = fn, on_start, on_stop, on_decision
        self.n_updates = self.n_decisions = 0

    def decision(self, d: Any) -> None:
        self.n_decisions += 1
        if self.on_decision is not None:
            self.on_decision(d)

    def start(self, info: StreamInfo) -> None:
        if self.on_start is not None:
            self.on_start(info)

    def update(self, shown: float, *, z: float, gated: bool, t_s: float, phase: str) -> None:
        self.n_updates += 1
        if self.fn is not None:
            self.fn(shown, z=z, gated=gated, t_s=t_s, phase=phase)

    def stop(self) -> None:
        if self.on_stop is not None:
            self.on_stop()
