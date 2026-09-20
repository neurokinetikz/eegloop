"""The one interface every acquisition path implements, and the two helpers that use it."""
from __future__ import annotations

import time
from typing import Any, Iterator, Mapping, Protocol, runtime_checkable

from ..block import Block, StreamInfo
from ..stream import Reblocker

__all__ = ["SOURCE_KINDS", "Source", "open_source", "blocks"]

SOURCE_KINDS: tuple[str, ...] = ("synthetic", "replay", "brainflow", "lsl")


@runtime_checkable
class Source(Protocol):
    """A stream of blocks, read non-blockingly.

    ``read()`` returns whatever has arrived since the last call as one :class:`Block` -- of any
    length -- or ``None`` when nothing has. ``done`` becomes true when nothing more will ever
    arrive (a file ran out; a device never sets it). ``clock_now()`` is the source's own clock, so
    a cue stamped with it lands on the same axis as the samples.
    """

    info: StreamInfo
    done: bool

    def start(self) -> None: ...
    def read(self, max_samples: int | None = None) -> Block | None: ...
    def stop(self) -> None: ...
    def clock_now(self) -> float: ...


def open_source(spec: str | Mapping[str, Any], **kw: Any) -> Source:
    """``'synthetic'``, ``'synthetic:<scenario>'``, ``'replay:<path>'`` -- or a mapping with a ``kind``.

    The hardware kinds are named here so a protocol file can already say what it means; they arrive
    in a later phase and until then say so instead of failing obscurely.
    """
    if isinstance(spec, Mapping):
        opts = {k: v for k, v in spec.items() if k != "kind"}
        kind = str(spec.get("kind", ""))
    else:
        kind, _, arg = str(spec).partition(":")
        opts = {}
        if arg:
            opts[{"synthetic": "scenario", "replay": "source", "brainflow": "board", "lsl": "name"}.get(kind, "arg")] = arg
    opts.update(kw)
    if kind == "synthetic":
        from .synthetic import SyntheticSource

        return SyntheticSource(**opts)
    if kind == "replay":
        from .replay import ReplaySource

        return ReplaySource(**opts)
    if kind in ("brainflow", "lsl"):
        raise ValueError(f"{kind!r} sources arrive in a later phase of eegloop; see loop/README.md")
    raise ValueError(f"unknown source kind {kind!r}; expected one of {', '.join(SOURCE_KINDS)}")


def blocks(source: Source, block_samples: int, *, timeout_s: float | None = None,
           idle_sleep_s: float = 0.002) -> Iterator[Block]:
    """Fixed-size blocks from any source, until it is done or ``timeout_s`` passes with nothing.

    A trailing remainder shorter than ``block_samples`` is not yielded: the block length is a term
    in the latency budget, and a short last block would be a different pipeline for one step.
    """
    rb = Reblocker(block_samples)
    if not getattr(source, "_started", True):
        source.start()
    idle_since: float | None = None
    while True:
        raw = source.read()
        if raw is None:
            if source.done:
                return
            now = time.perf_counter()
            idle_since = now if idle_since is None else idle_since
            if timeout_s is not None and now - idle_since > timeout_s:
                return
            time.sleep(idle_sleep_s)
            continue
        idle_since = None
        yield from rb.push(raw)
