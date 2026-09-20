"""Online steps and the chain that runs them, in an order that is argued rather than configured.

Every step exposes the same surface::

    name: str                      # this instance, for the session log
    kind: str                      # one of CANONICAL_ONLINE_ORDER
    process(block) -> Block        # a new block; the input is never mutated
    reset() -> None                # forget the past -- what a chain does after a long gap
    latency_samples: float | None  # EXACT delay in samples, or None when there is no single number
    latency_note: str              # what the delay is, and how it is measured when it is not exact
    describe() -> dict             # resolved parameters for the session log

Two rules shape this module, both learned the hard way in the design review.

**Quality runs first, on the raw block.** A gate placed after an 8–12 Hz band-pass cannot see line
noise, muscle or a flat channel: the filter has removed exactly the evidence the gate needs. Lesson
L7.5 makes this point about devices that notch before the file exists; a chain that filtered before
it gated would have built the same mistake into its own fixed order.

**Latency is summed only where it is exact.** A linear-phase FIR delays every frequency by
``(N - 1) / 2`` samples; a Butterworth has a delay at a stated frequency. Those are rows in the
budget. A one-pole smoother has no single delay (its impulse response never ends), a quality gate's
window delays the *decision* and not the signal, and a Welch window's centroid sits at half its
length. Those are not rows; they are measured with the loopback probe, or reported as decision
delays, and never added into ``total_ms``. ``notebooks/_shared/helpers_l7.py`` takes the same stance
-- "smoothing costs more latency; the notebook measures how much" -- and this package keeps it.

The order below is fixed for the same reason ``pipelines/eegpipe`` fixes its offline order: a
pipeline whose order is a parameter is a pipeline whose order nobody has argued. Configuration may
leave a step out; it may not move one.
"""
from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from ..block import Block

__all__ = ["OnlineStep", "Chain", "CANONICAL_ONLINE_ORDER"]

#: quality (on the raw block) -> re-reference -> causal filter -> frozen spatial filter ->
#: envelope / band power -> features. Later phases fill the kinds this one does not ship.
CANONICAL_ONLINE_ORDER: tuple[str, ...] = (
    "quality", "reference", "causal_filter", "spatial", "envelope", "features",
)


@runtime_checkable
class OnlineStep(Protocol):
    name: str
    kind: str

    def process(self, block: Block) -> Block: ...
    def reset(self) -> None: ...

    @property
    def latency_samples(self) -> float | None: ...

    @property
    def latency_note(self) -> str: ...

    def describe(self) -> dict[str, Any]: ...


class Chain:
    """Steps run in the canonical order, with one decision about gaps made in one place.

    ``reset_on_gap_samples`` is the longest hole the chain will carry filter state across. A block
    whose ``dropped_before`` exceeds it resets every step first and is flagged ``chain_reset`` with
    the count, so the session log can say where the filters started over. ``None`` never resets,
    which is the right choice only for a source that cannot drop samples (a file).
    """

    def __init__(self, steps: Sequence[OnlineStep], *, reset_on_gap_samples: int | None = None,
                 check_order: bool = True) -> None:
        self.steps: tuple[OnlineStep, ...] = tuple(steps)
        self.reset_on_gap_samples = None if reset_on_gap_samples is None else int(reset_on_gap_samples)
        self.n_resets = 0
        if check_order:
            self.check_order()

    def check_order(self) -> None:
        """Raise unless every step's kind is known and the kinds never run backwards."""
        last = -1
        for step in self.steps:
            kind = getattr(step, "kind", None)
            if kind not in CANONICAL_ONLINE_ORDER:
                raise ValueError(
                    f"step {getattr(step, 'name', step)!r} has kind {kind!r}; "
                    f"expected one of {', '.join(CANONICAL_ONLINE_ORDER)}"
                )
            here = CANONICAL_ONLINE_ORDER.index(kind)
            if here < last:
                raise ValueError(
                    f"step {step.name!r} ({kind}) cannot follow a {CANONICAL_ONLINE_ORDER[last]!r} step: "
                    f"the online order is {' -> '.join(CANONICAL_ONLINE_ORDER)}"
                )
            last = here

    def process(self, block: Block) -> Block:
        if self.reset_on_gap_samples is not None and block.dropped_before > self.reset_on_gap_samples:
            self.reset()
            block = block.with_flag("chain_reset", int(block.dropped_before))
        for step in self.steps:
            block = step.process(block)
        return block

    def reset(self) -> None:
        for step in self.steps:
            step.reset()
        self.n_resets += 1

    @property
    def exact_latency_samples(self) -> float:
        """The sum of the delays that are exact; everything else is in :attr:`inexact`."""
        return float(sum(s.latency_samples for s in self.steps if s.latency_samples is not None))

    @property
    def exact_steps(self) -> list[OnlineStep]:
        return [s for s in self.steps if s.latency_samples is not None]

    @property
    def inexact(self) -> list[tuple[str, str]]:
        """``(name, latency_note)`` for every step whose delay is not one number."""
        return [(s.name, s.latency_note) for s in self.steps if s.latency_samples is None]

    def describe(self) -> list[dict[str, Any]]:
        return [dict(s.describe(), kind=s.kind, latency_samples=s.latency_samples,
                     latency_note=s.latency_note) for s in self.steps]
