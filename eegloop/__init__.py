"""eegloop — streaming EEG loops for the Scalp to Source course.

The package a learner builds a neurofeedback or BCI application on. Its shape is set by two
decisions:

* **One ``Source``, one ``Block``.** Every acquisition path — a synthetic generator, a replayed
  recording, a headset over a driver — emits the same frozen :class:`~eegloop.block.Block`, in
  microvolts, carrying per-sample timestamps and a count of the samples the source believes it lost.
  Everything above the source is therefore device-free, and so is every test.
* **Every step declares what it costs in time, and only where that is exact.** A linear-phase FIR
  delays every frequency by ``(N - 1) / 2`` samples and says so; a smoother has no single delay and
  says *that* instead, so the budget reports it as measured rather than adding a number that is not
  one. Lesson L7.3 spends its length on why this distinction matters.

This is version 0.1: the block contract, the ring buffer, causal filters with their state carried
across blocks, the latency budget and the loopback probe — a port of the real-time section of
``notebooks/_shared/helpers_l7.py`` into something installable, multichannel and unit-tested. The
notebook helpers are not edited; ``tests/test_parity_helpers_l7.py`` holds the two to agreement.
Sources, the quality gate, the feedback loop, the BCI loop and the session runner follow in later
phases; ``README.md`` says which.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .block import Block, StreamInfo
from .latency import (
    BUFFER_CONVENTIONS,
    fir_group_delay_samples,
    iir_group_delay_by_phase,
    iir_group_delay_samples,
    latency_budget,
    latency_budget_for,
    print_budget,
    zero_phase_lookahead_samples,
)
from .probe import energy_centroid, make_probe, measure_loop_delay, run_offline
from .ring import RingBuffer
from .steps import CANONICAL_ONLINE_ORDER, Chain, OnlineStep
from .steps.causal_filter import CausalFIR, CausalSOS, Notch, butter_sos, fir_taps
from .stream import DropAccount, Reblocker, detect_gaps
from .versions import check_pins, config_hash, installed_versions, seed_everything

__all__ = [
    "__version__",
    "Block", "StreamInfo",
    "detect_gaps", "DropAccount", "Reblocker",
    "RingBuffer",
    "OnlineStep", "Chain", "CANONICAL_ONLINE_ORDER",
    "fir_taps", "butter_sos", "CausalFIR", "CausalSOS", "Notch",
    "BUFFER_CONVENTIONS", "fir_group_delay_samples", "iir_group_delay_samples",
    "iir_group_delay_by_phase", "zero_phase_lookahead_samples", "latency_budget",
    "latency_budget_for", "print_budget",
    "make_probe", "energy_centroid", "run_offline", "measure_loop_delay",
    "installed_versions", "check_pins", "seed_everything", "config_hash",
]
