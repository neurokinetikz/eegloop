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
``notebooks/_shared/helpers_l7.py`` into something installable, multichannel and unit-tested (the
notebook helpers are not edited; ``tests/test_parity_helpers_l7.py`` holds the two to agreement) —
plus the sources (a synthetic stream with planted answers, replay of arrays, files and the site's own
assets, cues on the source clock) and the online steps (the quality gate that runs first on the raw
block, re-referencing, frozen spatial filters, envelopes, features). The feedback loop, the BCI
loop, the session runner and the hardware sources follow; ``README.md`` says which.
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
from .sources import (
    SCENARIOS, ListMarkers, MarkerSource, ReplaySource, Source, SyntheticSource, blocks,
    make_synthetic_stream, open_source, read_recording,
)
from .ring import RingBuffer
from .steps import CANONICAL_ONLINE_ORDER, Chain, OnlineStep
from .steps.causal_filter import CausalFIR, CausalSOS, Notch, butter_sos, fir_taps
from .steps.envelope import BandPower, BlockRMS, Smoother
from .steps.features import FeatureExtractor
from .steps.quality import QUALITY_LABELS, QualityGate
from .steps.reference import Reference
from .steps.spatial import SpatialFilter
from .feedback import (
    CREDNF_ITEMS, SHAM_MODES, AdaptiveBaseline, ContinuousMapping, FeedbackLoop, FixedBaseline, LoopStats,
    RewardEvent, ShamPolicy, SignalSpec, ThresholdReward, build_chain, combine,
)
from .present import CallbackPresenter, ConsolePresenter, NullPresenter, Presenter
from .session import (
    Protocol, ProtocolError, Recorder, SessionLog, SessionReader, SessionResult, build_source, export_fif,
    load_protocol, resolved, run_protocol, scrub_paths, validate_protocol,
)
from .analysis import crednf_report, learning_test, naive_trend, session_change
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
    "Source", "open_source", "blocks", "SyntheticSource", "make_synthetic_stream", "SCENARIOS",
    "ReplaySource", "read_recording", "MarkerSource", "ListMarkers",
    "QualityGate", "QUALITY_LABELS", "Reference", "SpatialFilter", "BlockRMS", "Smoother", "BandPower",
    "FeatureExtractor",
    "SignalSpec", "build_chain", "combine", "FixedBaseline", "AdaptiveBaseline", "ContinuousMapping",
    "ThresholdReward", "RewardEvent", "SHAM_MODES", "CREDNF_ITEMS", "ShamPolicy", "FeedbackLoop", "LoopStats",
    "Presenter", "NullPresenter", "ConsolePresenter", "CallbackPresenter",
    "Protocol", "ProtocolError", "load_protocol", "validate_protocol", "resolved", "SessionLog", "SessionReader",
    "Recorder", "export_fif", "SessionResult", "run_protocol", "build_source", "scrub_paths",
    "naive_trend", "session_change", "learning_test", "crednf_report",
    "installed_versions", "check_pins", "seed_everything", "config_hash",
]
