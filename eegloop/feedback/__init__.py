"""The neurofeedback loop: lesson L7.3's five decisions as five objects, plus the sham modes."""
from __future__ import annotations

from .baseline import AdaptiveBaseline, FixedBaseline
from .loop import FeedbackLoop, LoopStats
from .reward import ContinuousMapping, RewardEvent, ThresholdReward
from .sham import CREDNF_ITEMS, SHAM_MODES, ShamPolicy
from .signal import SignalSpec, build_chain, combine

__all__ = [
    "SignalSpec", "build_chain", "combine",
    "FixedBaseline", "AdaptiveBaseline",
    "ContinuousMapping", "ThresholdReward", "RewardEvent",
    "SHAM_MODES", "CREDNF_ITEMS", "ShamPolicy",
    "FeedbackLoop", "LoopStats",
]
