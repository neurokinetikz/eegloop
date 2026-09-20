"""The BCI loop: calibrate once, freeze, apply -- and say what the frozen thing costs in time.

Lesson L7.15 re-teaches decoding for a reader who skipped Level 6: what is decodable at rank 4, the
chance band, leakage, calibrate-then-apply. The library's shape follows that lesson. Epochs are cut
from the *same* causal chain the online windows come from, so offline and online features are one
computation; the decoder is fitted with folds in cue order, never shuffled, because adjacent trials
share slow state; and the fitted pipeline is **frozen** into numpy arrays that replay it exactly, so
the application that runs it needs no scikit-learn, and cannot quietly refit.
"""
from __future__ import annotations

from .decoder import FrozenDecoder, chance_interval, fit_frozen, freeze, temporal_split
from .epochs import Calibrator, EpochCutter, cut_epochs
from .pipelines import PIPELINE_NOTES, PIPELINES, csp_lda, make_pipeline, riemann_ts, xdawn_lda
from .ssvep import SSVEP_NOTE, SSVEPDetector, cca_correlation, ssvep_cca, ssvep_decide
from .stream import BCILoop, BCIStats, Decision, DwellDecision, StreamingPosterior

__all__ = [
    "PIPELINES", "PIPELINE_NOTES", "csp_lda", "riemann_ts", "xdawn_lda", "make_pipeline",
    "EpochCutter", "Calibrator", "cut_epochs",
    "FrozenDecoder", "fit_frozen", "freeze", "chance_interval", "temporal_split",
    "StreamingPosterior", "DwellDecision", "Decision", "BCILoop", "BCIStats",
    "cca_correlation", "ssvep_cca", "ssvep_decide", "SSVEPDetector", "SSVEP_NOTE",
]
