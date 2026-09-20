"""Where blocks come from: a synthetic generator with planted answers, a replayed recording, and --
in a later phase -- a headset behind the same interface.

Everything above a source is device-free because every source emits the same
:class:`~eegloop.block.Block`. :func:`open_source` is the one factory; :func:`blocks` turns any
source into fixed-size blocks; :class:`ListMarkers` delivers cues in the same clock as the EEG.
"""
from __future__ import annotations

from .base import SOURCE_KINDS, Source, blocks, open_source
from .markers import ListMarkers, MarkerSource
from .replay import ReplaySource, read_recording
from .synthetic import SCENARIOS, SyntheticSource, make_synthetic_stream

__all__ = [
    "SOURCE_KINDS", "Source", "blocks", "open_source",
    "MarkerSource", "ListMarkers",
    "ReplaySource", "read_recording",
    "SyntheticSource", "make_synthetic_stream", "SCENARIOS",
]
