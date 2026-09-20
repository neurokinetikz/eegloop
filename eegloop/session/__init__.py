"""A protocol file, a session log a notebook can read, a recorder, and the runner that ties them.

The session log is the product the course's evaluation lessons analyse and the donor a yoked sham
replays from. Giving it a reader inside the package means an offline notebook needs nothing but
numpy to open one.
"""
from __future__ import annotations

from .log import SessionLog, SessionReader, scrub_paths, scrub_value
from .protocol import (
    BaselineConfig, BCIConfig, OutputConfig, PhaseConfig, Protocol, ProtocolError, QualityConfig, RewardConfig,
    ShamConfig, SignalConfig, SourceConfig, VersionsConfig, load_protocol, resolved, validate_protocol,
)
from .record import Recorder, export_fif
from .runner import SessionResult, build_source, run_protocol

__all__ = [
    "Protocol", "ProtocolError", "load_protocol", "validate_protocol", "resolved",
    "SourceConfig", "SignalConfig", "QualityConfig", "BaselineConfig", "RewardConfig", "ShamConfig", "BCIConfig",
    "PhaseConfig", "OutputConfig", "VersionsConfig",
    "SessionLog", "SessionReader", "scrub_paths", "scrub_value",
    "Recorder", "export_fif",
    "SessionResult", "run_protocol", "build_source",
]
