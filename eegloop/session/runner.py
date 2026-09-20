"""Run a protocol end to end: build the source, the loop and the log, walk the phases, write the session."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..feedback.baseline import AdaptiveBaseline, FixedBaseline
from ..feedback.loop import FeedbackLoop
from ..feedback.reward import ContinuousMapping, ThresholdReward
from ..feedback.sham import ShamPolicy
from ..feedback.signal import SignalSpec
from ..sources.base import Source
from ..sources.replay import ReplaySource
from ..sources.synthetic import SyntheticSource
from .log import SessionLog, SessionReader, scrub_value
from .protocol import Protocol, resolved

__all__ = ["SessionResult", "run_protocol", "build_source"]


@dataclass
class SessionResult:
    status: str
    out_dir: str                       # the directory's name, never its absolute path
    phases: list[dict[str, Any]] = field(default_factory=list)
    budget: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    sealed_sham: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return scrub_value({"status": self.status, "out_dir": self.out_dir, "phases": self.phases, "budget": self.budget,
                            "stats": self.stats, "sealed_sham": self.sealed_sham, "error": self.error})


def _base_dir(protocol: Protocol) -> Path:
    p = protocol.__dict__.get("_path")
    return Path(p).parent if p is not None else Path.cwd()


def build_source(protocol: Protocol) -> Source:
    """The source a protocol names. Hardware kinds arrive in a later phase and say so."""
    cfg = protocol.source
    if cfg.kind == "synthetic":
        return SyntheticSource(cfg.scenario, seed=cfg.seed if cfg.seed is not None else protocol.seed, pace=cfg.pace,
                               block_samples=protocol.block_samples, loop=cfg.loop,
                               duration_s=cfg.duration_s if cfg.duration_s is not None else protocol.total_duration_s + 1.0)
    if cfg.kind == "replay":
        path = Path(cfg.path or "")
        if not path.is_absolute():
            path = _base_dir(protocol) / path
        return ReplaySource(path, pace=cfg.pace, loop=cfg.loop, block_samples=protocol.block_samples)
    raise ValueError(f"{cfg.kind!r} sources arrive in a later phase of eegloop; see loop/README.md")


def run_protocol(protocol: Protocol, source: Source | None = None, *, presenter: Any = None,
                 out_dir: str | Path | None = None, fail_fast: bool = False) -> SessionResult:
    src = source if source is not None else build_source(protocol)
    spec = SignalSpec.from_config(protocol.signal)
    donor = None
    if protocol.sham.mode == "sham-yoked":
        dp = Path(protocol.sham.donor or "")
        if not dp.is_absolute():
            dp = _base_dir(protocol) / dp
        donor = SessionReader(dp).feedback[1]
    sham = ShamPolicy(protocol.sham.mode, donor=donor, control_band=(float(protocol.signal.control_band[0]), float(protocol.signal.control_band[1])),
                      seed=protocol.seed)
    b = protocol.baseline
    if b.mode == "adaptive":
        baseline: Any = AdaptiveBaseline(b.window_s, src.info.fs, protocol.block_samples, percentile=b.percentile, spread=b.spread)
        factory = None
    else:
        baseline = None
        factory = lambda vals: FixedBaseline.from_values(vals, centre=b.centre, spread=b.spread)  # noqa: E731
    r = protocol.reward
    mapping = ContinuousMapping(r.lo_z, r.hi_z)
    reward = ThresholdReward(r.threshold_z, dwell_s=r.dwell_s, refractory_s=r.refractory_s, on_miss=r.on_miss) if r.mode == "threshold" else None
    where = Path(out_dir) if out_dir is not None else _base_dir(protocol) / protocol.output.dir / protocol.name
    log = SessionLog(where, protocol=resolved(protocol), info=src.info, record_raw=protocol.output.record_raw, sealed_sham=sham.seal())
    loop = FeedbackLoop(src, spec, block_samples=protocol.block_samples, baseline=baseline, baseline_factory=factory,
                        mapping=mapping, reward=reward, sham=sham, presenter=presenter, log=log, quality=protocol.quality,
                        gate_policy=protocol.quality.policy, processing_ms_stated=protocol.processing_ms,
                        convention=protocol.buffer_convention, markers=getattr(src, "marker_source", lambda: None)())
    if log.recorder is not None:
        # the raw stream is recorded as it arrives, before any step touches it
        original = loop.chain.process

        def recording_process(block):  # type: ignore[no-untyped-def]
            log.raw(block)
            return original(block)

        loop.chain.process = recording_process  # type: ignore[method-assign]
    phases = [(p.kind, p.duration_s) for p in protocol.phases]
    status, error = "ok", None
    try:
        loop.run(phases)
    except Exception as exc:  # noqa: BLE001 - a failure is a row in the log
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        if fail_fast:
            log.close(session={"status": status, "error": error})
            raise
    budget = loop.budget()
    stats = loop.stats.describe()
    log.event("end", loop.stats.duration_s, status=status)
    log.close(session={"status": status, "error": error, "budget": budget, "stats": stats,
                       "signal": spec.describe(src.info.fs, protocol.block_samples), "sham_note": sham.note,
                       "baseline": None if loop.baseline is None else loop.baseline.describe(),
                       "chain": loop.chain.describe(), "phases": [{"kind": k, "duration_s": d} for k, d in phases]})
    return SessionResult(status=status, out_dir=where.name, phases=[{"kind": k, "duration_s": d} for k, d in phases],
                         budget=scrub_value(budget), stats=stats, sealed_sham=sham.seal(), error=error)
