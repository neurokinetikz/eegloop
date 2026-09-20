"""Run a protocol end to end: build the source, the loop and the log, walk the phases, write the session."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..bci.decoder import fit_frozen
from ..bci.epochs import Calibrator
from ..bci.stream import BCILoop
from ..feedback.baseline import AdaptiveBaseline, FixedBaseline
from ..feedback.loop import FeedbackLoop
from ..feedback.reward import ContinuousMapping, ThresholdReward
from ..feedback.sham import ShamPolicy
from ..feedback.signal import SignalSpec, build_chain
from ..latency import latency_budget_for
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


def _record_through(log: SessionLog, chain: Any) -> None:
    """The raw stream is recorded as it arrives, before any step touches it."""
    if log.recorder is None:
        return
    original = chain.process

    def recording_process(block):  # type: ignore[no-untyped-def]
        log.raw(block)
        return original(block)

    chain.process = recording_process  # type: ignore[method-assign]


def _session_dir(protocol: Protocol, out_dir: str | Path | None) -> Path:
    return Path(out_dir) if out_dir is not None else _base_dir(protocol) / protocol.output.dir / protocol.name


def run_protocol(protocol: Protocol, source: Source | None = None, *, presenter: Any = None,
                 out_dir: str | Path | None = None, fail_fast: bool = False) -> SessionResult:
    src = source if source is not None else build_source(protocol)
    if protocol.bci is not None:
        return _run_bci(protocol, src, presenter=presenter, out_dir=out_dir, fail_fast=fail_fast)
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
    where = _session_dir(protocol, out_dir)
    log = SessionLog(where, protocol=resolved(protocol), info=src.info, record_raw=protocol.output.record_raw, sealed_sham=sham.seal())
    loop = FeedbackLoop(src, spec, block_samples=protocol.block_samples, baseline=baseline, baseline_factory=factory,
                        mapping=mapping, reward=reward, sham=sham, presenter=presenter, log=log, quality=protocol.quality,
                        gate_policy=protocol.quality.policy, processing_ms_stated=protocol.processing_ms,
                        convention=protocol.buffer_convention, markers=getattr(src, "marker_source", lambda: None)())
    _record_through(log, loop.chain)
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


def _run_bci(protocol: Protocol, src: Source, *, presenter: Any, out_dir: str | Path | None, fail_fast: bool) -> SessionResult:
    """Calibrate on the cued phase, fit and freeze at the first apply phase, apply, log, save the decoder."""
    b = protocol.bci
    assert b is not None
    spec = SignalSpec(band=(float(b.band[0]), float(b.band[1])), n_taps=int(b.n_taps), reference=protocol.signal.reference,
                      reference_channels=tuple(protocol.signal.reference_channels))
    chain = build_chain(spec, src.info, protocol.block_samples, quality=protocol.quality, envelope=False)
    markers = getattr(src, "marker_source", lambda: None)()
    where = _session_dir(protocol, out_dir)
    log = SessionLog(where, protocol=resolved(protocol), info=src.info, record_raw=protocol.output.record_raw, sealed_sham=None)
    calibrator = Calibrator(src.info, tmin_s=b.tmin_s, tmax_s=b.tmax_s, classes=b.classes)
    delay = chain.exact_latency_samples

    def fitter(X, y):  # type: ignore[no-untyped-def]
        return fit_frozen(b.pipeline, X, y, classes=b.classes, fs=src.info.fs, ch_names=src.info.ch_names, tmin_s=b.tmin_s,
                          tmax_s=b.tmax_s, folds=b.folds, seed=protocol.seed, band=(float(b.band[0]), float(b.band[1])),
                          filter_delay_samples=delay, n_components=b.n_components)

    loop = BCILoop(src, chain, block_samples=protocol.block_samples, calibrator=calibrator, fitter=fitter, mode=b.mode,
                   step_samples=b.step_samples, smooth_s=b.smooth_s, threshold=b.threshold, dwell_s=b.dwell_s,
                   refractory_s=b.refractory_s, presenter=presenter, log=log, markers=markers, classes=b.classes, grace_s=b.grace_s)
    _record_through(log, chain)
    phases = [(p.kind, p.duration_s) for p in protocol.phases]
    status, error = "ok", None
    try:
        loop.run(phases)
    except Exception as exc:  # noqa: BLE001 - a failure is a row in the log
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        if fail_fast:
            log.close(session={"status": status, "error": error})
            raise
    if loop.decoder is not None:
        log.decoder_file = loop.decoder.save(where / "decoder.npz").name
    budget = latency_budget_for(chain, fs=src.info.fs, block_samples=protocol.block_samples,
                                processing_ms=protocol.processing_ms, convention=protocol.buffer_convention)
    if loop.decoder is not None:
        budget["decoder"] = {"latency_samples": loop.decoder.latency_samples, "latency_ms": loop.decoder.latency_samples / src.info.fs * 1000.0,
                             "note": loop.decoder.latency_note}
    stats = loop.stats.describe(b.classes)
    if loop.stats.processing_ms:
        p = stats["processing_ms"]
        budget["processing_measured_ms"] = p
        budget["rows"][2]["detail"] = f"measured on this machine: p50 {p['p50']:.2f} ms, p95 {p['p95']:.2f} ms (stated {protocol.processing_ms:g})"
    log.event("end", loop.stats.duration_s, status=status)
    log.close(session={"status": status, "error": error, "budget": budget, "stats": stats, "bci": loop.describe(),
                       "signal": (f"{b.band[0]:g}–{b.band[1]:g} Hz on every channel, a causal linear-phase FIR of {b.n_taps} taps "
                                  f"(group delay {delay / src.info.fs * 1000:.1f} ms), windows of {b.tmax_s - b.tmin_s:g} s from "
                                  f"{b.tmin_s:g} s after each cue, decoded by {b.pipeline} in {b.mode} mode"),
                       "chain": chain.describe(), "phases": [{"kind": k, "duration_s": d} for k, d in phases]})
    return SessionResult(status=status, out_dir=where.name, phases=[{"kind": k, "duration_s": d} for k, d in phases],
                         budget=scrub_value(budget), stats=stats, sealed_sham=None, error=error)
