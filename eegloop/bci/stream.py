"""Apply a frozen decoder to the stream, and turn posteriors into decisions the application can use.

Two ways to apply. *Sliding*: every ``step_samples`` the latest window goes through the decoder and
the posterior is smoothed; a :class:`DwellDecision` fires when one class has stayed above threshold
long enough -- the shape of a self-paced control. *Cue-locked*: the window after each cue is
classified once -- the shape of an oddball speller. :class:`BCILoop` runs either, walks the phases
(calibrate → fit → apply), scores decisions against the cues it can see, and logs all of it.

Everything is on the block clock. A gated block stops the posterior and marks the stretch unusable
for any epoch that overlaps it; nothing is cleaned.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from ..block import Block
from ..ring import RingBuffer
from ..sources.base import Source, blocks
from ..sources.markers import MarkerSource
from ..steps import Chain
from .decoder import FrozenDecoder, chance_interval
from .epochs import Calibrator, EpochCutter

__all__ = ["Decision", "StreamingPosterior", "DwellDecision", "BCILoop", "BCIStats"]


@dataclass(frozen=True)
class Decision:
    t_s: float
    index: int
    label: str
    posterior: float
    dwell_s: float
    mode: str


class StreamingPosterior:
    """The latest ``epoch_samples`` through the decoder every ``step_samples``, optionally smoothed."""

    def __init__(self, decoder: FrozenDecoder, *, step_samples: int, smooth_s: float = 0.0) -> None:
        self.decoder, self.step, self.smooth_s = decoder, int(step_samples), float(smooth_s)
        self.ring = RingBuffer(len(decoder.ch_names), decoder.epoch_samples)
        self._since = 0
        self._p: np.ndarray | None = None
        self.n_updates = 0

    def update(self, block: Block) -> np.ndarray | None:
        self.ring.push(block)
        self._since += block.n_samples
        if self.ring.filled < self.decoder.epoch_samples or self._since < self.step:
            return None
        self._since = 0
        p = self.decoder.predict_proba(self.ring.latest(self.decoder.epoch_samples))
        if self.smooth_s > 0 and self._p is not None:
            a = float(np.exp(-(self.step / self.decoder.fs) / self.smooth_s))
            p = a * self._p + (1.0 - a) * p
        self._p = p
        self.n_updates += 1
        return p

    def reset(self) -> None:
        self.ring = RingBuffer(len(self.decoder.ch_names), self.decoder.epoch_samples)
        self._since, self._p = 0, None

    def reset_smoothing(self) -> None:
        """Forget the smoothed history but keep the signal: a cued trial starts fresh."""
        self._p = None

    @property
    def latency_samples(self) -> float:
        return self.decoder.latency_samples

    @property
    def latency_note(self) -> str:
        s = f"; one-pole smoothing with tau {self.smooth_s:g} s adds a tail with no single delay -- measured, not summed" if self.smooth_s > 0 else ""
        return self.decoder.latency_note + s

    def describe(self) -> dict[str, Any]:
        return {"step_samples": self.step, "smooth_s": self.smooth_s, "epoch_samples": self.decoder.epoch_samples,
                "latency_note": self.latency_note}


class DwellDecision:
    """A decision when one class's posterior has stayed above ``threshold`` for ``dwell_s``, at most one per ``refractory_s``."""

    def __init__(self, classes: Sequence[str], *, threshold: float = 0.7, dwell_s: float = 0.5, refractory_s: float = 1.0) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.classes = tuple(classes)
        self.threshold, self.dwell_s, self.refractory_s = float(threshold), float(dwell_s), float(refractory_s)
        self._k: int | None = None
        self._since: float | None = None
        self._last_t: float | None = None
        self.n_decisions = 0

    def update(self, p: np.ndarray, t_s: float) -> Decision | None:
        k = int(np.argmax(p))
        if p[k] >= self.threshold:
            if self._k != k:
                self._k, self._since = k, t_s
            dwelt = t_s - float(self._since)
            ready = self._last_t is None or t_s - self._last_t >= self.refractory_s
            if dwelt >= self.dwell_s and ready:
                self._last_t = t_s
                self.n_decisions += 1
                return Decision(float(t_s), k, self.classes[k], float(p[k]), float(dwelt), "sliding")
        else:
            self._k = self._since = None
        return None

    def reset(self) -> None:
        self._k = self._since = self._last_t = None

    def describe(self) -> dict[str, Any]:
        return {"threshold": self.threshold, "dwell_s": self.dwell_s, "refractory_s": self.refractory_s,
                "note": "the dwell is a choice, not a cost of the decoder; it trades decisions per minute for false ones"}


@dataclass
class BCIStats:
    n_blocks: int = 0
    n_gated: int = 0
    n_posteriors: int = 0
    n_epochs_calibration: int = 0
    duration_s: float = 0.0
    decisions: list[dict[str, Any]] = field(default_factory=list)
    trials: list[dict[str, Any]] = field(default_factory=list)
    processing_ms: list[float] = field(default_factory=list)
    fit_ms: float | None = None

    def describe(self, classes: Sequence[str] = ()) -> dict[str, Any]:
        scored = [t for t in self.trials if t["decided"]]
        correct = sum(1 for t in scored if t["correct"])
        n = len(scored)
        acc = correct / n if n else None
        band = chance_interval(n, 1.0 / max(2, len(classes))) if n and classes else None
        p = np.asarray(self.processing_ms, dtype=float) if self.processing_ms else np.zeros(1)
        return {"n_blocks": self.n_blocks, "n_gated": self.n_gated, "n_posteriors": self.n_posteriors,
                "n_epochs_calibration": self.n_epochs_calibration, "duration_s": self.duration_s,
                "n_decisions": len(self.decisions),
                "per_class": {c: int(sum(1 for d in self.decisions if d["label"] == c)) for c in classes},
                "online": {"n_trials": len(self.trials), "n_decided": n, "n_correct": int(correct), "accuracy": acc,
                           "chance_band_95": band,
                           "note": "each cued trial in an apply phase is scored by its first decision inside the window; "
                                   "undecided trials count as misses of the schedule, not of the class"},
                "fit_ms": self.fit_ms,
                "processing_ms": {"p50": float(np.percentile(p, 50)), "p95": float(np.percentile(p, 95)), "max": float(p.max()),
                                  "n": int(p.size), "note": "measured with time.perf_counter around each block's work"}}


class BCILoop:
    def __init__(self, source: Source, chain: Chain, *, block_samples: int, decoder: FrozenDecoder | None = None,
                 calibrator: Calibrator | None = None, fitter: Callable[[np.ndarray, np.ndarray], FrozenDecoder] | None = None,
                 mode: str = "sliding", step_samples: int = 32, smooth_s: float = 0.0, threshold: float = 0.7,
                 dwell_s: float = 0.5, refractory_s: float = 1.0, presenter: Any = None, log: Any = None,
                 markers: MarkerSource | None = None, classes: Sequence[str] | None = None, grace_s: float = 1.0) -> None:
        if mode not in ("sliding", "cue-locked"):
            raise ValueError("mode must be 'sliding' or 'cue-locked'")
        self.source, self.info, self.chain = source, source.info, chain
        self.block_samples, self.mode = int(block_samples), mode
        self.decoder, self.calibrator, self.fitter = decoder, calibrator, fitter
        self.step_samples, self.smooth_s = int(step_samples), float(smooth_s)
        self.threshold, self.dwell_s, self.refractory_s, self.grace_s = float(threshold), float(dwell_s), float(refractory_s), float(grace_s)
        self.presenter, self.log, self.markers = presenter, log, markers
        self.classes = tuple(classes) if classes is not None else (decoder.classes if decoder else (calibrator.classes if calibrator else ()))
        self.stats = BCIStats()
        self.posterior: StreamingPosterior | None = None
        self.decision: DwellDecision | None = None
        self.cutter: EpochCutter | None = None
        self._trial: dict[str, Any] | None = None
        if decoder is not None:
            self._arm(decoder)

    # -- set-up ----------------------------------------------------------------------------------------
    def _arm(self, decoder: FrozenDecoder) -> None:
        self.decoder = decoder
        if self.mode == "sliding":
            self.posterior = StreamingPosterior(decoder, step_samples=self.step_samples, smooth_s=self.smooth_s)
            self.decision = DwellDecision(decoder.classes, threshold=self.threshold, dwell_s=self.dwell_s, refractory_s=self.refractory_s)
        else:
            self.cutter = EpochCutter(self.info, tmin_s=decoder.tmin_s, tmax_s=decoder.tmax_s)

    def fit(self) -> FrozenDecoder:
        """Fit from the calibrator's harvest (the runner calls this when the first apply phase begins)."""
        if self.calibrator is None or self.fitter is None:
            raise RuntimeError("no calibrator and fitter: give the loop a decoder, or both")
        X, y, _ = self.calibrator.harvest()
        self.stats.n_epochs_calibration = int(len(y))
        t0 = time.perf_counter()
        dec = self.fitter(X, y)
        self.stats.fit_ms = (time.perf_counter() - t0) * 1000.0
        if self.log is not None:
            self.log.event("decoder", self.stats.duration_s, decoder=dec.describe(), calibration=self.calibrator.describe(), fit_ms=self.stats.fit_ms)
        self._arm(dec)
        return dec

    # -- scoring against cues --------------------------------------------------------------------------
    def _open_trial(self, t_c: float, label: str) -> None:
        """Sliding mode: a cued trial is scored by its first decision from the first window that lies
        entirely after the cue (``t_cue + tmax``) until ``grace_s`` later; the dwell and the smoothing
        start fresh so a decision carried over from before the cue cannot be credited to it."""
        if self._trial is not None and not self._trial["decided"]:
            self.stats.trials.append(self._trial)
        tmax = self.decoder.tmax_s if self.decoder else 0.0
        self._trial = {"t_cue_s": t_c, "label": label, "t_open_s": t_c + tmax, "t_close_s": t_c + tmax + self.grace_s,
                       "decided": False, "correct": None, "t_decision_s": None, "decided_label": None}
        if self.posterior is not None:
            self.posterior.reset_smoothing()
        if self.decision is not None:
            self.decision.reset()

    def _score(self, d: Decision) -> None:
        tr = self._trial
        if tr is None or tr["decided"] or not (tr["t_open_s"] <= d.t_s <= tr["t_close_s"]):
            return
        tr.update(decided=True, correct=(d.label == tr["label"]), t_decision_s=d.t_s, decided_label=d.label)
        self.stats.trials.append(tr)

    def _emit(self, d: Decision, phase: str) -> None:
        rec = {"t_s": d.t_s, "index": d.index, "label": d.label, "posterior": d.posterior, "dwell_s": d.dwell_s, "mode": d.mode, "phase": phase}
        self.stats.decisions.append(rec)
        if self.log is not None:
            self.log.event("decision", d.t_s, **{k: v for k, v in rec.items() if k != "t_s"})
        if self.presenter is not None and hasattr(self.presenter, "decision"):
            self.presenter.decision(d)

    # -- one block -------------------------------------------------------------------------------------
    def step(self, block: Block, *, phase: str = "test") -> dict[str, Any]:
        t0 = time.perf_counter()
        out = self.chain.process(block)
        q = out.flags.get("quality", {"ok": True, "state": "good", "labels": []})
        gated = not q.get("ok", True)
        cues = self.markers.read() if self.markers is not None else []
        if self.log is not None:
            for t_c, label in cues:
                self.log.event("cue", t_c, label=label, phase=phase)
            if gated:
                self.log.event("gate", block.t_end_s, state=q.get("state"), labels=q.get("labels", []))
            if block.dropped_before:
                self.log.event("gap", block.t_start_s, n_missing=int(block.dropped_before))
        if gated:
            self.stats.n_gated += 1
        rec: dict[str, Any] = {"t_s": block.t_end_s, "phase": phase, "gate_ok": not gated, "cues": cues, "posterior": None, "decision": None}
        if phase in ("rest", "calibrate"):
            if self.calibrator is not None:
                if gated:
                    self.calibrator.mark_bad_block(out)
                self.calibrator.push(out, cues)
        elif phase in ("train", "test"):
            if self.decoder is None:
                self.fit()
            if self.mode == "sliding":
                for t_c, label in cues:
                    if label in self.classes:
                        self._open_trial(t_c, label)
                assert self.posterior is not None and self.decision is not None
                if gated:
                    self.posterior.reset()
                    self.decision.reset()
                else:
                    p = self.posterior.update(out)
                    if p is not None:
                        self.stats.n_posteriors += 1
                        rec["posterior"] = p
                        if self.log is not None:
                            self.log.posterior(block.t_end_s, p, True, phase)
                        d = self.decision.update(p, block.t_end_s)
                        if d is not None:
                            rec["decision"] = d
                            self._score(d)
                            self._emit(d, phase)
                        if self.presenter is not None:
                            self.presenter.update(float(p.max()), z=float("nan"), gated=False, t_s=block.t_end_s, phase=phase)
            else:
                assert self.cutter is not None and self.decoder is not None
                if gated:
                    self.cutter.mark_bad_block(out)
                for t_c, label, x in self.cutter.push(out, cues):
                    p = self.decoder.predict_proba(x)
                    self.stats.n_posteriors += 1
                    if self.log is not None:
                        self.log.posterior(t_c + self.decoder.tmax_s, p, True, phase)
                    k = int(np.argmax(p))
                    d = Decision(float(t_c + self.decoder.tmax_s), k, self.decoder.classes[k], float(p[k]), 0.0, "cue-locked")
                    if label in self.classes:
                        self.stats.trials.append({"t_cue_s": t_c, "label": label, "decided": True, "correct": d.label == label,
                                                  "t_decision_s": d.t_s, "decided_label": d.label})
                    rec["decision"] = d
                    self._emit(d, phase)
        self.stats.n_blocks += 1
        self.stats.duration_s = block.t_end_s
        self.stats.processing_ms.append((time.perf_counter() - t0) * 1000.0)
        return rec

    # -- a whole session -------------------------------------------------------------------------------
    def run(self, phases: Sequence[tuple[str, float]] | None = None, *, max_s: float | None = None) -> BCIStats:
        phases = list(phases) if phases else [("test", float("inf"))]
        bounds, t = [], 0.0
        for kind, dur in phases:
            t += float(dur)
            bounds.append((kind, t))
        if self.presenter is not None:
            self.presenter.start(self.info)
        idx = 0
        if self.log is not None:
            self.log.event("phase", 0.0, phase=bounds[0][0], index=0)
        try:
            for block in blocks(self.source, self.block_samples):
                while idx < len(bounds) - 1 and block.t_end_s > bounds[idx][1]:
                    idx += 1
                    if self.log is not None:
                        self.log.event("phase", block.t_start_s, phase=bounds[idx][0], index=idx)
                self.step(block, phase=bounds[idx][0])
                if block.t_end_s >= bounds[-1][1] or (max_s is not None and block.t_end_s >= max_s):
                    break
        finally:
            if self._trial is not None and not self._trial["decided"]:
                self.stats.trials.append(self._trial)
                self._trial = None
            if self.presenter is not None:
                self.presenter.stop()
        return self.stats

    def describe(self) -> dict[str, Any]:
        return {"mode": self.mode, "classes": list(self.classes),
                "decoder": None if self.decoder is None else self.decoder.describe(),
                "posterior": None if self.posterior is None else self.posterior.describe(),
                "decision": None if self.decision is None else self.decision.describe(),
                "cutter": None if self.cutter is None else self.cutter.describe(),
                "calibration": None if self.calibrator is None else self.calibrator.describe()}
