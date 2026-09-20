"""The loop: source → chain → value → baseline → mapping → what the participant sees, block by block.

Everything the loop decides is an object it was given, so a protocol file can name each decision and
the session log can record it. Two policies are built in because L7.3 says they are the honest
defaults: the value is shown one block after it exists (a per-block value cannot be shown during the
block that produced it), and when the gate says the signal is unusable the display is **frozen or
zeroed, never fed a cleaned estimate** of something that was not measurable.

The processing row of the budget is *measured* here -- ``time.perf_counter`` around each block's
work -- because no arithmetic can know what compute costs on the machine running the loop.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from ..block import Block
from ..latency import latency_budget_for
from ..probe import run_offline
from ..sources.base import Source, blocks
from ..sources.markers import MarkerSource
from .baseline import AdaptiveBaseline, FixedBaseline
from .reward import ContinuousMapping, RewardEvent, ThresholdReward
from .sham import ShamPolicy
from .signal import SignalSpec, build_chain, combine

__all__ = ["FeedbackLoop", "LoopStats"]


@dataclass
class LoopStats:
    n_blocks: int = 0
    n_gated: int = 0
    n_rewards: int = 0
    drops_total: int = 0
    duration_s: float = 0.0
    processing_ms: list[float] = field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        p = np.asarray(self.processing_ms, dtype=float) if self.processing_ms else np.zeros(1)
        return {"n_blocks": self.n_blocks, "n_gated": self.n_gated, "n_rewards": self.n_rewards,
                "drops_total": self.drops_total, "duration_s": self.duration_s,
                "processing_ms": {"p50": float(np.percentile(p, 50)), "p95": float(np.percentile(p, 95)),
                                  "max": float(p.max()), "n": int(p.size),
                                  "note": "measured with time.perf_counter around each block's work on this machine"}}


class FeedbackLoop:
    def __init__(self, source: Source, spec: SignalSpec, *, block_samples: int,
                 baseline: FixedBaseline | AdaptiveBaseline | None,
                 mapping: ContinuousMapping | None = None, reward: ThresholdReward | None = None,
                 sham: ShamPolicy | None = None, presenter: Any = None, log: Any = None,
                 quality: Any = None, gate_policy: str = "freeze", processing_ms_stated: float = 10.0,
                 convention: str = "block-period", markers: MarkerSource | None = None,
                 baseline_factory: Callable[[Sequence[float]], FixedBaseline] | None = None) -> None:
        self.source, self.info = source, source.info
        self.sham = sham or ShamPolicy("veridical")
        self.spec = self.sham.adapt_spec(spec)
        self.block_samples = int(block_samples)
        self.chain = build_chain(self.spec, self.info, self.block_samples, quality=quality)
        self.picks = self.spec.picks(self.info.ch_names)
        self.baseline = baseline
        self.baseline_factory = baseline_factory
        self.mapping = mapping or ContinuousMapping()
        self.reward = reward
        self.presenter, self.log, self.markers = presenter, log, markers
        if gate_policy not in ("freeze", "zero"):
            raise ValueError("gate_policy must be 'freeze' or 'zero'")
        self.gate_policy = gate_policy
        self.processing_ms_stated, self.convention = float(processing_ms_stated), convention
        self.stats = LoopStats()
        self._k = 0
        self._shown = float("nan")
        self._calibration: list[float] = []

    # -- one block --------------------------------------------------------------------------------
    def step(self, block: Block, *, phase: str = "train") -> dict[str, Any]:
        t0 = time.perf_counter()
        out = self.chain.process(block)
        q = out.flags.get("quality", {"ok": True, "labels": [], "state": "good", "ready": True})
        v_true = combine(out.flags["envelope"], self.picks, self.spec.derivation)
        centre = getattr(self.baseline, "centre", None) if self.baseline is not None else None
        v = self.sham.value(v_true, self._k, centre)
        calibrating = phase in ("rest", "calibrate")
        if calibrating:
            self._calibration.append(v)
        if self.baseline is not None and not calibrating:
            self.baseline.update(v)
        z = float("nan") if self.baseline is None or calibrating else self.baseline.z(v)
        reward: RewardEvent | None = None
        gated = not q.get("ok", True)
        if gated:
            self.stats.n_gated += 1
            shown = 0.0 if self.gate_policy == "zero" else self._shown
        elif calibrating or z != z:
            shown = float("nan")
        else:
            if self.reward is not None:
                reward = self.reward.update(z, block.t_end_s)
                shown = self.reward.map(z)
                if reward is not None:
                    self.stats.n_rewards += 1
            else:
                shown = self.mapping.map(z)
        self._shown = shown
        self.stats.n_blocks += 1
        self.stats.drops_total += int(block.dropped_before)
        self.stats.duration_s = block.t_end_s
        ms = (time.perf_counter() - t0) * 1000.0
        self.stats.processing_ms.append(ms)
        rec = {"k": self._k, "t_s": block.t_end_s, "phase": phase, "value": v_true, "shown_value": v, "z": z,
               "shown": shown, "gate_ok": not gated, "gate_state": q.get("state"), "labels": q.get("labels", []),
               "reward": None if reward is None else {"t_s": reward.t_s, "z": reward.z, "dwell_s": reward.dwell_s},
               "dropped_before": int(block.dropped_before), "processing_ms": ms}
        if self.markers is not None:
            cues = self.markers.read()
            if cues:
                rec["cues"] = cues
                if self.log is not None:
                    for t_c, label in cues:
                        self.log.event("cue", t_c, label=label)
        if self.log is not None:
            self.log.value(block.t_end_s, v_true, z, shown, not gated, phase, shown_value=v, gate_state=str(q.get("state")))
            if gated:
                self.log.event("gate", block.t_end_s, state=q.get("state"), labels=q.get("labels", []))
            if reward is not None:
                self.log.event("reward", reward.t_s, z=reward.z, dwell_s=reward.dwell_s)
            if block.dropped_before:
                self.log.event("gap", block.t_start_s, n_missing=int(block.dropped_before))
        if self.presenter is not None:
            self.presenter.update(shown, z=z, gated=gated, t_s=block.t_end_s, phase=phase)
        self._k += 1
        return rec

    def fix_baseline(self) -> FixedBaseline:
        """Fix the baseline from the values collected during rest/calibrate phases."""
        factory = self.baseline_factory or (lambda vals: FixedBaseline.from_values(vals))
        self.baseline = factory(self._calibration)
        if self.log is not None:
            self.log.event("baseline", self.stats.duration_s, **self.baseline.describe())
        return self.baseline

    # -- a whole session --------------------------------------------------------------------------
    def run(self, phases: Sequence[tuple[str, float]] | None = None, *, max_s: float | None = None) -> LoopStats:
        """Run through the source's blocks, walking the phases by the block clock."""
        phases = list(phases) if phases else [("train", float("inf"))]
        bounds, t = [], 0.0
        for kind, dur in phases:
            t += float(dur)
            bounds.append((kind, t))
        if self.presenter is not None:
            self.presenter.start(self.info)
        idx, fixed = 0, self.baseline is not None
        if self.log is not None:
            self.log.event("phase", 0.0, phase=bounds[0][0], index=0)
        try:
            for block in blocks(self.source, self.block_samples):
                while idx < len(bounds) - 1 and block.t_end_s > bounds[idx][1]:
                    idx += 1
                    if self.log is not None:
                        self.log.event("phase", block.t_start_s, phase=bounds[idx][0], index=idx)
                kind = bounds[idx][0]
                if kind in ("train", "test") and not fixed and self._calibration:
                    self.fix_baseline()
                    fixed = True
                self.step(block, phase=kind)
                if block.t_end_s >= bounds[-1][1] or (max_s is not None and block.t_end_s >= max_s):
                    break
        finally:
            if self.presenter is not None:
                self.presenter.stop()
        return self.stats

    def budget(self) -> dict[str, Any]:
        """The chain's declared budget, with the processing row measured when the loop has run."""
        b = latency_budget_for(self.chain, fs=self.info.fs, block_samples=self.block_samples,
                               processing_ms=self.processing_ms_stated, convention=self.convention)
        if self.stats.processing_ms:
            p = self.stats.describe()["processing_ms"]
            b["processing_measured_ms"] = p
            b["rows"][2]["detail"] = f"measured on this machine: p50 {p['p50']:.2f} ms, p95 {p['p95']:.2f} ms (stated {self.processing_ms_stated:g})"
        return b

    def run_offline(self, x: np.ndarray) -> dict[str, Any]:
        """The loop over an array, ``helpers_l7.alpha_feedback``-compatible (values, times_s, held, filtered)."""
        chain = build_chain(self.spec, self.info, self.block_samples, quality=None, include_smoother=False)
        return run_offline(chain, x, fs=self.info.fs, block_samples=self.block_samples, smooth_s=self.spec.smooth_s)
