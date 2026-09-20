"""Decision 1 of L7.3: what the signal is -- exactly -- and the chain that computes it.

"'Alpha at Pz' is not a definition." A :class:`SignalSpec` is: the band, the channels and how they
are combined, the reference, the filter and its length, the derivation, and the smoothing (which is
latency). :func:`build_chain` turns it into the canonical online chain, quality gate first.
:meth:`SignalSpec.describe` is the sentence CRED-nf item 4 asks a report to contain.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from ..block import StreamInfo
from ..steps import Chain
from ..steps.causal_filter import CausalFIR, fir_taps
from ..steps.envelope import BlockRMS, Smoother
from ..steps.quality import QualityGate
from ..steps.reference import Reference

if TYPE_CHECKING:  # pragma: no cover
    from ..session.protocol import QualityConfig, SignalConfig

__all__ = ["SignalSpec", "build_chain", "combine"]


@dataclass(frozen=True)
class SignalSpec:
    band: tuple[float, float] = (8.0, 12.0)
    channels: tuple[str, ...] = ()            # empty: every channel of the source
    reference: str = "none"
    reference_channels: tuple[str, ...] = ()
    n_taps: int = 129
    derivation: str = "rms"                   # rms | log-power
    control_band: tuple[float, float] = (16.0, 20.0)
    smooth_s: float = 0.0

    @classmethod
    def from_config(cls, cfg: "SignalConfig") -> "SignalSpec":
        return cls(band=(float(cfg.band[0]), float(cfg.band[1])), channels=tuple(cfg.channels),
                   reference=cfg.reference, reference_channels=tuple(cfg.reference_channels),
                   n_taps=int(cfg.n_taps), derivation=cfg.derivation,
                   control_band=(float(cfg.control_band[0]), float(cfg.control_band[1])),
                   smooth_s=float(cfg.smooth_s))

    def replace(self, **changes: Any) -> "SignalSpec":
        return replace(self, **changes)

    def picks(self, ch_names: tuple[str, ...]) -> list[int]:
        if not self.channels:
            return list(range(len(ch_names)))
        missing = [c for c in self.channels if c not in ch_names]
        if missing:
            raise ValueError(f"signal channels not in the stream: {', '.join(missing)} (stream has {', '.join(ch_names)})")
        return [ch_names.index(c) for c in self.channels]

    def describe(self, fs: float, block_samples: int) -> str:
        """The feedback signal reported in full (CRED-nf item 4), as one sentence."""
        chans = ", ".join(self.channels) if self.channels else "every channel"
        ref = {"none": "no re-referencing", "average": "an average reference",
               "channels": f"referenced to {', '.join(self.reference_channels)}"}[self.reference]
        how = {"rms": "the root-mean-square per block", "log-power": "the log mean power per block"}[self.derivation]
        smooth = (f", smoothed by a one-pole filter with a {self.smooth_s:g} s time constant"
                  if self.smooth_s > 0 else ", unsmoothed")
        return (f"{self.band[0]:g}–{self.band[1]:g} Hz at {chans} with {ref}, a causal linear-phase FIR of "
                f"{self.n_taps} taps (group delay {(self.n_taps - 1) / 2 / fs * 1000:.1f} ms), {how} averaged over "
                f"those channels, updated every {block_samples / fs * 1000:.0f} ms ({fs / block_samples:.1f} Hz){smooth}.")


def combine(envelope: np.ndarray, picks: list[int], derivation: str) -> float:
    """One feedback value from a per-channel envelope: the mean over the chosen channels."""
    e = np.asarray(envelope, dtype=float)[picks]
    if derivation == "log-power":
        return float(np.log(np.mean(e ** 2) + 1e-12))
    return float(np.mean(e))


def build_chain(spec: SignalSpec, info: StreamInfo, block_samples: int, *,
                quality: "QualityConfig | None" = None, include_smoother: bool = True, envelope: bool = True) -> Chain:
    """The canonical online chain for a signal spec: gate (raw) → reference → causal FIR → RMS → smoother.
    ``envelope=False`` stops after the filter: the band-limited signal a decoder cuts its windows from."""
    steps: list[Any] = []
    if quality is not None and quality.enabled:
        steps.append(QualityGate(info.fs, info.ch_names, window_s=quality.window_s, hold_s=quality.hold_s,
                                 mains_hz=quality.mains_hz))
    if spec.reference != "none":
        steps.append(Reference(spec.reference, info.ch_names, channels=spec.reference_channels))
    steps.append(CausalFIR(fir_taps(spec.n_taps, spec.band, info.fs), info.n_channels, name="fir"))
    if envelope:
        steps.append(BlockRMS())
        if include_smoother and spec.smooth_s > 0:
            steps.append(Smoother(spec.smooth_s, info.fs, block_samples))
    return Chain(steps, reset_on_gap_samples=spec.n_taps)
