"""The control conditions a neurofeedback claim needs, and what each one does and does not rule out.

``SHAM_MODES`` and ``CREDNF_ITEMS`` are copied verbatim from ``notebooks/_shared/helpers_l7.py``
(2026-09-20). The control condition is part of the library, not an afterthought: a loop without a
sham mode cannot support the claim its participant is being asked to believe.

One divergence, recorded rather than reconciled. ``helpers_l7.sham_feedback`` inverts the feedback
about the *whole-session mean* -- which needs the future, and so is only possible offline. Here
``sham-inverted`` reflects about the *calibration baseline*, the centre that exists at the moment
the value is shown. The two are the same idea with different anchors; the notebook that compares
them prints both and changes neither.
"""
from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .signal import SignalSpec

__all__ = ["SHAM_MODES", "CREDNF_ITEMS", "ShamPolicy"]

#: The control conditions a neurofeedback claim needs, and what each one does and does not rule out.
SHAM_MODES: dict[str, str] = {
    "veridical": "Feedback computed from this participant's own band-limited signal, in real time.  "
                 "The experimental condition.",
    "sham-yoked": "Feedback replayed from a DIFFERENT recording, played back on the same schedule.  "
                  "It looks and moves like feedback and carries no information about this "
                  "participant's brain, so it controls for the display, the task, the time on task "
                  "and the expectation, and it is the control CRED-nf asks for by name.",
    "sham-band": "Feedback computed from the same participant, in real time, from a control band "
                 "the protocol does not target.  It controls for the participant's own arousal and "
                 "signal quality, which the yoked sham does not, and it does NOT control for the "
                 "possibility that the control band moves with the target band.",
    "sham-inverted": "Feedback computed from the target band and then inverted.  It controls for "
                     "everything the veridical condition does except the direction of the "
                     "contingency, and a participant who learns to lower the number is doing the "
                     "same thing in reverse, which makes it hard to interpret.",
}

CREDNF_ITEMS: tuple[str, ...] = (
    "Pre-registration of the primary outcome and the control condition, before data collection.",
    "A control group or condition (sham feedback, an active alternative task, or both).",
    "Blinding: the participant, and where possible the experimenter and the analyst.",
    "The feedback signal reported in full — the band, the derivation, the reference, the online "
    "filter, the update rate and the end-to-end latency.",
    "Evidence that the participants learned to change the targeted signal, reported separately "
    "from whether the clinical or behavioural outcome changed.",
    "The trial-level and session-level data made available so the analysis can be repeated.",
)

_INVERTED_NOTE = ("reflected about the calibration baseline, the centre that exists when the value is shown; "
                  "helpers_l7.sham_feedback reflects about the whole-session mean, which needs the future -- "
                  "a recorded divergence, not a reconciled one")


class ShamPolicy:
    """Run the loop in one of :data:`SHAM_MODES`; seal the mode so the log can be blinded."""

    def __init__(self, mode: str, *, donor: np.ndarray | None = None, control_band: tuple[float, float] = (16.0, 20.0),
                 seed: int = 0) -> None:
        if mode not in SHAM_MODES:
            raise ValueError(f"mode must be one of {sorted(SHAM_MODES)}")
        self.mode = mode
        self.control_band = (float(control_band[0]), float(control_band[1]))
        self.seed = int(seed)
        self.donor = None if donor is None else np.asarray(donor, dtype=float).reshape(-1)
        if mode == "sham-yoked" and (self.donor is None or self.donor.size == 0):
            raise ValueError("sham-yoked needs a donor: another session's feedback values, replayed on the same schedule")

    @property
    def note(self) -> str:
        return SHAM_MODES[self.mode] + (f"  Here: {_INVERTED_NOTE}." if self.mode == "sham-inverted" else "")

    def adapt_spec(self, spec: "SignalSpec") -> "SignalSpec":
        """``sham-band`` computes the same feedback from the control band; the others use the spec as given."""
        if self.mode == "sham-band":
            return spec.replace(band=self.control_band)
        return spec

    def value(self, v: float, k: int, centre: float | None) -> float:
        """The value the participant is shown, given the veridical value ``v`` for block ``k``."""
        if self.mode == "sham-yoked":
            assert self.donor is not None
            return float(self.donor[k % self.donor.size])
        if self.mode == "sham-inverted":
            return float(v) if centre is None else float(2.0 * centre - v)
        return float(v)

    def seal(self) -> str:
        """An opaque token for the log: the mode is recoverable from it only with the seed."""
        return hashlib.sha256(f"{self.mode}:{self.seed}".encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def unblind(token: str, seed: int) -> str:
        """The mode behind a sealed token, given the seed -- the unblinding step of the report."""
        for mode in SHAM_MODES:
            if hashlib.sha256(f"{mode}:{int(seed)}".encode("utf-8")).hexdigest()[:16] == token:
                return mode
        raise ValueError("no sham mode matches this token with this seed")

    def describe(self) -> dict[str, Any]:
        return {"mode": self.mode, "control_band": list(self.control_band),
                "donor_n": None if self.donor is None else int(self.donor.size), "note": self.note}
