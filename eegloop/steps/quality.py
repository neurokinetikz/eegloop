"""The quality gate: measure the raw block, decide, and say so -- never clean what you cannot inspect.

Lesson L7.3's online artifact policy ends in one rule: *gate rather than clean*. Offline you inspect
what a cleaner removed and re-run; online you get one pass, so the honest move is to detect a bad
stretch and suspend the feedback rather than feed the participant a cleaned estimate of something
that was not measurable. "A frozen display is honest; a feedback signal driven by a blink is not."

This step runs **first**, on the raw block, because every one of its criteria needs what a band-pass
removes: the mains line, the high-frequency muscle band, the DC level of a flat channel. The
criteria and thresholds are ported from ``data/scripts/detectors.py`` (2026-09-20) -- the detectors
that produced the site's shipped artifact annotations -- made causal:

* ``flat``   -- window SD below ``flat_sd_uv`` (0.5 µV), or a run of unchanging samples;
* ``pop``    -- either of the repository's two criteria, since it holds both: ``detectors.py``'s
                step between the two halves of the window on one channel above
                ``max(pop_k × robust SD, pop_min_step_uv)`` (6, 100 µV) that its neighbours do not share,
                or ``helpers.detect_artifacts``'s sample-to-sample jump above ``pop_jump_uv`` (100 µV) on
                one channel with every other channel below 30 % of it -- the latter fires in the block the
                jump lands in, the former once the window shows the level has stayed shifted. Both need a
                neighbour: on a one-channel stream locality is untestable and the gate makes no pop claim
                (a 280-µV eyes-closed alpha wave at 160 Hz clears the jump threshold on its own);
* ``blink``  -- on frontal channels, a same-sign 0.5-15 Hz deflection above
                ``max(blink_k × robust SD, blink_min_amp_uv)`` (4, 50 µV);
* ``emg``    -- 25-100 Hz RMS above ``max(emg_k × running median, emg_min_rms_uv)`` (3, 4 µV);
* ``line-noise`` -- Welch power at mains more than ``line_min_ratio`` (10) times the neighbourhood;
* ``gap``    -- the block arrived after a loss the source reported.

Its delay is a *decision* delay: a verdict needs a window, and the signal itself is not delayed at
all. That is why ``latency_samples`` is ``None`` here and the budget lists the window under
``decision`` rather than adding it to a total that means something else.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Iterable

import numpy as np

from ..block import Block
from ..ring import RingBuffer

__all__ = ["QualityGate", "QUALITY_LABELS", "robust_sd"]

QUALITY_LABELS: tuple[str, ...] = ("flat", "pop", "blink", "emg", "line-noise", "gap", "warming-up")
_FRONTAL_DEFAULT: tuple[str, ...] = ("Fp1", "Fp2", "Fpz", "AF7", "AF8", "AF3", "AF4", "AFz")


def robust_sd(x: np.ndarray) -> float:
    """MAD-based standard deviation, as ``data/scripts/detectors.py`` defines it."""
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    return float(np.median(np.abs(x - med)) * 1.4826 + 1e-12)


def _longest_run(mask: np.ndarray) -> int:
    if mask.size == 0 or not mask.any():
        return 0
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts, stops = np.where(d == 1)[0], np.where(d == -1)[0]
    return int((stops - starts).max())


class _CausalBand:
    """A causal Butterworth band-pass with state, one per channel set."""

    def __init__(self, fs: float, lo: float, hi: float, n_channels: int, order: int = 4) -> None:
        from scipy.signal import butter, sosfilt_zi

        hi = min(hi, 0.45 * fs)
        self.sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
        self.zi = np.repeat(sosfilt_zi(self.sos)[:, None, :], n_channels, axis=1) * 0.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        from scipy.signal import sosfilt

        y, self.zi = sosfilt(self.sos, x, axis=-1, zi=self.zi)
        return y

    def reset(self) -> None:
        self.zi[:] = 0.0


class QualityGate:
    kind = "quality"

    def __init__(self, fs: float, ch_names: Iterable[str], *, window_s: float = 1.0, hold_s: float = 0.5,
                 mains_hz: float | None = None, frontal: Iterable[str] = _FRONTAL_DEFAULT,
                 flat_sd_uv: float = 0.5, flat_run_s: float = 0.5,
                 pop_k: float = 6.0, pop_min_step_uv: float = 100.0, pop_neighbour_frac: float = 0.35,
                 pop_jump_uv: float = 100.0, pop_jump_neighbour_frac: float = 0.30,
                 blink_k: float = 4.0, blink_min_amp_uv: float = 50.0,
                 emg_k: float = 3.0, emg_min_rms_uv: float = 4.0, emg_band: tuple[float, float] = (25.0, 100.0),
                 emg_history: int = 60, line_min_ratio: float = 10.0,
                 min_good_channels: int = 1,
                 unusable_labels: Iterable[str] = ("flat", "pop", "blink", "emg", "gap"),
                 suspect_labels: Iterable[str] = ("line-noise", "warming-up"),
                 name: str = "quality") -> None:
        self.name = name
        self.fs = float(fs)
        self.ch_names = tuple(str(c) for c in ch_names)
        self.n_channels = len(self.ch_names)
        self.window_s, self.hold_s = float(window_s), float(hold_s)
        self.window = max(2, int(round(self.window_s * self.fs)))
        self.mains_hz = None if mains_hz is None else float(mains_hz)
        self.frontal_idx = [i for i, c in enumerate(self.ch_names) if c in set(frontal)]
        self.flat_sd_uv, self.flat_run = float(flat_sd_uv), int(round(flat_run_s * self.fs))
        self.pop_k, self.pop_min_step_uv, self.pop_neighbour_frac = float(pop_k), float(pop_min_step_uv), float(pop_neighbour_frac)
        self.pop_jump_uv, self.pop_jump_neighbour_frac = float(pop_jump_uv), float(pop_jump_neighbour_frac)
        self._last_sample: np.ndarray | None = None
        self.blink_k, self.blink_min_amp_uv = float(blink_k), float(blink_min_amp_uv)
        self.emg_k, self.emg_min_rms_uv = float(emg_k), float(emg_min_rms_uv)
        self.line_min_ratio = float(line_min_ratio)
        self.min_good_channels = int(min_good_channels)
        self.unusable_labels, self.suspect_labels = frozenset(unusable_labels), frozenset(suspect_labels)
        self._raw = RingBuffer(self.n_channels, self.window, fs=self.fs)
        self._emg_filter = _CausalBand(self.fs, emg_band[0], emg_band[1], self.n_channels) if self.fs > 2.5 * emg_band[0] else None
        self._emg_ring = RingBuffer(self.n_channels, self.window, fs=self.fs)
        self._emg_hist: list[deque] = [deque(maxlen=int(emg_history)) for _ in range(self.n_channels)]
        self._blink_filter = _CausalBand(self.fs, 0.5, 15.0, len(self.frontal_idx)) if self.frontal_idx else None
        self._blink_ring = RingBuffer(len(self.frontal_idx), self.window, fs=self.fs) if self.frontal_idx else None
        self._held_until_s: float = -1.0
        self._t_s: float = 0.0
        self.n_decisions = 0

    # -- the OnlineStep contract -----------------------------------------------------------------
    @property
    def latency_samples(self) -> None:
        return None

    @property
    def latency_note(self) -> str:
        return (f"decision delay: a {self.window_s:g}-s window (plus a {self.hold_s:g}-s hold after a violation); "
                f"the gate delays the decision, not the signal")

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "window_s": self.window_s, "hold_s": self.hold_s, "mains_hz": self.mains_hz,
                "frontal_channels": [self.ch_names[i] for i in self.frontal_idx],
                "blink_detection": bool(self.frontal_idx) or "no frontal channel in this montage",
                "pop_detection": self.n_channels > 1 or "one channel: a pop is a local event and locality is not testable without a neighbour",
                "thresholds": {"flat_sd_uv": self.flat_sd_uv, "pop_k": self.pop_k, "pop_min_step_uv": self.pop_min_step_uv,
                               "pop_jump_uv": self.pop_jump_uv,
                               "blink_k": self.blink_k, "blink_min_amp_uv": self.blink_min_amp_uv,
                               "emg_k": self.emg_k, "emg_min_rms_uv": self.emg_min_rms_uv,
                               "line_min_ratio": self.line_min_ratio},
                "criteria_from": ["data/scripts/detectors.py", "notebooks/_shared/helpers.py:detect_artifacts (pop jump)"],
                "min_good_channels": self.min_good_channels,
                "verdict": "ok unless a channel is unusable, fewer than min_good_channels are good-or-suspect, the hold is running, or the window is not yet full; suspect channels are fed back and labelled"}

    def reset(self) -> None:
        self._raw = RingBuffer(self.n_channels, self.window, fs=self.fs)
        self._emg_ring = RingBuffer(self.n_channels, self.window, fs=self.fs)
        if self._emg_filter is not None:
            self._emg_filter.reset()
        if self._blink_filter is not None:
            self._blink_filter.reset()
            self._blink_ring = RingBuffer(len(self.frontal_idx), self.window, fs=self.fs)
        for h in self._emg_hist:
            h.clear()
        self._held_until_s = -1.0
        self._last_sample = None

    def process(self, block: Block) -> Block:
        if block.n_channels != self.n_channels:
            raise ValueError(f"{self.name}: expected {self.n_channels} channels, got {block.n_channels}")
        self._t_s = block.t_end_s
        self._raw.push(block.data)
        emg_block = None
        if self._emg_filter is not None:
            emg_block = self._emg_filter(block.data)
            self._emg_ring.push(emg_block)
        blink_block = None
        if self._blink_filter is not None and self._blink_ring is not None:
            blink_block = self._blink_filter(block.data[self.frontal_idx])
            self._blink_ring.push(blink_block)

        labels: dict[int, set[str]] = {i: set() for i in range(self.n_channels)}
        metrics: dict[int, dict[str, float]] = {i: {} for i in range(self.n_channels)}
        ready = self._raw.filled >= self.window
        if block.dropped_before > 0:
            for i in range(self.n_channels):
                labels[i].add("gap")
        # a sample-to-sample jump is judged on the new samples, across the seam with the previous block
        prev = self._last_sample if (self._last_sample is not None and block.dropped_before == 0) else block.data[:, :1]
        jumps = np.abs(np.diff(np.concatenate([prev.reshape(-1, 1), block.data], axis=1), axis=1))
        for i in range(self.n_channels):
            j = int(np.argmax(jumps[i]))
            metrics[i]["max_jump_uv"] = float(jumps[i, j])
            if jumps[i, j] > self.pop_jump_uv:
                others = np.delete(jumps[:, j], i)
                # a pop is a local event; with no other channel its locality is untestable, so no claim
                if others.size and others.max() < self.pop_jump_neighbour_frac * jumps[i, j]:
                    labels[i].add("pop")
        self._last_sample = block.data[:, -1].copy()
        if not ready:
            for i in range(self.n_channels):
                labels[i].add("warming-up")
        else:
            self._judge(labels, metrics, emg_block, blink_block)
        self.n_decisions += 1

        per_channel: dict[str, dict[str, Any]] = {}
        n_good = n_suspect = n_unusable = 0
        for i, ch in enumerate(self.ch_names):
            ls = labels[i]
            status = "unusable" if ls & self.unusable_labels else ("suspect" if ls & self.suspect_labels else "good")
            n_good += status == "good"
            n_suspect += status == "suspect"
            n_unusable += status == "unusable"
            per_channel[ch] = {"status": status, "labels": sorted(ls), "metrics": metrics[i]}
        # a suspect channel is still fed back -- a visible mains line is the normal state of an
        # un-notched recording, and the band-pass removes it -- so the verdict is 'suspect', not closed;
        # the gate closes on an unusable channel, or on too few usable (good or suspect) ones
        violated = n_unusable > 0 or (n_good + n_suspect) < self.min_good_channels
        if violated:
            self._held_until_s = self._t_s + self.hold_s
        held = self._t_s < self._held_until_s
        ok = not violated and not held and ready
        all_labels = sorted(set().union(*labels.values()))
        state = "unusable" if (n_unusable > 0 or held) else ("suspect" if (not ok or n_suspect) else "good")
        return block.with_flag("quality", {
            "ok": bool(ok), "state": state, "labels": all_labels, "per_channel": per_channel,
            "n_good": n_good, "n_suspect": n_suspect, "ready": ready, "held": bool(held and not violated),
            "held_until_s": self._held_until_s if held else None, "window_s": self.window_s,
        })

    # -- the criteria ----------------------------------------------------------------------------
    def _judge(self, labels: dict[int, set[str]], metrics: dict[int, dict[str, float]],
               emg_block: np.ndarray | None, blink_block: np.ndarray | None) -> None:
        win = self._raw.latest(self.window)
        half = self.window // 2
        sd = win.std(axis=1)
        rsd = np.array([robust_sd(w) for w in win])
        step = win[:, half:].mean(axis=1) - win[:, :half].mean(axis=1)
        # flat
        for i in range(self.n_channels):
            metrics[i]["sd_uv"] = float(sd[i])
            if sd[i] < self.flat_sd_uv or _longest_run(np.abs(np.diff(win[i])) < 1e-3) >= self.flat_run:
                labels[i].add("flat")
        # pop: a step on one channel that the others do not share
        for i in range(self.n_channels):
            thr = max(self.pop_k * rsd[i], self.pop_min_step_uv)
            metrics[i]["step_uv"] = float(step[i])
            if abs(step[i]) > thr:
                others = np.delete(np.abs(step), i)
                if others.size and others.max() < self.pop_neighbour_frac * abs(step[i]):
                    labels[i].add("pop")
        # line noise: Welch ratio at mains against the neighbourhood 2-8 Hz away
        if self.mains_hz is not None and self.mains_hz < 0.95 * self.fs / 2:
            from scipy.signal import welch

            fr, p = welch(win, fs=self.fs, nperseg=min(self.window, int(self.fs)), axis=-1)
            k = int(np.argmin(np.abs(fr - self.mains_hz)))
            nb = ((fr > self.mains_hz - 8) & (fr < self.mains_hz - 2)) | ((fr > self.mains_hz + 2) & (fr < self.mains_hz + 8))
            if nb.any():
                ratio = p[:, k] / (np.median(p[:, nb], axis=-1) + 1e-18)
                for i in range(self.n_channels):
                    metrics[i]["line_ratio"] = float(ratio[i])
                    if ratio[i] > self.line_min_ratio:
                        labels[i].add("line-noise")
        # emg: high-frequency RMS against a running median of past windows
        if self._emg_filter is not None and self._emg_ring.filled >= self.window:
            hf = self._emg_ring.latest(self.window)
            rms = np.sqrt(np.mean(hf ** 2, axis=1))
            for i in range(self.n_channels):
                metrics[i]["hf_rms_uv"] = float(rms[i])
                hist = self._emg_hist[i]
                if len(hist) >= 5:
                    thr = max(self.emg_k * float(np.median(hist)), self.emg_min_rms_uv)
                    if rms[i] > thr:
                        labels[i].add("emg")
                        continue  # a burst does not enter its own baseline
                hist.append(float(rms[i]))
        # blink: on the frontal channels only, judged on the NEW samples against the window's spread
        if self._blink_filter is not None and self._blink_ring is not None and blink_block is not None \
                and self._blink_ring.filled >= self.window:
            m_win = self._blink_ring.latest(self.window).mean(axis=0)
            thr = max(self.blink_k * robust_sd(m_win), self.blink_min_amp_uv)
            m_new = blink_block.mean(axis=0)
            j = int(np.argmax(np.abs(m_new)))
            same_sign = len(self.frontal_idx) < 2 or np.all(np.sign(blink_block[:, j]) == np.sign(m_new[j]))
            for i in self.frontal_idx:
                metrics[i]["blink_peak_uv"] = float(m_new[j])
                metrics[i]["blink_threshold_uv"] = float(thr)
            if abs(m_new[j]) > thr and same_sign:
                for i in self.frontal_idx:
                    labels[i].add("blink")
