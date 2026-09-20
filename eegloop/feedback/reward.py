"""Decisions 2 and 4 of L7.3: the mapping to the display, and the reinforcement schedule.

The transfer function from signal to display -- scale, clipping, smoothing -- and the schedule --
continuous feedback, or discrete rewards at threshold crossings, and what happens on a miss. Both
take their time from the block's clock, never from ``time.time()``, so a replay and a device give
the same decisions for the same signal and a test is deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["ContinuousMapping", "ThresholdReward", "RewardEvent"]


@dataclass(frozen=True)
class RewardEvent:
    t_s: float
    z: float
    dwell_s: float


class ContinuousMapping:
    """``z`` in ``[lo_z, hi_z]`` maps linearly onto ``[0, 1]``; outside is clipped (or not)."""

    def __init__(self, lo_z: float = -1.0, hi_z: float = 2.0, *, clip: bool = True) -> None:
        if not lo_z < hi_z:
            raise ValueError("lo_z must be below hi_z")
        self.lo_z, self.hi_z, self.clip = float(lo_z), float(hi_z), bool(clip)

    def map(self, z: float) -> float:
        if z != z:  # nan: nothing to show yet
            return float("nan")
        m = (float(z) - self.lo_z) / (self.hi_z - self.lo_z)
        return min(1.0, max(0.0, m)) if self.clip else m

    def describe(self) -> dict[str, Any]:
        return {"mode": "continuous", "lo_z": self.lo_z, "hi_z": self.hi_z, "clip": self.clip}


class ThresholdReward:
    """A reward when ``z`` has stayed above ``threshold_z`` for ``dwell_s``, at most once per ``refractory_s``."""

    def __init__(self, threshold_z: float = 1.0, *, dwell_s: float = 0.5, refractory_s: float = 1.0,
                 on_miss: str = "hold") -> None:
        if on_miss not in ("hold", "zero"):
            raise ValueError("on_miss must be 'hold' or 'zero'")
        self.threshold_z, self.dwell_s, self.refractory_s, self.on_miss = float(threshold_z), float(dwell_s), float(refractory_s), on_miss
        self._above_since: float | None = None
        self._last_reward_t: float | None = None
        self._last_shown = 0.0
        self.n_rewards = 0

    def update(self, z: float, t_s: float) -> RewardEvent | None:
        if z == z and z >= self.threshold_z:
            self._above_since = t_s if self._above_since is None else self._above_since
            dwelt = t_s - self._above_since
            ready = self._last_reward_t is None or t_s - self._last_reward_t >= self.refractory_s
            if dwelt >= self.dwell_s and ready:
                self._last_reward_t = t_s
                self.n_rewards += 1
                self._last_shown = 1.0
                return RewardEvent(float(t_s), float(z), float(dwelt))
        else:
            self._above_since = None
            if self.on_miss == "zero":
                self._last_shown = 0.0
        return None

    def map(self, z: float) -> float:
        """What the display shows: 1 on a reward, otherwise the last value (hold) or 0 (zero)."""
        return self._last_shown

    def reset(self) -> None:
        self._above_since = self._last_reward_t = None
        self._last_shown, self.n_rewards = 0.0, 0

    def describe(self) -> dict[str, Any]:
        return {"mode": "threshold", "threshold_z": self.threshold_z, "dwell_s": self.dwell_s,
                "refractory_s": self.refractory_s, "on_miss": self.on_miss}
