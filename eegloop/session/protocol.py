"""A protocol: every decision a feedback session makes, named in one file, validated all at once.

Lesson L7.3 lists five decisions a reader of a neurofeedback protocol will want stated -- what the
signal is, how it maps to the display, what the baseline is and whether it moves, what the
reinforcement schedule is, what the participant experiences -- and adds the controls a claim needs:
the sham condition, blinding, a pre-specified outcome. A :class:`Protocol` is those decisions as
data, so the session log can record them and a report can quote them.

The validation machinery -- ``_coerce``, ``_build``, ``_validate_tree``, ``_one_of``, ``_positive``,
``_fraction``, ``_type_name``, ``_got``, ``_Invalid`` and ``_apply_overrides`` -- is copied from
pipelines/eegpipe/config.py @ 2026-09-20 rather than imported, because importing eegpipe would pull
MNE into a package built to run without it. The copy is byte-for-byte except that ``_validate_tree``
also descends into lists of blocks (a protocol has a list of phases; a pipeline config has none),
``_coerce`` builds an optional block (``BCIConfig | None``) directly so its problems keep their paths, and
``ProtocolError`` is the exception's name. ``tests/test_protocol.py`` holds the message format to
``pipelines/tests/test_config.py``'s assertions, so the two CLIs read the same to a learner.
"""
from __future__ import annotations

import difflib
import json
import types
import typing
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..bci.pipelines import PIPELINES
from ..latency import BUFFER_CONVENTIONS
from ..feedback.sham import SHAM_MODES
from ..sources.base import SOURCE_KINDS
from ..sources.synthetic import SCENARIOS

__all__ = [
    "ProtocolError", "Protocol", "SourceConfig", "SignalConfig", "QualityConfig", "BaselineConfig",
    "RewardConfig", "ShamConfig", "BCIConfig", "PhaseConfig", "OutputConfig", "VersionsConfig",
    "load_protocol", "validate_protocol", "resolved",
]

PHASE_KINDS: tuple[str, ...] = ("rest", "calibrate", "train", "test")


class ProtocolError(ValueError):
    """Raised when a protocol is not usable. Lists every problem found."""

    def __init__(self, problems: Sequence[str], source: str = "<protocol>") -> None:
        self.problems = list(problems)
        self.source = source
        n = len(self.problems)
        head = f"{source}: {n} problem{'s' if n != 1 else ''}"
        super().__init__("\n".join([head, *(f"  {p}" for p in self.problems)]))


# --------------------------------------------------------------------------- #
# Validation machinery -- copied from pipelines/eegpipe/config.py @ 2026-09-20
# --------------------------------------------------------------------------- #
def _type_name(hint: Any) -> str:
    if hint is Any:
        return "anything"
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        parts = [_type_name(a) for a in typing.get_args(hint) if a is not type(None)]
        optional = type(None) in typing.get_args(hint)
        return " or ".join(parts) + (" or null" if optional else "")
    if origin in (list, tuple):
        args = typing.get_args(hint)
        return f"a list of {_type_name(args[0])}" if args else "a list"
    if origin is dict:
        args = typing.get_args(hint)
        return f"a mapping of {_type_name(args[1])}" if len(args) == 2 else "a mapping"
    return {int: "an integer", float: "a number", str: "a string", bool: "true or false",
            type(None): "null"}.get(hint, getattr(hint, "__name__", str(hint)))


def _got(value: Any) -> str:
    return f"got {value!r} ({type(value).__name__})"


class _Invalid:
    def __repr__(self) -> str:  # pragma: no cover - debugging only
        return "<invalid>"


INVALID = _Invalid()


def _coerce(value: Any, hint: Any, path: str, problems: list[str]) -> Any:
    if hint is Any:
        return value
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        args = typing.get_args(hint)
        if value is None and type(None) in args:
            return None
        concrete = [a for a in args if a is not type(None)]
        if len(concrete) == 1 and is_dataclass(concrete[0]):  # an optional block: its problems keep their paths
            return _build(concrete[0], value, path, problems)
        for arg in args:
            if arg is type(None):
                continue
            probe: list[str] = []
            out = _coerce(value, arg, path, probe)
            if not probe:
                return out
        problems.append(f"{path}: expected {_type_name(hint)}, {_got(value)}")
        return INVALID
    if hint is bool:
        if isinstance(value, bool):
            return value
        problems.append(f"{path}: expected true or false, {_got(value)}")
        return INVALID
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            problems.append(f"{path}: expected an integer, {_got(value)}")
            return INVALID
        return int(value)
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"{path}: expected a number, {_got(value)}")
            return INVALID
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            problems.append(f"{path}: expected a string, {_got(value)}")
            return INVALID
        return value
    if origin in (list, tuple):
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            problems.append(f"{path}: expected {_type_name(hint)}, {_got(value)}")
            return INVALID
        args = typing.get_args(hint)
        item_hint = args[0] if args else Any
        items = [_coerce(v, item_hint, f"{path}[{i}]", problems) for i, v in enumerate(value)]
        return [v for v in items if v is not INVALID]
    if origin is dict:
        if not isinstance(value, Mapping):
            problems.append(f"{path}: expected {_type_name(hint)}, {_got(value)}")
            return INVALID
        args = typing.get_args(hint)
        val_hint = args[1] if len(args) == 2 else Any
        coerced = {str(k): _coerce(v, val_hint, f"{path}.{k}", problems) for k, v in value.items()}
        return {k: v for k, v in coerced.items() if v is not INVALID}
    if is_dataclass(hint):
        return _build(hint, value, path, problems)
    return value


def _build(cls: type, mapping: Any, path: str, problems: list[str]) -> Any:
    if mapping is None:
        mapping = {}
    if not isinstance(mapping, Mapping):
        problems.append(f"{path or cls.__name__}: expected a mapping of settings, {_got(mapping)}")
        return cls()
    normalise = getattr(cls, "_normalise", None)
    if normalise is not None:
        mapping = normalise(dict(mapping))
    hints = typing.get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    prefix = f"{path}." if path else ""
    for key in mapping:
        if key not in known:
            close = difflib.get_close_matches(str(key), sorted(known), n=1, cutoff=0.6)
            hint = f" (did you mean {close[0]!r}?)" if close else ""
            problems.append(f"{prefix}{key}: unknown setting{hint}; this block takes {', '.join(sorted(known))}")
    for name in getattr(cls, "_REQUIRED", ()):
        if name not in mapping or mapping[name] is None:
            why = getattr(cls, "_REQUIRED_WHY", {}).get(name, "")
            problems.append(f"{prefix}{name}: required{' -- ' + why if why else ''}")
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in mapping:
            continue
        value = _coerce(mapping[f.name], hints[f.name], f"{prefix}{f.name}", problems)
        if value is not INVALID:
            kwargs[f.name] = value
    try:
        return cls(**kwargs)
    except TypeError as exc:  # pragma: no cover
        problems.append(f"{path or cls.__name__}: {exc}")
        return cls()


def _validate_tree(obj: Any, path: str, problems: list[str]) -> None:
    check = getattr(obj, "_check", None)
    if check is not None:
        check(path, problems)
    for f in fields(obj):
        value = getattr(obj, f.name)
        child = f"{path}.{f.name}" if path else f.name
        if is_dataclass(value) and not isinstance(value, type):
            _validate_tree(value, child, problems)
        elif isinstance(value, list):  # the one addition: a protocol holds a list of phase blocks
            for i, item in enumerate(value):
                if is_dataclass(item) and not isinstance(item, type):
                    _validate_tree(item, f"{child}[{i}]", problems)


def _one_of(value: Any, options: Sequence[str], path: str, problems: list[str]) -> None:
    if value in options:
        return
    close = difflib.get_close_matches(str(value), [str(o) for o in options], n=1, cutoff=0.5)
    hint = f" (did you mean {close[0]!r}?)" if close else ""
    problems.append(f"{path}: {value!r} is not valid; expected one of {', '.join(str(o) for o in options)}{hint}")


def _positive(value: Any, path: str, problems: list[str], *, allow_zero: bool = False) -> None:
    if value is None or not isinstance(value, (int, float)) or isinstance(value, bool):
        return
    if value < 0 or (value == 0 and not allow_zero):
        problems.append(f"{path}: expected a positive number, got {value}")


def _fraction(value: Any, path: str, problems: list[str]) -> None:
    if value is None or not isinstance(value, (int, float)) or isinstance(value, bool):
        return
    if not 0.0 <= value <= 1.0:
        problems.append(f"{path}: expected a fraction between 0 and 1, got {value}")


def _apply_overrides(mapping: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(mapping)
    for dotted, value in overrides.items():
        parts = str(dotted).split(".")
        node: dict[str, Any] = out
        for part in parts[:-1]:
            child = node.get(part)
            node[part] = dict(child) if isinstance(child, Mapping) else {}
            node = node[part]
        node[parts[-1]] = value
    return out


def _band(value: Any, path: str, problems: list[str]) -> None:
    if not isinstance(value, list) or len(value) != 2:
        problems.append(f"{path}: expected [low_hz, high_hz]")
        return
    lo, hi = value
    if not (isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and 0 < lo < hi):
        problems.append(f"{path}: expected 0 < low < high in Hz, got {value}")


# --------------------------------------------------------------------------- #
# The blocks
# --------------------------------------------------------------------------- #
@dataclass
class SourceConfig:
    """Where the blocks come from. ``synthetic`` and ``replay`` run anywhere; the hardware kinds
    need a headset and arrive in a later phase."""

    kind: str = "synthetic"
    scenario: str = "alpha-schedule"       # synthetic only
    path: str | None = None                # replay only, relative to the protocol file
    board: str | None = None               # hardware only
    pace: str = "fast"                     # 'wall' releases samples on the clock; 'fast' does not wait
    loop: bool = False
    seed: int | None = None                # synthetic only; null means the protocol's seed
    duration_s: float | None = None        # synthetic only; null means the phases' total

    def _check(self, path: str, problems: list[str]) -> None:
        _one_of(self.kind, SOURCE_KINDS, f"{path}.kind", problems)
        _one_of(self.pace, ("wall", "fast"), f"{path}.pace", problems)
        if self.kind == "synthetic":
            _one_of(self.scenario, tuple(SCENARIOS), f"{path}.scenario", problems)
        if self.kind == "replay" and not self.path:
            problems.append(f"{path}.path: required -- a replay source needs a recording to play")
        if self.kind in ("brainflow", "lsl") and not self.board and self.kind == "brainflow":
            problems.append(f"{path}.board: required -- a hardware source needs a board key")


@dataclass
class SignalConfig:
    """Decision 1 of L7.3: what the signal is. Exact computation, window, channels -- '"alpha at
    Pz" is not a definition'."""

    band: list[float] = field(default_factory=lambda: [8.0, 12.0])
    channels: list[str] = field(default_factory=list)          # empty: every channel of the source
    reference: str = "none"                                    # none | average | channels
    reference_channels: list[str] = field(default_factory=list)
    n_taps: int = 129
    derivation: str = "rms"                                    # rms | log-power
    control_band: list[float] = field(default_factory=lambda: [16.0, 20.0])  # the sham-band band
    smooth_s: float = 0.0                                      # decision 2's smoothing: it is latency

    def _check(self, path: str, problems: list[str]) -> None:
        _band(self.band, f"{path}.band", problems)
        _band(self.control_band, f"{path}.control_band", problems)
        _one_of(self.reference, ("none", "average", "channels"), f"{path}.reference", problems)
        if self.reference == "channels" and not self.reference_channels:
            problems.append(f"{path}.reference_channels: required when reference is 'channels'")
        _one_of(self.derivation, ("rms", "log-power"), f"{path}.derivation", problems)
        if isinstance(self.n_taps, int) and (self.n_taps < 3 or self.n_taps % 2 == 0):
            problems.append(f"{path}.n_taps: expected an odd tap count of at least 3 (linear phase), got {self.n_taps}")
        _positive(self.smooth_s, f"{path}.smooth_s", problems, allow_zero=True)


@dataclass
class QualityConfig:
    """The online artifact policy: gate rather than clean (L7.3)."""

    enabled: bool = True
    window_s: float = 1.0
    hold_s: float = 0.5
    mains_hz: float | None = None
    policy: str = "freeze"                                     # freeze the display, or show zero

    def _check(self, path: str, problems: list[str]) -> None:
        _positive(self.window_s, f"{path}.window_s", problems)
        _positive(self.hold_s, f"{path}.hold_s", problems, allow_zero=True)
        _one_of(self.policy, ("freeze", "zero"), f"{path}.policy", problems)


@dataclass
class BaselineConfig:
    """Decision 3: what the baseline is, and whether it moves."""

    mode: str = "fixed"                                        # fixed | adaptive
    centre: str = "median"                                     # median | mean
    spread: str = "mad"                                        # mad | sd
    window_s: float = 30.0                                     # adaptive only
    percentile: float = 50.0                                   # adaptive only

    def _check(self, path: str, problems: list[str]) -> None:
        _one_of(self.mode, ("fixed", "adaptive"), f"{path}.mode", problems)
        _one_of(self.centre, ("median", "mean"), f"{path}.centre", problems)
        _one_of(self.spread, ("mad", "sd"), f"{path}.spread", problems)
        _positive(self.window_s, f"{path}.window_s", problems)
        if isinstance(self.percentile, (int, float)) and not 0 < self.percentile < 100:
            problems.append(f"{path}.percentile: expected a percentile strictly between 0 and 100, got {self.percentile}")


@dataclass
class RewardConfig:
    """Decisions 2 and 4: the mapping to the display, and the reinforcement schedule."""

    mode: str = "continuous"                                   # continuous | threshold
    lo_z: float = -1.0
    hi_z: float = 2.0
    threshold_z: float = 1.0
    dwell_s: float = 0.5
    refractory_s: float = 1.0
    on_miss: str = "hold"                                      # hold | zero

    def _check(self, path: str, problems: list[str]) -> None:
        _one_of(self.mode, ("continuous", "threshold"), f"{path}.mode", problems)
        _one_of(self.on_miss, ("hold", "zero"), f"{path}.on_miss", problems)
        if isinstance(self.lo_z, (int, float)) and isinstance(self.hi_z, (int, float)) and not self.lo_z < self.hi_z:
            problems.append(f"{path}.lo_z/hi_z: expected lo_z < hi_z, got {self.lo_z} and {self.hi_z}")
        _positive(self.dwell_s, f"{path}.dwell_s", problems, allow_zero=True)
        _positive(self.refractory_s, f"{path}.refractory_s", problems, allow_zero=True)


@dataclass
class ShamConfig:
    """The control condition. ``sham-yoked`` needs a donor session to replay."""

    mode: str = "veridical"
    donor: str | None = None                                   # a session directory, relative to the protocol file

    def _check(self, path: str, problems: list[str]) -> None:
        _one_of(self.mode, tuple(SHAM_MODES), f"{path}.mode", problems)
        if self.mode == "sham-yoked" and not self.donor:
            problems.append(f"{path}.donor: required -- a yoked sham replays another session's feedback on the same schedule")


@dataclass
class PhaseConfig:
    kind: str = "train"                                        # rest | calibrate | train | test
    duration_s: float = 60.0
    label: str = ""

    def _check(self, path: str, problems: list[str]) -> None:
        _one_of(self.kind, PHASE_KINDS, f"{path}.kind", problems)
        _positive(self.duration_s, f"{path}.duration_s", problems)


@dataclass
class BCIConfig:
    """Calibrate once, freeze, apply. Present in a protocol that runs a decoder instead of feedback."""

    pipeline: str = "csp_lda"                                  # csp_lda | riemann_ts | xdawn_lda
    classes: list[str] = field(default_factory=lambda: ["left", "right"])
    tmin_s: float = 0.5                                        # window after the cue, seconds
    tmax_s: float = 3.5
    band: list[float] = field(default_factory=lambda: [8.0, 30.0])
    n_taps: int = 129
    n_components: int = 4                                      # CSP filters, or Xdawn filters per class
    mode: str = "sliding"                                      # sliding | cue-locked
    step_samples: int = 32                                     # sliding: posterior every this many samples
    smooth_s: float = 0.5                                      # sliding: posterior smoothing (measured, not summed)
    threshold: float = 0.7                                     # sliding: dwell decision
    dwell_s: float = 0.5
    refractory_s: float = 1.0
    folds: int = 5
    grace_s: float = 1.0                                       # how long after tmax a decision still counts for a cue

    def _check(self, path: str, problems: list[str]) -> None:
        _one_of(self.pipeline, PIPELINES, f"{path}.pipeline", problems)
        _one_of(self.mode, ("sliding", "cue-locked"), f"{path}.mode", problems)
        if isinstance(self.classes, list) and len(set(self.classes)) < 2:
            problems.append(f"{path}.classes: expected at least two distinct class labels, got {self.classes}")
        if isinstance(self.tmin_s, (int, float)) and isinstance(self.tmax_s, (int, float)) and not self.tmin_s < self.tmax_s:
            problems.append(f"{path}.tmin_s/tmax_s: expected tmin_s < tmax_s, got {self.tmin_s} and {self.tmax_s}")
        _band(self.band, f"{path}.band", problems)
        if isinstance(self.n_taps, int) and (self.n_taps < 3 or self.n_taps % 2 == 0):
            problems.append(f"{path}.n_taps: expected an odd tap count of at least 3 (linear phase), got {self.n_taps}")
        _positive(self.n_components, f"{path}.n_components", problems)
        _positive(self.step_samples, f"{path}.step_samples", problems)
        _positive(self.smooth_s, f"{path}.smooth_s", problems, allow_zero=True)
        _fraction(self.threshold, f"{path}.threshold", problems)
        _positive(self.dwell_s, f"{path}.dwell_s", problems, allow_zero=True)
        _positive(self.refractory_s, f"{path}.refractory_s", problems, allow_zero=True)
        _positive(self.grace_s, f"{path}.grace_s", problems, allow_zero=True)
        if isinstance(self.folds, int) and self.folds < 2:
            problems.append(f"{path}.folds: expected at least 2, got {self.folds}")


@dataclass
class OutputConfig:
    dir: str = "sessions"
    record_raw: bool = False


@dataclass
class VersionsConfig:
    on_mismatch: str = "warn"                                  # warn | error | ignore
    pins: dict[str, str] = field(default_factory=dict)

    def _check(self, path: str, problems: list[str]) -> None:
        _one_of(self.on_mismatch, ("warn", "error", "ignore"), f"{path}.on_mismatch", problems)


@dataclass
class Protocol:
    name: str = ""
    seed: int = 0
    block_samples: int = 32
    processing_ms: float = 10.0                                # stated; the run measures the real one
    buffer_convention: str = "block-period"
    source: SourceConfig = field(default_factory=SourceConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    sham: ShamConfig = field(default_factory=ShamConfig)
    bci: BCIConfig | None = None                               # present: calibrate-then-apply instead of feedback
    phases: list[PhaseConfig] = field(default_factory=list)
    output: OutputConfig = field(default_factory=OutputConfig)
    versions: VersionsConfig = field(default_factory=VersionsConfig)

    _REQUIRED = ("name", "seed")
    _REQUIRED_WHY = {
        "seed": "every stochastic step and the sealed sham assignment derive from it; a session without one cannot be reproduced",
        "name": "the session directory and the log are named after it",
    }

    def _check(self, path: str, problems: list[str]) -> None:
        _positive(self.block_samples, "block_samples", problems)
        _positive(self.processing_ms, "processing_ms", problems, allow_zero=True)
        _one_of(self.buffer_convention, tuple(BUFFER_CONVENTIONS), "buffer_convention", problems)
        if not self.phases:
            problems.append("phases: required -- a protocol needs at least one phase")
            return
        kinds = [p.kind for p in self.phases]
        first_apply = next((i for i, k in enumerate(kinds) if k in ("train", "test")), None)
        if self.bci is not None:
            if first_apply is not None and "calibrate" not in kinds[:first_apply]:
                problems.append("phases: a bci protocol needs a calibrate phase (cued trials to fit from) before the "
                                "first train/test phase")
            return
        if self.baseline.mode == "fixed" and first_apply is not None \
                and not any(k in ("rest", "calibrate") for k in kinds[:first_apply]):
            problems.append("phases: baseline.mode is 'fixed' but no rest or calibrate phase precedes the first "
                            "train/test phase; a fixed baseline needs values to fix from")

    @property
    def total_duration_s(self) -> float:
        return float(sum(p.duration_s for p in self.phases))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def validate_protocol(mapping: Mapping[str, Any], source: str = "<dict>",
                      overrides: Mapping[str, Any] | None = None) -> Protocol:
    """Build a :class:`Protocol` from a mapping, or raise :class:`ProtocolError` listing every problem."""
    if not isinstance(mapping, Mapping):
        raise ProtocolError([f"expected a mapping of settings at the top level, {_got(mapping)}"], source)
    raw = _apply_overrides(dict(mapping), overrides or {})
    problems: list[str] = []
    protocol = _build(Protocol, raw, "", problems)
    _validate_tree(protocol, "", problems)
    if problems:
        raise ProtocolError(problems, source)
    return protocol


def load_protocol(path: str | Path, overrides: Mapping[str, Any] | None = None) -> Protocol:
    """Read a YAML (or JSON) protocol and validate it. YAML needs PyYAML; JSON needs nothing."""
    p = Path(path).expanduser()
    if not p.exists():
        raise ProtocolError([f"no such protocol file: {p.name}"], str(path))
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProtocolError([f"not valid JSON: {exc}"], str(p)) from None
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - exercised only without PyYAML
            raise ProtocolError(["reading a YAML protocol needs PyYAML: pip install pyyaml (or write the protocol as JSON)"], str(p)) from exc
        try:
            loaded = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ProtocolError([f"not valid YAML: {exc}"], str(p)) from None
    if loaded is None:
        raise ProtocolError(["the file is empty"], str(p))
    protocol = validate_protocol(loaded, source=str(p), overrides=overrides)
    protocol.__dict__["_path"] = p.resolve()  # for resolving replay paths and donors; never logged
    return protocol


def resolved(protocol: Protocol) -> dict[str, Any]:
    """The protocol as plain data, defaults filled in -- what the session log stores."""
    return asdict(protocol)
