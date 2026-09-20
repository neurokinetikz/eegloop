"""The session log: what was shown, when, under which sealed condition -- and a reader for it.

CRED-nf item 4 asks that the feedback signal be reported in full; item 6 that the trial- and
session-level data be available so the analysis can be repeated. A :class:`SessionLog` is those two
items as files: ``session.json`` (versions, the protocol's hash, the source's own facts, the budget,
the totals, the sealed sham token), ``events.jsonl`` (gate closures, rewards, cues, gaps, phase
changes, the baseline as fixed), ``signal.npz`` (the per-block series), ``protocol-resolved.json``
(every decision, defaults filled in), and optionally the raw stream. :class:`SessionReader` opens all
of it with numpy alone.

Nothing in a log names the machine it was made on. ``scrub_paths`` is copied from
pipelines/eegpipe/run.py @ 2026-09-20 and applied to **every** string in ``session.json``, not only to
tracebacks: a path in a stored output is the thing scripts/render-notebooks.py fails a build on.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from ..block import Block, StreamInfo
from ..versions import config_hash, installed_versions
from .record import Recorder

__all__ = ["SessionLog", "SessionReader", "scrub_paths", "scrub_value"]

_FILE_LINE = re.compile(r'(?m)^(\s*File ")([^"]+)(")')
_ABS_PATH = re.compile(r"(?<![\w/])(/(?:Users|home|Volumes|mnt|tmp|private|opt|var)/[^\s'\"]+)")


def scrub_paths(text: str) -> str:
    """Rewrite absolute file paths so a log travels. Copied from eegpipe.run.scrub_paths, then
    extended to plain paths outside tracebacks: only the last two segments survive."""
    root = Path(__file__).resolve().parent.parent.parent  # the loop/ directory

    def rewrite_frame(m: "re.Match[str]") -> str:
        path = Path(m.group(2))
        try:
            shown = path.resolve().relative_to(root).as_posix()
        except (ValueError, OSError):
            parts = path.parts[-2:]
            shown = ".../" + "/".join(parts) if len(path.parts) > 2 else path.as_posix()
        return f"{m.group(1)}{shown}{m.group(3)}"

    def rewrite_plain(m: "re.Match[str]") -> str:
        parts = Path(m.group(1)).parts[-2:]
        return ".../" + "/".join(parts)

    return _ABS_PATH.sub(rewrite_plain, _FILE_LINE.sub(rewrite_frame, text))


def scrub_value(value: Any) -> Any:
    """``scrub_paths`` over every string inside a nested structure; numpy made plain along the way."""
    if isinstance(value, str):
        return scrub_paths(value)
    if isinstance(value, dict):
        return {str(k): scrub_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [scrub_value(v) for v in value]
    if isinstance(value, Path):
        return scrub_paths(str(value))
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class SessionLog:
    def __init__(self, out_dir: str | Path, *, protocol: dict[str, Any] | None, info: StreamInfo,
                 record_raw: bool = False, sealed_sham: str | None = None) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.protocol = dict(protocol or {})
        self.info = info
        self.sealed_sham = sealed_sham
        self.started_at = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        self.events: list[dict[str, Any]] = []
        self._t: list[float] = []
        self._v: list[float] = []
        self._vs: list[float] = []
        self._z: list[float] = []
        self._shown: list[float] = []
        self._ok: list[bool] = []
        self._state: list[str] = []
        self._pt: list[float] = []
        self._pp: list[np.ndarray] = []
        self._pok: list[bool] = []
        self._pphase: list[str] = []
        self.decoder_file: str | None = None
        self._phase: list[str] = []
        self.recorder = Recorder(self.out_dir, info, name="raw") if record_raw else None
        self.closed = False

    def event(self, kind: str, t_s: float, **payload: Any) -> None:
        self.events.append({"kind": str(kind), "t_s": float(t_s), **scrub_value(payload)})

    def value(self, t_s: float, value: float, z: float, shown: float, gate_ok: bool, phase: str, *,
              shown_value: float | None = None, gate_state: str = "") -> None:
        self._t.append(float(t_s)); self._v.append(float(value)); self._z.append(float(z))
        self._shown.append(float(shown)); self._ok.append(bool(gate_ok)); self._phase.append(str(phase))
        self._vs.append(float(value if shown_value is None else shown_value)); self._state.append(str(gate_state))

    def posterior(self, t_s: float, p: np.ndarray, gate_ok: bool, phase: str) -> None:
        self._pt.append(float(t_s)); self._pp.append(np.asarray(p, dtype=float)); self._pok.append(bool(gate_ok)); self._pphase.append(str(phase))

    def raw(self, block: Block) -> None:
        if self.recorder is not None:
            self.recorder.write(block)

    def close(self, *, session: dict[str, Any] | None = None) -> Path:
        if self.closed:
            return self.out_dir
        raw_files = None
        if self.recorder is not None:
            b, s = self.recorder.close()
            raw_files = {"bin": b.name, "sidecar": s.name}
        head = {
            "eegloop_session": 1,
            "name": self.protocol.get("name"),
            "started_at": self.started_at,
            "finished_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "protocol_hash": config_hash(self.protocol) if self.protocol else None,
            "sealed_sham": self.sealed_sham,
            "versions": installed_versions(),
            "source": {"kind": self.info.kind, "fs": self.info.fs, "ch_names": list(self.info.ch_names),
                       "clock": self.info.clock, "nominal": self.info.nominal},
            "n_values": len(self._t), "n_posteriors": len(self._pt), "n_events": len(self.events),
            "raw": raw_files, "decoder": self.decoder_file,
            "files": {"session": "session.json", "events": "events.jsonl", "signal": "signal.npz",
                      "protocol": "protocol-resolved.json"},
        }
        head.update(session or {})
        (self.out_dir / "session.json").write_text(json.dumps(scrub_value(head), indent=2, default=str), encoding="utf-8")
        with (self.out_dir / "events.jsonl").open("w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps(scrub_value(e), default=str) + "\n")
        arrays = dict(times_s=np.asarray(self._t), values=np.asarray(self._v),
                      shown_values=np.asarray(self._vs), z=np.asarray(self._z), shown=np.asarray(self._shown),
                      gate_ok=np.asarray(self._ok, dtype=bool), gate_state=np.asarray(self._state, dtype=str),
                      phase=np.asarray(self._phase, dtype=str))
        if self._pt:
            arrays.update(post_times_s=np.asarray(self._pt), posteriors=np.asarray(self._pp),
                          post_gate_ok=np.asarray(self._pok, dtype=bool), post_phase=np.asarray(self._pphase, dtype=str))
        np.savez(self.out_dir / "signal.npz", **arrays)
        (self.out_dir / "protocol-resolved.json").write_text(json.dumps(scrub_value(self.protocol), indent=2, default=str), encoding="utf-8")
        self.closed = True
        return self.out_dir


class SessionReader:
    """A session directory, read back with numpy only."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not (self.path / "session.json").exists():
            raise FileNotFoundError(f"no session.json in {self.path.name}")
        self.session: dict[str, Any] = json.loads((self.path / "session.json").read_text(encoding="utf-8"))
        proto = self.path / "protocol-resolved.json"
        self.protocol: dict[str, Any] = json.loads(proto.read_text(encoding="utf-8")) if proto.exists() else {}
        with np.load(self.path / "signal.npz", allow_pickle=False) as z:
            self.signal = {k: z[k] for k in z.files}

    @property
    def feedback(self) -> tuple[np.ndarray, np.ndarray]:
        """``(times_s, values)`` -- the veridical per-block values, the donor a yoked sham replays."""
        return self.signal["times_s"], self.signal["values"]

    @property
    def shown(self) -> np.ndarray:
        return self.signal["shown"]

    def events(self, kind: str | None = None) -> list[dict[str, Any]]:
        out = []
        with (self.path / "events.jsonl").open(encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                if kind is None or e.get("kind") == kind:
                    out.append(e)
        return out

    @property
    def posteriors(self) -> tuple[np.ndarray, np.ndarray] | None:
        """``(times_s, posteriors)`` of a bci session, or ``None`` for a feedback session."""
        if "posteriors" not in self.signal:
            return None
        return self.signal["post_times_s"], self.signal["posteriors"]

    def decoder(self) -> Any:
        """The frozen decoder a bci session saved, loaded with numpy alone."""
        name = self.session.get("decoder")
        if not name:
            return None
        from ..bci.decoder import FrozenDecoder

        return FrozenDecoder.load(self.path / name)

    @property
    def budget(self) -> dict[str, Any] | None:
        return self.session.get("budget")

    @property
    def sealed_sham(self) -> str | None:
        return self.session.get("sealed_sham")

    @property
    def raw(self) -> tuple[np.ndarray, float, tuple[str, ...], np.ndarray | None] | None:
        """The recorded stream, if ``record_raw`` was on: ``(data_uv, fs, ch_names, timestamps_s)``."""
        files = self.session.get("raw")
        if not files:
            return None
        from ..sources.replay import read_recording

        data, fs, names, ts, _ = read_recording(self.path / files["bin"])
        tsf = self.path / (Path(files["bin"]).stem + ".timestamps.npy")
        if ts is None and tsf.exists():
            ts = np.load(tsf)
        return data, fs, names, ts
