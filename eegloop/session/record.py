"""Record a stream in the site's own asset format, so a session opens wherever an asset does.

The format is spec §4.5 as ``data/scripts/common.py`` writes it: a Float32 little-endian ``.bin``,
channels × samples, row-major, in microvolts, with a JSON sidecar beside it naming the channels, the
rate, the units and the provenance. That choice is deliberate: a learner's own recording and a
shipped course asset are then the same kind of file, ``ReplaySource.from_asset`` reads both, and
nothing about a learner's data has to be converted to be analysed with the course's tools.

A recording made here is the learner's. The sidecar says so -- ``license`` is "private" and
``label_source`` is "none" -- and no path inside it names the machine.

The layout is channel-major for the *whole* file (all of channel 0, then all of channel 1, ...), which a
stream cannot write as it arrives: each block is channels × a few samples. So the recorder keeps the
blocks in memory and writes the file once, at ``close()`` -- 4 channels at 256 Hz cost about 15 MB an
hour in float32, stated here so nobody is surprised. (The first version wrote each block's chunk in
turn, a block-interleaved layout no reader of the format understands; ``nb-7-11`` found it.)
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..block import Block, StreamInfo

__all__ = ["Recorder", "export_fif"]


class Recorder:
    def __init__(self, out_dir: str | Path, info: StreamInfo, *, name: str = "raw", dataset: str = "local-recording",
                 subject: str = "", run: str = "", reference: str | None = None,
                 generated_by: str = "eegloop.session.record.Recorder") -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.info, self.name = info, name
        self.dataset, self.subject, self.run = dataset, subject, run
        self.reference = reference if reference is not None else str(info.nominal.get("reference", "TODO(confirm) (not reported by the source)"))
        self.generated_by = generated_by
        self._chunks: list[np.ndarray] = []
        self._n = 0
        self._t0: float | None = None
        self._ts: list[np.ndarray] = []
        self._has_ts = True
        self.gaps: list[tuple[int, int]] = []
        self.closed = False

    def write(self, block: Block) -> None:
        if block.n_channels != self.info.n_channels:
            raise ValueError(f"recorder expects {self.info.n_channels} channels, got {block.n_channels}")
        if self._t0 is None:
            self._t0 = float(block.t_start_s)
        if block.dropped_before:
            self.gaps.append((int(block.sample_index), int(block.dropped_before)))
        self._chunks.append(np.asarray(block.data, dtype="<f4"))
        if block.timestamps_s is None:
            self._has_ts = False
        elif self._has_ts:
            self._ts.append(np.asarray(block.timestamps_s, dtype=np.float64))
        self._n += block.n_samples

    @property
    def n_samples(self) -> int:
        return self._n

    def close(self) -> tuple[Path, Path]:
        if self.closed:
            return self.out_dir / f"{self.name}.bin", self.out_dir / f"{self.name}.json"
        data = (np.concatenate(self._chunks, axis=1) if self._chunks
                else np.zeros((self.info.n_channels, 0), dtype="<f4"))
        np.ascontiguousarray(data, dtype="<f4").tofile(self.out_dir / f"{self.name}.bin")   # channels x samples, row-major
        self._chunks = []
        fs = self.info.fs
        side: dict[str, Any] = {
            "channels": list(self.info.ch_names), "sfreq": fs, "source_sfreq": float(self.info.nominal.get("source_sfreq", fs)),
            "units": "uV", "dataset": self.dataset, "subject": self.subject, "run": self.run,
            "t0_s": self._t0 if self._t0 is not None else 0.0, "duration_s": self._n / fs,
            "reference": self.reference,
            "filters_applied": ["none: the stream as the source delivered it"],
            "hardware_filters": self.info.nominal.get("reported_filters", self.info.nominal.get("hardware_filters", "TODO(confirm) (not reported by the source)")),
            "mains_hz": self.info.nominal.get("mains_hz"),
            "license": "private — the recorded person's own data; not for redistribution unless they say so",
            "generated_by": self.generated_by,
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "label_source": "none",
            "source_kind": self.info.kind, "clock": self.info.clock,
            "n_samples": self._n, "dtype": "float32-le", "layout": "channels x samples, row-major",
            "bytes": self._n * self.info.n_channels * 4,
            "gaps": [{"sample_index": i, "n_missing": n} for i, n in self.gaps],
            "timestamps_file": f"{self.name}.timestamps.npy" if self._has_ts and self._ts else None,
        }
        if self._has_ts and self._ts:
            np.save(self.out_dir / f"{self.name}.timestamps.npy", np.concatenate(self._ts))
        (self.out_dir / f"{self.name}.json").write_text(json.dumps(side, indent=2, default=str), encoding="utf-8")
        self.closed = True
        return self.out_dir / f"{self.name}.bin", self.out_dir / f"{self.name}.json"


def export_fif(bin_path: str | Path, out_path: str | Path | None = None) -> Path:
    """A recorded ``.bin`` as an MNE ``_raw.fif`` (needs the ``mne`` extra), so eegpipe can run on it."""
    try:
        import mne
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("export_fif needs MNE: pip install 'eegloop[mne]'") from exc
    from ..sources.replay import read_recording

    data, fs, names, _, nominal = read_recording(Path(bin_path))
    info = mne.create_info(list(names), sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(data * 1e-6, info, verbose="ERROR")
    out = Path(out_path) if out_path is not None else Path(bin_path).with_name(Path(bin_path).stem + "_raw.fif")
    raw.save(out, overwrite=True, verbose="ERROR")
    return out
