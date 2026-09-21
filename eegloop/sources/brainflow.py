"""A headset through the BrainFlow driver, behind the same interface as a replayed file.

This is the one module in the package that names a device (``site/CONTRACTS.md``, Phase 5, rule 4):
the mapping from a board key to the driver's board id lives in :data:`BOARDS` and in
``loop/README.md``, and nowhere else. Everything above a source is device-free because every source
emits the same :class:`~eegloop.block.Block`; lessons say "the driver's board table".

What is verified and what is not is written down per board -- a ``verified`` date on each fact, a
``TODO(confirm)`` on each one the author's own headset has yet to settle -- and ``eegloop check``
exists to settle them: it measures the rate, the drop accounting, the timestamp structure and the
per-channel signal facts from a live stream and prints a report a human files under
``site/notes/device-<board>.md`` with no serial number in it.

The driver is imported lazily and named when missing. A pairing address or serial number reaches the
driver from an argument or from the environment (``EEGLOOP_MAC_ADDRESS``, ``EEGLOOP_SERIAL_NUMBER``)
and is never placed in ``StreamInfo.nominal``, so no session log, sidecar or report can carry it.
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable

import numpy as np

from ..block import Block, StreamInfo

__all__ = ["BOARDS", "DEPRECATED_IDS", "BrainFlowSource", "list_boards", "board_table"]

#: Board key → verified facts. ``board_id`` is the name in the driver's ``BoardIds`` enum.
#: ``channels`` are the names the course uses (10-20 equivalents where the driver's differ).
BOARDS: dict[str, dict[str, Any]] = {
    "muse-2": {
        "board_id": "MUSE_2_BOARD", "product": "Muse 2", "montage": "frontotemporal-4",
        "channels": ("TP9", "AF7", "AF8", "TP10"), "fs": 256.0, "reference": "Fpz (driven), TODO(confirm) as the driver reports it",
        "transport": "native Bluetooth LE, no dongle", "driver_version_min": "5.22.0",
        "reported_filters": "TODO(confirm): none known on the raw EEG preset; the companion app's stream is not this stream",
        "units": "uV, TODO(confirm) against a known-amplitude signal", "verified": "2026-09-20",
        "note": "the *_BLED ids (a dongle route) are deprecated in the driver and refused here",
    },
    "muse-s": {
        "board_id": "MUSE_S_BOARD", "product": "Muse S", "montage": "frontotemporal-4",
        "channels": ("TP9", "AF7", "AF8", "TP10"), "fs": 256.0, "reference": "Fpz (driven), TODO(confirm) as the driver reports it",
        "transport": "native Bluetooth LE, no dongle", "driver_version_min": "5.22.0",
        "reported_filters": "TODO(confirm)", "units": "uV, TODO(confirm)", "verified": "2026-09-20",
        "note": "the *_BLED ids are deprecated in the driver and refused here",
    },
    "muse-s-athena": {
        "board_id": "MUSE_S_ANTHENA_BOARD", "product": "Muse S Athena", "montage": "frontotemporal-4",
        "channels": ("TP9", "AF7", "AF8", "TP10"), "fs": 256.0, "reference": "TODO(confirm)",
        "transport": "native Bluetooth LE, no dongle", "driver_version_min": "5.22.0",
        "reported_filters": "TODO(confirm)", "units": "uV, TODO(confirm)", "verified": "2026-09-20",
        "note": "the driver spells the id ANTHENA; the extra optical rows are not EEG and are not read",
    },
    "brainbit": {
        "board_id": "BRAINBIT_BOARD", "product": "BrainBit", "montage": "occipitotemporal-4",
        "channels": ("O1", "O2", "T7", "T8"), "driver_channel_names": ("O1", "O2", "T3", "T4"),
        "fs": 250.0, "bandwidth": "0–100 Hz", "reference": "TODO(confirm)",
        "transport": "native Bluetooth LE, no dongle", "driver_version_min": "5.22.0",
        "reported_filters": "an on-device notch and a 1–40 Hz band-pass are OPTIONAL and must be OFF for the course; "
                            "TODO(confirm) the driver switch and its default, then log the state as a fact",
        "units": "uV, TODO(confirm)", "verified": "2026-09-20",
        "note": "T3/T4 are the older names of T7/T8; the course uses the 10-20 names",
    },
}

#: Driver ids this package refuses, with the key to use instead.
DEPRECATED_IDS: dict[str, str] = {"MUSE_2_BLED_BOARD": "muse-2", "MUSE_S_BLED_BOARD": "muse-s"}

#: Keys of :data:`BOARDS` entries a report may quote; nothing else about a device is quoted.
PUBLIC_FACTS: tuple[str, ...] = ("board_id", "product", "montage", "channels", "driver_channel_names", "fs", "bandwidth",
                                 "reference", "transport", "driver_version_min", "reported_filters", "units", "verified", "note")


def list_boards() -> list[dict[str, Any]]:
    return [{"key": k, **{f: v.get(f) for f in PUBLIC_FACTS if f in v}} for k, v in BOARDS.items()]


def board_table() -> str:
    """The table as markdown, for ``eegloop devices`` and the README."""
    lines = ["| key | product | driver id | montage | channels | rate | transport | verified |", "|---|---|---|---|---|---|---|---|"]
    for k, b in BOARDS.items():
        lines.append(f"| `{k}` | {b['product']} | `{b['board_id']}` | {b['montage']} | {', '.join(b['channels'])} | {b['fs']:g} Hz | {b['transport']} | {b['verified']} |")
    return "\n".join(lines)


def _driver() -> Any:
    try:
        import brainflow  # noqa: F401
        from brainflow import board_shim
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("a headset source needs the driver: pip install 'eegloop[brainflow]'") from exc
    return board_shim


class BrainFlowSource:
    """A board key from :data:`BOARDS`, streamed as blocks in microvolts on the driver's clock.

    ``shim_factory`` and ``descriptor`` exist for tests: a fake board with the driver's interface can
    stand in for the driver, so the drop accounting, the timestamps and the reblocking are tested on
    every runner while the driver itself is proven by ``pytest -m hardware`` on a paired headset.
    """

    def __init__(self, board: str, *, mac_address: str | None = None, serial_number: str | None = None,
                 timeout_s: float = 15.0, block_samples: int = 32, buffer_samples: int = 45000,
                 package_modulus: int | None = None, shim_factory: Callable[..., Any] | None = None,
                 descriptor: dict[str, Any] | None = None, driver_version: str | None = None) -> None:
        key = str(board)
        if key in DEPRECATED_IDS or key.upper() in DEPRECATED_IDS:
            use = DEPRECATED_IDS.get(key, DEPRECATED_IDS.get(key.upper()))
            raise ValueError(f"{key!r} is a deprecated dongle route in the driver; use the board key {use!r}")
        if key not in BOARDS:
            by_id = {v["board_id"]: k for k, v in BOARDS.items()}
            hint = f" (that driver id is board key {by_id[key]!r})" if key in by_id else ""
            raise ValueError(f"unknown board key {key!r}{hint}; one of {', '.join(BOARDS)}")
        self.key, self.facts = key, BOARDS[key]
        self.block_samples, self.timeout_s, self.buffer_samples = int(block_samples), float(timeout_s), int(buffer_samples)
        # pairing details: arguments or the environment, never the protocol file and never the log
        self._mac = mac_address if mac_address is not None else os.environ.get("EEGLOOP_MAC_ADDRESS", "")
        self._serial = serial_number if serial_number is not None else os.environ.get("EEGLOOP_SERIAL_NUMBER", "")
        self._shim_factory = shim_factory
        self._descr = descriptor
        self._driver_version = driver_version
        self._shim: Any = None
        self._started = False
        self.done = False
        self._count = 0                     # session sample index of the next sample, losses included
        self._ts0: float | None = None      # the driver's stamp of the first sample: session time zero
        self._last_pkg: float | None = None
        self._last_ts: float | None = None
        self._package_modulus = package_modulus
        self._pending: list[Block] = []
        self._t_wall0: float | None = None
        self.n_drops = 0
        self._pkg_steps: dict[int, int] = {}   # histogram of package-number increments, for the report
        self.info = self._build_info()

    # -- set-up ------------------------------------------------------------------------------------
    def _resolve_descriptor(self) -> tuple[dict[str, Any], int | None, str]:
        if self._descr is not None:
            return dict(self._descr), self._descr.get("board_id_value"), self._driver_version or "fake"
        bs = _driver()
        import brainflow

        board_id = int(bs.BoardIds[self.facts["board_id"]].value)
        descr = dict(bs.BoardShim.get_board_descr(board_id))
        return descr, board_id, str(getattr(brainflow, "__version__", "unknown"))

    def _build_info(self) -> StreamInfo:
        descr, board_id, version = self._resolve_descriptor()
        self._board_id = board_id
        self._descr = descr
        fs = float(descr.get("sampling_rate", self.facts["fs"]))
        self._eeg_rows = [int(r) for r in descr.get("eeg_channels", [])]
        names_raw = descr.get("eeg_names")
        driver_names = tuple(s.strip() for s in names_raw.split(",")) if isinstance(names_raw, str) else tuple(names_raw or ())
        expected = tuple(self.facts.get("driver_channel_names", self.facts["channels"]))
        mismatch: list[str] = []
        if abs(fs - float(self.facts["fs"])) > 1e-9:
            mismatch.append(f"rate: table {self.facts['fs']:g} Hz, driver {fs:g} Hz")
        if driver_names and tuple(driver_names[:len(expected)]) != expected:
            mismatch.append(f"channel names: table {expected}, driver {driver_names}")
        if self._eeg_rows and len(self._eeg_rows) != len(self.facts["channels"]):
            mismatch.append(f"EEG rows: table {len(self.facts['channels'])}, driver {len(self._eeg_rows)}")
        n = len(self._eeg_rows) if self._eeg_rows else len(self.facts["channels"])
        names = tuple(self.facts["channels"][:n]) if n <= len(self.facts["channels"]) else tuple(driver_names[:n])
        self._ts_row = descr.get("timestamp_channel")
        self._pkg_row = descr.get("package_num_channel")
        self._marker_row = descr.get("marker_channel")
        nominal = {
            "board": self.key, "board_id": self.facts["board_id"], "product": self.facts["product"], "driver": "brainflow",
            "driver_version": version, "montage": self.facts["montage"], "expected_fs": float(self.facts["fs"]),
            "descriptor": {k: v for k, v in descr.items() if k in ("name", "sampling_rate", "eeg_channels", "eeg_names",
                                                                   "timestamp_channel", "package_num_channel", "marker_channel", "num_rows")},
            "descriptor_mismatch": mismatch,
            "channel_names_note": self.facts.get("note", ""),
            "reference": self.facts.get("reference"),
            "reported_filters": self.facts["reported_filters"],
            "units_note": self.facts["units"],
            "mains_hz": None,
            "clock": "the driver's timestamps are the computer's clock at packet arrival (UNIX seconds, rebased to the first "
                     "sample); Bluetooth packets arrive in bursts, so per-sample intervals are not the sample period -- "
                     "eegloop check measures the structure",
            "drop_accounting": "package-number increments; whether the counter steps per sample or per packet, and its "
                               "modulus, are TODO(confirm) per board -- eegloop check prints the increment histogram",
            "package_modulus": self._package_modulus if self._package_modulus is not None else "inferred at the first wrap",
            "verified": self.facts["verified"],
        }
        return StreamInfo(fs, names, kind="brainflow", clock="wall", units="uV", nominal=nominal)

    # -- the Source interface ----------------------------------------------------------------------
    def start(self) -> None:
        if self._started:
            return
        if self._shim_factory is not None:
            self._shim = self._shim_factory(self._board_id, {"mac_address": self._mac, "serial_number": self._serial, "timeout": self.timeout_s})
        else:
            bs = _driver()
            params = bs.BrainFlowInputParams()
            params.mac_address = self._mac
            params.serial_number = self._serial
            params.timeout = int(self.timeout_s)
            bs.BoardShim.disable_board_logger()
            self._shim = bs.BoardShim(self._board_id, params)
        self._shim.prepare_session()
        self._shim.start_stream(self.buffer_samples)
        self._t_wall0 = time.perf_counter()
        self._started = True

    def stop(self) -> None:
        if self._shim is not None:
            try:
                self._shim.stop_stream()
            finally:
                self._shim.release_session()
        self._shim, self._started, self.done = None, False, True

    def clock_now(self) -> float:
        """Session seconds of the newest sample received (the driver's stamp, rebased)."""
        if self._last_ts is None or self._ts0 is None:
            return 0.0
        return float(self._last_ts - self._ts0)

    @property
    def elapsed_wall_s(self) -> float:
        return 0.0 if self._t_wall0 is None else time.perf_counter() - self._t_wall0

    def read(self, max_samples: int | None = None) -> Block | None:
        if not self._started:
            self.start()
        if not self._pending:
            self._poll()
        if not self._pending:
            return None
        block = self._pending.pop(0)
        if max_samples is not None and block.n_samples > int(max_samples):
            head, tail = block.data[:, :int(max_samples)], block.data[:, int(max_samples):]
            ts = block.timestamps_s
            k = head.shape[1]
            self._pending.insert(0, Block(tail, block.fs, seq=block.seq + 1, sample_index=block.sample_index + k,
                                          t_start_s=float(ts[k]) if ts is not None else block.t_start_s + k / block.fs,
                                          timestamps_s=None if ts is None else ts[k:], dropped_before=0))
            block = block.replace(data=head, timestamps_s=None if ts is None else ts[:head.shape[1]])
        return block

    # -- the driver's rows → runs of contiguous samples ----------------------------------------------
    def _poll(self) -> None:
        raw = np.asarray(self._shim.get_board_data(), dtype=float)
        if raw.ndim != 2 or raw.shape[1] == 0:
            return
        eeg = raw[self._eeg_rows] if self._eeg_rows else raw[: self.info.n_channels]
        n = eeg.shape[1]
        ts = raw[int(self._ts_row)] if self._ts_row is not None else None
        pkg = raw[int(self._pkg_row)] if self._pkg_row is not None else None
        if ts is not None and self._ts0 is None:
            self._ts0 = float(ts[0])
        missing = self._missing_before_each(pkg, n)
        fs = self.info.fs
        # split at every loss so each emitted raw block is one contiguous run with its own dropped_before
        cut = [0] + [i for i in range(1, n) if missing[i] > 0] + [n]
        for a, b in zip(cut[:-1], cut[1:]):
            drop = int(missing[a])
            self._count += drop
            self.n_drops += drop
            chunk_ts = None if ts is None else ts[a:b] - self._ts0
            t0 = float(chunk_ts[0]) if chunk_ts is not None else self._count / fs
            self._pending.append(Block(eeg[:, a:b], fs, seq=0, sample_index=self._count, t_start_s=t0,
                                       timestamps_s=chunk_ts, dropped_before=drop))
            self._count += b - a
        if ts is not None:
            self._last_ts = float(ts[-1])
        if pkg is not None:
            self._last_pkg = float(pkg[-1])

    def _missing_before_each(self, pkg: np.ndarray | None, n: int) -> np.ndarray:
        """Samples lost before each sample of the chunk, from the package counter; zeros without one."""
        if pkg is None:
            return np.zeros(int(n), dtype=int)
        seq = pkg if self._last_pkg is None else np.concatenate([[self._last_pkg], pkg])
        d = np.diff(seq)
        if self._package_modulus is None and (d < 0).any():
            self._package_modulus = 256 if float(np.max(seq)) < 256 else 65536
            self.info.nominal["package_modulus"] = self._package_modulus
        if self._package_modulus:
            d = np.mod(d, self._package_modulus)
        for step in np.unique(d).astype(int):
            self._pkg_steps[int(step)] = self._pkg_steps.get(int(step), 0) + int(np.sum(d == step))
        miss = np.where(d > 1, d - 1, 0).astype(int)
        out = np.zeros(pkg.shape[0], dtype=int)
        if self._last_pkg is None:
            out[1:] = miss
        else:
            out[:] = miss
        return out

    @property
    def package_steps(self) -> dict[int, int]:
        """Histogram of package-number increments seen: ``{1: n}`` means one count per sample."""
        return dict(sorted(self._pkg_steps.items()))

    def describe(self) -> dict[str, Any]:
        """Everything a log may carry about this source. The pairing details are not in it by construction."""
        return {"key": self.key, "info": {"fs": self.info.fs, "ch_names": list(self.info.ch_names), "clock": self.info.clock},
                "nominal": dict(self.info.nominal), "n_drops": self.n_drops, "package_steps": self.package_steps}
