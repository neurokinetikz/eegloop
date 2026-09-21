"""The headset source against a fake board with the driver's interface: drop accounting from the package
counter, timestamps rebased to the session, the split into contiguous runs, and nothing private in the
facts. The driver itself is proven by ``pytest -m hardware`` on a paired headset (tests/test_hardware.py)."""
from __future__ import annotations

import numpy as np
import pytest

from eegloop.session import ProtocolError, validate_protocol
from eegloop.sources import BOARDS, BrainFlowSource, blocks, board_table, list_boards, open_source
from eegloop.sources.brainflow import DEPRECATED_IDS

KEY = next(iter(BOARDS))            # a board key; the test never spells a product name
FS = 256.0
PACKET = 12                          # samples per transport packet: identical stamps within one
LOSS_AT, LOSS_PACKETS = 2560, 5      # a planted loss of five packets after ten seconds


class FakeShim:
    """The driver's board interface, fed by a deterministic generator."""

    descr = {"name": "fake-board", "sampling_rate": int(FS), "eeg_channels": [1, 2, 3, 4], "eeg_names": "TP9,AF7,AF8,TP10",
             "timestamp_channel": 5, "package_num_channel": 0, "marker_channel": 6, "num_rows": 7, "board_id_value": -99}
    instances: list["FakeShim"] = []

    def __init__(self, board_id, params, *, total_s: float = 20.0, chunk: int = 37, wrap: int = 256, seed: int = 3) -> None:
        self.board_id, self.params = board_id, dict(params)
        self.prepared = self.streaming = self.released = False
        rng = np.random.default_rng(seed)
        n = int(total_s * FS)
        t = np.arange(n) / FS
        eeg = 8.0 * rng.standard_normal((4, n)) + 12.0 * np.sin(2 * np.pi * 10.0 * t)[None, :]
        pkg = np.arange(n, dtype=float)
        pkg[LOSS_AT:] += LOSS_PACKETS * PACKET          # the counter skips: a loss the driver never saw
        pkg = np.mod(pkg, wrap)
        packet_idx = np.arange(n) // PACKET
        ts = 1.7e9 + packet_idx * (PACKET / FS)         # stamped at packet arrival: identical within a packet
        ts[LOSS_AT:] += LOSS_PACKETS * PACKET / FS
        rows = np.zeros((7, n))
        rows[0], rows[1:5], rows[5] = pkg, eeg, ts
        self._rows, self._pos, self._chunk = rows, 0, int(chunk)
        FakeShim.instances.append(self)

    def prepare_session(self) -> None:
        self.prepared = True

    def start_stream(self, buffer_size: int) -> None:
        self.streaming = True

    def get_board_data(self) -> np.ndarray:
        a, b = self._pos, min(self._pos + self._chunk, self._rows.shape[1])
        self._pos = b
        return self._rows[:, a:b]

    def stop_stream(self) -> None:
        self.streaming = False

    def release_session(self) -> None:
        self.released = True


def _fake_source(**kw) -> BrainFlowSource:
    return BrainFlowSource(KEY, descriptor=FakeShim.descr, shim_factory=FakeShim, driver_version="fake", **kw)


def test_board_table_names_only_verified_facts_and_refuses_the_deprecated_route():
    assert set(BOARDS) >= {"muse-2", "muse-s", "muse-s-athena", "brainbit"}
    for k, b in BOARDS.items():
        assert b["board_id"].endswith("_BOARD") and b["verified"] == "2026-09-20" and len(b["channels"]) == 4
    assert [row["key"] for row in list_boards()] == list(BOARDS)
    assert board_table().startswith("| key |") and "TODO" not in board_table()
    for dep, use in DEPRECATED_IDS.items():
        with pytest.raises(ValueError, match="deprecated"):
            BrainFlowSource(dep)
    with pytest.raises(ValueError, match="unknown board key 'nope'"):
        BrainFlowSource("nope")
    with pytest.raises(ValueError, match="that driver id is board key"):
        BrainFlowSource(BOARDS[KEY]["board_id"])


def test_missing_driver_names_the_extra():
    try:
        import brainflow  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match=r"eegloop\[brainflow\]"):
            BrainFlowSource(KEY)
    else:
        pytest.skip("the driver is installed here; the message for its absence cannot be exercised")


def test_fake_board_streams_blocks_with_drops_accounted_and_timestamps_rebased(monkeypatch):
    monkeypatch.setenv("EEGLOOP_MAC_ADDRESS", "AA:BB:CC:DD:EE:FF")
    monkeypatch.setenv("EEGLOOP_SERIAL_NUMBER", "SN-000123")
    src = _fake_source()
    assert src.info.fs == FS and src.info.ch_names == BOARDS[KEY]["channels"] and src.info.kind == "brainflow" and src.info.clock == "wall"
    nominal_text = str(src.info.nominal) + str(src.describe())
    assert "AA:BB" not in nominal_text and "SN-000123" not in nominal_text
    assert src.info.nominal["descriptor_mismatch"] == [] and src.info.nominal["driver_version"] == "fake"
    got = list(blocks(src, 32, timeout_s=0.3))
    shim = FakeShim.instances[-1]
    assert shim.prepared and shim.streaming and shim.params["mac_address"] == "AA:BB:CC:DD:EE:FF"  # the driver got it; the log did not
    n = sum(b.n_samples for b in got)
    assert n >= 5120 - 64 and all(b.n_samples == 32 for b in got)
    assert sum(b.dropped_before for b in got) == LOSS_PACKETS * PACKET == src.n_drops
    hit = [b for b in got if b.dropped_before]
    assert len(hit) == 1 and hit[0].sample_index == LOSS_AT + LOSS_PACKETS * PACKET
    assert got[0].t_start_s == 0.0 and got[0].timestamps_s is not None and got[0].timestamps_s[0] == 0.0
    # the stamp jumped by the loss, as it should -- to within a packet, because a packet's samples share its arrival stamp
    assert abs(hit[0].t_start_s - hit[0].sample_index / FS) < PACKET / FS
    assert src.package_steps == {1: 5119 - 1, LOSS_PACKETS * PACKET + 1: 1} or src.package_steps[LOSS_PACKETS * PACKET + 1] == 1
    assert 19.8 < src.clock_now() < 20.5  # 20 s of samples plus the 0.23 s the loss moved the stamps
    src.stop()
    assert shim.released and src.done


def test_read_honours_max_samples_and_the_wrap_of_the_counter():
    src = _fake_source(package_modulus=None)
    src.start()
    first = src.read(max_samples=10)
    assert first is not None and first.n_samples == 10 and first.sample_index == 0
    second = src.read(max_samples=10)
    assert second is not None and second.sample_index == 10 and second.t_start_s == pytest.approx(0.0)  # same packet: same stamp
    rest = [b for b in iter(lambda: src.read(), None)]
    assert src._package_modulus == 256 and src.info.nominal["package_modulus"] == 256
    src.stop()


def test_open_source_and_protocol_paths():
    src = open_source(f"brainflow:{KEY}", descriptor=FakeShim.descr, shim_factory=FakeShim, driver_version="fake")
    assert isinstance(src, BrainFlowSource) and src.key == KEY
    with pytest.raises(ValueError, match="binding decision"):
        open_source("lsl:anything")
    base = {"name": "hw", "seed": 1, "phases": [{"kind": "calibrate", "duration_s": 5}, {"kind": "train", "duration_s": 5}]}
    with pytest.raises(ProtocolError) as e:
        validate_protocol(dict(base, source={"kind": "brainflow", "board": "nope"}))
    assert any(p.startswith("source.board: 'nope' is not valid") for p in e.value.problems)
    with pytest.raises(ProtocolError) as e:
        validate_protocol(dict(base, source={"kind": "brainflow"}))
    assert any(p.startswith("source.board: required") for p in e.value.problems)
    ok = validate_protocol(dict(base, source={"kind": "brainflow", "board": KEY}))
    assert ok.source.timeout_s == 15.0
    assert "mac" not in str(ok.source).lower() and "serial" not in str(ok.source).lower()
