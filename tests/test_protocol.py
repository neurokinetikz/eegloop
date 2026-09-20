"""The protocol schema: every problem listed at once, in the message format eegpipe's tests assert."""
from __future__ import annotations

import json

import pytest

from eegloop.session import Protocol, ProtocolError, load_protocol, resolved, validate_protocol

GOOD = {
    "name": "t", "seed": 1, "block_samples": 32,
    "source": {"kind": "synthetic", "scenario": "clean"},
    "phases": [{"kind": "calibrate", "duration_s": 5}, {"kind": "train", "duration_s": 10}],
}


def test_a_minimal_protocol_validates_with_defaults_filled():
    p = validate_protocol(GOOD)
    assert isinstance(p, Protocol) and p.signal.band == [8.0, 12.0] and p.sham.mode == "veridical"
    assert p.total_duration_s == 15.0
    r = resolved(p)
    assert r["quality"]["policy"] == "freeze" and r["phases"][1]["kind"] == "train"


def test_required_fields_say_why():
    with pytest.raises(ProtocolError) as excinfo:
        validate_protocol({"phases": GOOD["phases"]}, source="x.yaml")
    problems = excinfo.value.problems
    assert any(p.startswith("seed: required") for p in problems)
    assert any(p.startswith("name: required") for p in problems)
    assert "sealed sham assignment" in "\n".join(problems)
    assert str(excinfo.value).startswith("x.yaml: ")


def test_type_and_did_you_mean_messages_match_eegpipe():
    bad = dict(GOOD, seed="twenty", signal={"bands": [8, 12], "n_taps": 128, "derivation": "rmz"},
               quality={"policy": "freeze", "window_s": -1})
    with pytest.raises(ProtocolError) as excinfo:
        validate_protocol(bad, source="x.yaml")
    problems = excinfo.value.problems
    assert "seed: expected an integer, got 'twenty' (str)" in problems
    assert any("signal.bands: unknown setting (did you mean 'band'?)" in p for p in problems)
    assert any("signal.n_taps" in p and "odd" in p for p in problems)
    assert any("signal.derivation: 'rmz' is not valid" in p and "did you mean 'rms'" in p for p in problems)
    assert any("quality.window_s: expected a positive number, got -1" in p for p in problems)
    assert "x.yaml: 5 problems" in str(excinfo.value)


def test_semantic_checks_across_blocks():
    with pytest.raises(ProtocolError) as e:
        validate_protocol(dict(GOOD, phases=[{"kind": "train", "duration_s": 10}]))
    assert any("fixed baseline needs values to fix from" in p for p in e.value.problems)
    with pytest.raises(ProtocolError) as e:
        validate_protocol(dict(GOOD, sham={"mode": "sham-yoked"}))
    assert any("sham.donor: required" in p for p in e.value.problems)
    with pytest.raises(ProtocolError) as e:
        validate_protocol(dict(GOOD, source={"kind": "replay"}))
    assert any("source.path: required" in p for p in e.value.problems)
    with pytest.raises(ProtocolError) as e:
        validate_protocol(dict(GOOD, phases=[]))
    assert any("phases: required" in p for p in e.value.problems)
    with pytest.raises(ProtocolError) as e:
        validate_protocol(dict(GOOD, phases=[{"kind": "nap", "duration_s": 3}]))
    assert any("phases[0].kind: 'nap' is not valid" in p for p in e.value.problems)


def test_overrides_use_dotted_paths():
    p = validate_protocol(GOOD, overrides={"output.record_raw": True, "signal.smooth_s": 0.5})
    assert p.output.record_raw is True and p.signal.smooth_s == 0.5


def test_load_json_and_yaml(tmp_path):
    j = tmp_path / "p.json"
    j.write_text(json.dumps(GOOD), encoding="utf-8")
    p = load_protocol(j)
    assert p.name == "t" and p.__dict__["_path"] == j.resolve()
    with pytest.raises(ProtocolError, match="no such protocol file"):
        load_protocol(tmp_path / "missing.json")
    (tmp_path / "empty.json").write_text("null", encoding="utf-8")
    with pytest.raises(ProtocolError, match="empty"):
        load_protocol(tmp_path / "empty.json")
    yaml = pytest.importorskip("yaml")
    y = tmp_path / "p.yaml"
    y.write_text(yaml.safe_dump(GOOD), encoding="utf-8")
    assert load_protocol(y).seed == 1


def test_shipped_protocols_validate():
    from pathlib import Path

    pytest.importorskip("yaml")
    here = Path(__file__).resolve().parents[1] / "configs"
    for f in sorted(here.glob("*.yaml")):
        p = load_protocol(f)
        assert p.name == f.stem, f.name
