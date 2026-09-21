"""``eegloop run --protocol …`` -- one command, no prompts.

    eegloop validate --protocol loop/configs/alpha-up-synthetic.yaml
    eegloop budget   --protocol loop/configs/alpha-up-synthetic.yaml
    eegloop probe    --protocol loop/configs/alpha-up-synthetic.yaml
    eegloop run      --protocol loop/configs/alpha-up-synthetic.yaml [--out DIR] [--record-raw] [--quiet]
    eegloop versions
    eegloop devices                                                   # the driver's board table
    eegloop check --source brainflow:<key> --seconds 30 [--mains 60] [--out site/notes/device-<key>.md]

Exit codes follow ``eegpipe``: 0 on success, 2 for a protocol problem (every problem is listed), 1 for
anything else -- including a ``check`` with a failed row.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from .feedback.signal import SignalSpec, build_chain
from .latency import with_probe, latency_budget_for, print_budget
from .probe import measure_loop_delay
from .session.protocol import ProtocolError, load_protocol, resolved
from .versions import installed_versions

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eegloop", description="Streaming EEG loops for the Scalp to Source course.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser("run", help="run a protocol and write a session directory")
    run_cmd.add_argument("--protocol", required=True, help="path to the YAML or JSON protocol")
    run_cmd.add_argument("--out", default=None, help="session directory (default: <protocol dir>/<output.dir>/<name>)")
    run_cmd.add_argument("--source", default=None, help="override the source, e.g. synthetic:clean or replay:file.npz")
    run_cmd.add_argument("--record-raw", action="store_true", help="also record the raw stream as a .bin + sidecar")
    run_cmd.add_argument("--quiet", action="store_true", help="no console bar")
    run_cmd.add_argument("--fail-fast", action="store_true", help="raise on the first failure instead of logging it")

    v = sub.add_parser("validate", help="check a protocol and print it as resolved")
    v.add_argument("--protocol", required=True)
    v.add_argument("--json", action="store_true")

    b = sub.add_parser("budget", help="the latency budget the protocol's chain declares")
    b.add_argument("--protocol", required=True)
    b.add_argument("--json", action="store_true")

    p = sub.add_parser("probe", help="measure the chain's delay with the loopback probe")
    p.add_argument("--protocol", required=True)
    p.add_argument("--alignments", type=int, default=None)

    ver = sub.add_parser("versions", help="print the versions this environment would record")
    ver.add_argument("--json", action="store_true")

    dev = sub.add_parser("devices", help="the driver's board table: keys, ids, montages, verified facts")
    dev.add_argument("--json", action="store_true")

    chk = sub.add_parser("check", help="stream from a source and report rate, drops, timestamps and per-channel facts")
    chk.add_argument("--source", required=True, help="brainflow:<key> | synthetic[:scenario] | replay:<path>")
    chk.add_argument("--seconds", type=float, default=30.0)
    chk.add_argument("--mains", type=float, default=None, help="mains frequency for the gate's line-noise label")
    chk.add_argument("--block", type=int, default=32)
    chk.add_argument("--out", default=None, help="write the markdown report here (e.g. site/notes/device-<key>.md)")
    chk.add_argument("--json", action="store_true")
    return parser


def _source_and_info(protocol, override: str | None):  # type: ignore[no-untyped-def]
    from .session.runner import build_source
    from .sources.base import open_source

    if override:
        return open_source(override, block_samples=protocol.block_samples)
    return build_source(protocol)


def _run(args: argparse.Namespace) -> int:
    from .present import ConsolePresenter, NullPresenter
    from .session.runner import run_protocol

    overrides: dict[str, Any] = {}
    if args.record_raw:
        overrides["output.record_raw"] = True
    protocol = load_protocol(args.protocol, overrides)
    source = _source_and_info(protocol, args.source) if args.source else None
    result = run_protocol(protocol, source, presenter=NullPresenter() if args.quiet else ConsolePresenter(block_samples=protocol.block_samples),
                          out_dir=args.out, fail_fast=args.fail_fast)
    print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0 if result.status == "ok" else 1


def _validate(args: argparse.Namespace) -> int:
    protocol = load_protocol(args.protocol)
    if args.json:
        print(json.dumps(resolved(protocol), indent=2, default=str))
    else:
        print(f"{protocol.name}: valid — {len(protocol.phases)} phase(s), {protocol.total_duration_s:g} s, "
              f"source {protocol.source.kind}, sham {protocol.sham.mode}, baseline {protocol.baseline.mode}")
    return 0


def _chain_for(protocol, override: str | None = None):  # type: ignore[no-untyped-def]
    src = _source_and_info(protocol, override)
    spec = SignalSpec.from_config(protocol.signal)
    return src, build_chain(spec, src.info, protocol.block_samples, quality=protocol.quality)


def _budget(args: argparse.Namespace) -> int:
    protocol = load_protocol(args.protocol)
    src, chain = _chain_for(protocol)
    budget = latency_budget_for(chain, fs=src.info.fs, block_samples=protocol.block_samples,
                                processing_ms=protocol.processing_ms, convention=protocol.buffer_convention)
    if args.json:
        print(json.dumps(budget, indent=2, default=str))
    else:
        print_budget(budget, title=f"{protocol.name}: declared budget ({src.info.fs:g} Hz)")
    return 0


def _probe(args: argparse.Namespace) -> int:
    protocol = load_protocol(args.protocol)
    src, _ = _chain_for(protocol)
    # the probe is one channel: the filter and the block clock are what it times, not the montage
    from .block import StreamInfo

    spec = SignalSpec.from_config(protocol.signal).replace(channels=(), reference="none", reference_channels=())
    one = StreamInfo(src.info.fs, (src.info.ch_names[0],), kind=src.info.kind)
    factory = lambda: build_chain(spec, one, protocol.block_samples, quality=None, include_smoother=False)  # noqa: E731
    r = measure_loop_delay(factory, fs=src.info.fs, block_samples=protocol.block_samples,
                           centre_hz=float(sum(spec.band) / 2), alignments=args.alignments, smooth_s=spec.smooth_s)
    print(f"{protocol.name}: loopback probe at {r['burst_hz']:g} Hz, {r['alignments']} alignments")
    print(f"  filter delay  measured {r['filter_ms']:8.2f} ms   expected {r['expected_filter_ms']:8.2f} ms")
    print(f"  value delay   measured {r['value_ms']:8.2f} ms   expected {r['expected_value_ms']:8.2f} ms   "
          f"(range {r['value_min_ms']:.2f}–{r['value_max_ms']:.2f} over the block grid)")
    if spec.smooth_s > 0:
        print(f"  smoothing tau {spec.smooth_s:g} s: its cost is the gap between measured and expected above")
    budget = with_probe(latency_budget_for(factory(), fs=src.info.fs, block_samples=protocol.block_samples,
                                           processing_ms=protocol.processing_ms, convention=protocol.buffer_convention), r)
    print(f"  within one block of the arithmetic: {'yes' if budget['probe_within_one_block'] else 'NO'}")
    return 0


def _versions(args: argparse.Namespace) -> int:
    v = installed_versions()
    if args.json:
        print(json.dumps(v, indent=2))
    else:
        for k, val in v.items():
            print(f"{k}: {val}")
    return 0


def _devices(args: argparse.Namespace) -> int:
    from .sources.brainflow import board_table, list_boards

    if args.json:
        print(json.dumps(list_boards(), indent=2, default=str))
    else:
        print(board_table())
        try:
            import brainflow

            print(f"\ndriver installed: brainflow {getattr(brainflow, '__version__', '?')}")
        except ImportError:
            print("\ndriver not installed: pip install 'eegloop[brainflow]'")
    return 0


def _check(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .check import render_check, run_check
    from .sources.base import open_source

    src = open_source(args.source, block_samples=args.block)
    try:
        report = run_check(src, seconds=args.seconds, block_samples=args.block, mains_hz=args.mains)
    finally:
        src.stop()
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        text = render_check(report)
        print(text)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"written to {Path(args.out).name}")
    return 0 if report.get("ok") else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
        if args.command == "validate":
            return _validate(args)
        if args.command == "budget":
            return _budget(args)
        if args.command == "probe":
            return _probe(args)
        if args.command == "versions":
            return _versions(args)
        if args.command == "devices":
            return _devices(args)
        if args.command == "check":
            return _check(args)
    except ProtocolError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 1  # pragma: no cover - argparse rejects anything else
