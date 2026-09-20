"""``eegloop run --protocol …`` -- one command, no prompts.

    eegloop validate --protocol loop/configs/alpha-up-synthetic.yaml
    eegloop budget   --protocol loop/configs/alpha-up-synthetic.yaml
    eegloop probe    --protocol loop/configs/alpha-up-synthetic.yaml
    eegloop run      --protocol loop/configs/alpha-up-synthetic.yaml [--out DIR] [--record-raw] [--quiet]
    eegloop versions

Exit codes follow ``eegpipe``: 0 on success, 2 for a protocol problem (every problem is listed), 1 for
anything else. ``devices`` and ``check`` arrive with the hardware sources.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from .feedback.signal import SignalSpec, build_chain
from .latency import latency_budget_for, print_budget
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
    result = run_protocol(protocol, source, presenter=NullPresenter() if args.quiet else ConsolePresenter(),
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
    return 0


def _versions(args: argparse.Namespace) -> int:
    v = installed_versions()
    if args.json:
        print(json.dumps(v, indent=2))
    else:
        for k, val in v.items():
            print(f"{k}: {val}")
    return 0


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
    except ProtocolError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 1  # pragma: no cover - argparse rejects anything else
