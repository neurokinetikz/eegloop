"""The smallest application: a protocol, a source, and a presenter you wrote.

    python loop/examples/alpha_bar.py                       # synthetic, fast
    python loop/examples/alpha_bar.py --wall                 # synthetic, at its real update rate
    python loop/examples/alpha_bar.py --protocol my.yaml     # your own protocol

Everything the loop decides is in the protocol file; everything the participant sees goes through
``show`` below. Replace ``show`` with a tone, a game, a web page -- that is where your application
begins, and nothing above it changes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # import without installing; `pip install -e ./loop` makes this unnecessary

from eegloop.present import CallbackPresenter  # noqa: E402
from eegloop.session import load_protocol, run_protocol  # noqa: E402


def show(shown: float, *, z: float, gated: bool, t_s: float, phase: str) -> None:
    """Your display. This one is a line of text; yours need not be."""
    if gated:
        line = "signal unusable — feedback held"
    elif shown != shown:
        line = f"{phase}…"
    else:
        line = "#" * int(round(shown * 40)) + f"  z {z:+.2f}"
    print(f"\r{t_s:6.1f} s  {line:<60}", end="")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default=str(HERE.parent / "configs" / "alpha-up-synthetic.yaml"))
    ap.add_argument("--wall", action="store_true", help="release samples on the wall clock")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    overrides = {"source.pace": "wall"} if args.wall else {}
    protocol = load_protocol(args.protocol, overrides)
    result = run_protocol(protocol, presenter=CallbackPresenter(show), out_dir=args.out)
    print()
    b = result.budget
    print(f"session {result.status}: {result.stats['n_blocks']} blocks, {result.stats['n_gated']} gated, "
          f"budget {b['total_ms']:.1f} ms declared, processing measured p50 {result.stats['processing_ms']['p50']:.2f} ms")
    print(f"written to {result.out_dir}/ (session.json, events.jsonl, signal.npz, protocol-resolved.json)")
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
