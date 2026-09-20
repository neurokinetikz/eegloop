"""A two-way switch: the decoder's decisions, delivered to a function you wrote.

    python loop/examples/cue_switch.py                       # synthetic imagery, fast
    python loop/examples/cue_switch.py --wall                 # at the real update rate
    python loop/examples/cue_switch.py --protocol my.yaml     # your own protocol

The protocol calibrates on cued trials, fits and freezes a decoder, then applies it; every decision
arrives at ``switch`` below with its posterior and the block time. Replace ``switch`` with whatever
the decision should do -- that is the application, and nothing above it changes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # import without installing; `pip install -e ./loop` makes this unnecessary

from eegloop.present import CallbackPresenter  # noqa: E402
from eegloop.session import load_protocol, run_protocol  # noqa: E402

ARROWS = {"left": "◀ LEFT ", "right": " RIGHT ▶"}


def switch(d) -> None:  # type: ignore[no-untyped-def]
    """Your application. This one prints; yours might move a cursor or select a letter."""
    print(f"{d.t_s:7.1f} s  {ARROWS.get(d.label, d.label):10s}  p = {d.posterior:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default=str(HERE.parent / "configs" / "mi-2class-synthetic.yaml"))
    ap.add_argument("--wall", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    protocol = load_protocol(args.protocol, {"source.pace": "wall"} if args.wall else {})
    result = run_protocol(protocol, presenter=CallbackPresenter(on_decision=switch), out_dir=args.out)
    o = result.stats["online"]
    band = o["chance_band_95"]
    print(f"\nsession {result.status}: {result.stats['n_decisions']} decisions; {o['n_correct']}/{o['n_decided']} cued trials "
          f"correct" + (f" (chance band {band[0]:.2f}–{band[1]:.2f} for n = {o['n_decided']})" if band else ""))
    print(f"written to {result.out_dir}/ (decoder.npz, session.json, events.jsonl, signal.npz)")
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
