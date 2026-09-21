"""A switch that closing the eyes turns on: calibrate once, freeze, apply, on a montage the course names.

    python loop/examples/eo_ec_switch.py                       # synthetic eyes-closed / eyes-open periods, fast
    python loop/examples/eo_ec_switch.py --wall                 # at the real update rate
    python loop/examples/eo_ec_switch.py --protocol my.yaml     # your own protocol: a replay with cues, a headset

``cue_switch.py`` shows the loop's shape on planted motor imagery, which no four-channel headband the
course names can record. This one decodes the state both montage classes can: eyes closed against
eyes open, by posterior alpha (lesson L7.15). The protocol calibrates on cued periods, fits with folds
in cue order, freezes the decoder into arrays and applies them; every decision reaches ``lamp`` with
its posterior and the block time. Replace ``lamp`` with whatever the state should switch -- that is
the application, and nothing above it changes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # import without installing; `pip install -e ./loop` makes this unnecessary

from eegloop.present import CallbackPresenter  # noqa: E402
from eegloop.session import load_protocol, run_protocol  # noqa: E402

LAMP = {"closed": "● on ", "open": "○ off"}


def lamp(d) -> None:  # type: ignore[no-untyped-def]
    """Your application. This one prints a lamp; yours might dim a screen or pause a player."""
    print(f"{d.t_s:7.1f} s  {LAMP.get(d.label, d.label)}  p = {d.posterior:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default=str(HERE.parent / "configs" / "eo-ec-synthetic.yaml"))
    ap.add_argument("--wall", action="store_true", help="release samples on the wall clock")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    protocol = load_protocol(args.protocol, {"source.pace": "wall"} if args.wall else {})
    result = run_protocol(protocol, presenter=CallbackPresenter(on_decision=lamp), out_dir=args.out)
    o = result.stats["online"]
    band = o["chance_band_95"]
    dec = result.budget.get("decoder") or {}
    print(f"\nsession {result.status}: {result.stats['n_decisions']} decisions; {o['n_correct']}/{o['n_decided']} cued periods "
          f"correct" + (f" (chance band {band[0]:.2f}–{band[1]:.2f} for n = {o['n_decided']})" if band else ""))
    if dec:
        print(f"decoder latency {dec['latency_ms']:.0f} ms — {dec['note']}")
    print(f"written to {result.out_dir}/ (decoder.npz, session.json, events.jsonl, signal.npz)")
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
