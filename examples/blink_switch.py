"""A blink switch -- an EOG switch, labelled as such, not a brain-computer interface.

    python loop/examples/blink_switch.py                                 # synthetic planted blinks, fast
    python loop/examples/blink_switch.py --source replay:recording.npz   # a recording of your own
    python loop/examples/blink_switch.py --source brainflow:<key>        # a headset (eegloop devices lists the keys)

On a frontal geometry the fastest control there is has nothing to do with the brain: a voluntary
blink is a same-sign deflection of a hundred microvolts or more on the frontal pair, and the quality
gate of lesson L7.12 already labels it -- the gate runs FIRST, on the raw block, which is the only
place a blink is still visible. The switch is that label with a refractory period. It is honest, it
works, and every page of the course that mentions it calls it what it is.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # import without installing; `pip install -e ./loop` makes this unnecessary

from eegloop.sources import blocks, open_source  # noqa: E402
from eegloop.steps.quality import QualityGate  # noqa: E402


def press(t_s: float) -> None:
    """Your application. This one prints; yours might advance a slide."""
    print(f"{t_s:7.1f} s  press")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic:blinks")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--mains", type=float, default=60.0)
    ap.add_argument("--refractory", type=float, default=1.5, help="seconds after a press before another can register")
    ap.add_argument("--wall", action="store_true", help="release samples on the wall clock (synthetic and replay)")
    args = ap.parse_args()
    kw = {"pace": "wall" if args.wall else "fast"} if args.source.split(":")[0] in ("synthetic", "replay") else {}
    if args.source.split(":")[0] == "synthetic":
        kw["duration_s"] = args.seconds + 1.0
    src = open_source(args.source, block_samples=32, **kw)
    gate = QualityGate(src.info.fs, src.info.ch_names, window_s=1.0, hold_s=0.5, mains_hz=args.mains)
    presses: list[float] = []
    try:
        for b in blocks(src, 32):
            if b.t_end_s > args.seconds:
                break
            labels = gate.process(b).flags["quality"]["labels"]
            if "blink" in labels and (not presses or b.t_end_s - presses[-1] > args.refractory):
                presses.append(b.t_end_s)
                press(b.t_end_s)
    finally:
        src.stop()
    truth = getattr(src, "truth", None)
    if truth and truth.get("blink_onsets_s") is not None:
        onsets = [o for o in truth["blink_onsets_s"] if o <= args.seconds]
        hit = sum(any(abs(p - o) <= args.refractory for p in presses) for o in onsets)
        print(f"\n{len(onsets)} planted blinks, {len(presses)} presses, {hit} within {args.refractory:g} s of a planted blink, "
              f"{len(presses) - hit} spurious")
    else:
        print(f"\n{len(presses)} presses in {args.seconds:g} s — an EOG switch on the frontal pair; no ground truth on a live stream")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
