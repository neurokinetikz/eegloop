"""``eegloop check``: the acceptance report for a stream, by the course's own criteria.

Lesson L0.6's first-look checklist, made live: the rate as measured against the rate as claimed, the
samples lost and when, what the timestamps actually look like, and per channel the facts an analysis
depends on -- DC level, spread, flatness, the mains line at 50 and at 60 Hz, the frequency above
which there is nothing (an on-device low-pass shows here as a cliff), and how often the quality gate
would have closed. Verdicts use the names ``helpers.first_look_checks`` uses: ok / warn / fail / info.

It runs on any source. On a synthetic or replayed stream it is a test of itself; on a headset it is
the report a human files under ``site/notes/device-<board>.md``. Nothing in it can carry a pairing
address or a serial number, because no source puts one where a report could read it.
"""
from __future__ import annotations

import datetime as _dt
import time
from typing import Any, Sequence

import numpy as np

from .block import StreamInfo
from .sources.base import Source, blocks
from .steps.quality import QualityGate, robust_sd
from .stream import detect_gaps
from .versions import installed_versions

__all__ = ["run_check", "render_check", "line_ratio", "bandwidth_edge_hz"]


def _welch(x: np.ndarray, fs: float, nperseg: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import welch

    n = int(min(x.shape[-1], nperseg or int(4 * fs)))
    return welch(x, fs=fs, nperseg=n)


def line_ratio(x: np.ndarray, fs: float, f_hz: float, *, half_width_hz: float = 0.5, neighbour_hz: tuple[float, float] = (2.0, 8.0)) -> float:
    """Power within ±0.5 Hz of ``f_hz`` over the median power 2–8 Hz away -- the gate's ``line-noise`` metric."""
    if f_hz >= fs / 2:
        return float("nan")
    f, p = _welch(x, fs)
    at = p[(f >= f_hz - half_width_hz) & (f <= f_hz + half_width_hz)]
    off = np.abs(f - f_hz)
    nb = p[(off >= neighbour_hz[0]) & (off <= neighbour_hz[1])]
    if at.size == 0 or nb.size == 0:
        return float("nan")
    return float(at.max() / (np.median(nb) + 1e-24))


def bandwidth_edge_hz(x: np.ndarray, fs: float, *, ref_band: tuple[float, float] = (4.0, 20.0), drop_db: float = 20.0) -> float | None:
    """The lowest frequency above ``ref_band`` beyond which the spectrum stays ``drop_db`` below the
    band's median for the rest of the way to Nyquist -- ``None`` when it never does. An on-device
    low-pass (the ≈43 Hz ceiling of some consumer front-ends) shows here; open bandwidth returns None."""
    f, p = _welch(x, fs)
    ref = np.median(p[(f >= ref_band[0]) & (f <= ref_band[1])])
    if not np.isfinite(ref) or ref <= 0:
        return None
    below = p < ref * 10 ** (-drop_db / 10.0)
    # the Nyquist bin is half-weighted by the one-sided estimate and never counts; an edge needs a band
    # of at least 3 Hz below the level, not a bin
    usable = f < fs / 2 - 1e-9
    cand = np.nonzero((f > ref_band[1]) & (f <= fs / 2 - 3.0) & below)[0]
    for i in cand:
        if below[i:][usable[i:]].all():
            return float(f[i])
    return None


def run_check(source: Source, *, seconds: float = 30.0, block_samples: int = 32, mains_hz: float | None = None,
              expected_fs: float | None = None, timeout_s: float | None = None) -> dict[str, Any]:
    """Stream for ``seconds`` of source time and measure. Returns a plain dict; :func:`render_check` prints it."""
    info: StreamInfo = source.info
    fs = info.fs
    if expected_fs is None:
        expected_fs = float(info.nominal.get("expected_fs", fs))
    gate = QualityGate(fs, info.ch_names, window_s=1.0, hold_s=0.0, mains_hz=mains_hz)
    data: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    drops: list[tuple[float, int]] = []
    labels_per_ch: dict[str, dict[str, int]] = {c: {} for c in info.ch_names}
    n_blocks = n_decided = 0
    t_wall0 = time.perf_counter()
    paced = getattr(source, "pace", "wall") != "fast"
    for block in blocks(source, block_samples, timeout_s=timeout_s):
        data.append(block.data)
        if block.timestamps_s is not None:
            ts.append(np.asarray(block.timestamps_s, dtype=float))
        if block.dropped_before:
            drops.append((float(block.t_start_s), int(block.dropped_before)))
        q = gate.process(block).flags["quality"]
        n_blocks += 1
        if q["ready"]:
            n_decided += 1
            for c, row in q["per_channel"].items():
                for lab in row["labels"]:
                    labels_per_ch[c][lab] = labels_per_ch[c].get(lab, 0) + 1
        if block.t_end_s >= seconds:
            break
    wall = time.perf_counter() - t_wall0
    if not data:
        return {"ok": False, "error": "no samples arrived", "source": _source_facts(info), "seconds_requested": seconds}
    x = np.concatenate(data, axis=1)
    n = x.shape[1]
    tsa = np.concatenate(ts) if ts else None
    rows: list[dict[str, Any]] = []

    def add(item: str, status: str, value: str, note: str = "") -> None:
        rows.append({"item": item, "status": status, "value": value, "note": note})

    # rate
    fs_ts = float((tsa.size - 1) / (tsa[-1] - tsa[0])) if tsa is not None and tsa.size > 1 and tsa[-1] > tsa[0] else None
    fs_wall = float(n / wall) if paced and wall > 0 else None
    measured = fs_ts if fs_ts is not None else fs_wall
    if measured is None:
        add("sampling rate", "info", f"{fs:g} Hz claimed", "not paced and no timestamps: nothing to measure against")
    else:
        dev = abs(measured - expected_fs) / expected_fs * 100.0
        how = "from the timestamps" if fs_ts is not None else "from the wall clock"
        status = "ok" if dev < 1.0 else ("warn" if dev < 3.0 else "fail")
        add("sampling rate", status, f"{measured:.2f} Hz measured {how}, {expected_fs:g} Hz claimed ({dev:.2f} % off)",
            "over 1 % means the claimed rate is not the rate: every latency and every band edge is scaled by it" if status != "ok" else "")
    add("channel count", "info", f"{info.n_channels} EEG: {', '.join(info.ch_names)}", str(info.nominal.get("channel_names_note", "")))
    add("duration", "info", f"{n / fs:.1f} s = {n} samples / {fs:g} Hz; {n_blocks} blocks of {block_samples}", f"{wall:.1f} s of wall clock")
    # drops
    n_dropped = sum(k for _, k in drops)
    rate = n_dropped / max(n / fs, 1e-9) * 60.0
    status = "ok" if n_dropped == 0 else ("warn" if rate < 60.0 else "fail")
    add("dropped samples", status, f"{n_dropped} in {len(drops)} events ({rate:.1f} samples/min)",
        "a loss is a hole in time: the chain resets across a long one and the log records each" if n_dropped else "")
    # timestamps
    if tsa is not None and tsa.size > 2:
        per = np.diff(tsa) * fs
        gaps = detect_gaps(tsa, fs)
        bursty = float(np.mean(per < 0.1))
        add("timestamps", "info",
            f"inter-sample interval median {np.median(per):.2f}, p5 {np.percentile(per, 5):.2f}, p95 {np.percentile(per, 95):.2f}, "
            f"max {per.max():.1f} sample periods; {bursty * 100:.0f} % of intervals under 0.1 period; {len(gaps)} interval(s) over 1.5 periods",
            "identical stamps within a packet and one long interval per packet is the signature of arrival-time stamping over Bluetooth; "
            "cues must then be aligned by the sample clock, not by these stamps" if bursty > 0.2 else "")
    else:
        add("timestamps", "info", "none", "the source carries no per-sample timestamps")
    steps = getattr(source, "package_steps", None)
    if steps:
        add("package counter", "info", "increments seen " + ", ".join(f"{k}: {v}" for k, v in steps.items()),
            "1 per sample means the counter steps per sample; 0 within a packet then 1 means it steps per packet, and a lost "
            "packet is then undercounted by the packet length (TODO(confirm) per board)")
    # per channel
    per_channel: list[dict[str, Any]] = []
    flat: list[str] = []
    for i, c in enumerate(info.ch_names):
        xi = x[i]
        sd, rsd = float(xi.std()), robust_sd(xi)
        is_flat = sd < 0.5
        if is_flat:
            flat.append(c)
        l50, l60 = line_ratio(xi, fs, 50.0), line_ratio(xi, fs, 60.0)
        edge = bandwidth_edge_hz(xi, fs)
        labs = labels_per_ch[c]
        per_channel.append({"channel": c, "mean_uv": float(xi.mean()), "sd_uv": sd, "robust_sd_uv": rsd, "max_abs_uv": float(np.abs(xi).max()),
                            "flat": is_flat, "line_50_ratio": l50, "line_60_ratio": l60, "bandwidth_edge_hz": edge,
                            "gate_labels": {k: v / max(n_decided, 1) for k, v in sorted(labs.items())}})
    add("flat channels", "fail" if flat else "ok", ", ".join(flat) if flat else "none",
        "a flat channel is an open contact or a disconnected lead: nothing is measurable on it" if flat else "")
    strongest = max(per_channel, key=lambda r: max(r["line_50_ratio"] or 0, r["line_60_ratio"] or 0))
    which = 50 if (strongest["line_50_ratio"] or 0) >= (strongest["line_60_ratio"] or 0) else 60
    ratio = strongest[f"line_{which}_ratio"]
    add("line noise", "info" if ratio < 10 else "warn", f"strongest at {which} Hz on {strongest['channel']}: ratio {ratio:.1f}",
        "over 10 the gate labels the channel suspect; the band-pass removes the line but a notch on the device would have removed the diagnostic too")
    edges = [r["bandwidth_edge_hz"] for r in per_channel if r["bandwidth_edge_hz"] is not None]
    if edges:
        e = min(edges)
        add("bandwidth edge", "warn" if e < 0.4 * fs / 2 else "info", f"the spectrum falls 20 dB below its 4–20 Hz level from {e:.0f} Hz on",
            "well below Nyquist: an on-device low-pass or the transport's own filtering -- record it as a fact of the stream")
    else:
        add("bandwidth edge", "info", "none below Nyquist", "no cliff: the band is open to the rate's limit")
    gate_any = sum(sum(v.values()) for v in labels_per_ch.values())
    add("quality gate", "info", f"{n_decided} decisions; labels per channel: " + "; ".join(
        f"{c} " + (", ".join(f"{k} {v * 100 / max(n_decided, 1):.0f} %" for k, v in sorted(labels_per_ch[c].items())) or "none") for c in info.ch_names),
        "the fraction of one-second windows each label would have closed or flagged" if gate_any else "")
    return {
        "ok": all(r["status"] != "fail" for r in rows),
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "source": _source_facts(info), "seconds_requested": seconds, "seconds_received": n / fs, "wall_s": wall,
        "fs_claimed": fs, "fs_expected": expected_fs, "fs_measured_timestamps": fs_ts, "fs_measured_wall": fs_wall,
        "n_samples": n, "n_blocks": n_blocks, "drops": {"total": n_dropped, "events": drops, "per_min": rate},
        "rows": rows, "per_channel": per_channel, "versions": installed_versions(),
    }


def _source_facts(info: StreamInfo) -> dict[str, Any]:
    nominal = {k: v for k, v in info.nominal.items() if "mac" not in k.lower() and "serial" not in k.lower()}
    return {"kind": info.kind, "fs": info.fs, "ch_names": list(info.ch_names), "clock": info.clock, "units": info.units, "nominal": nominal}


def render_check(report: dict[str, Any], *, title: str | None = None) -> str:
    """The report as markdown in the shape of ``site/notes/device-<board>.md``."""
    src = report["source"]
    nom = src.get("nominal", {})
    head = title or f"Device check — {nom.get('product', src['kind'])}"
    out = [f"# {head}", "", f"Generated {report.get('generated_at', '')} by `eegloop check`. No serial number or pairing address appears "
           "in this report by construction: the source never exposes one.", ""]
    if "error" in report:
        out += [f"**{report['error']}**", ""]
        return "\n".join(out)
    out += ["## Stream", "", f"- kind: `{src['kind']}`; channels: {', '.join(src['ch_names'])}; claimed rate {src['fs']:g} Hz; clock: {src['clock']}; units: {src['units']}"]
    for k in ("board", "board_id", "product", "driver", "driver_version", "montage", "reference", "reported_filters", "units_note", "verified"):
        if k in nom:
            out.append(f"- {k.replace('_', ' ')}: {nom[k]}")
    if nom.get("descriptor_mismatch"):
        out.append("- **driver descriptor disagrees with the board table:** " + "; ".join(nom["descriptor_mismatch"]))
    out += ["", "## Checks", "", "| item | status | value | note |", "|---|---|---|---|"]
    for r in report["rows"]:
        out.append(f"| {r['item']} | {r['status']} | {r['value']} | {r['note']} |")
    out += ["", "## Per channel", "", "| channel | mean µV | SD µV | robust SD | max |x| | flat | line 50 | line 60 | bandwidth edge | gate labels |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in report["per_channel"]:
        labs = ", ".join(f"{k} {v * 100:.0f} %" for k, v in r["gate_labels"].items()) or "none"
        edge = "none" if r["bandwidth_edge_hz"] is None else f"{r['bandwidth_edge_hz']:.0f} Hz"
        out.append(f"| {r['channel']} | {r['mean_uv']:.1f} | {r['sd_uv']:.1f} | {r['robust_sd_uv']:.1f} | {r['max_abs_uv']:.0f} | "
                   f"{'yes' if r['flat'] else 'no'} | {r['line_50_ratio']:.1f} | {r['line_60_ratio']:.1f} | {edge} | {labs} |")
    v = report.get("versions", {})
    out += ["", "## Versions", "", ", ".join(f"{k} {val}" for k, val in v.items() if val not in (None, "not installed")), ""]
    return "\n".join(out)
