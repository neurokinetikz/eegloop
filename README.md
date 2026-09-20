# eegloop — streaming EEG loops

The package a learner builds a neurofeedback or BCI application on, for the Scalp to Source course.
It is a sibling of `pipelines/eegpipe`, built the same way — `pyproject.toml`, pinned to
`notebooks/requirements.txt`, tests that run on synthetic data with no network and no device — and
deliberately not part of it: `eegpipe` is a batch pipeline over a whole recording and depends on MNE;
this is a loop over blocks of a stream and depends on numpy and scipy.

**This is version 0.1.** What is here: the block contract, the ring buffer, causal filters with their
state carried across blocks, the latency budget, the loopback probe that measures a loop's delay
instead of trusting the arithmetic — a port of the real-time section of
`notebooks/_shared/helpers_l7.py` into something installable, multichannel and unit-tested — and, on
top of it, the sources (a synthetic stream with planted answers; replay of arrays, `.npz`, `.csv`, the
site's own `.bin` + sidecar assets, and MNE formats through an extra; cues on the source clock) and
the online steps (a quality gate that runs first on the raw block, re-referencing, frozen spatial
filters, envelopes, features). What is not here yet, and arrives in later phases: the feedback loop,
the BCI loop, the session runner, the command line, and the hardware sources. The proposal that lays
them out is `site/notes/proposal-phase5-consumer-nf-bci.md`.

## Quick start (nothing is downloaded, no device)

```sh
pip install -e './loop[dev]'
python -m pytest loop
```

The budget lesson L7.3 spends its length on, computed from a chain's own declared delays:

```python
import numpy as np
from eegloop import Block, CausalFIR, Chain, fir_taps, latency_budget_for, print_budget, measure_loop_delay

fs, band = 160.0, (8.0, 12.0)
chain = Chain([CausalFIR(fir_taps(129, band, fs), n_channels=1)])
print_budget(latency_budget_for(chain, fs=fs, block_samples=32, processing_ms=10.0))
# filter 400.00 + buffer 200.00 + processing 10.00 = 610.00 ms worst case, 510.00 mean

probe = measure_loop_delay(lambda: Chain([CausalFIR(fir_taps(129, band, fs), 1)]),
                           fs=fs, block_samples=32, centre_hz=10.0)
print(f"measured filter delay {probe['filter_ms']:.2f} ms, value delay {probe['value_ms']:.2f} ms")
```

## Import without installing

The notebooks find the package by walking up from their own directory, as they do for `eegpipe`:

```python
import sys
from pathlib import Path
loop = next(d / "loop" for d in (Path.cwd(), *Path.cwd().parents) if (d / "loop" / "eegloop" / "__init__.py").exists())
sys.path.insert(0, str(loop))
import eegloop
```

## Modules

| Module | What it holds |
|---|---|
| `eegloop.block` | `Block` — `n_channels × n_samples` in µV with `seq`, `sample_index` (drops included), `t_start_s`, per-sample `timestamps_s`, `dropped_before`, `flags`. Frozen. `StreamInfo` — what a source knows before the first block, including any filtering the device applied. |
| `eegloop.stream` | `detect_gaps(timestamps, fs)`, `DropAccount`, `Reblocker` — fixed blocks out of whatever a driver delivered, losses carried to the next block or recorded inside it. |
| `eegloop.ring` | `RingBuffer` — fixed capacity, never reallocates. |
| `eegloop.steps` | The `OnlineStep` contract, `Chain`, and `CANONICAL_ONLINE_ORDER`. |
| `eegloop.steps.causal_filter` | `fir_taps`, `butter_sos`, `CausalFIR`, `CausalSOS`, `Notch` — state carried across blocks; block-wise output equals whole-signal filtering. |
| `eegloop.latency` | `latency_budget` (by tap count or sections), `latency_budget_for(chain)`, group-delay and look-ahead functions, `BUFFER_CONVENTIONS`, `print_budget`. |
| `eegloop.probe` | `make_probe`, `energy_centroid`, `run_offline(chain, x)`, `measure_loop_delay(chain_factory)`. |
| `eegloop.sources` | `Source` — the one interface (`info`, `start`, `read`, `stop`, `clock_now`, `done`); `open_source('synthetic:<scenario>' \| 'replay:<path>')`; `blocks(source, block_samples)`; `ListMarkers` — cues released as the source clock passes them. |
| `eegloop.sources.synthetic` | `make_synthetic_stream` and `SyntheticSource` with `SCENARIOS` (`clean`, `alpha-schedule`, `blinks`, `gaps`, `flat-channel`, `line-noise`, `pops`, `ssvep`, `mi-2class`, `p300`) and a `truth` dict naming every planted answer: alpha bursts, frontal blinks, a dropped-sample gap that shows in the timestamps, clock drift in ppm, a flicker component, cues with an evoked response and a lateralised power change, single-channel electrode pops, a flat channel. |
| `eegloop.sources.replay` | `ReplaySource` — arrays, `.npz`, `.csv`, `.bin` + JSON sidecar (`from_asset`), `.fif`/`.edf`/`.bdf` via the `mne` extra; `pace='wall'` releases samples on the clock; `inject_gaps` removes samples so the timestamps jump; reads never straddle a gap. `read_recording` is the reader alone. |
| `eegloop.steps.quality` | `QualityGate` — runs **first, on the raw block**; `flat`, `pop`, `blink` (frontal channels only, and it says when there are none), `emg`, `line-noise`, `gap`; thresholds ported from `data/scripts/detectors.py` and cited; a three-state verdict with a hold; a *decision* delay, never a signal delay. |
| `eegloop.steps.reference` | `Reference('none' \| 'average' \| 'channels')` — zero delay, exact; reports its rank cost. |
| `eegloop.steps.spatial` | `SpatialFilter` — frozen weights applied per block; `from_projector`, `from_ica(unmixing, mixing, exclude)`; zero delay, exact. |
| `eegloop.steps.envelope` | `BlockRMS` (exact: the block is the delay, already in the buffer row), `Smoother` (no single delay — measured), `BandPower` (Welch over a ring — measured). Each writes `flags['envelope']`. |
| `eegloop.steps.features` | `FeatureExtractor('log-var' \| 'band-power' \| 'band-ratio')` → `flags['features']`; zero delay, exact. |
| `eegloop.versions` | `installed_versions`, `check_pins`, `seed_everything`, `config_hash` — copied from `eegpipe`. |

## The loop, and why it is that shape

**Every step declares what it costs in time — and only where that is exact.** A linear-phase FIR
delays every frequency by `(N − 1)/2` samples and says so; a Butterworth has a delay at a stated
frequency, summed over its sections because expanding the cascade is numerically unstable at order 8
(it reports the steeper filter as *faster*). Those are rows in the budget. A one-pole smoother has no
single delay, a quality window delays the decision rather than the signal, and a Welch window's
centroid sits at half its length; those are not rows. `latency_budget_for` lists them under
`measured` and the probe fills that in. Nothing inexact is ever added into `total_ms`.

**Quality runs first, on the raw block.** `CANONICAL_ONLINE_ORDER` begins with `quality` because a
gate placed after an 8–12 Hz band-pass cannot see line noise, muscle or a flat channel: the filter has
removed exactly the evidence the gate needs. That is lesson L7.5's point about devices that notch
before the file exists, and a chain that filtered before it gated would have built the same mistake
into its own fixed order. The order is fixed for the reason `eegpipe` fixes its offline order: a
pipeline whose order is a parameter is a pipeline whose order nobody has argued.

**A gap resets the chain.** `Block.dropped_before` is on the block so a filter can see it; a chain
built with `reset_on_gap_samples` forgets its state across a hole longer than its memory and flags the
block, because carrying `zi` across 400 ms of nothing is the online form of
`pf-filter-across-boundaries`.

**The buffer term has two defensible conventions and the budget prints both.** `block-period`
(`B/fs`, the default and the one the site teaches) and `first-to-last` (`(B − 1)/fs`); the reason is
in `BUFFER_CONVENTIONS`. Quoting one while meaning the other is the commonest error in a latency
figure, and it is one sample.

## Two words that mean two things in this repository

**Pop.** `data/scripts/detectors.py` defines an electrode pop as an abrupt *sustained* step on one
channel; `notebooks/_shared/helpers.py` defines it as a sample-to-sample *jump* on one channel; and
`pipelines/eegpipe/synthetic.py` plants a 20-ms transient under the same name. The gate here fires
`pop` on either of the first two criteria, cites both, and the synthetic stream plants the first —
because the gate is a port of `detectors.py`, it plants what that criterion means. The divergence
from `eegpipe.synthetic` is recorded, not reconciled.

**Delay.** A filter's group delay is a fact about the signal. A gate's window, a smoother's time
constant and a Welch window are facts about a *decision*. The budget keeps them apart.

## Tests

`python -m pytest loop` — no network, no device, seconds. `tests/test_latency.py` reads the SciPy
reference the site's latency widget is itself tested against
(`site/public/data/widgets/w-latency-budget/traces.json`) rather than retyping its numbers, and
asserts the L7.3 pipeline at 610.00 ms causal and 1010.00 ms zero-phase-live.
`tests/test_parity_helpers_l7.py` holds this package to `notebooks/_shared/helpers_l7.py` to 1e-12;
under CI that import is hard, so the guard cannot skip itself away. `tests/test_neutrality.py`
applies the site's private-path and vendor-language patterns to this directory. Hardware and network
tests carry markers and never run by default.

## Dependencies

Core: `numpy>=2.1,<3`, `scipy>=1.15,<2` — the ranges `notebooks/requirements.txt` uses. MNE is an
extra (`mne`) for reading recording formats, not a dependency of the loop. Hardware drivers are extras
the later sources import lazily and fail without by naming the package. `dev` carries `mne==1.10.2`
only so the parity test runs.

## What this package does not do

It does not download data, preprocess a recording offline, or evaluate a decoder across subjects —
those are the notebooks, `eegpipe` and `helpers_l7`'s benchmarking harness. It does not name a
headset outside one board table (added with the hardware sources), and it never ships a recording.
