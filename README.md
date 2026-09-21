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
filters, envelopes, features); and, on top of those, the neurofeedback loop — lesson L7.3's five
decisions as five objects, the sham modes, a protocol file that names every decision and is validated
all at once, a session log a notebook opens with numpy alone, a recorder that writes the site's own
asset format, a command line, and the valid learning test beside the invalid one; and the BCI loop —
epochs cut on a stated clock, three decoders fitted in cue order and *frozen* into arrays that replay
them exactly without scikit-learn, a sliding and a cue-locked way to apply one, a steady-state
detector, and a protocol that calibrates, fits, applies and scores in one run; and the headset source —
one driver behind the same interface, a board table of verified facts, and `eegloop check`, the
acceptance report that measures on a live stream what a table cannot state. The proposal that lays it
all out is `site/notes/proposal-phase5-consumer-nf-bci.md`; `HARDWARE.md` is the device page.

## Quick start (nothing is downloaded, no device)

```sh
pip install -e './loop[dev]'
python -m pytest loop
eegloop run --protocol loop/configs/alpha-up-synthetic.yaml      # a 2-min alpha session, no device
eegloop budget --protocol loop/configs/alpha-up-synthetic.yaml   # what its chain declares
eegloop probe  --protocol loop/configs/alpha-up-replay.yaml      # what the loopback probe measures
eegloop run --protocol loop/configs/mi-2class-synthetic.yaml     # calibrate, fit, freeze, apply, score
eegloop run --protocol loop/configs/eo-ec-synthetic.yaml         # the same, on a state four channels can decode
eegloop run --protocol loop/configs/alpha-up-oc4-replay.yaml     # the alpha loop over the shipped four-channel asset
eegloop check --source synthetic:clean --seconds 10 --mains 60   # the acceptance report, on a stream with no device
```

`run` writes `sessions/<name>/` beside the protocol: `session.json` (versions, the protocol's hash,
the budget with the processing row *measured*, the sealed sham token), `events.jsonl` (gate closures,
rewards, cues, gaps, phases, the baseline as fixed), `signal.npz` (the per-block series) and
`protocol-resolved.json` (every decision, defaults filled in). Nothing in it names the machine.
`loop/examples/alpha_bar.py` is the smallest application: the same run with a presenter you wrote.
A BCI protocol adds `decoder.npz` — the frozen decoder, arrays and a report, loadable with numpy
alone — and its decisions are scored against the cues of the apply phase with the chance band beside
the number; `loop/examples/cue_switch.py` is the two-way switch on planted imagery, `loop/examples/eo_ec_switch.py` the same loop on eyes closed against eyes open — the state a
four-channel headband can actually decode — and `loop/examples/blink_switch.py` an EOG switch,
labelled as such. Lesson L7.16 builds an application from these three and `tests/test_examples.py`
runs them the way a learner does, on the synthetic stream and on the shipped asset.

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
| `eegloop.sources.brainflow` | `BrainFlowSource(board_key)` — the driver behind the `Source` interface; `BOARDS`, the one table that names a device; blocks split at every loss the package counter shows, timestamps rebased to the session; pairing details from the environment, never from a protocol or into a log. |
| `eegloop.check` | `run_check(source, seconds=…)` and `render_check` — rate measured against rate claimed, drops, timestamp structure, per-channel DC/spread/flat/line/bandwidth-edge and gate-label prevalence, verdicts in `first_look_checks`' names. Works on any source. |
| `eegloop.sources.replay` | `ReplaySource` — arrays, `.npz`, `.csv`, `.bin` + JSON sidecar (`from_asset`), `.fif`/`.edf`/`.bdf` via the `mne` extra; `pace='wall'` releases samples on the clock; `inject_gaps` removes samples so the timestamps jump; reads never straddle a gap. `read_recording` is the reader alone. |
| `eegloop.steps.quality` | `QualityGate` — runs **first, on the raw block**; `flat`, `pop`, `blink` (frontal channels only, and it says when there are none), `emg`, `line-noise`, `gap`; thresholds ported from `data/scripts/detectors.py` and cited; a three-state verdict with a hold; a *decision* delay, never a signal delay. |
| `eegloop.steps.reference` | `Reference('none' \| 'average' \| 'channels')` — zero delay, exact; reports its rank cost. |
| `eegloop.steps.spatial` | `SpatialFilter` — frozen weights applied per block; `from_projector`, `from_ica(unmixing, mixing, exclude)`; zero delay, exact. |
| `eegloop.steps.envelope` | `BlockRMS` (exact: the block is the delay, already in the buffer row), `Smoother` (no single delay — measured), `BandPower` (Welch over a ring — measured). Each writes `flags['envelope']`. |
| `eegloop.steps.features` | `FeatureExtractor('log-var' \| 'band-power' \| 'band-ratio')` → `flags['features']`; zero delay, exact. |
| `eegloop.versions` | `installed_versions`, `check_pins`, `seed_everything`, `config_hash` — copied from `eegpipe`. |
| `eegloop.feedback.signal` | `SignalSpec` — decision 1, *what the signal is*, exactly: band, channels, reference, taps, derivation, smoothing; `describe()` is the sentence CRED-nf item 4 asks for. `build_chain` — gate (raw) → reference → causal FIR → RMS → smoother. |
| `eegloop.feedback.baseline` | `FixedBaseline.from_values` (the threshold does not move) and `AdaptiveBaseline` (it follows the participant). Each carries a `note` saying what it does to a learning claim. |
| `eegloop.feedback.reward` | `ContinuousMapping` (z → [0, 1]) and `ThresholdReward` (dwell, refractory, what a miss shows). Time comes from the block clock, never `time.time()`. |
| `eegloop.feedback.sham` | `SHAM_MODES` and `CREDNF_ITEMS` verbatim from `helpers_l7`; `ShamPolicy` — veridical, yoked (replays a donor session on the same schedule), band, inverted — sealed into the log as a token the seed unblinds. |
| `eegloop.feedback.loop` | `FeedbackLoop` — one block: gate → value → sham → baseline → z → mapping → what is shown; gated blocks freeze or zero the display, never a cleaned guess; the processing row is measured with `perf_counter`. `budget()` fills that row in. |
| `eegloop.session.protocol` | `Protocol` and its blocks (`source`, `signal`, `quality`, `baseline`, `reward`, `sham`, `phases`, `output`, `versions`); `load_protocol` (YAML or JSON, dotted overrides); `ProtocolError` lists every problem at once in `eegpipe`'s message format, with did-you-mean. |
| `eegloop.session.log` | `SessionLog` and `SessionReader` — the files above, every string scrubbed of absolute paths (`scrub_paths`, from `eegpipe.run`); the reader needs numpy alone; `.feedback` is the donor a yoked sham replays. |
| `eegloop.session.record` | `Recorder` — the stream as it arrived, in spec §4.5's `.bin` + sidecar, so a learner's recording opens exactly as a shipped asset does (`ReplaySource.from_asset`); `export_fif` behind the `mne` extra. The sidecar says `license: private`. |
| `eegloop.session.runner` | `run_protocol(protocol, source=None, presenter=None)` → `SessionResult`; `build_source` (`brainflow` opens a headset through the driver extra; the `lsl` kind waits on the binding decision and says so). |
| `eegloop.present` | `Presenter` — `start`, `update(shown, z, gated, t_s, phase)`, `stop`. `ConsolePresenter` proves the loop is alive; `CallbackPresenter` is the seam where an application begins. |
| `eegloop.analysis` | `learning_test` — one change score per session, a sign-flip permutation test across sessions (the session is the unit; exact below 13); `naive_trend` — `helpers_l7.within_session_trend`, kept and labelled **invalid**; `crednf_report` — the six items, each *satisfied* / *not satisfiable alone* / *unsatisfied*, with why. |
| `eegloop.cli` | `eegloop run \| validate \| budget \| probe \| check \| devices \| versions`; exit 2 lists every protocol problem, exit 1 a `check` with a failed row. |
| `eegloop.bci.pipelines` | `csp_lda`, `riemann_ts`, `xdawn_lda` as scikit-learn pipelines from the `decode` extra, mirroring `helpers_l7`'s families by name; `PIPELINE_NOTES` states the one divergence (pyriemann's CSP, so no MNE). |
| `eegloop.bci.epochs` | `cut_epochs` (offline) and `EpochCutter` (online) — the same window to the sample; `clock='sample'` or `'timestamps'`, because cue alignment is a property of the clock the cues are on; epochs over a gap or a gated stretch are dropped and counted. `Calibrator` harvests labelled epochs as cues arrive. |
| `eegloop.bci.decoder` | `fit_frozen` — folds in cue order, never shuffled; the chance band (`chance_interval` = `helpers_l6.chance_band`); then `freeze`: the fitted pipeline's covariance estimator, spatial filters, tangent-space reference, scaler and classifier as arrays. `FrozenDecoder.predict_proba` replays it in numpy to 1e-6 (checked at fit time), `save`/`load` an `.npz` with nothing pickled; `latency_samples` = the window. `temporal_split` — disjoint *and* non-adjacent. |
| `eegloop.bci.stream` | `StreamingPosterior` (a window every step, smoothing measured not summed), `DwellDecision` (threshold, dwell, refractory — a choice, not a cost), `BCILoop` (calibrate → fit → apply in `sliding` or `cue-locked` mode; decisions scored against cues; a gated block stops the posterior and marks the stretch unusable). |
| `eegloop.bci.ssvep` | `cca_correlation`, `ssvep_cca`, `ssvep_decide`, `SSVEPDetector` — canonical correlation with sine/cosine references, one QR; `SSVEP_NOTE` on the standing-alpha caveat the synthetic scenario makes measurable. |

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

**Suspect is fed back and said so; unusable is closed.** The gate's verdict has three states. A
visible mains line is the normal state of an un-notched recording and the band-pass removes it, so
`line-noise` marks a channel *suspect* and the block is still fed back with that label in the log. A
flat channel, a pop, a blink, a muscle burst or a gap marks it *unusable*, and the display is frozen
or zeroed for the block and a hold afterwards — never fed a cleaned estimate. A pop is a *local*
event: on a one-channel stream its locality cannot be tested and the gate makes no pop claim (the
shipped eyes-closed trace has 280-µV alpha waves that clear the jump threshold on their own).

**The control condition is in the library, not the notebook.** A loop without a sham mode cannot
support the claim its participant is being asked to believe. `ShamPolicy` runs the four modes
`helpers_l7` names, seals the mode into the log as a token only the seed unblinds, and the yoked mode
replays another session's per-block values on the same schedule — which is why the session log has a
reader. One divergence is recorded: `helpers_l7.sham_feedback` inverts about the whole-session mean,
which needs the future; the live loop inverts about the calibration baseline, the centre that exists
when the value is shown.

**The test that is valid is beside the one that is not.** `naive_trend` is the analysis an
uncontrolled report runs — OLS on autocorrelated per-block values — and it is kept, labelled
`valid: False`, because the course shows what it does wrong. `learning_test` scores each session
once and permutes signs across sessions, so nothing inside a session can inflate it; it answers only
whether the targeted signal moved, and `crednf_report` keeps that apart from any outcome claim.

**A decoder is fitted once and frozen.** `fit_frozen` cross-validates in cue order, fits on every
calibration epoch, then turns the pipeline into arrays — CSP filters and a linear classifier, or a
tangent-space reference, a scaler and a classifier, or Xdawn filters and a classifier — and checks
that the arrays reproduce the pipeline's posteriors before returning. The loop runs the arrays. So the
application needs no scikit-learn, the decoder cannot quietly refit on what it sees, and what was
applied is exactly what was reported. The window is its delay, stated as such; the dwell before a
decision is a choice and is listed as one.

**Epochs are cut on the clock the cues are on.** A block carries its sample index and its
timestamps, and they drift apart — the synthetic stream's by 200 ppm, six samples in two minutes.
Cues from a replayed or synthetic recording are on the sample clock; cues an application stamps from
the wall clock are on the timestamps. The cutter is told which and a test shows the misalignment
when it is told wrong. That is lesson L7.11's point, made a parameter.

**The alpha rhythm looks like an SSVEP at 10 Hz.** The synthetic `ssvep` scenario shows it: the
canonical correlation with a 10 Hz reference is about 0.5 in every window, flicker or not, because a
standing posterior alpha *is* a 10 Hz oscillation. Stimulus frequencies belong outside 8–12 Hz, and a
score means nothing without a no-stimulus baseline. `SSVEP_NOTE` says so wherever the detector is
described.

**A device is named once.** The board table below is copied from `eegloop.sources.brainflow.BOARDS`;
`HARDWARE.md` holds the per-device facts with their verified dates and the list of what a report
must still settle. Everywhere else a device is its montage class. `tests/test_neutrality.py` holds the
package to that with an allowlist of exactly these three files.

| key | product | driver id | montage | channels | rate | transport | verified |
|---|---|---|---|---|---|---|---|
| `muse-2` | Muse 2 | `MUSE_2_BOARD` | frontotemporal-4 | TP9, AF7, AF8, TP10 | 256 Hz | native Bluetooth LE, no dongle | 2026-09-20 |
| `muse-s` | Muse S | `MUSE_S_BOARD` | frontotemporal-4 | TP9, AF7, AF8, TP10 | 256 Hz | native Bluetooth LE, no dongle | 2026-09-20 |
| `muse-s-athena` | Muse S Athena | `MUSE_S_ANTHENA_BOARD` | frontotemporal-4 | TP9, AF7, AF8, TP10 | 256 Hz | native Bluetooth LE, no dongle | 2026-09-20 |
| `brainbit` | BrainBit | `BRAINBIT_BOARD` | occipitotemporal-4 | O1, O2, T7, T8 | 250 Hz | native Bluetooth LE, no dongle | 2026-09-20 |

## The protocol schema

A protocol is a YAML (or JSON) file with the blocks below; `eegloop validate --protocol f.yaml` checks
it all at once and prints every problem with its dotted path (`reward.hi_z: expected a number, got
'high'`), and `eegloop validate --json` prints it resolved, every default filled in — which is also
what `protocol-resolved.json` in a session directory holds and what the log's `protocol_hash` is
computed over. `load_protocol(path, {"source.pace": "wall"})` applies dotted overrides before
validation. Every value below is the default; `null` means "not set".

| block | field | default | meaning |
|---|---|---|---|
| — | `name` | *(required)* | the session directory's name |
| — | `seed` | *(required)* | seeds the synthetic source and the sham token; the log records it |
| — | `block_samples` | `32` | samples per block: a row of the budget and the loop's update period |
| — | `processing_ms` | `10.0` | the stated processing time; the run replaces it with the measured one |
| — | `buffer_convention` | `block-period` | `block-period` or `first-to-last` (`BUFFER_CONVENTIONS`) |
| `source` | `kind` | `synthetic` | `synthetic` \| `replay` \| `brainflow` \| `lsl` (the last waits on its binding) |
| | `scenario` | `alpha-schedule` | synthetic only: a key of `SCENARIOS` |
| | `path` | `null` | replay only, relative to the protocol file: `.npz`, `.csv`, a `.bin` with its sidecar, `.fif`/`.edf` via the `mne` extra |
| | `board` | `null` | hardware only: a key of the board table (`eegloop devices`); pairing details come from the environment, never from here |
| | `timeout_s` | `15.0` | hardware only: how long to wait for the first samples |
| | `pace` | `fast` | `wall` releases samples on the clock; `fast` does not wait |
| | `loop` | `false` | replay a recording again when it ends |
| | `seed` | `null` | synthetic only; `null` means the protocol's seed |
| | `duration_s` | `null` | synthetic only; `null` means the phases' total |
| `signal` | `band` | `[8.0, 12.0]` | Hz |
| | `channels` | `[]` | empty: every channel of the source |
| | `reference` | `none` | `none` \| `average` \| `channels` |
| | `reference_channels` | `[]` | with `reference: channels` |
| | `n_taps` | `129` | the causal FIR's length; its exact delay is `(n_taps − 1) / 2` samples |
| | `derivation` | `rms` | `rms` \| `log-power` — the per-block value |
| | `control_band` | `[16.0, 20.0]` | the band `sham-band` computes from |
| | `smooth_s` | `0.0` | the smoother's time constant; it is latency, and the probe measures it |
| `quality` | `enabled` | `true` | the gate runs first, on the raw block |
| | `window_s` | `1.0` | the decision window (listed under the budget's `decision`) |
| | `hold_s` | `0.5` | how long the display stays held after a violation |
| | `mains_hz` | `null` | 50 or 60: the line-noise label needs it |
| | `policy` | `freeze` | `freeze` the display, or show `zero` |
| `baseline` | `mode` | `fixed` | `fixed` from the calibration phase, or `adaptive` (what that does to a learning claim: L7.13) |
| | `centre` | `median` | `median` \| `mean` |
| | `spread` | `mad` | `mad` \| `sd` |
| | `window_s` | `30.0` | adaptive only |
| | `percentile` | `50.0` | adaptive only |
| `reward` | `mode` | `continuous` | `continuous` (a bar) \| `threshold` (an event) |
| | `lo_z`, `hi_z` | `-1.0`, `2.0` | continuous: the z range mapped onto the bar |
| | `threshold_z` | `1.0` | threshold: the crossing |
| | `dwell_s` | `0.5` | threshold: how long above before an event |
| | `refractory_s` | `1.0` | threshold: at most one event per this many seconds |
| | `on_miss` | `hold` | `hold` the bar, or show `zero` |
| `sham` | `mode` | `veridical` | `veridical` \| `sham-band` \| `sham-yoked` \| `sham-inverted` (`SHAM_MODES`); sealed into the log as a token |
| | `donor` | `null` | `sham-yoked`: a session directory, relative to the protocol file |
| `bci` | *(absent)* | `null` | present: calibrate-then-apply instead of feedback |
| | `pipeline` | `csp_lda` | `csp_lda` \| `riemann_ts` \| `xdawn_lda` |
| | `classes` | `["left", "right"]` | the cue labels |
| | `tmin_s`, `tmax_s` | `0.5`, `3.5` | the window after each cue; the window is the decoder's delay |
| | `band` | `[8.0, 30.0]` | Hz |
| | `n_taps` | `129` | |
| | `n_components` | `4` | CSP filters, or Xdawn filters per class |
| | `mode` | `sliding` | `sliding` (a posterior every step) \| `cue-locked` (one per cue) |
| | `step_samples` | `32` | sliding |
| | `smooth_s` | `0.5` | sliding: posterior smoothing (measured, not summed) |
| | `threshold`, `dwell_s`, `refractory_s` | `0.7`, `0.5`, `1.0` | sliding: the dwell decision |
| | `folds` | `5` | cross-validation folds, in cue order |
| | `grace_s` | `1.0` | how long after `tmax` a decision still counts for a cue |
| `phases` | list of `{kind, duration_s, label}` | `[]` | `kind`: `rest` \| `calibrate` \| `train` \| `test`; a fixed baseline needs a `rest` or `calibrate` phase first; a bci protocol needs a `calibrate` phase before its first `train`/`test` |
| `output` | `dir` | `sessions` | where session directories go, beside the protocol |
| | `record_raw` | `false` | also write the raw stream as a `.bin` + sidecar |
| `versions` | `on_mismatch` | `warn` | `warn` \| `error` \| `ignore` when a pinned version differs |
| | `pins` | `{}` | `{package: version}` |

The six shipped protocols in `configs/` are complete examples: three feedback (`alpha-up-synthetic`,
`alpha-up-replay`, `alpha-up-oc4-replay` over the four-channel asset) and three BCI (`mi-2class-synthetic`,
`p300-synthetic`, `eo-ec-synthetic`).

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
applies the site's private-path and vendor-language patterns to this directory — including to the
session files a test writes. `tests/test_session.py` runs whole protocols: the feedback rises on the
planted bursts and nowhere else, the yoked participant sees the donor's values block for block, a
recording re-opens as an asset, and no file names the machine. `tests/test_protocol.py` asserts the
same message formats `pipelines/tests/test_config.py` does. `tests/test_bci.py` holds the frozen
arithmetic to scikit-learn's and pyriemann's, the online cut to the offline one to the sample, the
chance band to `helpers_l6`, and the decoders to the planted answers; `tests/test_bci_session.py` runs
the two BCI protocols end to end. `tests/test_brainflow_source.py` drives the headset source with a
fake board that has the driver's interface — a planted loss of five packets, a counter that wraps,
stamps identical within a packet — and `tests/test_check.py` runs the report on planted facts.
`tests/test_hardware.py` is the one test that needs a headset: it runs only with
`EEGLOOP_HARDWARE=<key>` and never by default.

## Dependencies

Core: `numpy>=2.1,<3`, `scipy>=1.15,<2` — the ranges `notebooks/requirements.txt` uses. MNE is an
extra (`mne`) for reading recording formats and `export_fif`, not a dependency of the loop. PyYAML is
imported lazily by `load_protocol` and named when missing; a JSON protocol needs nothing. Fitting a
decoder needs the `decode` extra (scikit-learn, pyriemann), imported lazily and named when missing;
*applying* a frozen decoder needs neither. The headset driver is the `brainflow` extra, imported
lazily and named when missing; the fake-board tests need nothing. `dev` carries `mne==1.10.2` so the
parity tests run, PyYAML so the shipped protocols load, and the `decode` pair so the BCI tests run
rather than skip.

## What this package does not do

It does not download data, preprocess a recording offline, or evaluate a decoder across subjects —
those are the notebooks, `eegpipe` and `helpers_l7`'s benchmarking harness. It does not name a
headset outside the board table, `HARDWARE.md` and this file's copy of the table, and it never ships
a recording.
