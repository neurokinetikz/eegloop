# Headsets

How a headset reaches `eegloop`, what is verified about each, and how the rest gets verified. This
file and `eegloop/sources/brainflow.py` (the board table) are the only places in the package that
name a device; everywhere else a device is its montage class — `frontotemporal-4` (TP9, AF7, AF8,
TP10) or `occipitotemporal-4` (O1, O2, T7, T8). The rules are `site/CONTRACTS.md`, Phase 5
addendum, "Naming a device".

## The route

One driver, one extra, one source:

```sh
pip install -e './loop[brainflow]'
eegloop devices                                   # the board table below, and whether the driver is installed
EEGLOOP_MAC_ADDRESS=… eegloop check --source brainflow:<key> --seconds 30 --mains 60 \
    --out site/notes/device-<key>.md              # the acceptance report, filed by a human
EEGLOOP_HARDWARE=<key> python -m pytest loop -m hardware -rs
```

The pairing address or serial number reaches the driver from the environment
(`EEGLOOP_MAC_ADDRESS`, `EEGLOOP_SERIAL_NUMBER`) or a constructor argument, and from nowhere else: it
is not a protocol field, it is not in `StreamInfo.nominal`, and so no resolved protocol, session log,
sidecar or report can carry it. The hardware test checks the report for its absence.

Recordings made from a headset during development live under `loop/recordings/` (gitignored) and
enter nothing shipped — not an asset, a sidecar, a notebook output or a figure — until spec §13
item 25 (registry entry, licence, public home, consent) is met.

## The board table

| key | product | driver id | montage | channels | rate | transport | verified |
|---|---|---|---|---|---|---|---|
| `muse-2` | Muse 2 | `MUSE_2_BOARD` | frontotemporal-4 | TP9, AF7, AF8, TP10 | 256 Hz | native Bluetooth LE, no dongle | 2026-09-20 |
| `muse-s` | Muse S | `MUSE_S_BOARD` | frontotemporal-4 | TP9, AF7, AF8, TP10 | 256 Hz | native Bluetooth LE, no dongle | 2026-09-20 |
| `muse-s-athena` | Muse S Athena | `MUSE_S_ANTHENA_BOARD` | frontotemporal-4 | TP9, AF7, AF8, TP10 | 256 Hz | native Bluetooth LE, no dongle | 2026-09-20 |
| `brainbit` | BrainBit | `BRAINBIT_BOARD` | occipitotemporal-4 | O1, O2, T7, T8 (the driver says T3, T4) | 250 Hz | native Bluetooth LE, no dongle | 2026-09-20 |

The `*_BLED` ids are a deprecated dongle route in the driver and the source refuses them by name.
Why these come first is a fact, stated once: each reaches Python through a free open-source driver
over native Bluetooth with no dongle (TODO(confirm): cite the driver's licence page for
"open-source").

## What `eegloop check` settles, per board

The source builds `StreamInfo` from the driver's board descriptor at runtime and records any
disagreement with the table under `nominal['descriptor_mismatch']`. The report then measures what a
table cannot state:

| fact | how it is measured | status |
|---|---|---|
| the rate | samples over the timestamps' span, and over the wall clock when paced | TODO(confirm) per board, from a report |
| drop accounting | package-number increments; the increment histogram shows whether the counter steps per sample or per packet, and the modulus is inferred at the first wrap | TODO(confirm) per board — a per-packet counter undercounts a lost packet by the packet length |
| timestamp structure | inter-sample intervals in sample periods; the fraction under 0.1 period is the packet-burst signature of arrival-time stamping | TODO(confirm) per board; cues are aligned on the sample clock regardless (`eegloop.bci.epochs`) |
| units | the table says µV; a known-amplitude signal into the input settles it | TODO(confirm) per board |
| on-device filters | the `brainbit` notch and 1–40 Hz band-pass are optional and must be OFF for the course; the driver switch and its default | TODO(confirm), then logged as a fact in `reported_filters` |
| the reference | as the driver documents it | TODO(confirm) per board |
| the AUX / optical / contact-quality rows | the descriptor's `eeg_channels` are the only rows read; whether a non-EEG row is listed among them | TODO(confirm) per board (the `muse-s-athena` optical rows are not EEG) |
| the bandwidth edge | the frequency above which the spectrum stays 20 dB under its 4–20 Hz level | measured per report; a cliff well below Nyquist is an on-device low-pass |

A filed report goes to `site/notes/device-<key>.md`. Nothing from it is quoted on a page; the pages
quote `ds-eegbci` channel subsets for every number, because a difference between devices is a
montage or bandwidth fact plus its analytic consequence, never a recording of a named product.

## Not settled

- **Which Emotiv, and how.** Raw EEG on EPOC X requires a paid Developer licence; Insight exposes it
  free (verified 2026-09-20, an access fact, not a ranking). Neither reaches this package until the
  author decides the device (spec §13 item 6) and the route is verified.
- **Neurosity Crown.** Eight channels at 256 Hz through its own Python SDK, and reportedly through
  the driver or LSL (2026-09-20); the channel names, the driver id and the route are TODO(confirm)
  before an 8-channel montage class is named (CONTRACTS rule 5).
- **LSL.** The binding (`mne-lsl`, which pulls MNE, or `pylsl`, which needs `liblsl`) is decided at
  a named release (`pyproject.toml`, the `lsl` extra); `open_source('lsl:…')` says so until then.
