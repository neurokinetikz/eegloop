# Shipped replay assets (a mirror)

Five files the package's replay protocols and its latency test read, copied byte for byte from the site's
`site/public/data/widgets/w-latency-budget/` so that a standalone checkout of `eegloop` runs without the
site beside it. The site's copies are the registered ones — `data/manifest.json` tracks their SHA-256 and
provenance — and `tests/test_assets_mirror.py` holds these equal to them whenever the site is present.

| file | what it is | written by |
|---|---|---|
| `alpha-ec.bin`, `alpha-ec.json` | 10 s of `ds-eegbci` S001, channel O1, eyes closed (R02, from 50.0 s), 160 Hz, unfiltered: the trace lesson L7.3 and `configs/alpha-up-replay.yaml` replay | `data/scripts/make_latency_fixtures.py` |
| `oc4-ec.bin`, `oc4-ec.json` | 60 s of `ds-eegbci` S001 R02 (eyes closed) on O1, O2, T7, T8 — the occipitotemporal-4 geometry cut from the 64-channel cap — 160 Hz, per-channel mean removed, nothing else: `configs/alpha-up-oc4-replay.yaml`, lesson L7.16's rubric and the site's `feedback` widget mode replay it | `data/scripts/make_consumer_replay.py` |
| `traces.json` | the index of the site's latency traces and SciPy's own group delays, look-aheads and stop-band attenuations for twelve filter designs (scipy 1.15.3): `tests/test_latency.py` checks this package against it rather than retyping the numbers | `data/scripts/make_latency_fixtures.py` |

Format: Float32 little-endian, channels × samples, with a JSON sidecar naming channels, rate, units, subject,
run, window and every processing step (the site's section 4.5 format; `ReplaySource.from_asset` reads it).

**Licence and attribution.** `ds-eegbci` is the EEG Motor Movement/Imagery Dataset (EEGMMIDB), PhysioNet
v1.0.0, **ODC-By 1.0**: Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., & Wolpaw, J.R. (2004),
BCI2000: A General-Purpose Brain-Computer Interface (BCI) System, IEEE Trans Biomed Eng 51(6), 1034–1043;
and Goldberger, A.L. et al. (2000), PhysioBank, PhysioToolkit, and PhysioNet, Circulation 101(23), e215–e220.
Dataset DOI 10.13026/C28G6P. These files are derivatives (cropped, channel-selected, demeaned) and carry the
same licence; the sidecars repeat the attribution. The `.bin` files are a montage emulation on a gel research
recording, not a consumer recording (the sidecars' `montage_note`).
