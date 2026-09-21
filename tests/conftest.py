"""Fixtures. Nothing here downloads anything or touches a device.

The package is imported from the repository without being installed: loop/ is the source root,
exactly as pipelines/tests does for eegpipe and the notebooks do for notebooks/_shared.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

LOOP = Path(__file__).resolve().parents[1]
REPO = LOOP.parent
if str(LOOP) not in sys.path:
    sys.path.insert(0, str(LOOP))

#: The pipeline lesson L7.3 quotes and helpers_l7.LATENCY_SPEC pins: 160 Hz, 8-12 Hz, 129 taps,
#: 32-sample blocks, 10 ms stated processing, group delay read at 10 Hz. Every latency number in the
#: course is quoted for this pipeline, so it is the fixture every cross-check runs on.
SPEC = {"fs": 160.0, "band": (8.0, 12.0), "n_taps": 129, "block_samples": 32,
        "processing_ms": 10.0, "ref_hz": 10.0}
SEED = 20260920

#: The SciPy reference the site's latency widget is tested against (data/scripts/make_latency_fixtures.py
#: wrote it; data/manifest.json tracks its sha256). Read here rather than retyped: retyping is the
#: drift the manifest exists to prevent. loop/assets/ mirrors the site's copy so a standalone checkout
#: of this package has it; tests/test_assets_mirror.py holds the two copies byte-equal.
REFERENCE = LOOP / "assets" / "traces.json"


@pytest.fixture(scope="session")
def spec() -> dict:
    return dict(SPEC)


@pytest.fixture(scope="session")
def reference() -> dict:
    if not REFERENCE.exists():
        pytest.skip(f"shipped reference not present: {REFERENCE.relative_to(LOOP)}")
    return json.loads(REFERENCE.read_text(encoding="utf-8"))["reference"]


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


@pytest.fixture
def signal(rng: np.random.Generator) -> np.ndarray:
    """Three channels, 20 s at the spec rate: pink-ish noise at a few tens of µV plus a 10 Hz line."""
    fs, n = SPEC["fs"], int(20 * SPEC["fs"])
    t = np.arange(n) / fs
    white = rng.standard_normal((3, n))
    # a one-pole low-pass makes the noise look like EEG rather than like a hiss
    pink = np.zeros_like(white)
    for c in range(3):
        acc = 0.0
        for i in range(n):
            acc = 0.9 * acc + white[c, i]
            pink[c, i] = acc
    pink *= 15.0 / pink.std()
    alpha = 20.0 * np.sin(2 * np.pi * 10.0 * t)[None, :] * np.array([[1.0], [0.6], [0.3]])
    return pink + alpha
