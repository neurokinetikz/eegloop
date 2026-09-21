"""loop/assets/ mirrors the site's shipped replay assets so a standalone checkout of this package runs its
replay protocols and its latency test. The site's copies are the registered ones (data/manifest.json tracks
their checksums); these must stay byte-identical to them. Skips when the site is not beside the package."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

LOOP = Path(__file__).resolve().parents[1]
ASSETS = LOOP / "assets"
SITE = LOOP.parent / "site" / "public" / "data" / "widgets" / "w-latency-budget"
MIRRORED = ("alpha-ec.bin", "alpha-ec.json", "oc4-ec.bin", "oc4-ec.json", "traces.json")


def test_every_mirrored_asset_is_present_and_documented():
    for name in MIRRORED:
        assert (ASSETS / name).exists(), name
    readme = (ASSETS / "README.md").read_text(encoding="utf-8")
    for name in MIRRORED:
        assert name in readme, f"assets/README.md does not list {name}"
    for name in ("alpha-ec.json", "oc4-ec.json"):
        side = json.loads((ASSETS / name).read_text(encoding="utf-8"))
        assert side["dataset"] == "ds-eegbci" and "license" in side, name


@pytest.mark.skipif(not SITE.exists(), reason="the site is not beside this package (standalone checkout)")
def test_mirrored_assets_are_byte_identical_to_the_site_copies():
    for name in MIRRORED:
        assert (ASSETS / name).read_bytes() == (SITE / name).read_bytes(), f"{name} differs from the site's copy: re-copy it"
