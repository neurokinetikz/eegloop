"""Package versions and seeding -- the two things that make a session reproducible.

Copied from pipelines/eegpipe/versions.py @ 2026-09-20 rather than imported: importing eegpipe would
pull mne into a package whose whole point is to run without it. The only change is the list of
packages worth tracking. Lesson L2.8's argument applies to a session exactly as it does to a run:
"a default change is a silent pipeline change", so what was installed goes in the session log.
"""
from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
from importlib import metadata
from typing import Any, Iterable

__all__ = [
    "TRACKED_PACKAGES",
    "installed_versions",
    "check_pins",
    "seed_everything",
    "config_hash",
]

#: Packages whose version can change a result. ``None`` for one that is not installed -- which is a
#: fact worth recording too, because it says which source or decoder could not have run.
TRACKED_PACKAGES: tuple[str, ...] = (
    "numpy",
    "scipy",
    "brainflow",
    "mne-lsl",
    "pylsl",
    "scikit-learn",
    "pyriemann",
    "mne",
    "PyYAML",
)

#: Distribution name -> import name, where they differ.
_IMPORT_NAME = {
    "scikit-learn": "sklearn",
    "mne-lsl": "mne_lsl",
    "PyYAML": "yaml",
}


def _version_of(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        pass
    mod_name = _IMPORT_NAME.get(dist, dist)
    mod = sys.modules.get(mod_name)
    if mod is None:
        try:
            __import__(mod_name)
        except Exception:
            return None
        mod = sys.modules.get(mod_name)
    version = getattr(mod, "__version__", None)
    return str(version) if version else None


def installed_versions(packages: Iterable[str] = TRACKED_PACKAGES) -> dict[str, str | None]:
    """The versions actually in use, plus the interpreter and platform."""
    out: dict[str, str | None] = {
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
    }
    for dist in packages:
        out[dist] = _version_of(dist)
    return out


def check_pins(pins: dict[str, str], installed: dict[str, str | None] | None = None) -> list[str]:
    """Compare a protocol's ``versions:`` block with what is installed.

    Only exact pins (``mne: "1.10.2"``) are compared; a range such as ``">=2.1,<3"`` is recorded but
    not parsed -- the package refuses to re-implement a resolver and says so rather than pretending
    to check. Returns one human-readable line per mismatch (empty when everything agrees).
    """
    installed = installed if installed is not None else installed_versions()
    problems: list[str] = []
    for name, pinned in pins.items():
        if name == "on_mismatch":
            continue
        have = installed.get(name)
        if have is None:
            problems.append(f"{name}: pinned {pinned!r}, not installed")
            continue
        pinned_s = str(pinned).strip()
        if any(c in pinned_s for c in "<>=~^,*"):
            continue  # a range: recorded in the log, not checked here
        if pinned_s != have:
            problems.append(f"{name}: pinned {pinned_s}, running {have}")
    return problems


def seed_everything(seed: int) -> dict[str, int]:
    """Seed Python's and NumPy's global generators; return what was seeded."""
    seed = int(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except Exception:  # pragma: no cover - numpy is a hard dependency
        pass
    return {"python_random": seed, "numpy_random": seed % (2**32)}


def config_hash(obj: Any) -> str:
    """A short, stable hash of a resolved configuration (sorted keys, compact separators)."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
