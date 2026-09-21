"""The package is held to the site's product-neutrality and private-path rules.

scripts/check-content.ts greps site content, the catalog, the notebooks and public data for the
patterns below; loop/ is outside those roots, so this test applies the same patterns here. Vendor
names are checked separately: a headset is named in the board table, in the README's copy of it and
in HARDWARE.md, and described by montage class everywhere else.
"""
from __future__ import annotations

import re
from pathlib import Path

LOOP = Path(__file__).resolve().parents[1]
SCAN = (".py", ".md", ".toml", ".yaml", ".yml", ".txt")

# The same patterns scripts/check-content.ts applies to content (assembled so this file carries none of them;
# since the Phase 5 addendum that script also greps loop/ itself).
FORBIDDEN = [
    (re.compile("research" + "-grade", re.I), "vendor marketing language (§10.10)"),
    (re.compile(r"/Users/[A-Za-z]"), "a home directory (§10.10)"),
    (re.compile(r"/home/[a-z]"), "a home directory (§10.10)"),
    (re.compile("gs:" + "//"), "a private bucket (§10.10)"),
    (re.compile("/" + "Volumes/"), "a local volume (§10.10)"),
    (re.compile("/" + "mnt/"), "a local mount (§10.10)"),
    (re.compile("(clinical|medical|lab)" + "-grade", re.I), "vendor marketing language"),
]

# Assembled at runtime so this file does not itself contain the names it forbids.
_VENDORS = ["".join(p) for p in (("Mu", "se"), ("Brain", "Bit"), ("Emo", "tiv"), ("Neuro", "sity"),
                                  ("Open", "BCI"), ("Inter", "aXon"), ("EP", "OC"))]
VENDOR = re.compile(r"\b(" + "|".join(_VENDORS) + r")\b")  # case-sensitive: lowercase board keys are public facts
#: Files allowed to name a vendor: the one board table (CONTRACTS Phase 5, rule 4), the README's copy of
#: it, and HARDWARE.md, which records per-device facts with their verified dates. Nothing else.
VENDOR_ALLOWLIST: set[str] = {"eegloop/sources/brainflow.py", "README.md", "HARDWARE.md"}


#: The notebook helpers the notebooks-L7c track appends to are held to the same rule as the package.
HELPERS_L7 = LOOP.parent / "notebooks" / "_shared" / "helpers_l7.py"


def _rel(p: Path) -> str:
    return str(p.relative_to(LOOP.parent))


def _files():
    for p in sorted(LOOP.rglob("*")):
        if p.is_file() and p.suffix in SCAN and "__pycache__" not in p.parts and p.name != Path(__file__).name:
            yield p
    if HELPERS_L7.exists():
        yield HELPERS_L7


def test_no_private_paths_or_marketing_language():
    hits = []
    for p in _files():
        text = p.read_text(encoding="utf-8", errors="replace")
        for pat, why in FORBIDDEN:
            for m in pat.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                hits.append(f"{_rel(p)}:{line}: {why}: {m.group(0)!r}")
    assert not hits, "\n".join(hits)


def test_vendors_named_only_where_allowed():
    hits = []
    for p in _files():
        rel = str(p.relative_to(LOOP)) if p.is_relative_to(LOOP) else _rel(p)
        if rel in VENDOR_ALLOWLIST:
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for m in VENDOR.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            hits.append(f"{rel}:{line}: {m.group(0)!r}")
    assert not hits, "a device is named once, in the board table, and described by class everywhere else:\n" + "\n".join(hits)
