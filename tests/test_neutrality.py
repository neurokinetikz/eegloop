"""The package is held to the site's product-neutrality and private-path rules.

scripts/check-content.ts greps site content, the catalog, the notebooks and public data for the
patterns below; loop/ is outside those roots, so this test applies the same patterns here. Vendor
names are checked separately: in this phase no headset vendor is named anywhere in the package; the
phase that adds hardware sources will add an allowlist of exactly one board table.
"""
from __future__ import annotations

import re
from pathlib import Path

LOOP = Path(__file__).resolve().parents[1]
SCAN = (".py", ".md", ".toml", ".yaml", ".yml", ".txt")

# The same patterns scripts/check-content.ts:906-916 applies to content.
FORBIDDEN = [
    (re.compile(r"research-grade", re.I), "vendor marketing language (§10.10)"),
    (re.compile(r"/Users/[A-Za-z]"), "a home directory (§10.10)"),
    (re.compile(r"/home/[a-z]"), "a home directory (§10.10)"),
    (re.compile(r"gs://"), "a private bucket (§10.10)"),
    (re.compile(r"/Volumes/"), "a local volume (§10.10)"),
    (re.compile(r"/mnt/"), "a local mount (§10.10)"),
    (re.compile(r"(clinical|medical|lab)-grade", re.I), "vendor marketing language"),
]

# Assembled at runtime so this file does not itself contain the names it forbids.
_VENDORS = ["".join(p) for p in (("Mu", "se"), ("Brain", "Bit"), ("Emo", "tiv"), ("Neuro", "sity"),
                                  ("Open", "BCI"), ("Inter", "aXon"), ("EP", "OC"))]
VENDOR = re.compile(r"\b(" + "|".join(_VENDORS) + r")\b")
#: Files allowed to name a vendor. Empty in this phase; the hardware phase adds its board table.
VENDOR_ALLOWLIST: set[str] = set()


def _files():
    for p in sorted(LOOP.rglob("*")):
        if p.is_file() and p.suffix in SCAN and "__pycache__" not in p.parts and p.name != Path(__file__).name:
            yield p


def test_no_private_paths_or_marketing_language():
    hits = []
    for p in _files():
        text = p.read_text(encoding="utf-8", errors="replace")
        for pat, why in FORBIDDEN:
            for m in pat.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                hits.append(f"{p.relative_to(LOOP)}:{line}: {why}: {m.group(0)!r}")
    assert not hits, "\n".join(hits)


def test_vendors_named_only_where_allowed():
    hits = []
    for p in _files():
        rel = str(p.relative_to(LOOP))
        if rel in VENDOR_ALLOWLIST:
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for m in VENDOR.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            hits.append(f"{rel}:{line}: {m.group(0)!r}")
    assert not hits, "a device is named once, in the board table, and described by class everywhere else:\n" + "\n".join(hits)
