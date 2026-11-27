#!/usr/bin/env python3
"""Check every file in this repository against tools/MANIFEST_SHA256.tsv.

Each published file must match its recorded SHA-256. Where the published copy differs from the file
as it was run (local paths written as ${WORKSPACE} or ${HOME}, or a declared edit), the manifest also
gives the as-run SHA-256 and the reason; the authors checked those as-run values against the
run-time hash lists before release.

Usage: python tools/verify_manifest.py   (from the repository root)
"""
import hashlib
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
bad, changes = [], Counter()
for line in (ROOT / "tools" / "MANIFEST_SHA256.tsv").read_text().splitlines():
    if line.startswith("#") or not line.strip():
        continue
    path, sha_pub, _origin, _sha_asrun, change = (line.split("\t") + [""] * 5)[:5]
    if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != sha_pub:
        bad.append(path)
    if change:
        changes["declared edit" if change.startswith("edited") else "path placeholders"] += 1
print(f"files not matching the manifest: {len(bad)}")
for p in bad:
    print("   ", p)
print("published copies that differ from the as-run file:", dict(changes) or "none")
sys.exit(1 if bad else 0)
