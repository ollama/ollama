"""Generate MANIFEST.json for a bring-up tree.

Records sha256 + byte size for every file and classifies each as archived
(ships in the PR archive) or regenerable (excluded for size; the manifest
records how to rebuild it). Regenerable patterns and commands are read from
an optional ARCHIVE_EXCLUDE.json in the tree root:

    {"reference/bf16/activations.safetensors": "<regeneration command>"}

Usage: python3 x/models/scripts/make_manifest.py x/models/bringup/<model>
"""
import hashlib
import json
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."

regenerable = {}
exclude_path = os.path.join(ROOT, "ARCHIVE_EXCLUDE.json")
if os.path.exists(exclude_path):
    regenerable = json.load(open(exclude_path))

manifest = {"files": {}, "root": os.path.basename(os.path.abspath(ROOT))}
for dirpath, dirnames, filenames in os.walk(ROOT):
    for f in sorted(filenames):
        if f == "MANIFEST.json":
            continue
        p = os.path.join(dirpath, f)
        rel = os.path.relpath(p, ROOT)
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        entry = {"sha256": h.hexdigest(), "bytes": os.path.getsize(p)}
        regen = next((cmd for pref, cmd in regenerable.items() if rel.startswith(pref)), None)
        if regen:
            entry["archived"] = False
            entry["regenerate"] = regen
        else:
            entry["archived"] = True
        manifest["files"][rel] = entry

out = os.path.join(ROOT, "MANIFEST.json")
with open(out, "w") as fh:
    json.dump(manifest, fh, indent=1, sort_keys=True)
archived = sum(1 for e in manifest["files"].values() if e["archived"])
regen = len(manifest["files"]) - archived
print(f"{out}: {len(manifest['files'])} files ({archived} archived, {regen} regenerable)")
