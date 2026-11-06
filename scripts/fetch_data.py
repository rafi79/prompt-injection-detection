"""
Download the raw source files at the exact upstream commits pinned in
src/data.py, verify their SHA-256 hashes, and write data/raw/MANIFEST.json.

    py scripts/fetch_data.py

Only raw.githubusercontent.com is contacted (no Hugging Face access needed).
Files are written byte-for-byte, so line endings are identical on Windows
and Linux (a git checkout with autocrlf could change them).
"""
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import PINNED, raw_path, _sha256, save_manifest, verify_raw


def main():
    for src, spec in PINNED.items():
        for rel, expected in spec["files"].items():
            dest = raw_path(src, rel)
            if dest.exists() and _sha256(dest) == expected:
                print(f"ok (cached)  {src}/{rel}")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/{spec['repo']}/{spec['sha']}/{rel}"
            print(f"downloading  {url}")
            with urllib.request.urlopen(url, timeout=120) as r:
                dest.write_bytes(r.read())
            got = _sha256(dest)
            if got != expected:
                raise SystemExit(f"SHA-256 mismatch for {dest}: {got} != {expected}")
            print(f"ok           {src}/{rel}")
    verify_raw()
    save_manifest()
    print("\nAll raw files present at pinned commits. Manifest: data/raw/MANIFEST.json")


if __name__ == "__main__":
    main()
