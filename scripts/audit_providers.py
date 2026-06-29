#!/usr/bin/env python3
"""Audit provider directory coverage under litellm/llms/.

Lists every provider subdirectory, reports which have the expected
core files (transformation.py, __init__.py), and summarises gaps.

Usage:
    uv run python scripts/audit_providers.py
    uv run python scripts/audit_providers.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LLMS_DIR = Path("litellm/llms")

# Files that indicate a provider is properly structured.
EXPECTED_FILES = {
    "__init__.py",
    "transformation.py",
}


def audit_providers(llms_dir: Path) -> list[dict]:
    """Return audit records for every provider directory."""
    if not llms_dir.is_dir():
        print(f"ERROR: {llms_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    results = []
    for entry in sorted(llms_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        existing = {f.name for f in entry.iterdir() if f.is_file()}
        missing = sorted(EXPECTED_FILES - existing)
        results.append(
            {
                "provider": entry.name,
                "files": sorted(existing),
                "missing": missing,
                "ok": len(missing) == 0,
            }
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args(argv)

    records = audit_providers(LLMS_DIR)
    if not records:
        print("No provider directories found.")
        return 1

    if args.json:
        print(json.dumps(records, indent=2))
        return 0

    ok = [r for r in records if r["ok"]]
    bad = [r for r in records if not r["ok"]]

    print(f"Providers audited : {len(records)}")
    print(f"  Fully structured : {len(ok)}")
    print(f"  Missing files    : {len(bad)}")

    if bad:
        print("\nProviders with missing files:")
        for r in bad:
            print(f"  {r['provider']:30s}  missing: {', '.join(r['missing'])}")
        return 1

    print("\nAll providers look good.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
