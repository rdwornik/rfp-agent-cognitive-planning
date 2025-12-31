"""
kb_merge_canonical.py
Merge multiple canonical KBs into one unified file.

Usage:
  python scripts/core/kb_merge_canonical.py

  # Or specify custom output
  python scripts/core/kb_merge_canonical.py --output custom_unified.json
"""

import json
import argparse
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_DIR = PROJECT_ROOT / "data_kb/canonical"

# Default output file
UNIFIED_KB = CANONICAL_DIR / "RFP_Database_UNIFIED_CANONICAL.json"

# Domain mapping for legacy files (files without domain in name)
DOMAIN_MAP = {
    "RFP_Database_Cognitive_Planning_CANONICAL.json": "planning",
    "RFP_Database_AIML_CANONICAL.json": "aiml",
}


def discover_canonical_files() -> List[Tuple[Path, str]]:
    """
    Auto-discover all canonical KB files in the canonical directory.

    Returns:
        List of (file_path, domain_name) tuples
    """
    files = []

    if not CANONICAL_DIR.exists():
        print(f"[ERROR] Canonical directory not found: {CANONICAL_DIR}")
        return files

    # Find all *_CANONICAL.json files (excluding UNIFIED)
    for file_path in CANONICAL_DIR.glob("*_CANONICAL.json"):
        if "UNIFIED" in file_path.name:
            continue

        # Try to extract domain from filename
        # Expected pattern: RFP_Database_{DOMAIN}_CANONICAL.json
        parts = file_path.stem.split("_")

        domain = None

        # Check legacy mapping first
        if file_path.name in DOMAIN_MAP:
            domain = DOMAIN_MAP[file_path.name]
        # Try to extract from filename pattern
        elif len(parts) >= 3:
            # e.g., RFP_Database_WMS_CANONICAL -> wms
            domain_part = parts[-2].lower()
            if domain_part != "database":
                domain = domain_part

        if domain:
            files.append((file_path, domain))
        else:
            print(f"[WARNING] Skipping {file_path.name} - cannot determine domain")

    return files


def merge(output_path: Path = UNIFIED_KB):
    """
    Merge all discovered canonical KB files.

    Args:
        output_path: Path to output unified file
    """
    print("[INFO] Discovering canonical KB files...")

    canonical_files = discover_canonical_files()

    if not canonical_files:
        print("[ERROR] No canonical files found to merge")
        return

    print(f"   -> Found {len(canonical_files)} files:\n")
    for file_path, domain in canonical_files:
        print(f"      * {file_path.name} (domain: {domain})")

    print()

    # Load and merge all files
    unified = []
    stats = {}

    for file_path, domain in canonical_files:
        print(f"[INFO] Loading {file_path.name}...")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                entries = json.load(f)

            print(f"   -> {len(entries)} entries")

            # Ensure domain field is set
            for item in entries:
                if "domain" not in item or not item["domain"]:
                    item["domain"] = domain

            unified.extend(entries)
            stats[domain] = len(entries)

        except Exception as e:
            print(f"   [ERROR] Error loading {file_path.name}: {e}")
            continue

    # Save merged KB
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unified, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n[SUCCESS] Merged successfully!")
    print(f"[OUTPUT] {output_path}")
    print(f"[STATS] Total entries: {len(unified)}\n")

    print("Domain breakdown:")
    for domain, count in sorted(stats.items()):
        print(f"   {domain}: {count}")

    # Scope breakdown (if available)
    scope_stats = {"platform": 0, "product_specific": 0}
    for item in unified:
        scope = item.get("scope")
        if scope in scope_stats:
            scope_stats[scope] += 1

    if any(scope_stats.values()):
        print(f"\nScope breakdown:")
        for scope, count in scope_stats.items():
            if count > 0:
                print(f"   {scope}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge all canonical KB files into unified database"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help=f"Output file path (default: {UNIFIED_KB})"
    )

    args = parser.parse_args()

    output = Path(args.output) if args.output else UNIFIED_KB
    merge(output)


if __name__ == "__main__":
    main()