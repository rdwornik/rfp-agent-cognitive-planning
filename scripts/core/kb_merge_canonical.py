"""
kb_merge_canonical.py
Merge multiple canonical KBs into one unified file.

Usage:
  python scripts/core/kb_merge_canonical.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input files
PLANNING_KB = PROJECT_ROOT / "data_kb/canonical/RFP_Database_Cognitive_Planning_CANONICAL.json"
AIML_KB = PROJECT_ROOT / "data_kb/canonical/RFP_Database_AIML_CANONICAL.json"

# Output file
UNIFIED_KB = PROJECT_ROOT / "data_kb/canonical/RFP_Database_UNIFIED_CANONICAL.json"


def merge():
    # Load planning
    print(f"📖 Loading planning KB...")
    with open(PLANNING_KB, 'r', encoding='utf-8') as f:
        planning = json.load(f)
    print(f"   → {len(planning)} entries")
    
    # Add domain field
    for item in planning:
        item["domain"] = "planning"
    
    # Load AI/ML
    print(f"📖 Loading AI/ML KB...")
    with open(AIML_KB, 'r', encoding='utf-8') as f:
        aiml = json.load(f)
    print(f"   → {len(aiml)} entries")
    
    # Add domain field
    for item in aiml:
        item["domain"] = "aiml"
    
    # Merge
    unified = planning + aiml
    
    # Save
    with open(UNIFIED_KB, 'w', encoding='utf-8') as f:
        json.dump(unified, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Merged! Total: {len(unified)} entries")
    print(f"📄 Output: {UNIFIED_KB}")


if __name__ == "__main__":
    merge()