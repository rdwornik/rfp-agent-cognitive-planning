# upload_canonical_kb.py
#
# Usage (from RFPs folder):
#   python .\upload_canonical_kb.py
#
# This uploads the canonical KB JSON file into an existing File Search store.

from pathlib import Path
from google import genai

# 🔧 Use your existing File Search store name here
STORE_NAME = "fileSearchStores/rfpcognitiveplanningkbv2-6pqup4g1x9sm"

# Canonical KB file (we copied the .jsonl to .json so mime-type is easier)
KB_FILE = "RFP_Database_Cognitive_Planning_CANONICAL.json"


def main():
    base_dir = Path(__file__).resolve().parent
    kb_path = base_dir / KB_FILE

    if not kb_path.exists():
        raise FileNotFoundError(f"KB file not found: {kb_path}")

    client = genai.Client()

    print(f"Uploading '{KB_FILE}' to '{STORE_NAME}' ...")

    # IMPORTANT: your SDK expects a single "request" dict argument:
    #   { "name": <store>, "file": { "display_name": ..., "path": ... } }
    request = {
        "name": STORE_NAME,
        "file": {
            "display_name": KB_FILE,
            "path": str(kb_path),
        },
    }

    operation = client.file_search_stores.upload_to_file_search_store(request)

    print("Upload request sent. Operation object:")
    print(operation)


if __name__ == "__main__":
    main()
