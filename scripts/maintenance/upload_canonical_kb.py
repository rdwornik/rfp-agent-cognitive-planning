"""
upload_canonical_kb.py

Uploads the canonical KB JSON file into an existing Gemini File Search store.

Usage (from repo root):
  python scripts/maintenance/upload_canonical_kb.py

Reads:
  data_kb/canonical/RFP_Database_Cognitive_Planning_CANONICAL.json

On success:
  - prints the document name inside the store (documents/...).
"""

from pathlib import Path
from google import genai


# --- config ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# parents[0] = maintenance
# parents[1] = scripts
# parents[2] = repo root

FILE_SEARCH_STORE_NAME = "fileSearchStores/rfpcognitiveplanningkbv2-6pqup4g1x9sm"

KB_FILE = "RFP_Database_Cognitive_Planning_CANONICAL.json"
CANONICAL_KB_PATH = PROJECT_ROOT / "data_kb" / "canonical" / KB_FILE
# ---------------


def main():
    kb_path = CANONICAL_KB_PATH

    if not kb_path.exists():
        raise FileNotFoundError(f"KB file not found: {kb_path}")

    client = genai.Client()

    print(f"Uploading '{KB_FILE}' to '{FILE_SEARCH_STORE_NAME}' ...")
    print("The script will wait for this operation to complete.")

    try:
        # This is a synchronous call.
        # It waits for the operation to finish.
        operation = client.file_search_stores.upload_to_file_search_store(
            file_search_store_name=FILE_SEARCH_STORE_NAME,
            file=kb_path,
            config={
                "display_name": KB_FILE,
            },
        )

        # --- THIS IS THE CORRECT CHECK ---
        # If an error occurred, the 'error' attribute will be populated.
        if operation.error:
            print(f"\n--- UPLOAD FAILED ---")
            print("Error details:")
            print(operation.error)

        # If 'error' is None, we check for a valid 'response'.
        # A populated 'response' object IS the success signal.
        elif operation.response and operation.response.document_name:
            print(f"\n--- UPLOAD SUCCESSFUL ---")
            print(f"File uploaded as: {operation.response.document_name}")
            print(
                "It may take ~60 seconds for this file to be visible in 'list_files'."
            )

        # Fallback for an unexpected state
        else:
            print("\n--- UPLOAD STATUS UNKNOWN ---")
            print(
                "Operation did not raise an exception, but 'error' and 'response' are both empty."
            )
            print("Full object:")
            print(operation)

    except Exception as e:
        print(f"\n--- UPLOAD FAILED (Exception) ---")
        print(f"This error (e.g., PERMISSION_DENIED) happens before the API call.")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
