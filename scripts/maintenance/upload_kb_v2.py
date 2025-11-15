"""
upload_kb_v2.py

Legacy/auxiliary uploader for the *raw* KB into a File Search store.
Kept for cases where you want to experiment with the non-canonical KB.

Usage (from repo root):
  python scripts/maintenance/upload_kb_v2.py

Reads:
  data_kb/raw/RFP_Database_Cognitive_Planning.json (or similar)
"""

from google import genai
import time

# 👇 Use your NEW store name here:
FILE_SEARCH_STORE_NAME = "fileSearchStores/rfpcognitiveplanningkbv2-6pqup4g1x9sm"

# 👇 Updated KB file (no DAS, etc.)
KB_PATH = "RFP_Database_Cognitive_Planning_CANONICAL.json"


def main():
    client = genai.Client()  # uses GEMINI_API_KEY

    print(f"Uploading '{KB_PATH}' to '{FILE_SEARCH_STORE_NAME}' ...")

    # Depending on SDK version, this may return either:
    # - an Operation object with `.name`, or
    # - a plain string with the operation name.
    operation = client.file_search_stores.upload_to_file_search_store(
        file=KB_PATH,
        file_search_store_name=FILE_SEARCH_STORE_NAME,
        config={"display_name": "RFP_Database_Cognitive_Planning_CANONICAL.json"},
    )

    # Normalise to a string operation name
    if hasattr(operation, "name"):
        op_name = operation.name
    else:
        op_name = str(operation)

    # Try to poll the operation until it's done (if supported)
    try:
        while True:
            op = client.operations.get(
                op_name
            )  # in your SDK this was used as positional
            if getattr(op, "done", True):
                break
            print("  - Indexing in progress... waiting 5 seconds...")
            time.sleep(5)

        print("✅ Upload + indexing completed.")
    except Exception as e:
        # If polling isn't supported, we still don't want to crash here.
        print(f"⚠️ Could not poll operation status ({e}).")
        print("   Assuming upload request was accepted by the API.")


if __name__ == "__main__":
    main()
