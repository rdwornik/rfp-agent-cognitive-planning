# setup_rfp_file_search_store.py
import os
import time
from google import genai
from google.genai import types

# 🔧 CHANGE THIS IF YOUR FILE NAME IS DIFFERENT
KB_FILE = "RFP_Database_Cognitive_Planning.jsonl"
STORE_DISPLAY_NAME = "rfp_cognitive_planning_kb"

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable GEMINI_API_KEY is not set.")

    # Create Gemini client (Developer API)
    client = genai.Client(api_key=api_key)

    # 1) Create File Search store
    print(f"Creating File Search store: {STORE_DISPLAY_NAME!r} ...")
    file_search_store = client.file_search_stores.create(
        config={"display_name": STORE_DISPLAY_NAME}
    )
    print("✅ Store created.")
    print("   Store name (copy this somewhere safe):")
    print(f"   {file_search_store.name}")
    print()

    # 2) Upload and index the KB JSONL file
    if not os.path.exists(KB_FILE):
        raise FileNotFoundError(f"KB file not found: {KB_FILE}")

    print(f"Uploading KB file: {KB_FILE!r} ...")
    operation = client.file_search_stores.upload_to_file_search_store(
        file=KB_FILE,
        file_search_store_name=file_search_store.name,
        config={
            # This can be anything you like; it shows up in citations
            "display_name": "rfp_cognitive_planning_kb_file",
        },
    )

    # 3) Poll until indexing is done
    while not operation.done:
        print("⌛ Indexing in progress... (waiting 5 seconds)")
        time.sleep(5)
        # Refresh the operation
        operation = client.operations.get(operation)

    print("✅ File successfully uploaded and indexed.")
    print()
    print("IMPORTANT: use this store name in your RFP scripts:")
    print(f"FILE_SEARCH_STORE_NAME = {file_search_store.name!r}")

if __name__ == "__main__":
    main()
