import os
import time
from google import genai

# 🔧 Adjust if your file name is different
KB_FILE = "RFP_Database_Cognitive_Planning.json"

# 🔧 Use the store name you already got from the previous script
FILE_SEARCH_STORE_NAME = "fileSearchStores/rfpcognitiveplanningkb-omdvy46zlxon"

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    if not os.path.exists(KB_FILE):
        raise FileNotFoundError(f"KB file not found: {KB_FILE}")

    print(f"Uploading KB file {KB_FILE!r} to {FILE_SEARCH_STORE_NAME!r} ...")

    # 👇 key fix: explicitly set mime_type for .jsonl
    operation = client.file_search_stores.upload_to_file_search_store(
        file=KB_FILE,
        file_search_store_name=FILE_SEARCH_STORE_NAME,
        config={
            "display_name": "rfp_cognitive_planning_kb_file",
        },
    )

    while not operation.done:
        print("Indexing in progress...")
        time.sleep(5)
        operation = client.operations.get(operation)

    print("✅ File successfully uploaded and indexed.")
    print("Use this store name in your query scripts:")
    print(f"FILE_SEARCH_STORE_NAME = {FILE_SEARCH_STORE_NAME!r}")

if __name__ == "__main__":
    main()
