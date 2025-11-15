from google import genai
from google.genai import types

# 🔧 Your existing store name
FILE_SEARCH_STORE_NAME = "fileSearchStores/rfpcognitiveplanningkb-omdvy46zlxon"


def main():
    client = genai.Client()  # uses GEMINI_API_KEY

    print(f"\nListing files in store: {FILE_SEARCH_STORE_NAME}\n")

    # This returns a pager of "file search store file" objects
    pager = client.file_search_stores.list_file_search_store_files(
        file_search_store=FILE_SEARCH_STORE_NAME
    )

    found_any = False
    for fs_file in pager:
        found_any = True
        # fs_file.file is the underlying File object
        file_obj = getattr(fs_file, "file", None)

        print("--------------------------------------------------")
        print("file_search_store_file name:", getattr(fs_file, "name", ""))
        if file_obj is not None:
            print("  file.name:        ", getattr(file_obj, "name", ""))
            print("  file.display_name:", getattr(file_obj, "display_name", ""))
            print("  file.mime_type:   ", getattr(file_obj, "mime_type", ""))
            print("  file.size_bytes:  ", getattr(file_obj, "size_bytes", ""))
        else:
            print("  (no embedded file object in this SDK version)")
        print("raw store file object:", fs_file)
        print()

    if not found_any:
        print("⚠️  No files are attached to this File Search store.")


if __name__ == "__main__":
    main()
