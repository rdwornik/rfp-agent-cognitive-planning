# create_canonical_store.py
from google import genai


def main():
    client = genai.Client()
    store = client.file_search_stores.create_file_search_store(
        display_name="rfp_cognitive_planning_canonical"
    )
    print("Canonical store created:")
    print("  name:", store.name)


if __name__ == "__main__":
    main()
