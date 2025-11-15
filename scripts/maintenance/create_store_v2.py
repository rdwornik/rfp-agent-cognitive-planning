from google import genai

def main():
    client = genai.Client()  # uses GEMINI_API_KEY env var

    store = client.file_search_stores.create(
        config={"display_name": "rfp_cognitive_planning_kb_v2"}
    )

    print("New File Search Store created:")
    print("  name =", store.name)

if __name__ == "__main__":
    main()
