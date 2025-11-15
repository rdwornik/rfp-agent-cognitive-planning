import os
from google import genai
from google.genai import types

# 👇 Use the store name you already have
FILE_SEARCH_STORE_NAME = "fileSearchStores/rfpcognitiveplanningkb-omdvy46zlxon"

def load_system_instructions(path: str = "rfp_system_prompt.txt") -> str:
    """Load the full SYSTEM prompt from a text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def ask_rfp_question(question: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    # File Search tool using your KB store
    file_search_tool = types.Tool(
        file_search=types.FileSearch(
            file_search_store_names=[FILE_SEARCH_STORE_NAME]
        )
    )

    system_text = load_system_instructions()

    response = client.models.generate_content(
        model="gemini-2.5-flash",   # or "gemini-2.5-pro" if you prefer
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=system_text,
            tools=[file_search_tool],
        ),
    )

    return response.text.strip()

if __name__ == "__main__":
    question = (
        "The SCP tool supports the creation of customizable key figure calculation, "
        "e.g. custom-calculation rules."
    )
    answer = ask_rfp_question(question)
    print("Q:", question)
    print("\nA:", answer)
