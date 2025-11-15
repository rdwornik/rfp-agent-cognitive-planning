"""
rfp_query_test.py

Simple smoke test to verify that:
  - GEMINI_API_KEY is configured
  - the File Search store is reachable
  - the canonical KB responds correctly to a sample question.

Usage (from repo root):
  python scripts/core/rfp_query_test.py

Prints the model's answer to the console.
"""

import os
from pathlib import Path

from google import genai
from google.genai import types


# --- config ---
# repo root = two levels up from scripts/core/...
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts_instructions" / "rfp_system_prompt.txt"

FILE_SEARCH_STORE_NAME = "fileSearchStores/rfpcognitiveplanningkbv2-6pqup4g1x9sm"

MODEL_NAME = "gemini-2.5-flash"
# ---------------


def load_system_instructions(path: Path) -> str:
    """Load the full SYSTEM prompt from a text file."""
    if not path.exists():
        raise FileNotFoundError(f"System prompt not found at: {path}")
    print(f"Loading system prompt from: {path}")
    return path.read_text(encoding="utf-8")


def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable GEMINI_API_KEY is not set.")
    return genai.Client(api_key=api_key)


def main():
    client = get_client()
    system_text = load_system_instructions(SYSTEM_PROMPT_PATH)

    # File Search tool (google.genai style)
    file_search_tool = types.Tool(
        file_search=types.FileSearch(file_search_store_names=[FILE_SEARCH_STORE_NAME])
    )
    tools = [file_search_tool]

    question = (
        "The SCP tool supports the creation of customizable key figure calculation, "
        "e.g. custom-calculation rules."
    )

    print(f"\nQuerying store '{FILE_SEARCH_STORE_NAME}' with:")
    print(f"Q: {question}\n")

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                {
                    "role": "user",
                    "parts": [{"text": question}],
                }
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_text,
                tools=tools,
            ),
        )

        print("--- Response ---")
        print(response.text)
        print("\n---")
        print("If this answer looks like it came from your KB, it worked!")

    except Exception as e:
        print("\n--- QUERY FAILED ---")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
