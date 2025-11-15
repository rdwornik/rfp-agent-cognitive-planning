import os
from google import genai

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set.")

client = genai.Client(api_key=api_key)

resp = client.models.generate_content(
    model="gemini-2.5-flash", contents="Say 'hello' in one short sentence."
)

print(resp.text)
