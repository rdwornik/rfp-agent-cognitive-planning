import os
import json
import chromadb
from chromadb.utils import embedding_functions
from google import genai 
from google.genai import types

# Defensive Import for Anthropic
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("⚠️  Anthropic SDK not found. Claude model will be disabled.")

# --- CONFIGURATION ---
KB_JSON_PATH = "data_kb/canonical/RFP_Database_Cognitive_Planning_CANONICAL.json"
DB_PATH = "data_kb/chroma_store"
COLLECTION_NAME = "rfp_knowledge_base"

# Models
GEMINI_MODEL = "gemini-2.5-pro"
CLAUDE_MODEL = "claude-sonnet-4-20250514"

class LLMRouter:
    def __init__(self):
        print("⚙️  Initializing Universal LLM Router (2025 Edition)...")
        
        # Load Lookup Map
        if not os.path.exists(KB_JSON_PATH):
            raise FileNotFoundError(f"❌ Canonical JSON not found at: {KB_JSON_PATH}")

        with open(KB_JSON_PATH, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            self.kb_lookup = {item.get("kb_id"): item for item in raw_data}

        # Connect to ChromaDB
        self.client_db = chromadb.PersistentClient(path=DB_PATH)
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client_db.get_collection(
            name=COLLECTION_NAME, 
            embedding_function=self.ef
        )
        
    def retrieve_context(self, query, k=5):
        print(f"🔎 Searching for: '{query}'...")
        results = self.collection.query(query_texts=[query], n_results=k)
        
        found_items = []
        if results['ids']:
            for kb_id in results['ids'][0]:
                if kb_id in self.kb_lookup:
                    found_items.append(self.kb_lookup[kb_id])
        return found_items

    def generate_answer(self, query, model="gemini"):
        # 1. Get Context
        context_items = self.retrieve_context(query)
        if not context_items:
            return "❌ No relevant information found."

        # Format Context
        context_str = ""
        for item in context_items:
            context_str += f"""
            ---
            [ID: {item.get('kb_id')}]
            Question: {item.get('canonical_question')}
            Answer: {item.get('canonical_answer')}
            ---
            """

        # 2. Refined System Prompt (Verbose & Professional)
        system_prompt = f"""
        You are an expert Proposal Writer. Use the provided Knowledge Base Context to answer the user's question.
        
        USER QUESTION: "{query}"
        
        KNOWLEDGE BASE CONTEXT:
        {context_str}
        
        INSTRUCTIONS:
        - Use ONLY the provided context.
        - Adopt a professional, persuasive RFP tone.
        - Provide a COMPREHENSIVE answer of 800-1500 characters.
        - Include specific product names, features, and compliance standards from the context.
        - detailed reasoning steps ("Chain of Thought").
        """

        print(f"🧠 Thinking with {model.upper()}...")

        try:
            # --- GEMINI 3 IMPLEMENTATION ---
            if model == "gemini":
                client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=system_prompt)]
                        )
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.3, # Lower for consistency
                        max_output_tokens=8192,
                    )
                )
                return response.text

            # --- CLAUDE 4 IMPLEMENTATION ---
            elif model == "claude":
                if not ANTHROPIC_AVAILABLE:
                    return "❌ Anthropic SDK not installed."
                
                client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
                message = client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=4096,
                    temperature=0.3, # Lower for consistency
                    messages=[
                        {"role": "user", "content": system_prompt}
                    ]
                )
                return message.content[0].text

        except Exception as e:
            return f"❌ Model Error: {e}"

if __name__ == "__main__":
    router = LLMRouter()
    # Test
    print(router.generate_answer("How do we handle data encryption?", model="gemini"))