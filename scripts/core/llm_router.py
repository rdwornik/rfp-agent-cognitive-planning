import os
import json
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from google import genai
from google.genai import types

# Optional imports
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
KB_JSON_PATH = PROJECT_ROOT / "data_kb/canonical/RFP_Database_Cognitive_Planning_CANONICAL.json"
DB_PATH = PROJECT_ROOT / "data_kb/chroma_store"
COLLECTION_NAME = "rfp_knowledge_base"
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts_instructions/rfp_system_prompt_universal.txt"

# --- MODEL REGISTRY ---
MODELS = {
    "gemini": {"name": "gemini-3-pro-preview", "provider": "google"},
    "gemini-flash": {"name": "gemini-3-flash-preview", "provider": "google"},
    "claude": {"name": "claude-sonnet-4-20250514", "provider": "anthropic"},
    "gpt5": {"name": "gpt-5", "provider": "openai"},
    "deepseek": {"name": "deepseek-chat", "provider": "deepseek"},
    "kimi": {"name": "kimi-k2-0905", "provider": "moonshot"},
    "llama": {"name": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "provider": "together"},
    "grok": {"name": "grok-3-beta", "provider": "xai"},
}


def load_system_prompt() -> str:
    """Load system prompt from external file."""
    if not SYSTEM_PROMPT_PATH.exists():
        raise FileNotFoundError(f"System prompt not found: {SYSTEM_PROMPT_PATH}")
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()

def clean_bold_markdown(text: str) -> str:
    """Remove bold/italic markdown but keep list structure."""
    import re
    # Remove bold **text** → text
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    # Remove italic *text* → text
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    # Remove headers
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    return text.strip()

class LLMRouter:
    def __init__(self):
        print("⚙️  Initializing Universal LLM Router...")
        
        # Load system prompt
        self.system_prompt_template = load_system_prompt()
        print(f"✅ Loaded prompt from: {SYSTEM_PROMPT_PATH.name}")
        
        # Load KB lookup
        with open(KB_JSON_PATH, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            self.kb_lookup = {item.get("kb_id"): item for item in raw_data}
        print(f"✅ Loaded {len(self.kb_lookup)} KB entries")
        
        # Connect to ChromaDB
        self.client_db = chromadb.PersistentClient(path=str(DB_PATH))
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client_db.get_collection(
            name=COLLECTION_NAME,
            embedding_function=self.ef
        )
        print(f"✅ Connected to ChromaDB: {COLLECTION_NAME}")

    def retrieve_context(self, query, k=8):
        results = self.collection.query(query_texts=[query], n_results=k)
        found_items = []
        if results['ids']:
            for kb_id in results['ids'][0]:
                if kb_id in self.kb_lookup:
                    found_items.append(self.kb_lookup[kb_id])
        return found_items

    def format_context(self, items) -> str:
        """Format KB items into context string."""
        parts = []
        for item in items:
            parts.append(
                f"---\n"
                f"Category: {item.get('category', '')} / {item.get('subcategory', '')}\n"
                f"Canonical Question: {item.get('canonical_question', '')}\n"
                f"Canonical Answer: {item.get('canonical_answer', '')}\n"
                f"---"
            )
        return "\n\n".join(parts)

    def generate_answer(self, query, model="gemini"):
        context_items = self.retrieve_context(query, k=8)
        
        if not context_items:
            return "Not in KB"

        # Build prompt from template
        context_str = self.format_context(context_items)
        prompt = self.system_prompt_template.format(
            context=context_str,
            query=query
        )

        model_config = MODELS.get(model, MODELS["gemini"])
        provider = model_config["provider"]
        model_name = model_config["name"]

        print(f"🧠 Generating with {model.upper()} ({model_name})...")

        try:
            # --- GOOGLE GEMINI ---
            if provider == "google":
                client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
                response = client.models.generate_content(
                    model=model_name,
                    contents=[{"role": "user", "parts": [{"text": prompt}]}],
                    config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=8192)
                )
                return clean_bold_markdown(response.text.strip()) if response.text else "Error: Empty response"

            # --- ANTHROPIC CLAUDE ---
            elif provider == "anthropic":
                if not ANTHROPIC_AVAILABLE:
                    return "Error: Anthropic SDK not installed"
                client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
                message = client.messages.create(
                    model=model_name,
                    max_tokens=4096,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}]
                )
                return message.content[0].text.strip()

            # --- OPENAI GPT-5 ---
            elif provider == "openai":
                if not OPENAI_AVAILABLE:
                    return "Error: OpenAI SDK not installed"
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4096,
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()

            # --- DEEPSEEK ---
            elif provider == "deepseek":
                if not OPENAI_AVAILABLE:
                    return "Error: OpenAI SDK not installed"
                client = OpenAI(
                    api_key=os.environ.get("DEEPSEEK_API_KEY"),
                    base_url="https://api.deepseek.com/v1"
                )
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4096
                )
                return response.choices[0].message.content.strip()

            # --- KIMI K2 (Moonshot) ---
            elif provider == "moonshot":
                if not OPENAI_AVAILABLE:
                    return "Error: OpenAI SDK not installed"
                client = OpenAI(
                    api_key=os.environ.get("MOONSHOT_API_KEY"),
                    base_url="https://api.moonshot.ai/v1"
                )
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4096
                )
                return response.choices[0].message.content.strip()

            # --- LLAMA 4 (via Together.ai) ---
            elif provider == "together":
                if not OPENAI_AVAILABLE:
                    return "Error: OpenAI SDK not installed"
                client = OpenAI(
                    api_key=os.environ.get("TOGETHER_API_KEY"),
                    base_url="https://api.together.xyz/v1"
                )
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4096
                )
                return response.choices[0].message.content.strip()

            # --- GROK (xAI) ---
            elif provider == "xai":
                if not OPENAI_AVAILABLE:
                    return "Error: OpenAI SDK not installed"
                client = OpenAI(
                    api_key=os.environ.get("XAI_API_KEY"),
                    base_url="https://api.x.ai/v1"
                )
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4096
                )
                return response.choices[0].message.content.strip()

            else:
                return f"Error: Unknown provider {provider}"

        except Exception as e:
            return f"Error: {str(e)}"


if __name__ == "__main__":
    router = LLMRouter()
    
    q = "How do you handle data encryption?"
    answer = router.generate_answer(q, model="gemini")
    
    print("\n" + "="*50)
    print("🤖 ANSWER:")
    print("="*50)
    print(answer)