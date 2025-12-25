import json
import os
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
# Updated path per Claude's feedback
KB_PATH = "data_kb/canonical/RFP_Database_Cognitive_Planning_CANONICAL.json"
DB_PATH = "data_kb/chroma_store"
COLLECTION_NAME = "rfp_knowledge_base"

def build_index():
    if not os.path.exists(KB_PATH):
        print(f"❌ Error: Could not find {KB_PATH}. Check your path.")
        return

    print(f"📖 Reading {KB_PATH}...")
    with open(KB_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Initialize ChromaDB (Persistent)
    client = chromadb.PersistentClient(path=DB_PATH)
    
    print("⚙️  Loading local embedding model (all-MiniLM-L6-v2)...")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Clean start
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("🗑️  Cleared old collection.")
    except Exception as e:
        # It's okay if it fails (it means the collection didn't exist yet)
        print(f"ℹ️  No existing collection to clear (creating new one).")

    collection = client.create_collection(
        name=COLLECTION_NAME, 
        embedding_function=ef
    )

    documents = []
    metadatas = []
    ids = []

    print(f"🧩 Processing {len(data)} items...")
    
    for idx, item in enumerate(data):
        # 1. Search Blob (The text the AI actually scans)
        # We search primarily on the 'search_blob' field
        doc_text = item.get("search_blob", "")
        if not doc_text:
            # Fallback if blob is missing, combine Q and A
            doc_text = f"{item.get('canonical_question', '')} {item.get('canonical_answer', '')}"
        
        if not doc_text.strip():
            continue

        documents.append(doc_text)
        
        # 2. Schema-Correct Metadata
        # We truncate canonical_answer in metadata to prevent ChromaDB errors (~8kb limit).
        # The full answer will be retrieved via ID lookup in the next stage if needed.
        full_answer = item.get("canonical_answer", "")
        safe_answer = full_answer[:1000] + "..." if len(full_answer) > 1000 else full_answer

        kb_id = item.get("kb_id", f"kb_{idx:04d}")

        meta = {
            "kb_id": str(kb_id),
            "category": str(item.get("category", "")),
            "subcategory": str(item.get("subcategory", "")),
            "canonical_question": str(item.get("canonical_question", "")),
            "canonical_answer": safe_answer, # Stores a preview
            "last_updated": str(item.get("last_updated", "")),
        }
        metadatas.append(meta)
        ids.append(str(kb_id))

    # 3. Batch Insert
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        end = min(i + batch_size, len(documents))
        print(f"   Writing batch {i} to {end}...")
        collection.add(
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end]
        )

    print(f"✅ Success! Indexed {len(documents)} items to {DB_PATH}")

if __name__ == "__main__":
    build_index()