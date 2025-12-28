"""
kb_embed_chroma.py
Index unified canonical KB to ChromaDB.

Usage:
  python scripts/core/kb_embed_chroma.py
"""

import json
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
KB_PATH = PROJECT_ROOT / "data_kb/canonical/RFP_Database_UNIFIED_CANONICAL.json"
DB_PATH = str(PROJECT_ROOT / "data_kb/chroma_store")
COLLECTION_NAME = "rfp_knowledge_base"


def build_index():
    if not KB_PATH.exists():
        print(f"❌ Error: Could not find {KB_PATH}")
        print(f"   Run kb_merge_canonical.py first!")
        return
    
    print(f"📖 Reading {KB_PATH.name}...")
    with open(KB_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 Total entries: {len(data)}")
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=DB_PATH)
    
    print("⚙️  Loading embedding model (BAAI/bge-large-en-v1.5)...")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-large-en-v1.5"
    )
    
    # Clean start
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("🗑️  Cleared old collection.")
    except Exception:
        print("ℹ️  No existing collection to clear.")
    
    collection = client.create_collection(
        name=COLLECTION_NAME, 
        embedding_function=ef
    )
    
    documents = []
    metadatas = []
    ids = []
    
    print(f"🧩 Processing {len(data)} items...")
    
    for idx, item in enumerate(data):
        doc_text = item.get("search_blob", "")
        if not doc_text:
            doc_text = f"{item.get('canonical_question', '')} {item.get('canonical_answer', '')}"
        
        if not doc_text.strip():
            continue
        
        documents.append(doc_text)
        
        full_answer = item.get("canonical_answer", "")
        safe_answer = full_answer[:1000] + "..." if len(full_answer) > 1000 else full_answer
        kb_id = item.get("kb_id", f"kb_{idx:04d}")
        
        meta = {
            "kb_id": str(kb_id),
            "domain": str(item.get("domain", "planning")),
            "category": str(item.get("category", "")),
            "subcategory": str(item.get("subcategory", "")),
            "canonical_question": str(item.get("canonical_question", "")),
            "canonical_answer": safe_answer,
            "last_updated": str(item.get("last_updated", "")),
        }
        metadatas.append(meta)
        ids.append(str(kb_id))
    
    # Batch Insert
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        end = min(i + batch_size, len(documents))
        print(f"   Writing batch {i} to {end}...")
        collection.add(
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end]
        )
    
    # Summary
    domain_counts = {}
    for m in metadatas:
        d = m.get("domain", "unknown")
        domain_counts[d] = domain_counts.get(d, 0) + 1
    
    print(f"\n✅ Indexed {len(documents)} items")
    print(f"   Breakdown: {domain_counts}")


if __name__ == "__main__":
    build_index()