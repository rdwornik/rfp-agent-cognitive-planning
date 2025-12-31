# BUGFIX: "Not in KB" Issue - RESOLVED

**Date:** 2024-12-31
**Status:** ✅ Fixed and Tested

## Problem Summary

The `rfp_batch_universal.py` script was returning "Not in KB" for ALL questions, even though ChromaDB had 899 indexed entries.

## Root Cause Analysis

### Issue 1: ID Mismatch (Primary Issue)
- **ChromaDB IDs:** `planning_kb_0001`, `aiml_kb_0001`, `wms_0001` (domain-prefixed)
- **KB Lookup Keys:** `kb_0001`, `kb_0002`, `wms_0001` (no domain prefix for legacy entries)
- **Result:** `llm_router.py` couldn't find entries because `planning_kb_0001` ≠ `kb_0001`

### Issue 2: Wrong KB File
- `llm_router.py` was loading only `RFP_Database_Cognitive_Planning_CANONICAL.json`
- Should have been loading `RFP_Database_UNIFIED_CANONICAL.json` with all domains

### Issue 3: No Debug Visibility
- No logging to show what ChromaDB was returning
- Impossible to diagnose without seeing the actual IDs

## Solutions Implemented

### 1. Fixed ID Lookup (llm_router.py:105-128)

Changed the KB lookup dictionary to support BOTH formats:

```python
self.kb_lookup = {}
for item in raw_data:
    kb_id = item.get("kb_id")
    domain = item.get("domain", "")

    # Store with original kb_id
    if kb_id:
        self.kb_lookup[kb_id] = item

        # Also store with domain-prefixed version
        if domain and not kb_id.startswith(f"{domain}_"):
            prefixed_id = f"{domain}_{kb_id}"
            self.kb_lookup[prefixed_id] = item
```

**Result:** Lookup works for both `kb_0001` AND `planning_kb_0001`

### 2. Changed KB Path (llm_router.py:24)

**Before:**
```python
KB_JSON_PATH = PROJECT_ROOT / "data_kb/canonical/RFP_Database_Cognitive_Planning_CANONICAL.json"
```

**After:**
```python
KB_JSON_PATH = PROJECT_ROOT / "data_kb/canonical/RFP_Database_UNIFIED_CANONICAL.json"
```

**Result:** Now loads all domains (planning, aiml, wms) instead of just planning

### 3. Added Debug Logging (llm_router.py:141-190)

Enhanced `retrieve_context()` with detailed debug output:

```python
if DEBUG:
    print(f"[DEBUG] === RAG Retrieval ===")
    print(f"[DEBUG] Query: {query}")
    print(f"[DEBUG] ChromaDB returned {len(chroma_ids)} IDs")

    for i, chroma_id in enumerate(chroma_ids):
        distance = distances[i]
        print(f"[DEBUG] {i+1}. ChromaDB ID: {chroma_id} | Distance: {distance}")

        if chroma_id in self.kb_lookup:
            print(f"[DEBUG]    [OK] Found: domain={domain} | kb_id={kb_id} | Q={question}...")
        else:
            print(f"[DEBUG]    [FAIL] NOT FOUND in kb_lookup!")
```

**Usage:**
```bash
# Windows
set DEBUG_RAG=1
python scripts/core/rfp_batch_universal.py --test

# Linux/Mac
DEBUG_RAG=1 python scripts/core/rfp_batch_universal.py --test
```

### 4. Windows Compatibility

Removed all emojis (✅, ❌, 🔍, etc.) that caused `UnicodeEncodeError` on Windows consoles.

**Before:**
```python
print("✅ Connected to ChromaDB")
print("❌ Error: Could not find file")
```

**After:**
```python
print("[SUCCESS] Connected to ChromaDB")
print("[ERROR] Could not find file")
```

## Verification Test Results

### ChromaDB Status
```
[SUCCESS] Collection 'rfp_knowledge_base' found!
[STATS] Total entries: 899

Sample ChromaDB IDs:
   aiml_kb_0001         | Domain: aiml       | KB ID: kb_0001
   planning_kb_0276     | Domain: planning   | KB ID: kb_0276
   wms_0006             | Domain: wms        | KB ID: wms_0006
```

### Retrieval Test
```
Query: What is the SLA for the platform?

[DEBUG] ChromaDB returned 5 IDs
[DEBUG] 1. ChromaDB ID: planning_kb_0656 | Distance: 0.307
[DEBUG]    [OK] Found: domain=planning | kb_id=kb_0656 | Q=What are the service levels...
[DEBUG] 2. ChromaDB ID: planning_kb_0276 | Distance: 0.308
[DEBUG]    [OK] Found: domain=planning | kb_id=kb_0276 | Q=What is the Service Level Agreement...
[DEBUG] Total items retrieved: 5/5

[SUCCESS] Retrieved 5 context items
```

### WMS Retrieval Test
```
Query: How does the WMS integrate with external systems?

[DEBUG] ChromaDB returned 5 IDs
[DEBUG] 1. ChromaDB ID: wms_0006 | Distance: 0.332
[DEBUG]    [OK] Found: domain=wms | kb_id=wms_0006 | Q=When working with warehouse automation...
[DEBUG] 2. ChromaDB ID: wms_0010 | Distance: 0.335
[DEBUG]    [OK] Found: domain=wms | kb_id=wms_0010 | Q=What functionality of the Integrator...
[DEBUG] Total items retrieved: 5/5

[SUCCESS] Retrieved 5 context items (including WMS entries!)
```

## Files Modified

1. **scripts/core/llm_router.py**
   - Changed KB path to unified DB
   - Fixed ID lookup to support both formats
   - Added DEBUG_RAG environment variable support
   - Added detailed debug logging
   - Removed emojis for Windows compatibility

2. **docs/KB_WORKFLOW.md**
   - Added "Debugging RAG Retrieval" section
   - Added troubleshooting for ChromaDB ID conflicts

3. **CLAUDE.md**
   - Updated "Recent Changes" with bugfix details

## Before vs After

### Before
```
Query: "What is the SLA?"
→ ChromaDB returns: planning_kb_0656
→ Lookup in dict for: planning_kb_0656
→ NOT FOUND (dict only has kb_0656)
→ context_items = []
→ Result: "Not in KB"
```

### After
```
Query: "What is the SLA?"
→ ChromaDB returns: planning_kb_0656
→ Lookup in dict for: planning_kb_0656
→ FOUND (dict has both kb_0656 AND planning_kb_0656)
→ context_items = [5 relevant entries]
→ Result: [Detailed answer from LLM using KB context]
```

## How to Use Debug Mode

Enable debug logging to troubleshoot retrieval issues:

```bash
# Windows (Command Prompt)
set DEBUG_RAG=1
python scripts/core/rfp_batch_universal.py --test --model gemini

# Windows (PowerShell)
$env:DEBUG_RAG="1"
python scripts/core/rfp_batch_universal.py --test --model gemini

# Linux/Mac
DEBUG_RAG=1 python scripts/core/rfp_batch_universal.py --test --model gemini
```

Debug output shows:
- ✅ Exact query text
- ✅ ChromaDB IDs returned
- ✅ Distance/similarity scores
- ✅ Whether each ID was found in KB lookup
- ✅ Sample questions from retrieved entries
- ✅ Total items retrieved vs requested

## Embedding Function Verification

Both indexing and querying use the same embedding model:

```python
# In kb_embed_chroma.py (indexing)
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-large-en-v1.5"
)

# In llm_router.py (querying)
self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-large-en-v1.5"
)
```

✅ **No embedding function mismatch**

## Conclusion

The "Not in KB" issue is now **completely resolved**. The system can successfully:

1. ✅ Retrieve entries from all domains (planning, aiml, wms)
2. ✅ Handle both legacy and new ID formats
3. ✅ Load the unified KB with 899 entries
4. ✅ Provide debug visibility into retrieval process
5. ✅ Run without Unicode errors on Windows

**Status:** Ready for production use with `rfp_batch_universal.py`
