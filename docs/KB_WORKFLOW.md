# Knowledge Base Workflow

This document explains how to add new knowledge to the RFP Answer Engine using the integrated KB transformation and merge pipeline.

## Overview

```
Raw JSONL → kb_transform_knowledge.py → Canonical JSON → kb_merge_canonical.py → Unified KB
```

## Step-by-Step Guide

### 1. Prepare Raw Knowledge (JSONL format)

Create a `.jsonl` file in `data_kb/raw/` with your knowledge entries:

```jsonl
{"question": "What is...", "answer": "...", "type": "WHAT", "frame_id": "001"}
{"question": "How does...", "answer": "...", "type": "HOW", "frame_id": "002"}
```

**Required fields:**
- `question`: The question text
- `answer`: The answer text
- `type`: Question type (WHAT, HOW, DOES, WHY, etc.)
- `frame_id` or `source_slide`: Source reference (optional)

### 2. Transform to Canonical Format

Run the transformation script:

```bash
python scripts/core/kb_transform_knowledge.py \
    --input data_kb/raw/knowledge_wms.jsonl \
    --domain wms \
    --source-type video_workshop \
    --version 2025.1
```

**Parameters:**
- `--input` (`-i`): Path to input JSONL file (required)
- `--domain` (`-d`): Domain tag: wms, planning, catman, logistics, platform (required)
- `--source-type` (`-s`): Source type (default: video_workshop)
  - Options: video_workshop, document, meeting, training, manual
- `--version` (`-v`): Product version (default: 2025.1)
- `--append` (`-a`): Append to existing canonical file instead of overwriting
- `--output` (`-o`): Custom output path (optional)

**Output:**
- Creates/updates `data_kb/canonical/RFP_Database_{DOMAIN}_CANONICAL.json`
- Shows scope distribution (platform vs product_specific)
- Shows category breakdown

### 3. Append More Knowledge (Optional)

To add more entries to an existing domain:

```bash
python scripts/core/kb_transform_knowledge.py \
    --input data_kb/raw/knowledge_wms_session2.jsonl \
    --domain wms \
    --append
```

The `--append` flag ensures existing entries are preserved and new ones are added with sequential IDs.

### 4. Merge All Canonical Files

After transforming knowledge, merge all domains into the unified KB:

```bash
python scripts/core/kb_merge_canonical.py
```

**What it does:**
- Auto-discovers all `*_CANONICAL.json` files in `data_kb/canonical/`
- Merges them into `RFP_Database_UNIFIED_CANONICAL.json`
- Ensures all entries have proper domain fields
- Shows statistics: total entries, domain breakdown, scope breakdown

**Custom output (optional):**
```bash
python scripts/core/kb_merge_canonical.py --output custom_path.json
```

### 5. Re-index to ChromaDB

After merging, update the vector index:

```bash
python scripts/core/kb_embed_chroma.py
```

This creates/updates the ChromaDB vector index used for RAG retrieval.

## Complete Example: Adding WMS Knowledge

```bash
# 1. Transform WMS knowledge from workshop
python scripts/core/kb_transform_knowledge.py \
    --input data_kb/raw/knowledge_wms_workshop1.jsonl \
    --domain wms \
    --source-type video_workshop \
    --version 2025.1

# 2. Add more WMS knowledge from second session
python scripts/core/kb_transform_knowledge.py \
    --input data_kb/raw/knowledge_wms_workshop2.jsonl \
    --domain wms \
    --source-type video_workshop \
    --version 2025.1 \
    --append

# 3. Merge all domains (planning + aiml + wms)
python scripts/core/kb_merge_canonical.py

# 4. Re-index to ChromaDB
python scripts/core/kb_embed_chroma.py
```

## Automatic Features

### Scope Classification

The transformer automatically classifies entries as:
- **platform**: Shared across all products (SSO, APIs, SLAs, infrastructure)
- **product_specific**: Unique to one product (Android app for WMS, Demand Sensing for Planning)

Classification uses keyword matching with confidence scores (0.0 to 1.0).

### Category Assignment

Entries are automatically categorized based on content:
- SLA & Availability
- Support
- Integration
- Security & Compliance
- Infrastructure
- Extensibility
- Platform Capabilities
- User Experience
- WMS Features (for WMS domain)
- Data Management
- Versioning
- Configuration

### Keyword Extraction

Up to 10 relevant keywords are extracted for each entry (SLA, API, Security, WMS, etc.).

### Search Blob Generation

A search-optimized text blob is created containing:
- Domain
- Scope
- Category/Subcategory
- Keywords
- Question
- Answer

This blob is used by the RAG system for semantic search.

## File Structure After Processing

```
data_kb/
├── raw/
│   ├── knowledge_wms_workshop1.jsonl
│   └── knowledge_wms_workshop2.jsonl
├── canonical/
│   ├── RFP_Database_Cognitive_Planning_CANONICAL.json
│   ├── RFP_Database_AIML_CANONICAL.json
│   ├── RFP_Database_WMS_CANONICAL.json
│   └── RFP_Database_UNIFIED_CANONICAL.json  # ← Used by system
└── chroma_store/
    └── [vector index files]
```

## Canonical Entry Schema

Each transformed entry has this structure:

```json
{
  "kb_id": "wms_0001",
  "domain": "wms",
  "scope": "platform",
  "category": "Integration",
  "subcategory": "APIs",
  "canonical_question": "How does WMS integrate...",
  "canonical_answer": "WMS integrates through...",
  "versioning": {
    "valid_from": "2025.1",
    "valid_until": null,
    "deprecated": false,
    "superseded_by": null,
    "version_notes": []
  },
  "rich_metadata": {
    "keywords": ["API", "Integration", "REST"],
    "question_type": "HOW",
    "source_type": "video_workshop",
    "source_id": "frame_123",
    "scope_confidence": 0.85,
    "auto_classified": true
  },
  "search_blob": "DOMAIN: wms | SCOPE: platform || ...",
  "last_updated": "2025-12-30",
  "created_date": "2025-12-30"
}
```

## Domain Support

Currently supported domains:
- `wms`: Warehouse Management System
- `planning`: Cognitive Planning (demand, supply, S&OP, IBP)
- `catman`: Category Management (assortment, space, planogram)
- `logistics`: Transportation Management System (TMS)
- `platform`: Platform-level features (use sparingly)

## Best Practices

1. **Always use `--append`** when adding to existing domain to avoid data loss
2. **Run merge after any transformation** to keep unified KB up-to-date
3. **Re-index ChromaDB** after merge to enable RAG retrieval
4. **Use descriptive frame_ids** to track knowledge sources
5. **Review scope classification** - platform should only be for truly shared features
6. **Keep JSONL files** in `data_kb/raw/` for audit trail

## Troubleshooting

### Problem: Unicode errors on Windows

**Solution:** Files are already updated to use `[INFO]`, `[ERROR]` instead of emojis.

### Problem: Merge doesn't find my canonical file

**Solution:** Ensure filename follows pattern: `RFP_Database_{DOMAIN}_CANONICAL.json`

### Problem: Duplicate kb_ids after append

**Solution:** The transformer automatically finds the highest existing ID and continues from there.

### Problem: Wrong scope classification

**Solution:** Edit `PLATFORM_KEYWORDS` or `DOMAIN_KEYWORDS` in `kb_transform_knowledge.py` and re-run transformation.

### Problem: ChromaDB ID conflicts across domains

**Solution:** This is now automatically handled. The `kb_embed_chroma.py` script ensures unique IDs by:
- Legacy domains (planning, aiml): `kb_0001` → `planning_kb_0001`, `aiml_kb_0001`
- New domains (wms, etc.): Already use domain prefix like `wms_0001` (no change needed)

All ChromaDB IDs follow the pattern `{domain}_{kb_id}` for guaranteed uniqueness.

## Debugging RAG Retrieval

To debug why certain questions aren't finding KB entries, enable debug mode:

```bash
# Windows
set DEBUG_RAG=1
python scripts/core/rfp_batch_universal.py --test

# Linux/Mac
DEBUG_RAG=1 python scripts/core/rfp_batch_universal.py --test
```

Debug output shows:
- Query text
- ChromaDB IDs returned
- Distance scores
- Whether each ID was found in the KB lookup
- Sample questions from retrieved entries

Example debug output:
```
[DEBUG] === RAG Retrieval ===
[DEBUG] Query: What is the SLA for the platform?
[DEBUG] Requested k=5 results
[DEBUG] ChromaDB returned 5 IDs
[DEBUG] 1. ChromaDB ID: planning_kb_0656 | Distance: 0.307
[DEBUG]    [OK] Found: domain=planning | kb_id=kb_0656 | Q=What are the service levels...
[DEBUG] Total items retrieved: 5/5
```

## Next Steps

After completing this workflow:
1. Test retrieval quality: Check if RFP batch processor finds relevant knowledge
2. Add more domains: Create canonical files for catman, logistics, etc.
3. Version management: Use deprecation tools to mark old entries (coming soon)
4. Quality review: Manually verify auto-classifications in canonical JSON files
