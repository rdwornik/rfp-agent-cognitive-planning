# RFP Answer Engine - Project Context

> This file provides context for Claude Code. Update it after significant changes.

## Project Overview

**Purpose:** AI-powered RFP (Request for Proposal) answering system for Blue Yonder solutions.

**Core Flow:**
```
RFP Questions (Excel) → Anonymization → RAG Retrieval → LLM Generation → De-anonymization → Answers
```

## Current State (v0.2)

### Architecture
- **Embeddings:** Local BGE-large-en-v1.5 (no API cost)
- **Vector DB:** ChromaDB (local, persistent)
- **LLM Router:** Multi-provider (Gemini, Claude, GPT-5, DeepSeek, GLM, Kimi, Llama, Grok)
- **Anonymization:** YAML-based blocklist with middleware pattern

### Knowledge Base Structure
| Domain | File | Entries | Description |
|--------|------|---------|-------------|
| planning | RFP_Database_Cognitive_Planning_CANONICAL.json | 807 | Cognitive Planning Q&A (most complete) |
| aiml | RFP_Database_AIML_CANONICAL.json | 54 | AI/ML capabilities |
| wms | RFP_Database_WMS_CANONICAL.json | 38 | Warehouse Management (from video workshops) |

### KB Entry Schema
```json
{
  "kb_id": "wms_0001",
  "domain": "wms",
  "scope": "platform | product_specific",
  "category": "Integration",
  "subcategory": "APIs",
  "canonical_question": "...",
  "canonical_answer": "...",
  "versioning": {
    "valid_from": "2025.1",
    "valid_until": null,
    "deprecated": false,
    "superseded_by": null,
    "version_notes": ["2024", "2025"]
  },
  "rich_metadata": {
    "keywords": ["API", "REST"],
    "question_type": "WHAT",
    "source_type": "video_workshop",
    "source_id": "frame_123",
    "scope_confidence": 0.85,
    "auto_classified": true
  },
  "search_blob": "...",
  "last_updated": "2025-01-15",
  "created_date": "2025-01-15"
}
```

## File Structure

```
rfp-answer-engine/
├── CLAUDE.md                          # THIS FILE - project context
├── README.md                          # User documentation
│
├── config/
│   ├── anonymization.yaml             # Blocklist and session config
│   └── platform_matrix.json           # Platform services matrix (from Excel)
│
├── data_kb/
│   ├── raw/                           # Source files (JSONL from workshops)
│   │   └── knowledge_wms.jsonl
│   ├── canonical/                     # Transformed KB files
│   │   ├── RFP_Database_Cognitive_Planning_CANONICAL.json
│   │   ├── RFP_Database_AIML_CANONICAL.json
│   │   ├── RFP_Database_WMS_CANONICAL.json
│   │   └── RFP_Database_UNIFIED_CANONICAL.json  # Merged
│   └── chroma_store/                  # ChromaDB vector index
│
├── scripts/core/
│   ├── rfp_batch_universal.py         # Main batch processor
│   ├── llm_router.py                  # Multi-LLM provider router
│   ├── kb_build_canonical.py          # Build KB from raw sources
│   ├── kb_transform_knowledge.py      # Transform JSONL → Canonical (NEW)
│   ├── kb_merge_canonical.py          # Merge all KBs into unified
│   ├── kb_embed_chroma.py             # Index to ChromaDB
│   └── anonymization/                 # Anonymization package
│       ├── __init__.py
│       ├── config.py                  # YAML loader
│       ├── core.py                    # anonymize(), deanonymize()
│       ├── middleware.py              # Pipeline middleware
│       ├── scan_kb.py                 # CLI: scan KB for sensitive terms
│       └── clean_kb.py                # CLI: clean KB with backup
│
├── logs/
│   └── anonymization.log              # Audit trail
│
└── outputs/                           # Generated RFP answers
```

## Key Commands

```bash
# Transform new knowledge from workshops
python scripts/core/kb_transform_knowledge.py \
    --input data_kb/raw/knowledge_wms.jsonl \
    --domain wms \
    --source-type video_workshop \
    --version 2025.1

# Append more knowledge to existing KB
python scripts/core/kb_transform_knowledge.py \
    --input data_kb/raw/knowledge_wms_session2.jsonl \
    --domain wms \
    --append

# Merge all KBs
python scripts/core/kb_merge_canonical.py

# Re-index to ChromaDB
python scripts/core/kb_embed_chroma.py

# Run batch processor
python scripts/core/rfp_batch_universal.py \
    --test \
    --model gemini \
    --anonymize \
    --workers 4

# Scan KB for sensitive terms
python -m scripts.core.anonymization.scan_kb

# Clean KB (dry run first!)
python -m scripts.core.anonymization.clean_kb --dry-run
```

## Environment Variables

```bash
GEMINI_API_KEY=...       # Google Gemini
ANTHROPIC_API_KEY=...    # Claude
OPENAI_API_KEY=...       # GPT-5
DEEPSEEK_API_KEY=...     # DeepSeek
ZHIPU_API_KEY=...        # GLM (use --workers 2)
MOONSHOT_API_KEY=...     # Kimi
TOGETHER_API_KEY=...     # Llama
XAI_API_KEY=...          # Grok
```

## Architecture Decisions

### Why local embeddings?
- Zero API cost
- Zero data exposure (embeddings never leave machine)
- BGE-large-en-v1.5 is high quality (top 10 on MTEB)

### Why YAML for anonymization config?
- Human-readable for non-technical users
- Easy to maintain blocklists
- Supports comments

### Why unified KB with domain metadata?
- RFPs often mix topics (planning + platform + AI/ML)
- Single ChromaDB collection is simpler
- Domain field enables optional filtering
- Scope field (platform/product_specific) helps with cross-solution questions

### Why versioning in KB entries?
- Product versions change (2024 → 2025 → 2026)
- Some features get deprecated
- Need to track when information becomes stale
- `valid_from`, `valid_until`, `deprecated`, `superseded_by` fields

## Scope Classification

| Scope | Description | Examples |
|-------|-------------|----------|
| `platform` | Shared across all products | SSO, API Management, Data Cloud, SLAs |
| `product_specific` | Unique to one product | Android app (WMS), Demand Sensing (Planning) |

**Auto-classification:** `kb_transform_knowledge.py` uses keyword matching with confidence scores.

## Current Tasks

- [ ] Create `kb_deprecate.py` CLI tool for marking old entries
- [ ] Add `--solution wms|planning|catman` flag to batch processor
- [ ] Update `kb_embed_chroma.py` to filter deprecated entries
- [ ] Test WMS knowledge retrieval quality
- [ ] Add more WMS knowledge from additional workshops

## Recent Changes

### 2025-01-01
- **FEATURE:** Solution-aware response system with platform service context
  - Added `--solution` flag to `rfp_batch_universal.py` (41 solutions available)
  - Integrated `config/platform_matrix.json` for service status lookup
  - Added `docs/platform_context.md` with response templates
  - Updated `llm_router.py` to inject solution-specific context before KB context
  - Context includes: native/planned/infrastructure service lists
  - Response framing adjusts based on integration level
  - NEVER exposes version numbers or roadmap dates to customer
  - Used Opus model for complex integration task

### 2024-12-31
- **BUGFIX:** Fixed "Not in KB" issue in `rfp_batch_universal.py`
  - Root cause: ChromaDB IDs (`planning_kb_0001`) didn't match KB lookup keys (`kb_0001`)
  - Solution: Updated `llm_router.py` to build lookup dict with both formats
  - Changed KB path from Planning-only to Unified KB (`RFP_Database_UNIFIED_CANONICAL.json`)
  - Added DEBUG_RAG=1 env var for retrieval debugging
  - Removed all emojis for Windows console compatibility
- **INTEGRATION:** Completed `kb_transform_knowledge.py` + `kb_merge_canonical.py` workflow
  - Auto-discovery of canonical files by domain
  - Dynamic merge without hardcoded file lists
  - Support for WMS and future domains
- **CHROMADB:** Fixed ID uniqueness across domains
  - Format: `{domain}_{kb_id}` (e.g., `planning_kb_0001`, `wms_0001`)
  - Updated `kb_embed_chroma.py` to generate domain-prefixed IDs
  - Verified 899 entries indexed successfully

### 2024-12-30
- Created `kb_transform_knowledge.py` - universal transformer for JSONL → Canonical
- Created `platform_matrix.json` from Platform_Usage_by_Product.xlsx
- Transformed WMS knowledge: 38 entries (10 platform, 28 product-specific)
- Added versioning schema to KB entries
- Added scope classification (platform vs product_specific)

### Previous (v0.2)
- Universal RAG architecture with ChromaDB + BGE embeddings
- Multi-LLM router with 8 providers
- Anonymization system with YAML config and middleware
- GLM retry logic with exponential backoff

## Notes for Claude Code

1. **Before editing KB-related files:** Check the schema in this document
2. **When adding new domains:** Update `DOMAIN_KEYWORDS` in `kb_transform_knowledge.py`
3. **When running batch processor:** Always test with `--test` flag first
4. **GLM rate limits:** Use `--workers 2` to avoid 429 errors
5. **Anonymization:** Run `scan_kb` before `clean_kb`, always use `--dry-run` first

## Contact

Project maintained by [Your Name]. For questions about architecture decisions, check the transcript history in Claude Chat.

## Git

Commit frequently with clear messages. Push after each working feature.

## Model Selection Guide

Use `/model claude-sonnet-4-20250514` (default) for:
- Creating simple files and functions
- Small edits, quick fixes
- Running tests and commands
- Iterative development
- Simple CRUD operations

Use `/model claude-opus-4-20250514` for:
- System architecture decisions
- Complex debugging (errors spanning multiple files)
- Refactoring across multiple files
- Large context analysis (understanding whole codebase)
- Code review and optimization
- When Sonnet fails 2+ times on same task

Rule: Start with Sonnet. Switch to Opus when stuck or task is complex.