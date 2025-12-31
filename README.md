# RFP Answer Engine – Multi-Domain (v0.2.1)

This repo hosts an **RFP answer engine** for Blue Yonder solutions, supporting **Planning, AI/ML, WMS**, and future product lines.

The idea:
Take historical presales answers → build a **canonical KB** → let LLMs draft answers for new RFPs in **CSV/Excel** form, in a way that is:

- **KB-first**: no hallucinated product facts.
- **Multi-domain**: Planning, AI/ML, WMS knowledge in unified database.
- **Multi-LLM**: supports Gemini, Claude, GPT-5, DeepSeek, GLM, and more.
- **Batch-oriented**: works well with Excel exports.
- **Privacy-aware**: anonymization layer protects customer names.
- **Auditable**: easy to review, tweak, and re-run.

---

## What's New in v0.2.1 (Dec 2024)

| Feature | Description |
|---------|-------------|
| **Solution-Aware Responses** | `--solution` flag injects platform service context for product-specific answers |
| **Multi-Domain KB** | Unified KB with Planning (807), AI/ML (54), WMS (38) entries |
| **KB Workflow Tools** | `kb_transform_knowledge.py` + `kb_merge_canonical.py` for easy domain additions |
| **Scope Classification** | Auto-classify entries as `platform` vs `product_specific` |
| **Fixed RAG Retrieval** | Resolved "Not in KB" issue with domain-prefixed ChromaDB IDs |
| **Debug Mode** | `DEBUG_RAG=1` for detailed retrieval logging |

### Previous (v0.2)

| Feature | Description |
|---------|-------------|
| **Universal RAG** | Local ChromaDB with BGE embeddings (no vendor lock-in) |
| **Multi-LLM Router** | Switch between 9+ LLM providers with one flag |
| **Anonymization** | Protect customer names before API calls |
| **BGE Embeddings** | Upgraded from MiniLM to `bge-large-en-v1.5` for better retrieval |

---

## High-level Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     RFP Questions (CSV/Excel)               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  ANONYMIZATION LAYER (optional)                             │
│  - Remove customer names from blocklist                     │
│  - Replace with [CUSTOMER] placeholder                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  RETRIEVAL (ChromaDB + BGE-large)                           │
│  - Local vector database                                    │
│  - Top-k similar KB entries                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  LLM ROUTER                                                 │
│  - Gemini, Claude, GPT-5, DeepSeek, GLM, Kimi, Llama, Grok  │
│  - KB-first system prompt                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  DE-ANONYMIZATION                                           │
│  - Restore customer names in output                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Answered RFP (CSV)                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Setup
```bash
pip install -r requirements.txt
```

Create `.env` file:
```env
GEMINI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key      # Optional: for Claude
OPENAI_API_KEY=your_key         # Optional: for GPT-5
DEEPSEEK_API_KEY=your_key       # Optional: for DeepSeek
ZHIPU_API_KEY=your_key          # Optional: for GLM
```

### 2. Build KB

#### Option A: Transform New Knowledge (Recommended)
```bash
# 1. Transform raw knowledge (JSONL) to canonical format
python scripts/core/kb_transform_knowledge.py \
    --input data_kb/raw/knowledge_wms.jsonl \
    --domain wms \
    --source-type video_workshop \
    --version 2025.1

# 2. Merge all domain KBs into unified database
python scripts/core/kb_merge_canonical.py

# 3. Index to ChromaDB (local, free)
python scripts/core/kb_embed_chroma.py
```

#### Option B: Use Existing KB (Legacy)
```bash
# Build canonical KB from raw data
python scripts/core/kb_build_canonical.py

# Index to ChromaDB
python scripts/core/kb_embed_chroma.py
```

See [docs/KB_WORKFLOW.md](docs/KB_WORKFLOW.md) for detailed workflow documentation.

### 3. Run Batch Processor
```bash
# Test mode with Gemini
python scripts/core/rfp_batch_universal.py --test --model gemini

# Production with Claude + anonymization
python scripts/core/rfp_batch_universal.py --model claude --anonymize

# Solution-aware mode (platform service integration context)
python scripts/core/rfp_batch_universal.py --solution wms_native
python scripts/core/rfp_batch_universal.py --solution planning -m claude

# Combined flags (all short form)
python scripts/core/rfp_batch_universal.py -t -m deepseek -a -w 8 -s wms

# Debug mode (see what KB entries are retrieved)
set DEBUG_RAG=1  # Windows
DEBUG_RAG=1 python scripts/core/rfp_batch_universal.py -t  # Linux/Mac
```

---

## Supported LLM Models

| Model | Provider | Flag | Cost (per 1M tokens) |
|-------|----------|------|----------------------|
| Gemini 2.5 Pro | Google | `gemini` | $2 in / $12 out |
| Gemini 2.5 Flash | Google | `gemini-flash` | $0.15 in / $0.60 out |
| Claude Sonnet 4 | Anthropic | `claude` | $3 in / $15 out |
| GPT-5 | OpenAI | `gpt5` | $1.25 in / $10 out |
| DeepSeek V3 | DeepSeek | `deepseek` | $0.28 in / $0.42 out |
| GLM 4.7 | Zhipu | `glm` | $0.60 in / $2.20 out |
| Kimi K2 | Moonshot | `kimi` | $0.60 in / $2.05 out |
| Llama 4 Maverick | Together | `llama` | $0.27 in / $0.80 out |
| Grok 3 | xAI | `grok` | $3 in / $15 out |

---

## Anonymization System

Protect customer names before sending to external LLM APIs.

### Configure Blocklist

Edit `config/anonymization.yaml`:
```yaml
blocklist:
  kb_sources:
    - Acme Corp           # Customers whose data built the KB
  customers:
    - Walmart             # Other names to protect
    - Target
  projects:
    - Project Phoenix     # Internal project names
  internal: []

session:
  customer_name: "Carrefour"   # Current RFP customer
  placeholder: "[CUSTOMER]"

settings:
  anonymize_api_calls: true
  anonymize_local_calls: false
```

### CLI Commands
```bash
# Scan KB for sensitive terms
python -m scripts.core.anonymization.scan_kb

# Preview cleaning (dry run)
python -m scripts.core.anonymization.clean_kb --dry-run

# Clean KB
python -m scripts.core.anonymization.clean_kb

# Run with anonymization
python scripts/core/rfp_batch_universal.py -t -m gemini -a
```

### How It Works
```
Input:  "Does Walmart need SSO integration?"
    ↓ anonymize()
API:    "Does [CUSTOMER] need SSO integration?"
    ↓ LLM response
Output: "Blue Yonder supports SSO for [CUSTOMER]..."
    ↓ deanonymize()
Final:  "Blue Yonder supports SSO for Walmart..."
```

---

## Project Structure
```
.
├── config/
│   ├── anonymization.yaml             # Blocklist and session config
│   └── platform_matrix.json           # Platform services matrix
├── data_kb/
│   ├── raw/                           # Raw knowledge (JSONL from workshops)
│   ├── canonical/                     # Canonical KB files by domain
│   │   ├── RFP_Database_AIML_CANONICAL.json
│   │   ├── RFP_Database_Cognitive_Planning_CANONICAL.json
│   │   ├── RFP_Database_WMS_CANONICAL.json
│   │   └── RFP_Database_UNIFIED_CANONICAL.json  # ← Used by system
│   └── chroma_store/                  # Local vector database
├── docs/
│   ├── KB_WORKFLOW.md                 # KB transformation workflow guide
│   └── BUGFIX_NOT_IN_KB.md            # Recent bugfix documentation
├── input_rfp/                         # Production RFP files
├── input_rfp_test/                    # Test RFP files
├── output_rfp_universal/              # Generated answers
├── logs/                              # Anonymization logs
├── prompts_instructions/
│   ├── rfp_system_prompt.txt               # Legacy (File Search)
│   └── rfp_system_prompt_universal.txt     # Universal RAG prompt
├── scripts/
│   └── core/
│       ├── kb_transform_knowledge.py       # Transform JSONL → Canonical
│       ├── kb_merge_canonical.py           # Merge domain KBs → Unified
│       ├── kb_build_canonical.py           # Legacy KB builder
│       ├── kb_embed_chroma.py              # Index to ChromaDB
│       ├── llm_router.py                   # Multi-LLM router + RAG
│       ├── rfp_batch_universal.py          # Universal batch processor
│       ├── rfp_batch_gemini_filesearch.py  # Legacy (File Search)
│       └── anonymization/                  # Anonymization package
│           ├── config.py
│           ├── core.py
│           ├── middleware.py
│           ├── scan_kb.py
│           └── clean_kb.py
├── CLAUDE.md                          # Project context for Claude Code
├── requirements.txt
└── README.md
```

---

## CLI Reference

### Batch Processor
```bash
python scripts/core/rfp_batch_universal.py [OPTIONS]

Options:
  -t, --test          Use input_rfp_test/ folder
  -m, --model MODEL   LLM to use (default: gemini)
  -w, --workers N     Parallel workers (default: 4)
  -a, --anonymize     Enable anonymization
  -s, --solution CODE Solution-aware context (41 solutions available; see config/platform_matrix.json)
```

### KB Management
```bash
# Transform new knowledge
python scripts/core/kb_transform_knowledge.py -i data_kb/raw/knowledge.jsonl -d wms

# Merge all domain KBs
python scripts/core/kb_merge_canonical.py

# Re-index ChromaDB
python scripts/core/kb_embed_chroma.py

# Legacy KB builder
python scripts/core/kb_build_canonical.py
```

### Anonymization
```bash
# Scan KB
python -m scripts.core.anonymization.scan_kb

# Clean KB
python -m scripts.core.anonymization.clean_kb [--dry-run]
```

---

## Legacy: Google File Search

The original File Search approach is still available:
```bash
python scripts/core/rfp_batch_gemini_filesearch.py
```

This uses Google's hosted File Search with the store:  
`fileSearchStores/rfpcognitiveplanningkbv2-6pqup4g1x9sm`

---

## Knowledge Base Statistics

| Domain | Entries | Scope: Platform | Scope: Product-Specific |
|--------|---------|-----------------|-------------------------|
| Planning | 807 | - | - |
| AI/ML | 54 | - | - |
| WMS | 38 | 10 | 28 |
| **Total** | **899** | **10** | **28** |

Knowledge is automatically classified by:
- **Domain**: planning, aiml, wms, catman, logistics
- **Scope**: platform (shared across products) vs product_specific
- **Category**: SLA, Integration, Security, WMS Features, etc.

---

## Documentation

- **[KB Workflow Guide](docs/KB_WORKFLOW.md)** - How to add new knowledge to the system
- **[Bugfix: "Not in KB"](docs/BUGFIX_NOT_IN_KB.md)** - Resolved RAG retrieval issue
- **[Platform Context Guide](docs/platform_context.md)** - Response framing for solution-aware RFP answers
- **[CLAUDE.md](CLAUDE.md)** - Project context for AI assistants

---

## Solution-Aware Response System

The `--solution` flag enables product-specific platform service context injection:

**How it works:**
1. Loads `config/platform_matrix.json` with 41 solutions and platform service statuses
2. Injects solution-specific context into LLM prompts
3. Adjusts response framing based on integration level:
   - **Native**: "[Capability] for [Product] is configured through Blue Yonder Platform"
   - **Planned**: "Blue Yonder Platform supports this on infrastructure level and full native integration is planned"
   - **Infrastructure**: "Blue Yonder Platform supports this functionality on an infrastructure level"

**Available Solutions (41):**
- Planning: `planning`, `planning_ibp`, `planning_pps`
- WMS: `wms`, `wms_native`, `wms_labor`, `wms_tasking`, `wms_billing`, `wms_robotics`
- Logistics: `logistics`, `logistics_ba`, `logistics_modeling`, `logistics_fom`, etc.
- Retail: `retail_ar`, `retail_ap`, `retail_clearance`, `retail_markdown`, etc.
- And more (see `config/platform_matrix.json`)

**Example Usage:**
```bash
python rfp_batch_universal.py --solution wms_native -m claude
```

---

## Roadmap

- [x] Multi-domain KB support (Planning, AI/ML, WMS)
- [x] Scope classification (platform vs product_specific)
- [x] Debug mode for RAG retrieval
- [x] Solution-aware response system with platform service context
- [ ] Deprecation system for versioned KB entries
- [ ] Add CatMan and Logistics domains
- [ ] Hybrid mode: Google File Search + local RAG
- [ ] Local LLM support (Ollama + Mistral)
- [ ] A/B testing across models
- [ ] Cost tracking per batch

---

## License

Internal use only – Blue Yonder Presales.