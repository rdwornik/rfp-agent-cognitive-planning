# RFP Answer Engine – Cognitive Planning (v0.2)

This repo hosts an **RFP answer engine** for Blue Yonder's planning platform, focused on **Cognitive / SCP** RFPs.

The idea:  
Take historical presales answers → build a **canonical KB** → let LLMs draft answers for new RFPs in **CSV/Excel** form, in a way that is:

- **KB-first**: no hallucinated product facts.
- **Multi-LLM**: supports Gemini, Claude, GPT-5, DeepSeek, GLM, and more.
- **Batch-oriented**: works well with Excel exports.
- **Privacy-aware**: anonymization layer protects customer names.
- **Auditable**: easy to review, tweak, and re-run.

---

## What's New in v0.2

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
```bash
GEMINI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key      # Optional: for Claude
OPENAI_API_KEY=your_key         # Optional: for GPT-5
DEEPSEEK_API_KEY=your_key       # Optional: for DeepSeek
ZHIPU_API_KEY=your_key          # Optional: for GLM
```

### 2. Build KB
```bash
# Build canonical KB from raw data
python scripts/core/kb_build_canonical.py

# Index to ChromaDB (local, free)
python scripts/core/kb_embed_chroma.py
```

### 3. Run Batch Processor
```bash
# Test mode with Gemini
python scripts/core/rfp_batch_universal.py --test --model gemini

# Production with Claude + anonymization
python scripts/core/rfp_batch_universal.py --model claude --anonymize

# Short flags
python scripts/core/rfp_batch_universal.py -t -m deepseek -a -w 8
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
│   └── anonymization.yaml      # Blocklist and session config
├── data_kb/
│   ├── raw/                    # Raw historical RFP answers (CSV)
│   ├── canonical/              # Distilled KB (JSON)
│   └── chroma_store/           # Local vector database
├── input_rfp/                  # Production RFP files
├── input_rfp_test/             # Test RFP files
├── output_rfp_universal/       # Generated answers
├── logs/                       # Anonymization logs
├── prompts_instructions/
│   ├── rfp_system_prompt.txt           # Legacy (File Search)
│   └── rfp_system_prompt_universal.txt # Universal RAG prompt
├── scripts/
│   └── core/
│       ├── kb_build_canonical.py       # Build canonical KB
│       ├── kb_embed_chroma.py          # Index to ChromaDB
│       ├── llm_router.py               # Multi-LLM router
│       ├── rfp_batch_universal.py      # Universal batch processor
│       ├── rfp_batch_gemini_filesearch.py  # Legacy (File Search)
│       └── anonymization/              # Anonymization package
│           ├── config.py
│           ├── core.py
│           ├── middleware.py
│           ├── scan_kb.py
│           └── clean_kb.py
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
```

### KB Management
```bash
# Build canonical KB
python scripts/core/kb_build_canonical.py

# Re-index ChromaDB
python scripts/core/kb_embed_chroma.py
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

## Roadmap

- [ ] Add `--solution` flag for WMS/CatMan RFPs
- [ ] Hybrid mode: Google File Search + local RAG
- [ ] Local LLM support (Ollama + Mistral)
- [ ] A/B testing across models
- [ ] Cost tracking per batch

---

## License

Internal use only – Blue Yonder Presales.