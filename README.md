# RFP Agent – Cognitive Planning (Gemini + File Search)

This project is a small Python-based toolkit that helps generate consistent,
canonical RFP answers for Blue Yonder Cognitive Planning using Gemini and
File Search.

The goal is to:
- Clean and canonicalize a messy historical RFP KB.
- Upload the canonical KB into a Gemini File Search store.
- Batch-answer new RFP Excel/CSV files with pragmatic, SaaS-first answers.

> **Important:** The repository intentionally excludes customer RFP files and
> internal KB content via `.gitignore`. Only code and prompts live in Git.

---

## Project layout

```text
.
├─ data_kb/
│  ├─ raw/         # internal KB (Excel/CSV/JSON/JSONL) – gitignored
│  └─ canonical/   # canonical KB JSON – gitignored
│
├─ input_rfp/          # full / production RFP input CSVs – gitignored
├─ input_rfp_test/     # small test CSVs (first 10 rows, etc.) – gitignored
├─ output_rfp_gemini/      # batch answers (prod) – gitignored
├─ output_rfp_gemini_test/ # batch answers (test) – gitignored
│
├─ prompts_instructions/
│  └─ rfp_system_prompt.txt   # system instructions used by batch scripts
│
├─ scripts/
│  ├─ core/
│  │  ├─ kb_build_canonical.py
│  │  ├─ rfp_batch_gemini_filesearch.py
│  │  └─ rfp_query_test.py
│  ├─ maintenance/
│  │  ├─ gemini_key_test.py
│  │  ├─ upload_canonical_kb.py
│  │  └─ upload_kb_v2.py
│  └─ archive/   # legacy / one-off scripts kept for reference
│
├─ .gitignore
├─ requirements.txt
└─ README.md
