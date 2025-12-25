# RFP Answer Engine – Cognitive Planning (v0.1)

This repo hosts an experimental **RFP answer engine** for Blue Yonder’s planning platform, focused on **Cognitive / SCP** RFPs.

The idea:  
Take historical presales answers → build a **canonical KB** → let Gemini + File Search draft answers for new RFPs in **CSV/Excel** form, in a way that is:

- **KB-first**: no hallucinated product facts.
- **Batch-oriented**: works well with Excel exports.
- **Auditable**: easy to review, tweak, and re-run.

We’re not “done”; this is a solid v0.1 with a clear, known limitation: **Gemini still struggles to consistently pull the full 1–2k-character canonical answers into the final RFP answers**. That’s the main open problem for the next iteration.

---

## High-level architecture

### 1. Canonical KB builder

**Script:** `scripts/core/kb_build_canonical.py`  
**Source:** `data_kb/raw/RFP_Database_Cognitive_Planning.csv`  
**Output:** `data_kb/canonical/RFP_Database_Cognitive_Planning_CANONICAL.json`

Pipeline:

1. Load raw historical RFP answers (Cognitive / SCP scope).
2. Normalize and cluster **similar questions** by:
   - Solution / category / subcategory
   - Normalized question text
3. For each cluster:
   - Ask Gemini to **distill a single canonical_answer**, respecting:
     - Newer “End Date” answers override older ones.
     - Tech Presales answers are preferred where available.
4. Write a canonical, de-duplicated KB JSON file.

This JSON is then loaded into a **Google Gemini File Search store**, which becomes the **only source of product truth** for the batch answering scripts.

---

### 2. Batch RFP answer engine (CSV in → CSV out)

**Script:** `scripts/core/rfp_batch_gemini_filesearch.py`  
**Inputs:**

- CSVs exported from Excel:
  - `input_rfp/`         (prod)
  - `input_rfp_test/`    (test)
- Environment:
  - `GEMINI_API_KEY` set
  - File Search store name:  
    `fileSearchStores/rfpcognitiveplanningkbv2-6pqup4g1x9sm`
- System prompt: `prompts_instructions/rfp_system_prompt.txt`

**Outputs:**

- `output_rfp_gemini/`  
- `output_rfp_gemini_test/`  
Each file:  
`<input_name>_answers_gemini.csv`

#### What the engine does per row

1. **Extract the question** from the RFP row using header heuristics:

   Priority:
   1. First non-empty column whose header contains `customer_question`
   2. Else first non-empty column whose header contains **both** `functionality` and `requirement`
   3. Else first non-empty column whose header contains `description`

   If no such column has text → we treat it as a **blank question row** and emit an empty output row (to keep the CSV aligned with Excel).

2. **Strict KB-first call (Gemini + File Search)**

   - Use the system prompt (`rfp_system_prompt.txt`) which enforces:
     - Canonical KB as **only** source of product facts.
     - Copy-first behaviour from `canonical_answer`.
     - SaaS-first, no new SLAs, no made-up certifications.
   - Model returns **one single-line kb_answer** (no headers, no CSV).

3. **Relaxed fallback when needed**

   - If the strict pass fails technically or responds with `Not in KB`, we:
     - Retry in **RELAXED FALLBACK MODE**:
       - Model may use generic SaaS / planning domain knowledge.
       - Must prefix answer with `[MODEL-GUESS] `.
       - Must stay high-level and conservative (no new numbers / SLAs).
   - If relaxed also fails:
     - We return a safe error text like `[Error v1] …`.

4. **Multi-variant answers per row**

   For each question we generate **three independent variants**:

   - `kb_answer_1`
   - `kb_answer_2`
   - `kb_answer_3`

   Each variant runs the strict-then-relaxed logic independently, so you get slightly different wording and emphasis. This is intentional: the human reviewer can pick their preferred version or use them as material to refine.

5. **Fusion step: aggregate the 3 variants into `kb_answer_final`**

   We run a **separate Gemini call** to fuse the three variants into a single answer:

   - Goal is to create a **UNION of details**, not a minimal summary.
   - Rules:
     - Include **all non-contradictory technical details** that appear in *any* variant:  
       PDC, BYDM, Workflow Orchestrator, Data Functions, Snowflake Secure Data Share, SFTP/AS2, Teams/SharePoint integration, etc.
     - Only deduplicate exact / obvious rephrasings.
     - Resolve contradictions by keeping the safest high-level statement and dropping conflicting specifics.
     - Treat `[MODEL-GUESS]` content as **lower trust**; only include it if generic and clearly safe.
     - Strip `[MODEL-GUESS]` from the final answer.
   - Output: `kb_answer_final` – one CSV-safe, single-line RFP answer.

6. **Parallel execution**

   - Rows are processed in parallel using a `ThreadPoolExecutor`.
   - Each worker thread has its own Gemini client (thread-local) to avoid issues with shared state.
   - `RFP_MAX_WORKERS` env var controls parallelism (default `4`).

Final CSV columns:

```text
customer_question_out,kb_answer_1,kb_answer_2,kb_answer_3,kb_answer_final
```

---

## Project Structure

```text
.
├── data_kb/
│   ├── raw/                # Raw historical RFP answers (CSV)
│   └── canonical/          # Distilled, de-duplicated KB (JSON)
├── input_rfp/              # Production RFP CSVs to be answered
├── input_rfp_test/         # Test RFP CSVs
├── output_rfp_gemini/      # Generated answers for production
├── output_rfp_gemini_test/ # Generated answers for testing
├── prompts_instructions/   # System prompts and instructions (Gemini logic)
│   ├── rfp_system_prompt.txt
│   └── kb_distiller_prompt.txt
├── scripts/
│   ├── core/               # Main logic: KB builder and Batch engine
│   ├── maintenance/        # KB upload and key verification
│   ├── custom/             # Custom processing scripts
│   └── archive/            # Old versions or experimental scripts
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Usage
1. **Setup:** `pip install -r requirements.txt` and create `.env` with `GEMINI_API_KEY`.
2. **Build KB:** `python scripts/core/kb_build_canonical.py`
3. **Run Batch:** `python scripts/core/rfp_batch_gemini_filesearch.py`
