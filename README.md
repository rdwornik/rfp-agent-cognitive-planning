# RFP Agent – Cognitive Planning

An experimental **RFP Answer Engine** for Blue Yonder cognitive planning deals.

The goal of this project is to:
1. Build a **canonical, de-duplicated knowledge base (KB)** from many historical RFP answers.
2. Upload that KB into a **Gemini File Search Store**.
3. Use that KB + a controlled system prompt to **auto-draft answers** to customer RFP CSVs.

This repo is designed so that:
- Humans can understand and modify the flow.
- Different LLMs (ChatGPT, Gemini, Grok, etc.) can safely help evolve the code and prompts.

---

## High-Level Architecture

There are three main workflows:

1. **Canonical KB Build**  
   Source of truth: `data_kb/raw/RFP_Database_Cognitive_Planning.csv`  
   - Group similar questions by `Category` + normalized question text.  
   - For each cluster, call Gemini once to **distill a single canonical answer**, taking
     the **newest End Date** as authoritative and treating older answers as “supporting detail”.  
   - Output: `data_kb/canonical/RFP_Database_Cognitive_Planning_CANONICAL.json`  
   - This JSON is what gets uploaded to the File Search store.

2. **KB Upload to Gemini File Search Store**  
   - Take the canonical JSON file and upload it into an existing  
     `FILE_SEARCH_STORE_NAME` using the `google.genai` SDK.
   - This store is then used by the batch RFP answering script.

3. **Batch RFP Answering (CSV → CSV)**  
   - Input: customer RFPs as CSVs (various column layouts).  
   - For each non-empty “question-like” row:
     - Derive a `customer_question_out` from the row
       (prefer `customer_question`, otherwise fall back to `Functionality/Requirement`, then `description`).
     - Call Gemini with:
       - The **system prompt**: `prompts_instructions/rfp_system_prompt.txt`
       - The File Search tool bound to your canonical KB.
     - Expect back a **single paragraph** answer only.
   - Output: simple, Excel-friendly CSV with two columns:  
     `customer_question_out,kb_answer`.

All **customer data** and **KB content** stay local and are `.gitignore`’d so the GitHub repo
only contains code, prompts, and project scaffolding.

---

## Repo Layout

```text
repo-root/
  data_kb/
    raw/         # raw, messy RFP KB (NOT tracked in git by default)
    canonical/   # distilled canonical KB JSON (also ignored by default)

  input_rfp/           # full / production RFP CSVs (ignored)
  input_rfp_test/      # small test CSVs for debugging (ignored)
  output_rfp_gemini/   # batch outputs for prod (ignored)
  output_rfp_gemini_test/ # batch outputs for test (ignored)

  prompts_instructions/
    rfp_system_prompt.txt       # main system prompt for answering RFPs
    kb_distiller_prompt.txt     # system prompt for canonical KB distillation
    ... possibly other prompt drafts

  scripts/
    core/
      kb_build_canonical.py         # build canonical KB (raw → canonical JSON)
      rfp_batch_gemini_filesearch.py# batch-answer RFP CSVs using File Search
      rfp_query_test.py             # quick single-question KB sanity check
      excel_to_jsonl.py             # helper to convert Excel/CSV → JSON/JSONL

    maintenance/
      create_store_v2.py            # create File Search store (one-time / rare)
      gemini_key_test.py            # test GEMINI_API_KEY & basic API connectivity
      upload_canonical_kb.py        # upload canonical JSON to File Search store
      upload_kb_v2.py               # upload raw KB (older / experimental)

    archive/
      create_canonical_store.py     # older, superseded scripts
      create_new_store_v2.py
      setup_rfp_file_search_store.py
      upload_kb_to_store.py

  tmp_rfps_to_be_answered/   # local parking lot of RFPs you haven’t run yet (ignored)

  .gitignore
  README.md
  requirements.txt
  project_tree.txt
  project_files.txt
