"""
rfp_batch_gemini_filesearch.py

Batch engine for answering RFP CSVs using the canonical KB + Gemini File Search.

Usage (from repo root):
  # Test mode (small sample CSVs)
  python scripts/core/rfp_batch_gemini_filesearch.py test

  # Prod mode (full RFP files)
  python scripts/core/rfp_batch_gemini_filesearch.py

Behaviour:
  - Reads CSV files from:
      input_rfp_test/  (MODE = 'test')
      input_rfp/       (MODE = 'prod')
  - For each row:
      * extracts the customer question according to header heuristics
      * calls Gemini (with File Search) THREE times to get 3 answer variants
        using strict KB-first logic + relaxed fallback if needed
      * calls Gemini once more to fuse the 3 variants into kb_answer_final
  - Writes clean, Excel-friendly CSVs with columns:
      customer_question_out,kb_answer_1,kb_answer_2,kb_answer_3,kb_answer_final
  - Outputs go to:
      output_rfp_gemini_test/
      output_rfp_gemini/
"""

import os
import time
import csv
import re
from pathlib import Path
import sys
import threading
import concurrent.futures

import pandas as pd
from google import genai
from google.genai import types, errors

# === CONFIG ===
FILE_SEARCH_STORE_NAME = "fileSearchStores/rfpcognitiveplanningkbv2-6pqup4g1x9sm"
MODEL_NAME = "gemini-2.5-pro"
OUTPUT_SUFFIX = "_answers_gemini"

# Prefix used when relaxed / fallback mode was used
RELAXED_PREFIX = "[MODEL-GUESS] "

# Number of answer variants per question
NUM_VARIANTS = 3

# Parallelism (rows in parallel)
MAX_WORKERS = int(os.environ.get("RFP_MAX_WORKERS", "4"))

# __file__ = .../scripts/core/rfp_batch_gemini_filesearch.py
# parents[0] = scripts/core
# parents[1] = scripts
# parents[2] = repo root  ✅
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts_instructions" / "rfp_system_prompt.txt"

# MODE: "prod" (default) or "test" (if first CLI arg == "test")
MODE = "prod"
if len(sys.argv) > 1 and sys.argv[1].lower() == "test":
    MODE = "test"

if MODE == "test":
    INPUT_DIR = PROJECT_ROOT / "input_rfp_test"
    OUTPUT_DIR = PROJECT_ROOT / "output_rfp_gemini_test"
else:
    INPUT_DIR = PROJECT_ROOT / "input_rfp"
    OUTPUT_DIR = PROJECT_ROOT / "output_rfp_gemini"
# ==============

# Thread-local Gemini clients (one per worker thread)
_thread_local = threading.local()


def load_system_instructions(path: Path = SYSTEM_PROMPT_PATH) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_client() -> genai.Client:
    """
    Return a per-thread Gemini client (avoids sharing a single client across threads).
    """
    client = getattr(_thread_local, "client", None)
    if client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Environment variable GEMINI_API_KEY is not set.")
        client = genai.Client(api_key=api_key)
        _thread_local.client = client
    return client


def build_tools():
    """
    Build the File Search tool bound to your canonical KB store.
    """
    file_search_tool = types.Tool(
        file_search=types.FileSearch(file_search_store_names=[FILE_SEARCH_STORE_NAME])
    )
    return [file_search_tool]


def get_response_text(response) -> str:
    """Extract plain text from GenerateContentResponse."""
    if getattr(response, "text", None):
        return response.text

    candidates = getattr(response, "candidates", None)
    if not candidates:
        raise ValueError("Model returned no text and no candidates.")

    parts = []
    for cand in candidates:
        content = getattr(cand, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            if hasattr(part, "text") and part.text:
                parts.append(part.text)

    if not parts:
        raise ValueError("Model returned no text content in candidates.")

    return "".join(parts)


def fix_mojibake(text: str) -> str:
    """
    Fix common Windows-1252 -> UTF-8 mojibake artefacts like:
    - â€™  -> ’
    - â€“  -> –
    - â€œ -> “
    - â€ -> ”
    - Ã©  -> é
    Extend this map if you see other recurring patterns.
    """
    if not text:
        return text

    replacements = {
        "â€™": "’",
        "â€˜": "‘",
        "â€“": "–",
        "â€”": "—",
        "â€œ": "“",
        "â€": "”",
        "â€¢": "•",
        "Ã©": "é",
        "Ã¨": "è",
        "Ã¡": "á",
        "Ã³": "ó",
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    return text


def clean_model_answer(raw: str) -> str:
    """
    Clean model output so we end up with a single-line kb_answer.

    - Remove ``` fences if present.
    - Strip any stray 'customer_question_out,kb_answer' headers if the model ignores instructions.
    - Collapse whitespace/newlines into a single space (safe for CSV).
    - Fix common mojibake artefacts (e.g. 'Blue Yonderâ€™s' -> 'Blue Yonder’s').
    """
    if raw is None:
        return ""

    txt = str(raw).strip().replace("\r\n", "\n")
    txt = txt.replace("```csv", "").replace("```", "")

    # Defensive: strip header if the model still emits it
    txt = re.sub(
        r"customer_question_out\s*,\s*kb_answer",
        "",
        txt,
        flags=re.IGNORECASE,
    )

    # Strip empty lines and collapse to a single line
    lines = [ln.strip() for ln in txt.split("\n") if ln.strip()]
    if not lines:
        return ""

    txt = " ".join(lines).strip()
    txt = " ".join(txt.split())

    # Fix common mojibake issues (e.g. â€™)
    txt = fix_mojibake(txt)

    return txt


def safe_str(val) -> str:
    if pd.isna(val):
        return ""
    return str(val)


def get_question_from_row(row: pd.Series) -> str:
    """
    Derive customer_question_out from the input row itself.
    Priority:
      1) customer_question*
      2) column containing both 'functionality' and 'requirement'
      3) column containing 'description'
    """
    # 1) customer_question*
    for col in row.index:
        name_clean = col.replace(" ", "").replace("\n", "").lower()
        if "customer_question" in name_clean:
            val = safe_str(row[col]).strip()
            if val and val.lower() != "nan":
                return val

    # 2) "functionality" AND "requirement"
    for col in row.index:
        name_clean = col.replace(" ", "").replace("\n", "").lower()
        if "functionality" in name_clean and "requirement" in name_clean:
            val = safe_str(row[col]).strip()
            if val and val.lower() != "nan":
                return val

    # 3) "description"
    for col in row.index:
        name_clean = col.replace(" ", "").replace("\n", "").lower()
        if "description" in name_clean:
            val = safe_str(row[col]).strip()
            if val and val.lower() != "nan":
                return val

    return ""


def call_model_for_row(
    tools,
    system_text: str,
    df_one_row: pd.DataFrame,
) -> tuple[str | None, str | None]:
    """
    Strict KB-first call for a single row.

    Returns:
      (answer_text, None)  on success.
      (None, err_msg)      on error or unusable output.
    """
    client = get_client()

    # Build a tiny CSV: header + one data row
    input_csv = df_one_row.to_csv(index=False, encoding="utf-8-sig")

    user_prompt = (
        "You are processing exactly one row of an RFP CSV (header + one data row).\n"
        "Use the system instructions and the canonical KB (via File Search) to answer this row.\n"
        "For this automated tool, return ONLY the kb_answer text for this row as a single line of natural language.\n"
        "- You may use multiple sentences and inline lists separated by semicolons.\n"
        "- Do NOT output CSV headers, code fences, or the text 'customer_question_out'.\n"
        "- Avoid newline characters in the answer.\n\n"
        "Here is the single-row RFC-4180 CSV:\n"
        "```csv\n"
        f"{input_csv}\n"
        "```\n"
    )

    delay = 3
    while True:
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_text,
                    tools=tools,
                    temperature=0.4,
                    # Large budget per row so answers can be as detailed as needed
                    max_output_tokens=8192,
                ),
            )
            raw = get_response_text(response)
            cleaned = clean_model_answer(raw)

            if not cleaned:
                raise ValueError("Empty answer after cleaning.")

            return cleaned, None

        except errors.ServerError as e:
            code = getattr(e, "status_code", None)
            if code == 503:
                print(
                    f"    ! 503 UNAVAILABLE for this row. Retrying in {delay} seconds..."
                )
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue

            msg = f"Server error ({code}): {str(e)}"
            print(f"    ! {msg}")
            return None, msg

        except ValueError as e:
            msg = f"Unusable model output: {str(e)}"
            print(f"    ! {msg}")
            return None, msg

        except Exception as e:
            msg = f"Unexpected error: {str(e)}"
            print(f"    ! {msg}")
            return None, msg


def call_model_for_row_relaxed(
    tools,
    relaxed_system_text: str,
    df_one_row: pd.DataFrame,
) -> tuple[str | None, str | None]:
    """
    RELAXED fallback call for a single row.

    Used when the strict KB-only pass failed or returned 'Not in KB'.

    Behaviour:
      - Still tries to use KB via File Search.
      - If KB evidence is weak/missing, model MAY rely on its own domain knowledge.
      - Any such answer MUST be prefixed with '[MODEL-GUESS] '.
    """
    client = get_client()
    input_csv = df_one_row.to_csv(index=False, encoding="utf-8-sig")

    user_prompt = (
        "RELAXED FALLBACK MODE.\n"
        "The strict KB-only pass could not provide an answer for this row (it failed or would answer 'Not in KB').\n"
        "In this fallback mode you MAY rely on your own general domain knowledge when KB evidence is weak or missing,\n"
        "but you MUST prefix your answer with '[MODEL-GUESS] '.\n"
        "- Keep the answer high-level and conservative.\n"
        "- Do NOT invent specific SLAs, numbers, certifications, or binding commitments.\n"
        "- Return ONLY the kb_answer as a single line of natural language (no headers, no code fences, no newlines).\n\n"
        "Here is the same single-row RFC-4180 CSV:\n"
        "```csv\n"
        f"{input_csv}\n"
        "```\n"
    )

    delay = 3
    while True:
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=relaxed_system_text,
                    tools=tools,
                    temperature=0.5,
                    max_output_tokens=8192,
                ),
            )
            raw = get_response_text(response)
            cleaned = clean_model_answer(raw)

            if not cleaned:
                raise ValueError("Empty answer after cleaning (relaxed mode).")

            # Ensure the relaxed prefix is present
            lower = cleaned.strip().lower()
            if not lower.startswith(RELAXED_PREFIX.lower()):
                cleaned = RELAXED_PREFIX + cleaned.lstrip()

            return cleaned, None

        except errors.ServerError as e:
            code = getattr(e, "status_code", None)
            if code == 503:
                print(
                    f"    ! 503 UNAVAILABLE in relaxed mode for this row. Retrying in {delay} seconds..."
                )
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue

            msg = f"Server error in relaxed mode ({code}): {str(e)}"
            print(f"    ! {msg}")
            return None, msg

        except ValueError as e:
            msg = f"Unusable model output in relaxed mode: {str(e)}"
            print(f"    ! {msg}")
            return None, msg

        except Exception as e:
            msg = f"Unexpected error in relaxed mode: {str(e)}"
            print(f"    ! {msg}")
            return None, msg


def get_final_answer_for_row(
    tools,
    strict_system_text: str,
    relaxed_system_text: str,
    df_one_row: pd.DataFrame,
    variant_index: int,
) -> str:
    """
    Helper that encapsulates:
      - strict KB-first attempt
      - 'Not in KB' detection
      - relaxed fallback with [MODEL-GUESS] if needed
      - error handling

    Returns a single kb_answer string (never None).
    """
    # --- Strict KB-first pass ---
    answer_text, err_msg = call_model_for_row(
        tools=tools,
        system_text=strict_system_text,
        df_one_row=df_one_row,
    )

    use_relaxed = False
    strict_answer = answer_text  # may be None

    if answer_text is None:
        # Technical / parsing problem
        use_relaxed = True
        print(
            f"    -> Strict pass failed for this row (variant {variant_index}): {err_msg}"
        )
    else:
        lowered = answer_text.strip().strip('"').lower()
        if lowered == "not in kb":
            use_relaxed = True
            print(
                f"    -> Strict pass returned 'Not in KB' (variant {variant_index}). "
                "Trying relaxed fallback..."
            )

    final_answer = None

    if use_relaxed:
        # --- Relaxed fallback pass ---
        relaxed_answer, relaxed_err = call_model_for_row_relaxed(
            tools=tools,
            relaxed_system_text=relaxed_system_text,
            df_one_row=df_one_row,
        )

        if relaxed_answer is not None:
            final_answer = relaxed_answer
            print(
                f"    -> Relaxed fallback used for this row (variant {variant_index})."
            )
        else:
            # Relaxed also failed – fall back to whatever we had
            if strict_answer is not None:
                final_answer = strict_answer
            else:
                safe_err = (
                    (relaxed_err or "Unknown error")
                    .replace('"', "'")
                    .replace("\n", " ")
                    .replace("\r", " ")
                )
                final_answer = f"[Error v{variant_index}] {safe_err}"
    else:
        final_answer = answer_text

    if final_answer is None:
        # Extremely defensive fallback
        final_answer = f"[Error v{variant_index}] Unknown error"

    return final_answer


def aggregate_variants_for_row(
    tools,
    strict_system_text: str,
    question_text: str,
    variants: list[str],
    row_index: int,
) -> str:
    """
    Fuse kb_answer_1..3 into a single kb_answer_final.

    Behaviour (UNION OF DETAILS):
      - Include ALL distinct, non-contradictory technical details that appear
        in ANY of the three variants.
      - Do not aggressively 'simplify' or shorten if that would drop information.
      - Deduplicate only exact or obvious rephrasings.
      - Treat '[MODEL-GUESS]' variants as lower trust: include their content only
        when it is generic, safe, and consistent with the other variants.
      - Final answer MUST NOT contain '[MODEL-GUESS]'.
      - Returns a single-line, CSV-safe string.
    """
    # If no question or everything is empty, nothing to aggregate
    if not question_text or not question_text.strip():
        return ""

    non_empty = [v for v in variants if v and v.strip()]
    if not non_empty:
        return ""

    client = get_client()

    # Strip the prefix only for the model's view; the raw variants stay in the CSV
    cleaned_variants = [v.replace(RELAXED_PREFIX, "").strip() for v in variants]

    user_prompt = (
        "You are assisting with RFP answers for a SaaS supply chain planning platform.\n"
        "For a single customer requirement, you have three candidate kb_answer variants that were generated from the KB.\n"
        "Your task is to synthesize them into ONE best kb_answer.\n\n"
        "IMPORTANT – INCLUDE ALL EXTRA INFORMATION:\n"
        "- Consider all three variants as independent snippets of valid product knowledge.\n"
        "- Build the final answer as a UNION of their non-contradictory details.\n"
        "- If a concrete technical detail appears in only one variant (e.g. Workflow Orchestrator, Teams/SharePoint integration,\n"
        "  Snowflake Secure Data Sharing, SFTP/AS2, unified data cloud, BYDM, etc.), INCLUDE it in the final answer as long as it\n"
        "  does not clearly conflict with another variant.\n"
        "- Do NOT drop useful details just because they are not repeated in multiple variants.\n"
        "- Only remove text when it is an exact repetition or an obvious rephrasing of the same idea.\n\n"
        "Consistency and safety:\n"
        "- If there is a true contradiction, resolve it by:\n"
        "  * keeping the safest, high-level formulation, and\n"
        "  * omitting the conflicting specific claim.\n"
        "- Do NOT invent new product features, SLAs, numbers, certifications, or guarantees beyond what the variants state.\n"
        "- If any variant was originally marked '[MODEL-GUESS]', treat that content as lower trust: you may include it only when it\n"
        "  is generic, clearly safe, and consistent with the rest (no new numbers or hard commitments).\n\n"
        "Style:\n"
        "- The final kb_answer should be rich and detailed when the variants are rich.\n"
        "- It can be multiple sentences, but MUST be on a SINGLE line (no newlines) and suitable for a presales RFP answer.\n"
        "- Do NOT include the tag '[MODEL-GUESS]' anywhere in the final answer.\n"
        "- Do NOT echo the question text; just answer it.\n\n"
        f"Customer requirement:\n{question_text}\n\n"
        "Variant 1:\n"
        f"{cleaned_variants[0] if len(cleaned_variants) > 0 else ''}\n\n"
        "Variant 2:\n"
        f"{cleaned_variants[1] if len(cleaned_variants) > 1 else ''}\n\n"
        "Variant 3:\n"
        f"{cleaned_variants[2] if len(cleaned_variants) > 2 else ''}\n\n"
        "Now respond with ONLY the final kb_answer as a single line of natural language "
        "(no bullets, no code fences, no newlines)."
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=strict_system_text,
                temperature=0.4,
                max_output_tokens=1024,
            ),
        )
        raw = get_response_text(response)
        cleaned = clean_model_answer(raw)

        if not cleaned:
            raise ValueError("Empty fused answer after cleaning.")

        # Ensure we don't leak '[MODEL-GUESS]' into the final
        cleaned = cleaned.replace(RELAXED_PREFIX, "").strip()
        return cleaned

    except Exception as e:
        # Fallback: choose the longest non-empty non-[MODEL-GUESS] variant,
        # or if none, longest overall.
        print(
            f"    ! Aggregation failed for row {row_index + 1}: {e}. Falling back to longest variant."
        )
        non_guess = [
            v
            for v in variants
            if v and not v.strip().lower().startswith(RELAXED_PREFIX.lower())
        ]
        candidates = non_guess or non_empty
        best = max(candidates, key=lambda s: len(s))
        return best.replace(RELAXED_PREFIX, "").strip()


def process_single_file(
    tools,
    strict_system_text: str,
    relaxed_system_text: str,
    input_path: Path,
    output_path: Path,
):
    print(f"\n=== Processing file: {input_path.name} ===")

    try:
        df_in = pd.read_csv(input_path, encoding="utf-8")
    except UnicodeDecodeError:
        print("  ! UTF-8 failed, retrying with Latin-1 encoding...")
        df_in = pd.read_csv(input_path, encoding="latin1")

    total_rows = len(df_in)
    print(f"  Total input rows: {total_rows}")
    print(f"  Using up to {MAX_WORKERS} worker(s) in parallel.")

    header = ["customer_question_out"]
    for i in range(1, NUM_VARIANTS + 1):
        header.append(f"kb_answer_{i}")
    header.append("kb_answer_final")

    out_rows: list[list[str]] = [None] * total_rows  # type: ignore

    progress_lock = threading.Lock()
    completed = 0

    def worker(row_index: int):
        nonlocal completed

        row_series = df_in.iloc[row_index]
        question_text = get_question_from_row(row_series)

        # No question -> keep row, but NEVER call the model
        if not question_text or not question_text.strip():
            result = [""] + [""] * NUM_VARIANTS + [""]
        else:
            df_one = df_in.iloc[[row_index]]
            answers: list[str] = []
            for variant_index in range(1, NUM_VARIANTS + 1):
                ans = get_final_answer_for_row(
                    tools=tools,
                    strict_system_text=strict_system_text,
                    relaxed_system_text=relaxed_system_text,
                    df_one_row=df_one,
                    variant_index=variant_index,
                )
                answers.append(ans)

            fused = aggregate_variants_for_row(
                tools=tools,
                strict_system_text=strict_system_text,
                question_text=question_text,
                variants=answers,
                row_index=row_index,
            )

            result = [question_text] + answers + [fused]

        with progress_lock:
            completed += 1
            print(f"  - Row {completed}/{total_rows} ...")

        return row_index, result

    # Run rows in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(worker, idx): idx for idx in range(total_rows)}
        for future in concurrent.futures.as_completed(futures):
            row_idx, row_values = future.result()
            out_rows[row_idx] = row_values

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in out_rows:
            writer.writerow(row)

    print(f"  ✔ Wrote {len(out_rows)} rows to {output_path}")


def main():
    input_dir = INPUT_DIR
    output_dir = OUTPUT_DIR

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}.")
        print("Drop your RFP CSV files into that folder and run this script again.")
        return

    # Fail fast if key missing
    os.environ.get("GEMINI_API_KEY") or (_ for _ in ()).throw(
        RuntimeError("Environment variable GEMINI_API_KEY is not set.")
    )

    base_system = load_system_instructions()

    # Strict system prompt: your normal KB-first rules
    strict_system_text = base_system

    # Relaxed system prompt: same base, with explicit relaxed override at the end
    relaxed_system_text = (
        base_system + "\n\nRELAXED FALLBACK MODE OVERRIDE:\n"
        "If you would otherwise answer 'Not in KB' because the KB evidence is weak or missing,\n"
        "you may rely on your own general domain knowledge about SaaS-based supply-chain planning and Blue Yonder-like platforms.\n"
        "In this mode you MUST prefix kb_answer with '[MODEL-GUESS] ' and keep claims high-level and conservative.\n"
        "Do NOT invent specific SLAs, numbers, certifications, or binding commitments."
    )

    tools = build_tools()

    print(f"Mode: {MODE}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input dir:  {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Found {len(csv_files)} input file(s) in {input_dir}:")
    print(f"Max workers (rows in parallel): {MAX_WORKERS}")

    total_files = len(csv_files)
    for idx, p in enumerate(csv_files, start=1):
        print(f" - [{idx}/{total_files}] {p.name}")

    for idx, input_path in enumerate(csv_files, start=1):
        output_name = f"{input_path.stem}{OUTPUT_SUFFIX}.csv"
        output_path = output_dir / output_name
        print(f"\n>>> File {idx}/{total_files}: {input_path.name}")
        process_single_file(
            tools=tools,
            strict_system_text=strict_system_text,
            relaxed_system_text=relaxed_system_text,
            input_path=input_path,
            output_path=output_path,
        )

    print("\nAll files processed.")


if __name__ == "__main__":
    main()
