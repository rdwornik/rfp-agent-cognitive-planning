"""
rfp_batch_gemini_filesearch.py

Main batch engine for answering RFP CSVs using the canonical KB + Gemini File Search.

Usage (from repo root):
  # Test mode (small sample CSVs)
  python scripts/core/rfp_batch_gemini_filesearch.py test

  # Prod mode (full RFP files)
  python scripts/core/rfp_batch_gemini_filesearch.py

Behaviour:
  - Reads CSV files from:
      input_rfp_test/  (MODE = 'test')
      input_rfp/       (MODE = 'prod')
  - For each row, derives customer_question_out from the input columns
  - Calls Gemini with the canonical KB (File Search store) to generate kb_answer
  - Writes clean, Excel-friendly CSVs to:
      output_rfp_gemini_test/
      output_rfp_gemini/
"""

import os
import time
import csv
import re
from pathlib import Path
import sys

import pandas as pd
from google import genai
from google.genai import types, errors

# === CONFIG ===
FILE_SEARCH_STORE_NAME = "fileSearchStores/rfpcognitiveplanningkbv2-6pqup4g1x9sm"
MODEL_NAME = "gemini-2.5-flash"
OUTPUT_SUFFIX = "_answers_gemini"

# __file__ = .../scripts/core/rfp_batch_gemini_filesearch.py
# parents[0] = scripts/core
# parents[1] = scripts
# parents[2] = repo root  ✅
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# System prompt lives under prompts_instructions/ at the root
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


def load_system_instructions(path: Path = SYSTEM_PROMPT_PATH) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable GEMINI_API_KEY is not set.")
    return genai.Client(api_key=api_key)


def build_tools():
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


def clean_model_answer(raw: str) -> str:
    """
    Clean whatever the model returned and try to get just the kb_answer paragraph.
    - Remove ``` fences
    - Remove CSV headers like 'customer_question_out,kb_answer'
    - Collapse whitespace
    """
    if raw is None:
        return ""

    txt = str(raw).strip().replace("\r\n", "\n")

    txt = txt.replace("```csv", "").replace("```", "")

    txt = re.sub(
        r"customer_question_out\s*,\s*kb_answer",
        "",
        txt,
        flags=re.IGNORECASE,
    )

    lines = [ln.strip() for ln in txt.split("\n") if ln.strip()]
    if not lines:
        return ""

    if "customer_question_out" in lines[0].lower() and "kb_answer" in lines[0].lower():
        lines = lines[1:] or []

    txt = " ".join(lines).strip()
    txt = " ".join(txt.split())
    return txt


def call_model_for_row(
    client,
    tools,
    system_text: str,
    df_one_row: pd.DataFrame,
) -> tuple[str | None, str | None]:
    """
    Call Gemini for a single row.

    Returns:
      (answer_text, None)  on success.
      (None, err_msg)      on non-503 error or unusable output.

    503 => retried forever for this row (with backoff).
    """
    input_csv = df_one_row.to_csv(index=False, encoding="utf-8-sig")
    user_prompt = (
        "Here is a single-row RFC-4180 CSV batch:\n"
        "```csv\n"
        f"{input_csv}\n"
        "```\n"
        "Answer ONLY with the kb_answer text as one plain paragraph. "
        "Do NOT return CSV, tables, code fences, or headers.\n"
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
    for col in row.index:
        name_clean = col.replace(" ", "").replace("\n", "").lower()
        if "customer_question" in name_clean:
            val = safe_str(row[col]).strip()
            if val and val.lower() != "nan":
                return val

    for col in row.index:
        name_clean = col.replace(" ", "").replace("\n", "").lower()
        if "functionality" in name_clean and "requirement" in name_clean:
            val = safe_str(row[col]).strip()
            if val and val.lower() != "nan":
                return val

    for col in row.index:
        name_clean = col.replace(" ", "").replace("\n", "").lower()
        if "description" in name_clean:
            val = safe_str(row[col]).strip()
            if val and val.lower() != "nan":
                return val

    return ""


def process_single_file(
    client, tools, system_text: str, input_path: Path, output_path: Path
):
    print(f"\n=== Processing file: {input_path.name} ===")

    try:
        df = pd.read_csv(input_path, encoding="utf-8")
    except UnicodeDecodeError:
        print("  ! UTF-8 failed, retrying with Latin-1 encoding...")
        df = pd.read_csv(input_path, encoding="latin1")

    total_rows = len(df)
    print(f"  Total rows: {total_rows}")

    header = ["customer_question_out", "kb_answer"]
    rows: list[tuple[str, str]] = []

    for idx in range(total_rows):
        print(f"  - Row {idx}...")
        row_series = df.iloc[idx]
        question_text = get_question_from_row(row_series)

        if not question_text:
            rows.append(("", ""))
            continue

        df_one = df.iloc[[idx]]

        answer_text, err_msg = call_model_for_row(client, tools, system_text, df_one)

        if answer_text is None:
            safe_err = (
                (err_msg or "Unknown error")
                .replace('"', "'")
                .replace("\n", " ")
                .replace("\r", " ")
            )
            rows.append((question_text, f"[Error] {safe_err}"))
        else:
            rows.append((question_text, answer_text))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for q, a in rows:
            writer.writerow([q, a])

    print(f"  ✔ Wrote {len(rows)} rows to {output_path}")


def main():
    # Use the already-resolved root-based dirs
    input_dir = INPUT_DIR
    output_dir = OUTPUT_DIR

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}.")
        print("Drop your RFP CSV files into that folder and run this script again.")
        return

    base_system = load_system_instructions()

    system_text = (
        "FOR THIS AUTOMATED TOOL:\n"
        "- The user sends a single-row CSV batch.\n"
        "- You MUST answer ONLY with the kb_answer text as one plain paragraph.\n"
        "- Do NOT return CSV, markdown, bullet points, headers, or code fences.\n"
        "- Do NOT include 'customer_question_out' or 'kb_answer' in your answer.\n"
        "- Just output the kb_answer sentence/paragraph, nothing else.\n\n"
        + base_system
    )

    client = get_client()
    tools = build_tools()

    print(f"Mode: {MODE}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input dir:  {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Found {len(csv_files)} input file(s) in {input_dir}:")
    for p in csv_files:
        print(f" - {p.name}")

    for input_path in csv_files:
        output_name = f"{input_path.stem}{OUTPUT_SUFFIX}.csv"
        output_path = output_dir / output_name
        process_single_file(client, tools, system_text, input_path, output_path)

    print("\nAll files processed.")


if __name__ == "__main__":
    main()
