"""
rfp_add_ootb_flag.py

Post-process RFP answer CSVs and add a custom evaluation column, e.g.
whether each requirement is Out of the box vs Development.

Pipeline (from repo root):
  1) Run the main batch engine to generate answers:
       python scripts/core/rfp_batch_gemini_filesearch.py
     -> writes CSVs to output_rfp_gemini/  (or output_rfp_gemini_test/ in test mode)

  2) Run this script to add the extra evaluation column:
       scripts/custom/rfp_add_ootb_flag.py
       scripts/custom/rfp_add_ootb_flag.py test   # for test mode

Behaviour:
  - Reads CSVs from:
      output_rfp_gemini/       (MODE = 'prod')
      output_rfp_gemini_test/  (MODE = 'test')
  - For each row, looks at:
      customer_question_out, kb_answer
  - Calls Gemini to classify each row as:
      "Development (Estimation of expenditure necessary)" OR
      "Out of the box (No estimation of expenditure necessary)"
  - Writes new CSVs with an extra column to:
      output_rfp_gemini_flagged/
      output_rfp_gemini_test_flagged/
"""

import os
import sys
import csv
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types, errors

# === CONFIG ===

MODEL_NAME = "gemini-2.5-pro"

# Name of the new column
FLAG_COLUMN_NAME = "Implementation_Nature"

# Exact labels required by the customer
FLAG_DEV = "Development (Estimation of expenditure necessary)"
FLAG_OOTB = "Out of the box (No estimation of expenditure necessary)"

# MODE: "prod" (default) or "test" (if first CLI arg == "test")
MODE = "prod"
if len(sys.argv) > 1 and sys.argv[1].lower() == "test":
    MODE = "test"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if MODE == "test":
    INPUT_DIR = PROJECT_ROOT / "output_rfp_gemini_test"
    OUTPUT_DIR = PROJECT_ROOT / "output_rfp_gemini_test_flagged"
else:
    INPUT_DIR = PROJECT_ROOT / "output_rfp_gemini"
    OUTPUT_DIR = PROJECT_ROOT / "output_rfp_gemini_flagged"
# =====================


def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable GEMINI_API_KEY is not set.")
    return genai.Client(api_key=api_key)


def safe_str(val) -> str:
    if pd.isna(val):
        return ""
    return str(val)


def classify_row(client, question: str, answer: str) -> str:
    """
    Ask Gemini to decide if a requirement is Out of the box or Development.

    Returns one of:
      FLAG_DEV  or  FLAG_OOTB

    If anything goes wrong, default to FLAG_DEV (conservative).
    """
    # Very small, focused prompt
    user_prompt = (
        "You are assisting with evaluating RFP requirements for a SaaS planning platform.\n"
        "Given the customer's requirement and our kb_answer, decide whether the requirement is\n"
        "covered Out of the box (configurable using standard capabilities) or requires Development\n"
        "(non-standard customization or additional development effort).\n\n"
        f"Requirement (customer_question_out):\n{question}\n\n"
        f"Answer (kb_answer):\n{answer}\n\n"
        "Choose exactly ONE of the following options:\n"
        f"1) {FLAG_DEV}\n"
        f"2) {FLAG_OOTB}\n\n"
        "Rules:\n"
        f"- If the answer clearly describes using standard, existing capabilities or configuration only,\n"
        f"  choose: {FLAG_OOTB}\n"
        f"- If the answer mentions custom development, custom code, bespoke extensions, or if you are unsure\n"
        f"  whether it is standard, choose: {FLAG_DEV} (be conservative).\n\n"
        "Respond with exactly ONE of the two strings above, nothing else."
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=64,
            ),
        )

        text = getattr(response, "text", "") or ""
        text = text.strip().replace("\r\n", "\n")

        # Use only the first non-empty line
        first_line = ""
        for line in text.split("\n"):
            line = line.strip()
            if line:
                first_line = line
                break

        # Normalise whitespace
        first_line = " ".join(first_line.split())

        # Map loosely to the exact labels
        lower = first_line.lower()
        if "out of the box" in lower:
            return FLAG_OOTB
        if "development" in lower:
            return FLAG_DEV

        # Fallback if model responded strangely
        return FLAG_DEV

    except errors.ServerError as e:
        print(f"    ! Server error during classification: {e}")
        return FLAG_DEV
    except Exception as e:
        print(f"    ! Unexpected error during classification: {e}")
        return FLAG_DEV


def process_single_file(client, input_path: Path, output_path: Path):
    print(f"\n=== Flagging file: {input_path.name} ===")

    try:
        df = pd.read_csv(input_path, encoding="utf-8")
    except UnicodeDecodeError:
        print("  ! UTF-8 failed, retrying with Latin-1 encoding...")
        df = pd.read_csv(input_path, encoding="latin1")

    if "customer_question_out" not in df.columns or "kb_answer" not in df.columns:
        print(
            "  ! Skipping: required columns 'customer_question_out' or 'kb_answer' not found."
        )
        return

    total_rows = len(df)
    print(f"  Total input rows: {total_rows}")

    flags: list[str] = []

    for idx in range(total_rows):
        print(f"  - Row {idx + 1}/{total_rows} ...")
        q = safe_str(df.loc[idx, "customer_question_out"]).strip()
        a = safe_str(df.loc[idx, "kb_answer"]).strip()

        if not q and not a:
            flags.append("")
            continue

        flag_value = classify_row(client, q, a)
        flags.append(flag_value)

    df[FLAG_COLUMN_NAME] = flags

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  ✔ Wrote flagged CSV to {output_path}")


def main():
    input_dir = INPUT_DIR
    output_dir = OUTPUT_DIR

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}.")
        print("Run rfp_batch_gemini_filesearch.py first to generate answer CSVs.")
        return

    client = get_client()

    print(f"Mode: {MODE}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input dir:  {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Found {len(csv_files)} input file(s) in {input_dir}:")

    total_files = len(csv_files)
    for idx, p in enumerate(csv_files, start=1):
        print(f" - [{idx}/{total_files}] {p.name}")

    for idx, input_path in enumerate(csv_files, start=1):
        output_name = input_path.name.replace(".csv", "_flagged.csv")
        output_path = output_dir / output_name
        print(f"\n>>> File {idx}/{total_files}: {input_path.name}")
        process_single_file(client, input_path, output_path)

    print("\nAll files processed.")


if __name__ == "__main__":
    main()
