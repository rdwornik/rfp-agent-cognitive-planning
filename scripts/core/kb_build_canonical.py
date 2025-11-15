"""
kb_build_canonical.py

Builds a canonical, de-duplicated KB from the raw RFP answers.

Pipeline (from repo root):
  - Load data_kb/raw/RFP_Database_Cognitive_Planning.csv
  - Group similar questions by category + normalized question text
  - For each cluster, call Gemini to distill ONE canonical answer
    (time-aware: newer End Date answers override older ones)
  - Write canonical KB to:
      data_kb/canonical/RFP_Database_Cognitive_Planning_CANONICAL.json

Notes:
- The JSON file is the source of truth that will be uploaded into
  the Gemini File Search store.
- Distillation behaviour is controlled by:
    prompts_instructions/kb_distiller_prompt.txt
"""

import json
import os
import re
import time
from pathlib import Path
import concurrent.futures
import threading

import pandas as pd
from google import genai
from google.genai import errors as genai_errors, types

# --- paths & constants -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repo root
RAW_KB_DIR = PROJECT_ROOT / "data_kb" / "raw"
CANONICAL_KB_DIR = PROJECT_ROOT / "data_kb" / "canonical"

RAW_CSV_PATH = RAW_KB_DIR / "RFP_Database_Cognitive_Planning.csv"
CANONICAL_JSON_PATH = (
    CANONICAL_KB_DIR / "RFP_Database_Cognitive_Planning_CANONICAL.json"
)

DISTILLER_PROMPT_PATH = (
    PROJECT_ROOT / "prompts_instructions" / "kb_distiller_prompt.txt"
)

MAX_CANONICAL_CHARS = 1600  # your requested limit
CANONICAL_MODEL_NAME = "gemini-2.5-flash"  # aligned with your batch script

# How many clusters to distill in parallel.
# If you see a lot of 503s, drop this to 4–6.
MAX_WORKERS = 8

# Thread-local Gemini clients (one per worker thread)
_thread_local = threading.local()


def get_gemini_client() -> genai.Client:
    """Return a per-thread Gemini client (avoids sharing a single client across threads)."""
    client = getattr(_thread_local, "client", None)
    if client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        client = genai.Client(api_key=api_key)
        _thread_local.client = client
    return client


# ---------- helpers for headers & normalization ----------


def norm_header(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def build_header_alias(df: pd.DataFrame) -> dict:
    alias = {}
    for col in df.columns:
        alias[norm_header(col)] = col
    return alias


def get_col(df: pd.DataFrame, alias: dict, candidates) -> str | None:
    for c in candidates:
        key = norm_header(c)
        if key in alias:
            return alias[key]
    return None


def normalize_question_text(q: str) -> str:
    if not isinstance(q, str):
        return ""
    q = q.strip().lower()
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"[^a-z0-9 ]+", "", q)
    return q


# ---------- load prompt & raw CSV ----------


def load_distiller_prompt(path: Path = DISTILLER_PROMPT_PATH) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Distiller prompt not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def load_raw_df(path: Path) -> tuple[pd.DataFrame, dict]:
    # Encoding fallback
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    alias = build_header_alias(df)

    col_customer = get_col(df, alias, ["Customer"])
    col_end_date = get_col(df, alias, ["End Date", "EndDate"])
    col_category = get_col(df, alias, ["Category"])
    col_subcat = get_col(df, alias, ["Subcategory", "Sub-category"])
    col_question = get_col(df, alias, ["Customer Question", "customer_question"])
    col_tech_ans = get_col(
        df, alias, ["Tech Presales Answers", "Tech Presales Answers "]
    )

    if col_question is None or col_tech_ans is None:
        raise ValueError(
            "Missing required columns 'Customer Question' and/or "
            "'Tech Presales Answers' in raw CSV."
        )

    # synthetic row_id
    df["_row_id"] = [f"row_{i:05d}" for i in range(len(df))]

    # parse date
    if col_end_date:
        df["_end_dt"] = pd.to_datetime(df[col_end_date], errors="coerce")
        df["_end_dt"] = df["_end_dt"].fillna(pd.Timestamp("1900-01-01"))
    else:
        df["_end_dt"] = pd.Timestamp("1900-01-01")

    # normalized question (for clustering)
    df["_q_norm"] = df[col_question].apply(normalize_question_text)

    cat_series = df[col_category] if col_category else ""
    df["_cluster_key"] = (
        cat_series.fillna("").astype(str).str.strip().str.lower() + "||" + df["_q_norm"]
    )

    meta = {
        "col_customer": col_customer,
        "col_end_date": col_end_date,
        "col_category": col_category,
        "col_subcat": col_subcat,
        "col_question": col_question,
        "col_tech_ans": col_tech_ans,
    }

    return df, meta


def build_cluster_payload(cluster_id: str, rows: pd.DataFrame, meta: dict) -> dict:
    rows_sorted = rows.sort_values(by="_end_dt", ascending=True)

    payload_rows = []
    for _, r in rows_sorted.iterrows():
        payload_rows.append(
            {
                "row_id": r["_row_id"],
                "customer": (
                    r.get(meta["col_customer"]) if meta["col_customer"] else ""
                ),
                "end_date": (
                    r.get(meta["col_end_date"]) if meta["col_end_date"] else ""
                ),
                "category": (
                    r.get(meta["col_category"]) if meta["col_category"] else ""
                ),
                "subcategory": (
                    r.get(meta["col_subcat"]) if meta["col_subcat"] else ""
                ),
                "customer_question": r.get(meta["col_question"], ""),
                "tech_presales_answer": r.get(meta["col_tech_ans"], ""),
            }
        )

    first = rows_sorted.iloc[0]
    return {
        "cluster_id": cluster_id,
        "category": first.get(meta["col_category"], "") if meta["col_category"] else "",
        "subcategory": first.get(meta["col_subcat"], "") if meta["col_subcat"] else "",
        "rows": payload_rows,
    }


# ---------- Gemini distiller call ----------


def call_gemini_distiller(
    client: genai.Client,
    distiller_prompt: str,
    cluster_payload: dict,
    max_retries: int = 5,
) -> dict:
    """
    Call Gemini to distill one cluster into a canonical Q/A.

    - Uses the external kb_distiller_prompt.txt as 'distiller_prompt'.
    - Asks the model to return strict JSON (response_mime_type='application/json').
    """

    user_text = (
        distiller_prompt
        + "\n\n=== CLUSTER PAYLOAD (JSON) ===\n"
        + json.dumps(cluster_payload, ensure_ascii=False)
    )

    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=CANONICAL_MODEL_NAME,
                contents=[{"role": "user", "parts": [{"text": user_text}]}],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
            )

            text = getattr(resp, "text", None)
            if text is None and getattr(resp, "candidates", None):
                first_cand = resp.candidates[0]
                if first_cand.content.parts:
                    text = first_cand.content.parts[0].text

            if not text:
                raise ValueError("Distiller returned no text content.")

            return json.loads(text)

        except genai_errors.ServerError as e:
            msg = str(e)
            if ("503" in msg or "UNAVAILABLE" in msg) and attempt < max_retries - 1:
                wait = 2.0 * (attempt + 1)
                print(f"  503 UNAVAILABLE from distiller, retrying in {wait:.1f}s...")
                time.sleep(wait)
                continue
            raise

        except Exception:
            raise


# ---------- per-cluster worker (for parallelism) ----------


def process_single_cluster(args) -> dict | None:
    """
    Worker function executed in parallel for each cluster.

    args = (i, rows, meta, distiller_prompt)
    Returns a canonical_record dict or None if skipped/failed.
    """
    i, rows, meta, distiller_prompt = args

    # Skip clusters with empty normalized question
    if not rows["_q_norm"].iloc[0]:
        return None

    cluster_id = f"cluster_{i:04d}"
    print(f"Processing {cluster_id} (rows={len(rows)}) ...")

    payload = build_cluster_payload(cluster_id, rows, meta)
    newest = rows.sort_values(by="_end_dt", ascending=True).iloc[-1]

    try:
        client = get_gemini_client()
        distilled = call_gemini_distiller(
            client=client,
            distiller_prompt=distiller_prompt,
            cluster_payload=payload,
        )
    except Exception as e:
        print(f"  ⚠️ Distillation failed for {cluster_id}: {e}")
        return None

    canonical_record = {
        "kb_id": f"kb_{i:04d}",
        "category": newest.get(meta["col_category"], "")
        if meta["col_category"]
        else "",
        "subcategory": newest.get(meta["col_subcat"], "") if meta["col_subcat"] else "",
        "canonical_question": distilled.get("canonical_question", "").strip(),
        "canonical_answer": distilled.get("canonical_answer", "").strip()[
            :MAX_CANONICAL_CHARS
        ],
        "last_updated": distilled.get(
            "last_updated", str(newest.get(meta["col_end_date"], ""))
        ),
        "source_rows": distilled.get("sources_used", []),
        "deprecated_removed": distilled.get("deprecated_removed", []),
    }

    return canonical_record


# ---------- main ----------


def main():
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(f"Raw KB CSV not found: {RAW_CSV_PATH}")

    # Fail fast if key missing
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not set.")

    df, meta = load_raw_df(RAW_CSV_PATH)
    distiller_prompt = load_distiller_prompt()

    CANONICAL_KB_DIR.mkdir(parents=True, exist_ok=True)

    group = df.groupby("_cluster_key")
    total_clusters = len(group)
    print(f"Total rows: {len(df)}")
    print(f"Total clusters: {total_clusters}")

    # Prepare work items
    tasks = []
    for i, (_, rows) in enumerate(group, start=1):
        tasks.append((i, rows, meta, distiller_prompt))

    canonical_records: list[dict] = []

    print(f"\nStarting parallel distillation with {MAX_WORKERS} workers...")

    # Run clusters in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for rec in executor.map(process_single_cluster, tasks):
            if rec:
                canonical_records.append(rec)

    print(f"\nFinished processing. {len(canonical_records)} canonical records created.")

    # Sort for stable output order
    canonical_records.sort(key=lambda r: r["kb_id"])

    # Write JSON array (valid JSON)
    with CANONICAL_JSON_PATH.open("w", encoding="utf-8") as f_json:
        json.dump(canonical_records, f_json, ensure_ascii=False, indent=2)

    print(f"\n✅ Canonical JSON written to: {CANONICAL_JSON_PATH}")


if __name__ == "__main__":
    main()
