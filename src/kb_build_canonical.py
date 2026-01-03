"""
kb_build_canonical.py

Builds a canonical, de-duplicated KB from the raw RFP answers.

Pipeline (from repo root):
  - Load data_kb/raw/RFP_Database_AIML.csv
  - Group similar questions by category + normalized question text
  - For each cluster, call Gemini to distill ONE canonical answer
    (time-aware: newer End Date answers override older ones)
  - Write canonical KB to:
      data_kb/canonical/RFP_Database_AIML_CANONICAL.json
      data_kb/canonical/RFP_Database_AIML_CANONICAL.csv

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
from dotenv import load_dotenv

# --- paths & constants -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # repo root
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)
RAW_KB_DIR = PROJECT_ROOT / "data" / "kb" / "raw"
CANONICAL_KB_DIR = PROJECT_ROOT / "data" / "kb" / "canonical"

RAW_CSV_PATH = RAW_KB_DIR / "RFP_Database_AIML.csv"

CANONICAL_JSON_PATH = (
    CANONICAL_KB_DIR / "RFP_Database_AIML_CANONICAL.json"
)
CANONICAL_CSV_PATH = CANONICAL_KB_DIR / "RFP_Database_AIML_CANONICAL.csv"

DISTILLER_PROMPT_PATH = (
    PROJECT_ROOT / "prompts" / "kb_distiller_prompt.txt"
)

# Allow long, rich answers. You can bump further if needed.
MAX_CANONICAL_CHARS = 6000

# Use Pro for high-quality, detail-preserving distillation
CANONICAL_MODEL_NAME = "gemini-2.5-pro"

# PERFORMANCE: parallelism & payload size tuning
MAX_WORKERS = int(os.environ.get("KB_MAX_WORKERS", "8"))

# Cap how many rows we send to Gemini per cluster (newest N rows).
# Set KB_MAX_ROWS_PER_CLUSTER=0 to disable and send all rows.
MAX_ROWS_PER_CLUSTER = int(os.environ.get("KB_MAX_ROWS_PER_CLUSTER", "8"))

# Control verbosity – per-cluster prints off by default for speed
VERBOSE = os.environ.get("KB_VERBOSE", "0") == "1"

# How often to print progress (in completed clusters)
PROGRESS_EVERY = int(os.environ.get("KB_PROGRESS_EVERY", "1"))

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


def build_cluster_payload(
    cluster_id: str, rows: pd.DataFrame, meta: dict
) -> tuple[dict, pd.Series]:
    """
    Build the JSON payload sent to the distiller AND return the newest row.

    PERFORMANCE:
    - Sort once by _end_dt (oldest → newest).
    - Optionally cap to the newest MAX_ROWS_PER_CLUSTER rows to keep
      payloads small without losing the most relevant information.
    """
    rows_sorted = rows.sort_values(by="_end_dt", ascending=True)

    if MAX_ROWS_PER_CLUSTER > 0 and len(rows_sorted) > MAX_ROWS_PER_CLUSTER:
        rows_sorted = rows_sorted.iloc[-MAX_ROWS_PER_CLUSTER:]

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

    # newest row after any trimming
    newest = rows_sorted.iloc[-1]

    payload = {
        "cluster_id": cluster_id,
        "category": newest.get(meta["col_category"], "")
        if meta["col_category"]
        else "",
        "subcategory": newest.get(meta["col_subcat"], "") if meta["col_subcat"] else "",
        "rows": payload_rows,
    }

    return payload, newest


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
    Now supports rich metadata (keywords, variants) and search_blob construction.
    """
    i, rows, meta, distiller_prompt = args

    # Skip clusters with empty normalized question
    if not rows["_q_norm"].iloc[0]:
        return None

    cluster_id = f"cluster_{i:04d}"

    if VERBOSE:
        print(f"Processing {cluster_id} (rows={len(rows)}) ...")

    payload, newest = build_cluster_payload(cluster_id, rows, meta)

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

    # Get answer text and apply optional length cap
    answer_text = distilled.get("canonical_answer", "").strip()
    if MAX_CANONICAL_CHARS and MAX_CANONICAL_CHARS > 0:
        answer_text = answer_text[:MAX_CANONICAL_CHARS]

    # --- NEW: Extract rich metadata ---
    keywords = distilled.get("keywords", [])
    variants = distilled.get("question_variants", [])
    
    # Ensure they are lists (fallback if LLM hallucinates format)
    if isinstance(keywords, str): keywords = [k.strip() for k in keywords.split(",")]
    if isinstance(variants, str): variants = [v.strip() for v in variants.split("|")]

    category = newest.get(meta["col_category"], "") if meta["col_category"] else ""
    subcategory = newest.get(meta["col_subcat"], "") if meta["col_subcat"] else ""
    canonical_q = distilled.get("canonical_question", "").strip()

    # --- NEW: Construct Search Blob (Dense Context) ---
    # Format: "CAT: ... | KEYWORDS: ... | VARIANTS: ... | Q: ... | A: ..."
    search_blob_parts = [
        f"CAT: {category} / {subcategory}",
        f"KEYWORDS: {', '.join(keywords)}",
        f"VARIANTS: {' | '.join(variants)}",
        f"Q: {canonical_q}",
        f"A: {answer_text}"
    ]
    search_blob = " || ".join(search_blob_parts)

    canonical_record = {
        "kb_id": f"kb_{i:04d}",
        "category": category,
        "subcategory": subcategory,
        "canonical_question": canonical_q,
        "canonical_answer": answer_text,
        "rich_metadata": {
            "keywords": keywords,
            "question_variants": variants,
            "source_rows_count": len(rows)
        },
        "search_blob": search_blob,  # <--- The magic field for File Search
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
    print(
        f"Using {MAX_WORKERS} workers, "
        f"MAX_ROWS_PER_CLUSTER={MAX_ROWS_PER_CLUSTER if MAX_ROWS_PER_CLUSTER > 0 else 'no limit'}"
    )

    # Prepare work items
    tasks = []
    for i, (_, rows) in enumerate(group, start=1):
        tasks.append((i, rows, meta, distiller_prompt))

    canonical_records: list[dict] = []

    print("\nStarting parallel distillation...")
    t0 = time.time()

    # Run clusters in parallel with progress reporting
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(process_single_cluster, task): idx
            for idx, task in enumerate(tasks, start=1)
        }

        completed = 0
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                rec = future.result()
            except Exception as e:
                print(f"  ⚠️ Cluster {idx:04d} raised an exception: {e}")
                rec = None

            if rec:
                canonical_records.append(rec)

            completed += 1
            if completed % PROGRESS_EVERY == 0 or completed == total_clusters:
                pct = completed * 100.0 / total_clusters
                print(f"  Progress: {completed}/{total_clusters} clusters ({pct:.1f}%)")

    elapsed = time.time() - t0
    print(
        f"\nFinished processing. {len(canonical_records)} canonical records created "
        f"in {elapsed:.1f}s "
        f"(~{elapsed / max(1, len(canonical_records)):.2f}s/record)."
    )

    # Sort for stable output order
    canonical_records.sort(key=lambda r: r["kb_id"])

    # --- JSON output (for File Search) ---
    with CANONICAL_JSON_PATH.open("w", encoding="utf-8") as f_json:
        json.dump(canonical_records, f_json, ensure_ascii=False, indent=2)

    print(f"\n✅ Canonical JSON written to: {CANONICAL_JSON_PATH}")

    # --- CSV output (for manual inspection) ---
    if canonical_records:
        df_can = pd.DataFrame(canonical_records)

        # Flatten list fields for CSV readability
        for col in ("source_rows", "deprecated_removed"):
            if col in df_can.columns:
                df_can[col] = df_can[col].apply(
                    lambda v: ";".join(map(str, v)) if isinstance(v, list) else str(v)
                )

        # utf-8-sig so Excel opens it correctly (no â€™ garbage)
        df_can.to_csv(CANONICAL_CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"✅ Canonical CSV written to:  {CANONICAL_CSV_PATH}")


if __name__ == "__main__":
    main()
