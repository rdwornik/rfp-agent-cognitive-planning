# kb_build_canonical.py
#
# Usage (from project root):
#   python .\scripts\kb_build_canonical.py
#
# Input (raw KB for manual review in Excel/CSV):
#   data_kb/raw/RFP_Database_Cognitive_Planning.csv
#
# Output (canonical KB for Gemini/File Search):
#   data_kb/canonical/RFP_Database_Cognitive_Planning_CANONICAL.json
#
# This script:
#   - Groups similar questions (by Category + normalized question)
#   - Sends each "cluster" of historical answers to Gemini as JSON
#   - Asks Gemini to produce ONE canonical, time-aware answer per cluster
#   - Writes the result as a JSON array (one object per canonical Q/A)

import json
import re
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import errors as genai_errors

# ---------- paths ----------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_KB_DIR = PROJECT_ROOT / "data_kb" / "raw"
CANONICAL_KB_DIR = PROJECT_ROOT / "data_kb" / "canonical"

RAW_CSV_PATH = RAW_KB_DIR / "RFP_Database_Cognitive_Planning.csv"
CANONICAL_JSON_PATH = (
    CANONICAL_KB_DIR / "RFP_Database_Cognitive_Planning_CANONICAL.json"
)


# ---------- helpers for headers & normalization ----------


def norm_header(name: str) -> str:
    """Normalize header names (lower, collapse spaces)."""
    if not isinstance(name, str):
        return ""
    s = name.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def build_header_alias(df: pd.DataFrame) -> dict:
    """Map normalized header -> original header."""
    alias = {}
    for col in df.columns:
        alias[norm_header(col)] = col
    return alias


def get_col(df: pd.DataFrame, alias: dict, candidates) -> str | None:
    """Pick first existing column among candidate names (case-insensitive)."""
    for c in candidates:
        key = norm_header(c)
        if key in alias:
            return alias[key]
    return None


def normalize_question_text(q: str) -> str:
    """Normalize question for clustering."""
    if not isinstance(q, str):
        return ""
    q = q.strip().lower()
    q = re.sub(r"\s+", " ", q)
    # keep letters/numbers/spaces only
    q = re.sub(r"[^a-z0-9 ]+", "", q)
    return q


# ---------- load & cluster raw CSV ----------


def load_raw_df() -> tuple[pd.DataFrame, dict]:
    """Load the raw KB CSV and prepare clustering metadata."""
    path = RAW_CSV_PATH

    # Encoding fallback
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    alias = build_header_alias(df)

    # Identify key columns via alias
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
            "Could not find required columns 'Customer Question' "
            "and 'Tech Presales Answers' in CSV."
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

    # cluster key: category + normalized question
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
    """Prepare JSON payload for Gemini for one cluster."""
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


# ---------- Gemini distiller instructions ----------

DISTILLER_SYSTEM = """
Blue Yonder KB Distiller (Time-Aware, SaaS-First, Specific-When-Specific)

You receive multiple historical RFP answers for similar questions, ordered by end_date
from oldest to newest in the "rows" array.

Your task:
Produce ONE canonical, current answer suitable for a knowledge base, following these rules.

1) TIME PRIORITY
- Newer rows override older rows when there are conflicts in terminology, scope, or positioning.
- Older rows may add detail ONLY if they do not contradict the newest rows.

2) SPECIFIC vs GENERIC
- If the question(s) in this cluster are inherently specific (e.g. ask for named regions,
  explicit deployment options, exact integration methods, etc.) and the NEWEST row(s)
  provide concrete details (lists, numbers, named regions, etc.), you MAY keep these details.
- Do NOT generalize away explicit facts that are clearly present in the newest row(s),
  unless they are obviously deprecated or contradicted.
- If the KB content for this cluster is broad/generic, keep the canonical answer broad.

3) DEPRECATED CONCEPTS & NAMING
- Follow the newest naming and positioning for:
  - Blue Yonder platform, SaaS, Platform Data Cloud, integration approach, etc.
- Treat legacy labels as deprecated when they appear only in older rows, such as:
  - "Luminate platform", "Luminate Cognitive Platform",
  - "Data Access Service", "DAS" (for integration),
  - other legacy offer names that are not present in the newest rows.
- Replace deprecated terms with current, generic wording consistent with the newest rows
  (e.g. "data extraction capabilities", "the platform data layer") rather than using
  deprecated brand names.

4) CONTENT SCOPE
- Use ONLY information from the provided rows; DO NOT invent new features, numbers, SLAs,
  certifications, or commitments.
- Style: SaaS-first, Blue Yonder platform-centric, pragmatic, configuration-focused.
- First reference: "The Blue Yonder platform ..." or "Blue Yonder’s platform ...".
- Later references: "the platform", "the solution", "the service".
- Keep the answer as one compact paragraph (up to ~1600 characters). Be detailed when the
  rows contain rich information, but stay concise and readable.

5) HIGH-RISK TOPICS (HANDLE CAREFULLY)
- SLA/uptime values, RPO/RTO, certifications (ISO, SOC, etc.), data residency,
  security architecture, upgrade cadence, custom code policy, PII/data protection:
  - Include these ONLY if they are clearly and consistently described in the newest rows.
  - If older and newer rows conflict, follow the newest.
  - If unclear, omit rather than speculate.

6) OUTPUT FORMAT
Return JSON ONLY (no markdown, no extra commentary) with:
{
  "canonical_question": "<short, normalized question capturing the essence of this cluster>",
  "canonical_answer": "<the distilled answer text>",
  "last_updated": "YYYY-MM-DD",   // usually the latest end_date in the cluster
  "sources_used": ["row_00010", "row_00015"],    // row_ids from input
  "deprecated_removed": ["...", "..."]           // list legacy terms you intentionally avoided
}
""".strip()


def call_gemini_distiller(
    client: genai.Client, cluster_payload: dict, max_retries: int = 5
) -> dict:
    """
    Call Gemini to distill one cluster into a canonical Q/A.
    IMPORTANT: uses ONLY a 'user' message (no system role/system_instruction),
    so it's compatible with your current SDK + backend.
    """
    import time

    user_text = (
        DISTILLER_SYSTEM
        + "\n\n=== CLUSTER PAYLOAD (JSON) ===\n"
        + json.dumps(cluster_payload, ensure_ascii=False)
    )

    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    {
                        "role": "user",
                        "parts": [{"text": user_text}],
                    }
                ],
                # Ask explicitly for JSON so we can json.loads()
                config={"response_mime_type": "application/json"},
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


# ---------- main ----------


def main():
    # Ensure canonical folder exists
    CANONICAL_KB_DIR.mkdir(parents=True, exist_ok=True)

    df, meta = load_raw_df()

    client = genai.Client()

    group = df.groupby("_cluster_key")
    total_clusters = len(group)
    print(f"Total rows: {len(df)}")
    print(f"Total clusters: {total_clusters}")

    canonical_records = []

    for i, (cluster_key, rows) in enumerate(group, start=1):
        # skip clusters with empty normalized question
        if not rows["_q_norm"].iloc[0]:
            continue

        cluster_id = f"cluster_{i:04d}"
        print(f"Processing {cluster_id} (rows={len(rows)}) ...")

        payload = build_cluster_payload(cluster_id, rows, meta)
        newest = rows.sort_values(by="_end_dt", ascending=True).iloc[-1]

        try:
            distilled = call_gemini_distiller(client, payload)
        except Exception as e:
            print(f"  ⚠️ Distillation failed for {cluster_id}: {e}")
            continue

        canonical_record = {
            "kb_id": f"kb_{i:04d}",
            "category": newest.get(meta["col_category"], "")
            if meta["col_category"]
            else "",
            "subcategory": newest.get(meta["col_subcat"], "")
            if meta["col_subcat"]
            else "",
            "canonical_question": distilled.get("canonical_question", "").strip(),
            "canonical_answer": distilled.get("canonical_answer", "").strip(),
            "last_updated": distilled.get(
                "last_updated", str(newest.get(meta["col_end_date"], ""))
            ),
            "source_rows": distilled.get("sources_used", []),
            "deprecated_removed": distilled.get("deprecated_removed", []),
        }

        canonical_records.append(canonical_record)

    # Write JSON array
    with CANONICAL_JSON_PATH.open("w", encoding="utf-8") as f_out:
        json.dump(canonical_records, f_out, ensure_ascii=False, indent=2)

    print(f"\n✅ Canonical KB written to: {CANONICAL_JSON_PATH}")


if __name__ == "__main__":
    main()
