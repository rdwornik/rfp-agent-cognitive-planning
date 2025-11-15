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

This file is the source of truth that will later be uploaded into
the Gemini File Search store.
"""

import csv
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import errors as genai_errors, types

# --- paths & constants -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repo root
RAW_KB_DIR = PROJECT_ROOT / "data_kb" / "raw"
CANONICAL_KB_DIR = PROJECT_ROOT / "data_kb" / "canonical"

RAW_CSV_PATH = RAW_KB_DIR / "RFP_Database_Cognitive_Planning.csv"
CANONICAL_CSV_PATH = CANONICAL_KB_DIR / "RFP_Database_Cognitive_Planning_CANONICAL.csv"
CANONICAL_JSON_PATH = (
    CANONICAL_KB_DIR / "RFP_Database_Cognitive_Planning_CANONICAL.json"
)

MAX_CANONICAL_CHARS = 1600  # your requested limit


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


# ---------- load & cluster raw CSV ----------


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


# ---------- Gemini distiller instructions ----------

DISTILLER_SYSTEM = f"""
Blue Yonder KB Distiller (Time-Aware, SaaS-First, Specific-When-Specific)

You receive multiple historical RFP answers for similar questions, ordered by end_date
from oldest to newest in the "rows" array.

Your task:
Produce ONE canonical, current answer suitable for a knowledge base, following these rules.

1) TIME PRIORITY
- Newer rows override older rows when there are conflicts in terminology, scope, or positioning.
- Older rows may add detail ONLY if they do not contradict the newest rows.

2) SPECIFIC vs GENERIC
- If the clustered questions are specific (e.g., named regions, explicit deployment options,
  concrete integration methods) and the NEWEST row(s) provide those details, you MAY keep them.
- Do NOT generalize away explicit facts that are clearly present in the newest row(s),
  unless they are obviously deprecated or contradicted.
- If the KB content is broad/generic, keep the canonical answer broad.

3) DEPRECATED CONCEPTS & NAMING
- Prefer the newest naming and positioning for:
  - Blue Yonder platform, SaaS, Platform Data Cloud, integration approach, etc.
- Treat legacy labels as deprecated when they appear only in older rows, such as:
  - "Luminate platform", "Luminate Cognitive Platform",
  - "Data Access Service", "DAS" (for integration),
  - other legacy offer names not present in the newest rows.
- Replace deprecated terms with current, generic wording consistent with the newest rows
  (e.g. "data extraction capabilities", "the platform data layer") rather than legacy brands.

4) CONTENT SCOPE & LENGTH
- Use ONLY information from the provided rows; DO NOT invent new features, numbers, SLAs,
  certifications, or commitments.
- Style: SaaS-first, Blue Yonder platform-centric, pragmatic, configuration-focused.
- First reference: "The Blue Yonder platform ..." or "Blue Yonder’s platform ...".
- Later references: "the platform", "the solution", "the service".
- One compact paragraph, up to ~{MAX_CANONICAL_CHARS} characters. Shorter is fine
  if that covers the content; only use the full budget when there is real detail to keep.

5) HIGH-RISK TOPICS (HANDLE CAREFULLY)
- SLA/uptime values, RPO/RTO, certifications (ISO, SOC, etc.), data residency,
  security architecture, upgrade cadence, custom code policy, PII/data protection:
  - Include ONLY if clearly and consistently described in the newest rows.
  - If older and newer rows conflict, follow the newest.
  - If unclear, omit rather than speculate.

6) OUTPUT FORMAT
Return JSON ONLY (no markdown) with:
{{
  "canonical_question": "<short, normalized question capturing the cluster>",
  "canonical_answer": "<the distilled answer text>",
  "last_updated": "YYYY-MM-DD",   // typically the latest end_date
  "sources_used": ["row_00010", "row_00015"],
  "deprecated_removed": ["...", "..."]
}}
""".strip()


def call_gemini_distiller(
    client: genai.Client, cluster_payload: dict, max_retries: int = 5
) -> dict:
    user_text = (
        DISTILLER_SYSTEM
        + "\n\n=== CLUSTER PAYLOAD (JSON) ===\n"
        + json.dumps(cluster_payload, ensure_ascii=False)
    )

    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    {
                        "role": "user",
                        "parts": [{"text": user_text}],
                    }
                ],
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


# ---------- main ----------


def main():
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(f"Raw KB CSV not found: {RAW_CSV_PATH}")

    df, meta = load_raw_df(RAW_CSV_PATH)

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    CANONICAL_KB_DIR.mkdir(parents=True, exist_ok=True)

    group = df.groupby("_cluster_key")
    total_clusters = len(group)
    print(f"Total rows: {len(df)}")
    print(f"Total clusters: {total_clusters}")

    canonical_records: list[dict] = []
    csv_rows: list[list[str]] = []

    for i, (cluster_key, rows) in enumerate(group, start=1):
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
            "canonical_answer": distilled.get("canonical_answer", "").strip()[
                :MAX_CANONICAL_CHARS
            ],
            "last_updated": distilled.get(
                "last_updated", str(newest.get(meta["col_end_date"], ""))
            ),
            "source_rows": distilled.get("sources_used", []),
            "deprecated_removed": distilled.get("deprecated_removed", []),
        }

        canonical_records.append(canonical_record)

        csv_rows.append(
            [
                canonical_record["kb_id"],
                canonical_record["category"],
                canonical_record["subcategory"],
                canonical_record["canonical_question"],
                canonical_record["canonical_answer"],
                canonical_record["last_updated"],
                "|".join(canonical_record["source_rows"]),
            ]
        )

    # Write CSV for Excel review
    with CANONICAL_CSV_PATH.open("w", encoding="utf-8", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(
            [
                "kb_id",
                "category",
                "subcategory",
                "canonical_question",
                "canonical_answer",
                "last_updated",
                "source_rows",
            ]
        )
        writer.writerows(csv_rows)

    # Write JSON array (valid JSON)
    with CANONICAL_JSON_PATH.open("w", encoding="utf-8") as f_json:
        json.dump(canonical_records, f_json, ensure_ascii=False, indent=2)

    print(f"\n✅ Canonical CSV written to:  {CANONICAL_CSV_PATH}")
    print(f"✅ Canonical JSON written to: {CANONICAL_JSON_PATH}")


if __name__ == "__main__":
    main()
