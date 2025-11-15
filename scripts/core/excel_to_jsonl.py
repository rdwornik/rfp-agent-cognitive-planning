import pandas as pd
from datetime import datetime
import json

# --- CONFIG ---
INPUT_XLSX = "RFP_Database_Cognitive_Planning.xlsx"
SHEET_NAME = "Main"
OUTPUT_JSONL = "RFP_Database_Cognitive_Planning.jsonl"
# ---------------

def to_iso_date(value):
    """Convert Excel/various date types to ISO yyyy-mm-dd string, or None."""
    if pd.isna(value):
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.date().isoformat()
    # try to parse string-like dates
    try:
        return datetime.fromisoformat(str(value)).date().isoformat()
    except Exception:
        # fallback: just string form
        return str(value)

# Read Excel
df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME)

records_written = 0

with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for idx, row in df.iterrows():
        rec = {
            "kb_id": int(idx) + 1,
            "customer": str(row.get("Customer")) if pd.notna(row.get("Customer")) else None,
            "end_date": to_iso_date(row.get("End Date")),
            "category": str(row.get("Category")) if pd.notna(row.get("Category")) else None,
            "subcategory": str(row.get("Subcategory")) if pd.notna(row.get("Subcategory")) else None,
            "customer_question": str(row.get("Customer Question")).strip() if pd.notna(row.get("Customer Question")) else "",
            "tech_presales_answer": str(row.get("Tech Presales Answers")).strip() if pd.notna(row.get("Tech Presales Answers")) else "",
            "vendor_response_flag": str(row.get("Vendor Response (Y or N)")) if pd.notna(row.get("Vendor Response (Y or N)")) else None,
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        records_written += 1

print(f"Wrote {records_written} rows to {OUTPUT_JSONL}")
