import pandas as pd
import os
import time
import concurrent.futures
from datetime import datetime
from llm_router import LLMRouter

# --- CONFIGURATION ---
TEST_MODE = os.environ.get("RFP_TEST_MODE", "0") == "1"
INPUT_DIR = "input_rfp_test/" if TEST_MODE else "input_rfp/"
OUTPUT_DIR = "output_rfp_universal/"
MAX_WORKERS = 4
MODEL_TO_USE = "gemini"  # "gemini" or "claude"

# --- HELPER: Find Column ---
def detect_question_column(df):
    candidates = [
        'Question', 'Requirement', 'Description', 'Customer Question', 
        'RFP Question', 'Functional Requirement', 'Question Text'
    ]
    # Exact + Case Insensitive
    col_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in col_map:
            return col_map[cand.lower()]
    # Fallback: Length heuristic
    for col in df.columns:
        if df[col].dtype == object:
            if df[col].astype(str).str.len().mean() > 20:
                return col
    return None

# --- WORKER FUNCTION ---
def process_single_row(router, row, question_col):
    question = str(row[question_col])
    if len(question.strip()) < 5:
        return "N/A (Question too short)"
    try:
        # Direct generation (Option A: Simplicity first)
        return router.generate_answer(question, model=MODEL_TO_USE)
    except Exception as e:
        return f"ERROR: {str(e)}"

# --- FILE PROCESSOR ---
def process_file(file_path, router):
    print(f"\n📂 Reading: {file_path}")
    try:
        if file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path, encoding="utf-8")
        else:
            print("❌ Unsupported file format")
            return
    except Exception as e:
        print(f"❌ Could not read file: {e}")
        return

    q_col = detect_question_column(df)
    if not q_col:
        print(f"❌ Error: No question column found in {os.path.basename(file_path)}")
        return
    print(f"✅ Detected Question Column: '{q_col}'")

    print(f"🚀 Processing {len(df)} items ({MAX_WORKERS} threads) with {MODEL_TO_USE.upper()}...")
    start_time = time.time()
    
    # Parallel Execution
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Use map to ensure results are returned in the same order as rows
        results = list(executor.map(lambda r: process_single_row(router, r, q_col), [row for _, row in df.iterrows()]))

    df[f"Univ_Answer_{MODEL_TO_USE}"] = results
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = os.path.basename(file_path)
    out_path = os.path.join(OUTPUT_DIR, f"Univ_{MODEL_TO_USE}_{timestamp}_{filename}")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    if out_path.endswith(".csv"):
        df.to_csv(out_path, index=False, encoding="utf-8")
    else:
        df.to_excel(out_path, index=False)
    
    elapsed = time.time() - start_time
    print(f"✅ Saved to: {out_path}")
    print(f"⏱️  Total: {elapsed:.2f}s (Avg: {elapsed/len(df):.2f}s/row)")

# --- MAIN ---
if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"⚠️ Created input folder '{INPUT_DIR}'. Add files and run again.")
    else:
        files = [f for f in os.listdir(INPUT_DIR) if f.endswith((".xlsx", ".csv")) and not f.startswith("~$")]
        
        if not files:
            print(f"⚠️ No files found in {INPUT_DIR}")
        else:
            # Init Router ONCE (efficient)
            print("⚙️  Initializing Universal Router...")
            try:
                main_router = LLMRouter()
                for f in files:
                    process_file(os.path.join(INPUT_DIR, f), main_router)
            except Exception as e:
                print(f"❌ CRITICAL ERROR: {e}")