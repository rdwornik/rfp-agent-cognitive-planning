"""
rfp_batch_universal.py
Universal RFP batch processor with multi-LLM support.

Usage:
  python rfp_batch_universal.py                     # Production mode, gemini
  python rfp_batch_universal.py --test              # Test mode, gemini
  python rfp_batch_universal.py --model claude      # Production mode, claude
  python rfp_batch_universal.py --test --model gpt5 # Test mode, gpt5
  python rfp_batch_universal.py -t -m deepseek      # Short flags
  python rfp_batch_universal.py -t -m gemini -a     # With anonymization
  python rfp_batch_universal.py --solution wms      # Solution-aware context
"""
import argparse
import pandas as pd
import os
import json
import time
import concurrent.futures
from datetime import datetime
from pathlib import Path
from llm_router import LLMRouter
from anonymization import AnonymizationMiddleware  # NEW

# --- PROJECT ROOT AND PLATFORM MATRIX ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLATFORM_MATRIX_PATH = PROJECT_ROOT / "config/platform_matrix.json"


def get_available_solutions():
    """Load available solutions from platform_matrix.json."""
    if not PLATFORM_MATRIX_PATH.exists():
        return []
    with open(PLATFORM_MATRIX_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return list(data.get("solutions", {}).keys())


# --- ARGUMENT PARSER ---
def parse_args():
    parser = argparse.ArgumentParser(
        description="Universal RFP Batch Processor with multi-LLM support"
    )
    parser.add_argument(
        "-t", "--test",
        action="store_true",
        help="Run in test mode (uses input_rfp_test/ folder)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="gemini",
        choices=["gemini", "gemini-flash", "claude", "gpt5", "deepseek", "kimi", "llama", "grok", "glm"],
        help="LLM model to use (default: gemini)"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "-a", "--anonymize",
        action="store_true",
        help="Anonymize customer names before sending to LLM API"
    )
    # Get available solutions dynamically from platform_matrix.json
    available_solutions = get_available_solutions()
    parser.add_argument(
        "-s", "--solution",
        type=str,
        default=None,
        choices=available_solutions if available_solutions else None,
        metavar="SOLUTION",
        help=f"Solution-aware context injection. Available: {', '.join(available_solutions[:5])}... (see platform_matrix.json)"
    )
    return parser.parse_args()


# --- HELPER: Find Column ---
def detect_question_column(df):
    candidates = [
        'Question', 'Requirement', 'Description', 'Customer Question', 
        'RFP Question', 'Functional Requirement', 'Question Text', 
        'customer_question'
    ]
    
    def normalize(s): 
        return str(s).lower().replace("_", " ").strip()
    
    col_map = {normalize(c): c for c in df.columns}
    
    for cand in candidates:
        norm_cand = normalize(cand)
        if norm_cand in col_map:
            return col_map[norm_cand]
            
    # Fallback: Length heuristic
    for col in df.columns:
        if df[col].dtype == object:
            non_empty = df[col].dropna().astype(str)
            if len(non_empty) > 0 and non_empty.str.len().mean() > 20:
                return col
    return None


# --- WORKER FUNCTION ---
def process_single_row(router, row, question_col, model, middleware):
    val = row[question_col]
    
    if pd.isna(val) or str(val).strip() == "":
        return ""
    
    question = str(val).strip()
    
    if len(question) < 3:
        return ""
    
    try:
        # Anonymize before LLM call
        clean_question, ctx = middleware.before(question)
        
        # Call LLM
        answer = router.generate_answer(clean_question, model=model)
        
        # De-anonymize response
        final_answer = middleware.after(answer, ctx)
        
        return final_answer
    except Exception as e:
        return f"ERROR: {str(e)}"


# --- FILE PROCESSOR ---
def process_file(file_path, router, model, max_workers, middleware):
    print(f"\n{'='*50}")
    print(f"📂 Reading: {file_path}")
    print(f"🤖 Model: {model.upper()}")
    print(f"🔒 Anonymization: {'ON' if middleware.enabled else 'OFF'}")
    print(f"{'='*50}")
    
    try:
        if file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        elif file_path.endswith(".csv"):
            try:
                df = pd.read_csv(file_path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding="latin-1")
        else:
            print("❌ Unsupported file format")
            return
    except Exception as e:
        print(f"❌ Could not read file: {e}")
        return

    q_col = detect_question_column(df)
    if not q_col:
        print(f"❌ Error: No question column found.")
        print(f"   Columns available: {list(df.columns)}")
        return

    print(f"✅ Question Column: '{q_col}'")
    print(f"📊 Total rows: {len(df)}")
    print(f"⚡ Workers: {max_workers}")
    print(f"\n🚀 Starting processing...")
    
    start_time = time.time()
    
    rows = [row for _, row in df.iterrows()]
    completed = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_row, router, row, q_col, model, middleware) 
            for row in rows
        ]
        
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            if completed % 5 == 0 or completed == len(rows):
                pct = completed * 100 / len(rows)
                print(f"   Progress: {completed}/{len(rows)} ({pct:.1f}%)")
        
        results = [f.result() for f in futures]

    output_df = pd.DataFrame({
        "customer_question": df[q_col].fillna(""),
        "answer": results
    })
    
    output_dir = "data/output/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    anon_suffix = "_anon" if middleware.enabled else ""
    out_path = os.path.join(output_dir, f"{base_name}_{model}{anon_suffix}_{timestamp}.csv")
    
    output_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    
    elapsed = time.time() - start_time
    non_empty = len([r for r in results if r and not r.startswith("ERROR")])
    
    print(f"\n{'='*50}")
    print(f"✅ COMPLETE!")
    print(f"📄 Output: {out_path}")
    print(f"📊 Processed: {non_empty}/{len(df)} questions")
    print(f"⏱️  Time: {elapsed:.1f}s ({elapsed/max(1,len(df)):.2f}s/row)")
    print(f"{'='*50}")


# --- MAIN ---
def main():
    args = parse_args()
    
    input_dir = "data/input/" if args.test else "data/input/"
    
    # Initialize middleware
    middleware = AnonymizationMiddleware(enabled=args.anonymize)
    
    print("\n" + "="*50)
    print("UNIVERSAL RFP BATCH PROCESSOR")
    print("="*50)
    print(f"Mode:        {'TEST' if args.test else 'PRODUCTION'}")
    print(f"Input:       {input_dir}")
    print(f"Model:       {args.model}")
    print(f"Workers:     {args.workers}")
    print(f"Anonymize:   {'YES' if args.anonymize else 'NO'}")
    print(f"Solution:    {args.solution.upper() if args.solution else 'NONE (generic)'}")
    print("="*50)
    
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"\n⚠️ Created '{input_dir}'. Add files and run again.")
        return
    
    files = [
        f for f in os.listdir(input_dir) 
        if f.endswith((".xlsx", ".csv")) and not f.startswith("~$")
    ]
    
    if not files:
        print(f"\n⚠️ No .xlsx or .csv files found in {input_dir}")
        return
    
    print(f"\n📁 Found {len(files)} file(s): {files}")
    
    try:
        print("\n[INFO] Initializing Universal Router...")
        router = LLMRouter(solution=args.solution)
        print("[SUCCESS] Router ready!\n")
        
        for f in files:
            process_file(os.path.join(input_dir, f), router, args.model, args.workers, middleware)
            
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()