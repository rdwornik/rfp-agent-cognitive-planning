"""
Main Application Entry Point

Provides CLI interface for the RFP Agent.
"""

import argparse
import sys
from pathlib import Path

from rfp_agent.config import Config
from rfp_agent.kb_builder import KnowledgeBaseBuilder
from rfp_agent.rfp_processor import RFPProcessor
from rfp_agent.gemini_agent import GeminiRFPAgent


def build_kb(args):
    """Build knowledge base from historical RFP answers."""
    config = Config()
    kb_builder = KnowledgeBaseBuilder(config.kb_path)
    
    print(f"Building knowledge base at: {config.kb_path}")
    
    if args.csv:
        print(f"Importing from CSV: {args.csv}")
        doc_ids = kb_builder.bulk_import_from_csv(
            args.csv,
            question_col=args.question_col,
            answer_col=args.answer_col
        )
        print(f"Imported {len(doc_ids)} Q&A pairs")
    
    docs = kb_builder.list_documents()
    print(f"\nKnowledge base now contains {len(docs)} documents")


def generate_answers(args):
    """Generate answers for a new RFP."""
    config = Config()
    config.validate()
    
    print("Initializing RFP Agent...")
    print(f"  Model: {config.model_name}")
    print(f"  KB Path: {config.kb_path}")
    
    # Load RFP questions
    processor = RFPProcessor()
    print(f"\nLoading RFP from: {args.rfp_csv}")
    processor.load_rfp_csv(args.rfp_csv, question_col=args.question_col)
    questions = processor.get_questions()
    print(f"Loaded {len(questions)} questions")
    
    # Initialize Gemini agent
    agent = GeminiRFPAgent(api_key=config.google_api_key, model_name=config.model_name)
    
    # Upload KB files if available
    kb_builder = KnowledgeBaseBuilder(config.kb_path)
    kb_files = kb_builder.get_kb_files_for_gemini()
    
    if kb_files:
        print(f"\nUploading {len(kb_files)} KB files to Gemini...")
        agent.upload_knowledge_base(kb_files)
        agent.wait_for_files_active()
    else:
        print("\nWarning: No KB files found. Generating answers without knowledge base.")
    
    # Generate answers
    print("\nGenerating answers...")
    answers = agent.batch_generate_answers(questions, delay=config.api_delay)
    
    # Save results
    output_path = args.output or args.rfp_csv.replace(".csv", "_answers.csv")
    processor.save_answers(answers, output_path)
    print(f"\n✓ Answers saved to: {output_path}")
    
    # Cleanup
    if kb_files:
        print("\nCleaning up uploaded files...")
        agent.cleanup_files()
    
    print("\n✓ Done!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RFP Agent - Semi-automated RFP answer engine"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Build KB command
    kb_parser = subparsers.add_parser("build-kb", help="Build knowledge base")
    kb_parser.add_argument("--csv", help="CSV file with historical Q&A")
    kb_parser.add_argument("--question-col", default="question", help="Question column name")
    kb_parser.add_argument("--answer-col", default="answer", help="Answer column name")
    
    # Generate answers command
    gen_parser = subparsers.add_parser("generate", help="Generate answers for RFP")
    gen_parser.add_argument("rfp_csv", help="CSV file with RFP questions")
    gen_parser.add_argument("--question-col", default="question", help="Question column name")
    gen_parser.add_argument("--output", "-o", help="Output file path")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == "build-kb":
            build_kb(args)
        elif args.command == "generate":
            generate_answers(args)
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
