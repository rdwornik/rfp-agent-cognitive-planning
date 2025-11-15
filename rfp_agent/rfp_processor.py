"""
RFP Processor Module

This module handles processing RFP CSV files and preparing questions for answer generation.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional


class RFPProcessor:
    """Processes RFP CSV files and manages question batching."""
    
    def __init__(self):
        """Initialize the RFP Processor."""
        self.current_rfp = None
        self.questions = []
        
    def load_rfp_csv(self, csv_path: str, question_col: str = "question", 
                     id_col: Optional[str] = None) -> pd.DataFrame:
        """
        Load an RFP CSV file.
        
        Args:
            csv_path: Path to the CSV file
            question_col: Column name containing questions
            id_col: Optional column name for question IDs
            
        Returns:
            DataFrame with RFP questions
        """
        df = pd.read_csv(csv_path)
        
        if question_col not in df.columns:
            raise ValueError(f"Column '{question_col}' not found in CSV")
        
        self.current_rfp = df
        
        # Extract questions
        self.questions = []
        for idx, row in df.iterrows():
            question_id = row[id_col] if id_col and id_col in df.columns else f"Q{idx+1}"
            question_text = row[question_col]
            
            self.questions.append({
                'id': question_id,
                'question': question_text,
                'metadata': {k: v for k, v in row.items() if k != question_col}
            })
        
        return df
    
    def get_questions(self) -> List[Dict[str, Any]]:
        """
        Get the list of questions from the current RFP.
        
        Returns:
            List of question dictionaries
        """
        return self.questions
    
    def get_questions_batch(self, start_idx: int = 0, batch_size: int = 10) -> List[Dict[str, Any]]:
        """
        Get a batch of questions.
        
        Args:
            start_idx: Starting index
            batch_size: Number of questions in batch
            
        Returns:
            List of question dictionaries
        """
        return self.questions[start_idx:start_idx + batch_size]
    
    def save_answers(self, answers: List[Dict[str, Any]], output_path: str):
        """
        Save generated answers to a CSV file.
        
        Args:
            answers: List of answer dictionaries with 'id', 'question', 'answer'
            output_path: Path to save the output CSV
        """
        df = pd.DataFrame(answers)
        df.to_csv(output_path, index=False)
        
    def merge_answers_with_rfp(self, answers: List[Dict[str, Any]], output_path: str):
        """
        Merge generated answers back into the original RFP structure.
        
        Args:
            answers: List of answer dictionaries
            output_path: Path to save the merged CSV
        """
        if self.current_rfp is None:
            raise ValueError("No RFP loaded. Call load_rfp_csv first.")
        
        # Create answers lookup
        answers_dict = {a['id']: a['answer'] for a in answers}
        
        # Add answer column to RFP
        df = self.current_rfp.copy()
        df['generated_answer'] = df.index.map(
            lambda idx: answers_dict.get(f"Q{idx+1}", "")
        )
        
        df.to_csv(output_path, index=False)
