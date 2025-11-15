"""
Knowledge Base Builder Module

This module handles building a canonical knowledge base from historical RFP answers.
It processes various document formats and organizes them into a structured KB.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any


class KnowledgeBaseBuilder:
    """Builds and manages the canonical knowledge base from historical RFP answers."""
    
    def __init__(self, kb_path: str = "knowledge_base"):
        """
        Initialize the Knowledge Base Builder.
        
        Args:
            kb_path: Path to the knowledge base directory
        """
        self.kb_path = Path(kb_path)
        self.kb_path.mkdir(parents=True, exist_ok=True)
        self.index_path = self.kb_path / "index.json"
        self.documents_path = self.kb_path / "documents"
        self.documents_path.mkdir(parents=True, exist_ok=True)
        
    def add_document(self, content: str, metadata: Dict[str, Any]) -> str:
        """
        Add a document to the knowledge base.
        
        Args:
            content: The document content
            metadata: Document metadata (title, category, tags, etc.)
            
        Returns:
            Document ID
        """
        # Generate document ID
        doc_id = f"doc_{len(self.list_documents()) + 1:05d}"
        
        # Save document
        doc_path = self.documents_path / f"{doc_id}.txt"
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Update index
        self._update_index(doc_id, metadata)
        
        return doc_id
    
    def add_rfp_answer(self, question: str, answer: str, metadata: Dict[str, Any] = None) -> str:
        """
        Add a historical RFP Q&A pair to the knowledge base.
        
        Args:
            question: The RFP question
            answer: The historical answer
            metadata: Additional metadata (client, date, category, etc.)
            
        Returns:
            Document ID
        """
        if metadata is None:
            metadata = {}
            
        metadata['type'] = 'rfp_qa'
        metadata['question'] = question
        
        content = f"Question: {question}\n\nAnswer: {answer}"
        
        return self.add_document(content, metadata)
    
    def bulk_import_from_csv(self, csv_path: str, question_col: str = "question", 
                            answer_col: str = "answer") -> List[str]:
        """
        Bulk import RFP Q&A pairs from a CSV file.
        
        Args:
            csv_path: Path to the CSV file
            question_col: Column name for questions
            answer_col: Column name for answers
            
        Returns:
            List of document IDs
        """
        import pandas as pd
        
        df = pd.read_csv(csv_path)
        doc_ids = []
        
        for _, row in df.iterrows():
            question = row.get(question_col, "")
            answer = row.get(answer_col, "")
            
            if question and answer:
                metadata = {k: v for k, v in row.items() 
                           if k not in [question_col, answer_col]}
                doc_id = self.add_rfp_answer(question, answer, metadata)
                doc_ids.append(doc_id)
        
        return doc_ids
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the knowledge base.
        
        Returns:
            List of document metadata
        """
        if not self.index_path.exists():
            return []
            
        with open(self.index_path, 'r', encoding='utf-8') as f:
            index = json.load(f)
            
        return list(index.values())
    
    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Retrieve a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document with content and metadata
        """
        doc_path = self.documents_path / f"{doc_id}.txt"
        
        if not doc_path.exists():
            raise ValueError(f"Document {doc_id} not found")
            
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        index = self._load_index()
        metadata = index.get(doc_id, {})
        
        return {
            'id': doc_id,
            'content': content,
            'metadata': metadata
        }
    
    def get_kb_files_for_gemini(self) -> List[str]:
        """
        Get list of file paths suitable for uploading to Gemini File API.
        
        Returns:
            List of file paths
        """
        return [str(p) for p in self.documents_path.glob("*.txt")]
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load the KB index."""
        if not self.index_path.exists():
            return {}
            
        with open(self.index_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _update_index(self, doc_id: str, metadata: Dict[str, Any]):
        """Update the KB index with new document metadata."""
        index = self._load_index()
        index[doc_id] = metadata
        
        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)
