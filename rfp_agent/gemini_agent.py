"""
Gemini Integration Module

This module integrates with Google's Gemini API using File Search to generate
answers for RFP questions based on the knowledge base.
"""

import os
import time
from typing import List, Dict, Any, Optional
import google.generativeai as genai


class GeminiRFPAgent:
    """
    Integrates with Gemini API to generate RFP answers using File Search.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-pro"):
        """
        Initialize the Gemini RFP Agent.
        
        Args:
            api_key: Google API key (if None, reads from GOOGLE_API_KEY env var)
            model_name: Gemini model to use
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("API key required. Set GOOGLE_API_KEY env var or pass api_key.")
        
        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        self.uploaded_files = []
        
    def upload_knowledge_base(self, file_paths: List[str]) -> List[Any]:
        """
        Upload knowledge base files to Gemini for File Search.
        
        Args:
            file_paths: List of file paths to upload
            
        Returns:
            List of uploaded file objects
        """
        uploaded = []
        
        for file_path in file_paths:
            try:
                print(f"Uploading {file_path}...")
                file = genai.upload_file(file_path)
                uploaded.append(file)
                print(f"  Uploaded: {file.display_name} ({file.name})")
            except Exception as e:
                print(f"  Error uploading {file_path}: {e}")
        
        self.uploaded_files = uploaded
        return uploaded
    
    def wait_for_files_active(self, timeout: int = 300):
        """
        Wait for uploaded files to be processed and active.
        
        Args:
            timeout: Maximum time to wait in seconds
        """
        start_time = time.time()
        
        for file in self.uploaded_files:
            while file.state.name == "PROCESSING":
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"File {file.name} processing timed out")
                
                print(f"  Waiting for {file.display_name} to be processed...")
                time.sleep(5)
                file = genai.get_file(file.name)
            
            if file.state.name != "ACTIVE":
                raise ValueError(f"File {file.name} failed to process: {file.state.name}")
        
        print("All files are active and ready.")
    
    def generate_answer(self, question: str, context: str = "") -> str:
        """
        Generate an answer for a single RFP question using the KB.
        
        Args:
            question: The RFP question
            context: Optional additional context
            
        Returns:
            Generated answer
        """
        # Create system prompt for SaaS-focused, pragmatic answers
        system_prompt = """You are an expert RFP response assistant for a SaaS company.
Your task is to provide pragmatic, professional answers to RFP questions based on the
knowledge base of historical answers.

Guidelines:
1. Provide clear, concise, and accurate answers
2. Focus on SaaS-relevant capabilities and features
3. Use professional business language
4. Be specific but not overly technical unless required
5. Draw from the knowledge base examples when available
6. If information is not in the KB, provide a general best-practice answer
7. Keep answers between 100-300 words unless the question requires more detail
"""
        
        # Prepare the prompt
        full_prompt = f"{system_prompt}\n\n"
        if context:
            full_prompt += f"Additional Context: {context}\n\n"
        full_prompt += f"Question: {question}\n\nAnswer:"
        
        try:
            # Create model with file search
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={"temperature": 0.7}
            )
            
            # Generate content with uploaded files
            if self.uploaded_files:
                response = model.generate_content([full_prompt] + self.uploaded_files)
            else:
                response = model.generate_content(full_prompt)
            
            return response.text.strip()
        
        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"[Error generating answer: {str(e)}]"
    
    def batch_generate_answers(self, questions: List[Dict[str, Any]], 
                               delay: float = 1.0) -> List[Dict[str, Any]]:
        """
        Generate answers for a batch of RFP questions.
        
        Args:
            questions: List of question dictionaries with 'id' and 'question'
            delay: Delay between API calls in seconds (for rate limiting)
            
        Returns:
            List of answer dictionaries with 'id', 'question', and 'answer'
        """
        answers = []
        total = len(questions)
        
        for idx, q in enumerate(questions, 1):
            question_id = q['id']
            question_text = q['question']
            
            print(f"\n[{idx}/{total}] Processing: {question_id}")
            print(f"  Q: {question_text[:80]}...")
            
            answer = self.generate_answer(question_text)
            
            answers.append({
                'id': question_id,
                'question': question_text,
                'answer': answer
            })
            
            print(f"  A: {answer[:80]}...")
            
            # Rate limiting delay
            if idx < total:
                time.sleep(delay)
        
        return answers
    
    def cleanup_files(self):
        """Delete uploaded files from Gemini."""
        for file in self.uploaded_files:
            try:
                genai.delete_file(file.name)
                print(f"Deleted file: {file.display_name}")
            except Exception as e:
                print(f"Error deleting {file.display_name}: {e}")
        
        self.uploaded_files = []
