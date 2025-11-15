"""
Configuration Management Module

Handles configuration loading from environment variables and config files.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class Config:
    """Configuration manager for RFP Agent."""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            env_file: Path to .env file (defaults to .env in current directory)
        """
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
        
        self.google_api_key = os.getenv("GOOGLE_API_KEY", "")
        self.kb_path = os.getenv("KB_PATH", "knowledge_base")
        self.data_path = os.getenv("DATA_PATH", "data")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
        self.batch_size = int(os.getenv("BATCH_SIZE", "10"))
        self.api_delay = float(os.getenv("API_DELAY", "1.0"))
        
    def validate(self) -> bool:
        """
        Validate that required configuration is present.
        
        Returns:
            True if valid, raises ValueError otherwise
        """
        if not self.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY not set. Please set it in .env file or environment."
            )
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "kb_path": self.kb_path,
            "data_path": self.data_path,
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "api_delay": self.api_delay,
        }
