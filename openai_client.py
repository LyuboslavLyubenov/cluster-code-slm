"""
OpenAI-compatible client for pattern extraction.
Provides an alternative to LM Studio for OpenAI-compatible API endpoints.
"""

from typing import List, Dict, Any, Optional
import requests
import json
import re
from slm_extractor import Segment, IdentifiedSegments


class OpenAIClient:
    """Client for interacting with OpenAI-compatible API endpoints."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.headers = {
            "Content-Type": "application/json"
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    def call_model(self, prompt: str) -> str:
        """Call OpenAI-compatible model with prompt and return raw response text."""
        try:
            return self._call_chat_completion(prompt)
        except Exception as e:
            print(f"⚠️  OpenAI-compatible API call failed: {e}")
            raise
    
    def _call_chat_completion(self, prompt: str) -> str:
        """Call OpenAI-compatible chat completion API."""
        url = f"{self.base_url}/v1/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.5,
            "max_tokens": 102_400
        }
        
        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]


class OpenAIExtractor:
    """Extracts code patterns using OpenAI-compatible API inference."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.model_client = OpenAIClient(base_url, api_key, model)
    
    def extract_patterns(self, code: str, language: str) -> IdentifiedSegments:
        """Extract patterns from code using OpenAI-compatible API."""
        prompt = self._build_prompt(code, language)
        
        try:
            raw_response = self.model_client.call_model(prompt)
            return IdentifiedSegments.from_response(raw_response)
        except Exception as e:
            print(f"⚠️  OpenAI extraction failed: {e}")
            return IdentifiedSegments()
    
    def _build_prompt(self, code: str, language: str) -> str:
        """Build the prompt for pattern extraction."""
        try:
            with open('analyze_file.md', 'r') as f:
                prompt_template = f.read()
        except FileNotFoundError:
            raise FileNotFoundError("analyze_file.md not found - please create the prompt file")
        
        return prompt_template.replace("{code}", code).replace("{language}", language)