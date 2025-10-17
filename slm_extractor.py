"""
SLM-powered pattern extractor using LM Studio.
Delegates pattern identification to local small language models.
"""

from typing import List, Dict, Any, Literal
from lmstudio import BaseModel, llm
import requests
import json

class Segment(BaseModel):
    """A single identified code segment."""
    code: str
    line_start: int
    line_end: int
    description: str

class IdentifiedSegments(BaseModel):
    """Container for multiple identified code segments."""
    segments: List[Segment] = []

class SLMExtractor:
    """Extracts code patterns using LM Studio SLM inference."""
    
    def __init__(self, model_name: str = "qwen/qwen3-4b-2507", use_http: bool = False, base_url: str = "http://localhost:1234"):
        self.model_name = model_name
        self.use_http = use_http
        self.base_url = base_url
        self.client = None
        if not use_http:
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize LM Studio client."""
        self.client = llm(self.model_name)
        print(f"✅ LM Studio client initialized with model: {self.model_name}")
    
    def extract_patterns(self, code: str, language: str) -> IdentifiedSegments:
        """Extract patterns from code using SLM."""
        prompt = self._build_prompt(code, language)
        
        try:
            return self._call_slm(prompt)
        except Exception as e:
            print(f"⚠️  SLM extraction failed: {e}")
            return IdentifiedSegments()
    
    def _build_prompt(self, code: str, language: str) -> str:
        """Build the SLM prompt for pattern extraction."""
        try:
            with open('analyze_file.md', 'r') as f:
                prompt_template = f.read()
        except FileNotFoundError:
            raise FileNotFoundError("analyze_file.md not found - please create the prompt file")
        
        return prompt_template.replace("{code}", code).replace("{language}", language)
    
    def _call_slm(self, prompt: str) -> IdentifiedSegments:
        """Call LM Studio SLM with the prompt."""
        try:
            if self.use_http:
                return self._call_slm_http(prompt)
            else:
                return self._call_slm_library(prompt)
        except Exception as e:
            print(f"⚠️  SLM call failed: {e}")
            return IdentifiedSegments()
    
    def _call_slm_library(self, prompt: str) -> IdentifiedSegments:
        """Call LM Studio using Python library."""
        if not self.client:
            raise RuntimeError("LM Studio client not initialized")
        result = self.client.respond(prompt)
        parsed_result = self._parse_llm_response(result.content)
        return parsed_result
    
    def _call_slm_http(self, prompt: str) -> IdentifiedSegments:
        """Call LM Studio using HTTP API."""
        url = f"{self.base_url}/v0/completions"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": 0.5,
            "stream": False,
            "max_tokens": 262144
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["text"]
        
        # Parse JSON response
        try:
            data = json.loads(content)
            return IdentifiedSegments(**data)
        except json.JSONDecodeError:
            # Fallback: try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return IdentifiedSegments(**data)
            raise ValueError("Could not parse JSON response from SLM")

    def _parse_llm_response(self, content: str) -> IdentifiedSegments:
        """Parse LLM response that may contain JSON code blocks."""
        import re
        
        # Try to extract JSON from code blocks first
        json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return IdentifiedSegments(**data)
            except json.JSONDecodeError:
                pass
        
        # Fallback: try to extract JSON without code blocks
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return IdentifiedSegments(**data)
            except json.JSONDecodeError:
                pass
        
        raise ValueError("Could not parse JSON response from LLM")
