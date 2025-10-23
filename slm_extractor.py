"""
SLM-powered pattern extractor supporting LM Studio and llama-cpp-python.
Delegates pattern identification to local small language models.
"""

from typing import List, Dict, Any, Literal, Optional
import requests
import json
import re

# Import shared models
from models import Segment, IdentifiedSegments

class ModelClient:
    """Client for interacting with language models via multiple backends."""
    
    def __init__(self, model_name: str = "qwen/qwen3-4b-2507", backend: Literal["lmstudio", "llamacpp"] = "lmstudio", 
                 use_http: bool = False, base_url: str = "http://localhost:1234"):
        self.model_name = model_name or "qwen/qwen3-4b-2507"
        self.backend = backend
        self.use_http = use_http
        self.base_url = base_url
        self.client = None
        if not use_http:
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize model client with specified backend."""
        try:
            if self.backend == "lmstudio":
                from lmstudio import llm
                self.client = llm(self.model_name)
                print(f"✅ LM Studio client initialized with model: {self.model_name}")
            else:  # llamacpp
                from llama_cpp import Llama
                self.client = Llama(model_path=self.model_name)
                print(f"✅ llama-cpp-python client initialized with model: {self.model_name}")
        except ImportError:
            print(f"⚠️  {self.backend} not available")
            self.client = None
        except Exception as e:
            print(f"⚠️  Failed to initialize {self.backend} client: {e}")
            self.client = None
    
    def call_model(self, prompt: str) -> str:
        """Call model with prompt and return raw response text."""
        try:
            if self.use_http:
                return self._call_model_http(prompt)
            else:
                return self._call_model_library(prompt)
        except Exception as e:
            print(f"⚠️  Model call failed: {e}")
            raise
    
    def _call_model_library(self, prompt: str) -> str:
        """Call model using Python library."""
        if not self.client:
            raise RuntimeError("Model client not initialized")
        
        if self.backend == "lmstudio":
            result = self.client.respond(prompt)
            return str(result)
        else:  # llamacpp
            result = self.client.create_completion(prompt, max_tokens=512, stop=["\n"], echo=False)
            return result["choices"][0]["text"]
    
    def _call_model_http(self, prompt: str) -> str:
        """Call model using HTTP API."""
        url = f"{self.base_url}/v1/completions"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": 0.5,
            "max_tokens": 512
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["text"]

class SLMExtractor:
    """Extracts code patterns using language model inference."""
    
    def __init__(self, model_name: str = "qwen/qwen3-4b-2507", backend: Literal["lmstudio", "llamacpp"] = "lmstudio", 
                 use_http: bool = False, base_url: str = "http://localhost:1234"):
        self.model_client = ModelClient(model_name, backend, use_http, base_url)
    
    def extract_patterns(self, code: str, language: str) -> IdentifiedSegments:
        """Extract patterns from code using language model."""
        prompt = self._build_prompt(code, language)
        
        try:
            raw_response = self.model_client.call_model(prompt)
            return IdentifiedSegments.from_response(raw_response)
        except Exception as e:
            print(f"⚠️  Model extraction failed: {e}")
            return IdentifiedSegments()
    
    def _build_prompt(self, code: str, language: str) -> str:
        """Build the model prompt for pattern extraction."""
        try:
            with open('./prompts/analyze_file.md', 'r') as f:
                prompt_template = f.read()
        except FileNotFoundError:
            raise FileNotFoundError("analyze_file.md not found - please create the prompt file")
        
        return prompt_template.replace("{code}", code).replace("{language}", language)