"""
Shared data models for pattern extraction.
"""

from typing import List, Dict, Any, Literal, Optional
import json
import re


class Segment:
    """A single identified code segment."""
    def __init__(self, code: str, line_start: int, line_end: int, description: str, confidence: Optional[str] = None):
        self.code = code
        self.line_start = line_start
        self.line_end = line_end
        self.description = description


class IdentifiedSegments:
    """Container for multiple identified code segments."""
    
    def __init__(self, segments: Optional[List[Segment]] = None):
        self.segments = segments or []
    
    @classmethod
    def from_response(cls, content: str) -> "IdentifiedSegments":
        """Parse LLM response that may contain JSON code blocks."""
        # Try to extract JSON from code blocks first
        json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                segments = [Segment(**seg_data) for seg_data in data.get("segments", [])]
                return cls(segments=segments)
            except json.JSONDecodeError:
                pass
        
        # Fallback: try to extract JSON without code blocks
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                segments = [Segment(**seg_data) for seg_data in data.get("segments", [])]
                return cls(segments=segments)
            except json.JSONDecodeError:
                pass
        
        print("⚠️  Failed to parse segments from response: returning empty list.")
        return cls(segments=[])