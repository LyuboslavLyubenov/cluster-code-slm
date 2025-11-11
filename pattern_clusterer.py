"""
Pattern clustering using UMAP and HDBSCAN on semantic embeddings.
Groups similar code patterns based on their semantic descriptions.
"""

import numpy as np
from typing import List, Dict, Any
import umap
import hdbscan
from models import Segment, IdentifiedSegments
from slm_extractor import ModelClient, SLMExtractor

class PatternClusterer:
    """Clusters code patterns using UMAP and HDBSCAN on semantic embeddings."""
    
    def __init__(self, min_cluster_size: int = 3, min_samples: int = 1, n_neighbors: int = 17, n_components: int = 10):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.default_n_neighbors = n_neighbors
        self.default_n_components = n_components
        self.llm_client = ModelClient()
        self._initialize_clustering()
    
    def _initialize_clustering(self):
        """Initialize UMAP and HDBSCAN clustering."""
        try:
            self.umap = umap.UMAP(n_neighbors=self.default_n_neighbors, n_components=self.default_n_components, metric='cosine', min_dist=0.1)
            self.hdbscan = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples)
            print(f"✅ UMAP and HDBSCAN initialized with min_cluster_size={self.min_cluster_size}, min_samples={self.min_samples}")
        except ImportError as e:
            print(f"⚠️  Required libraries not installed: {e}")
            print("   Install with: pip install umap-learn hdbscan")
            raise
    
    def cluster_patterns(self, segments: List[Dict[str, Any]], embeddings: np.ndarray) -> List[Dict[str, Any]]:
        """Cluster patterns based on semantic embeddings."""
        if len(segments) == 0 or len(embeddings) == 0:
            return []
        
        try:
            n_samples = len(embeddings)
            print(f"📊 Clustering {n_samples} samples...")
            
            if n_samples <= 5:
                print(f"📊 Using raw embeddings (too few samples for UMAP)")
                reduced_embeddings = embeddings
            else:
                n_neighbors = min(self.default_n_neighbors, n_samples - 1) if n_samples > 1 else 1
                n_components = min(self.default_n_components, n_samples - 1) if n_samples > 1 else 1
                
                if n_neighbors != self.default_n_neighbors or n_components != self.default_n_components:
                    print(f"📊 Adjusting UMAP parameters: n_neighbors={n_neighbors}, n_components={n_components}")
                    self.umap = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric='cosine')
                
                reduced_embeddings = self.umap.fit_transform(embeddings)
            
            reduced = np.asarray(reduced_embeddings, dtype=np.float64)
            if reduced.ndim == 1:
                reduced = reduced.reshape(-1, 1)
            labels = self.hdbscan.fit_predict(reduced)
            labels_list = labels.tolist() if hasattr(labels, 'tolist') else list(labels)

            clusters = self._group_by_cluster(segments, labels_list)
            patterns = self._format_patterns(clusters)
            
            return patterns
            
        except Exception as e:
            print(f"⚠️  Clustering failed: {e}")
            return self._format_patterns(self._group_by_cluster(segments, [0] * len(segments)))
    
    def _group_by_cluster(self, segments: List[Dict[str, Any]], labels: list) -> Dict[int, List[Dict[str, Any]]]:
        """Group segments by their cluster labels."""
        clusters = {}
        
        for segment, label in zip(segments, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(segment)

        return clusters
    
    def _format_patterns(self, clusters: Dict[int, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Format clustered segments into pattern objects."""
        patterns = []
        
        for cluster_id, segments in clusters.items():
            if cluster_id == -1:
                for segment in segments:
                    patterns.append({
                        'pattern_id': f"unique_{len(patterns)}",
                        'description': segment['description'],
                        'example_code': segment['code'],
                        'occurrences': [{
                            'file_path': segment['file_path'],
                        }],
                        'occurrence_count': 1
                    })
            else:
                if len(segments) >= self.min_samples:
                    cluster_description = self._generate_cluster_title(segments)
                    
                    patterns.append({
                        'pattern_id': f"pattern_{cluster_id}",
                        'description': cluster_description,
                        'example_code': segments[0]['code'],
                        'occurrences': [{
                            'file_path': seg['file_path'],
                            'code': seg['code']
                        } for seg in segments],
                        'occurrence_count': len(segments)
                    })
        
        patterns.sort(key=lambda x: x['occurrence_count'], reverse=True)
        
        return patterns
    
    def _generate_cluster_title(self, segments: List[Dict[str, Any]]) -> str:
        """Generate a cluster title using SLM based on all children descriptions."""

        descriptions = [seg['description'] for seg in segments]
        pattern_types = self._analyze_pattern_types(descriptions)
        prompt = self._build_cluster_title_prompt(descriptions, pattern_types, segments)
        
        print(f"🤖 Generating cluster title for {len(segments)} segments...")
        result = self.llm_client.call_model(prompt)
        
        import json
        parsed_result = json.loads(result)
        title = parsed_result.get("title", result)
        
        print(f"✅ Generated cluster title: {title}")
        return title
     
    
    def _analyze_pattern_types(self, descriptions: List[str]) -> Dict[str, Any]:
        """Analyze the types of patterns in the cluster."""
        type_counts = {}
        for desc in descriptions:
            # Extract pattern type from [TYPE] format
            if desc.startswith('[') and ']' in desc:
                pattern_type = desc.split(']')[0] + ']'
                type_counts[pattern_type] = type_counts.get(pattern_type, 0) + 1
        
        dominant_type = max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else "[PATTERN]"
        
        return {
            'types': type_counts,
            'dominant_type': dominant_type,
            'total_segments': len(descriptions)
        }
    
    def _build_cluster_title_prompt(self, descriptions: List[str], pattern_types: Dict[str, Any], segments: List[Dict[str, Any]]) -> str:
        """Build enhanced prompt for cluster title generation."""
        descriptions_text = "\n".join([f"- {desc}" for desc in descriptions])
        
        sample_codes = []
        for i, seg in enumerate(segments[:3]):
            code_preview = seg['code']
            sample_codes.append(f"Example {i+1}:\n```\n{code_preview}\n```")
        
        sample_codes_text = "\n\n".join(sample_codes)
        
        prompt = f"""
You are analyzing a cluster of {pattern_types['total_segments']} similar code patterns that have been grouped together by semantic similarity.

## PATTERN TYPE ANALYSIS:
- Dominant pattern type: {pattern_types['dominant_type']}
- Type distribution: {pattern_types['types']}

## INDIVIDUAL PATTERN DESCRIPTIONS:
{descriptions_text}

## SAMPLE CODE PATTERNS:
{sample_codes_text}

## TASK:
Generate a single, concise title that captures the ESSENCE of this code pattern cluster. The title should:

1. Use the same [TYPE] format as individual descriptions (e.g., {pattern_types['dominant_type']})
2. Be specific about the COMMON functionality across all patterns
3. Highlight the CORE purpose or behavior
4. Be 8-35 words maximum
5. Focus on WHAT the pattern does, not implementation details

## EXAMPLES OF GOOD CLUSTER TITLES:
- "[API] Fetches and processes user data with error handling and loading states"
- "[UI] Renders form validation feedback with conditional styling"
- "[HOOK] Manages component state with persistence and cleanup"
- "[UTILITY] Transforms and formats data for display purposes"

## OUTPUT FORMAT:
Return ONLY valid JSON in this exact format:
{{
    "title": "your generated title here"
}}

Your response:
"""
        return prompt
        """Generate a fallback title when SLM fails."""
        # Try to extract common words from descriptions
        descriptions = [seg['description'] for seg in segments]
        
        # Find most common type
        type_counts = {}
        for desc in descriptions:
            if desc.startswith('[') and ']' in desc:
                pattern_type = desc.split(']')[0] + ']'
                type_counts[pattern_type] = type_counts.get(pattern_type, 0) + 1
        
        dominant_type = max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else "[PATTERN]"
        
        # Extract common keywords
        words = []
        for desc in descriptions:
            # Remove type tags and split into words
            clean_desc = desc.split(']')[-1] if ']' in desc else desc
            words.extend(clean_desc.lower().split())
        
        from collections import Counter
        common_words = [word for word, count in Counter(words).most_common(5) 
                       if len(word) > 3 and count > 1]
        
        if common_words:
            fallback = f"{dominant_type} Pattern involving {', '.join(common_words[:3])}"
        else:
            # Use most frequent description
            desc_counts = {}
            for desc in descriptions:
                desc_counts[desc] = desc_counts.get(desc, 0) + 1
            fallback = max(desc_counts.items(), key=lambda x: x[1])[0]
        
        print(f"⚠️  Using enhanced fallback title: {fallback}")
        return fallback