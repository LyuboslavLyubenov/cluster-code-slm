"""
Pattern clustering using UMAP and HDBSCAN on semantic embeddings.
Groups similar code patterns based on their semantic descriptions.
"""

import numpy as np
from typing import List, Dict, Any
import umap
import hdbscan
from slm_extractor import LMStudioClient, SLMExtractor

class PatternClusterer:
    """Clusters code patterns using UMAP and HDBSCAN on semantic embeddings."""
    
    def __init__(self, min_cluster_size: int = 2, min_samples: int = 1, ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.default_n_neighbors = 5
        self.default_n_components = 5
        self.lm_studio_client = LMStudioClient()
        self._initialize_clustering()
    
    def _initialize_clustering(self):
        """Initialize UMAP and HDBSCAN clustering."""
        try:
            self.umap = umap.UMAP(n_neighbors=self.default_n_neighbors, n_components=self.default_n_components, metric='cosine')
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
            # Adjust UMAP parameters if embeddings are too small
            n_samples = len(embeddings)
            print(f"📊 Clustering {n_samples} samples...")
            
            # For very small datasets, skip UMAP and use raw embeddings
            if n_samples <= 5:
                print(f"📊 Using raw embeddings (too few samples for UMAP)")
                reduced_embeddings = embeddings
            else:
                n_neighbors = min(self.default_n_neighbors, n_samples - 1) if n_samples > 1 else 1
                n_components = min(self.default_n_components, n_samples - 1) if n_samples > 1 else 1
                
                if n_neighbors != self.default_n_neighbors or n_components != self.default_n_components:
                    print(f"📊 Adjusting UMAP parameters: n_neighbors={n_neighbors}, n_components={n_components}")
                    self.umap = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric='cosine')
                
                # Reduce dimensionality with UMAP
                reduced_embeddings = self.umap.fit_transform(embeddings)
            
            labels = self.hdbscan.fit_predict(reduced_embeddings)
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
                            'line_start': segment['line_start'],
                            'line_end': segment['line_end']
                        }],
                        'occurrence_count': 1
                    })
            else:
                # Grouped pattern
                if len(segments) >= self.min_samples:
                    # Generate cluster title using SLM based on all children descriptions
                    cluster_description = self._generate_cluster_title(segments)
                    
                    patterns.append({
                        'pattern_id': f"pattern_{cluster_id}",
                        'description': cluster_description,
                        'example_code': segments[0]['code'],
                        'occurrences': [{
                            'file_path': seg['file_path'],
                            'line_start': seg['line_start'],
                            'line_end': seg['line_end']
                        } for seg in segments],
                        'occurrence_count': len(segments)
                    })
        
        patterns.sort(key=lambda x: x['occurrence_count'], reverse=True)
        
        return patterns
    
    def _generate_cluster_title(self, segments: List[Dict[str, Any]]) -> str:
        """Generate a cluster title using SLM based on all children descriptions."""
        try:
            # Combine all descriptions from the cluster
            all_descriptions = "\n".join([seg['description'] for seg in segments])
            
            # Create prompt for SLM to generate a unified cluster title
            prompt = f"""
            You are analyzing a cluster of similar code patterns. Below are descriptions of individual code segments that belong to the same cluster:
            
            {all_descriptions}
            
            Please generate a single, concise title that captures the common theme or pattern across all these descriptions. 
            The title should be descriptive and specific to the code patterns.
            
            Return json object in the format:
            {{
                "title: "your generated title here"
            }}
            """
            
            # Use SLM to generate the title
            result = self.lm_studio_client.call_model(prompt)
            
            # Extract the title from the response
            if hasattr(result, 'segments') and result.segments:
                return result.segments[0].description
            else:
                # Fallback: use the most common description
                desc_counts = {}
                for seg in segments:
                    desc = seg['description']
                    desc_counts[desc] = desc_counts.get(desc, 0) + 1
                return max(desc_counts.items(), key=lambda x: x[1])[0]
                
        except Exception as e:
            print(f"⚠️  SLM cluster title generation failed: {e}")
            # Fallback to most common description
            desc_counts = {}
            for seg in segments:
                desc = seg['description']
                desc_counts[desc] = desc_counts.get(desc, 0) + 1
            return max(desc_counts.items(), key=lambda x: x[1])[0]