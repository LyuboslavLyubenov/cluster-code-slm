#!/usr/bin/env python3
"""
Code Pattern Analyzer MVP - SLM-First Edition
A lightweight tool that identifies recurring logic patterns in codebases
using local small language models for semantic understanding.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from file_processor import FileProcessor
from slm_extractor import SLMExtractor
from embedding_service import EmbeddingService
from pattern_clusterer import PatternClusterer


class PatternAnalyzer:
    """Main orchestrator for code pattern analysis pipeline."""
    
    def __init__(self, codebase_path: str, output_path: str = "patterns.json", 
                 model_name: str = "default", eps: float = 0.35):
        self.codebase_path = Path(codebase_path)
        self.output_path = output_path
        self.model_name = model_name
        self.eps = eps
        
        self.file_processor = FileProcessor()
        self.slm_extractor = SLMExtractor(model_name, use_http=False, base_url="http://192.168.50.184:1234")
        self.embedding_service = EmbeddingService()
        self.pattern_clusterer = PatternClusterer()
        
        self.stats = {
            'files_processed': 0,
            'segments_found': 0,
            'patterns_detected': 0
        }
    
    def analyze(self) -> Dict[str, Any]:
        """Run the complete analysis pipeline with cache support."""
        print("🔍 Starting code pattern analysis...")
        
        # Define cache file paths
        files_cache_path = self.output_path.replace('.json', '_files_cache.json')
        segments_cache_path = self.output_path.replace('.json', '_segments_cache.json')
        embeddings_cache_path = self.output_path.replace('.json', '_embeddings_cache.json')
        
        # Step 1: Process files from codebase (or load from cache)
        if os.path.exists(files_cache_path):
            print("📁 Loading files data from cache...")
            with open(files_cache_path, 'r') as f:
                try:
                    files_cache = json.load(f)
                except json.JSONDecodeError:
                    files_cache = {
                        "files_data": {},
                        "config": {}
                    }
            files_data = files_cache["files_data"]
            self.stats['files_processed'] = len(files_data)
        else:
            print("📁 Scanning codebase for files...")
            files_data = self.file_processor.process_directory(self.codebase_path)
            self.stats['files_processed'] = len(files_data)
            
            # Save files data cache
            files_cache = {
                "files_data": files_data,
                "config": {
                    "codebase_path": str(self.codebase_path),
                    "model_name": self.model_name,
                    "eps": self.eps
                }
            }
            with open(files_cache_path, 'w') as f:
                json.dump(files_cache, f, indent=2)
        
        if os.path.exists(segments_cache_path):
            print("🤖 Loading segments from cache...")
            with open(segments_cache_path, 'r') as f:
                segments_cache = json.load(f)
            all_segments = segments_cache["segments"]
            self.stats['segments_found'] = len(all_segments)
        else:
            print("🤖 Extracting patterns with SLM...")
            all_segments = []
            for file_path, file_data in files_data.items():
                print(f"📄 Analyzing file: {file_path}")
                segments_result = self.slm_extractor.extract_patterns(file_data['content'], file_data['language'])
                for segment in segments_result.segments:
                    segment_dict = {
                        'code': segment['code'],
                        'line_start': segment['line_start'],
                        'line_end': segment['line_end'],
                        'description': segment['description'],
                        'file_path': str(file_path),
                        'file_name': file_data['file_name']
                    }
                    all_segments.append(segment_dict)
            
            self.stats['segments_found'] = len(all_segments)
            
            segments_cache = {
                "segments": all_segments,
                "config": {
                    "codebase_path": str(self.codebase_path),
                    "model_name": self.model_name,
                    "eps": self.eps
                }
            }
            with open(segments_cache_path, 'w') as f:
                json.dump(segments_cache, f, indent=2)
        
        if not all_segments:
            print("⚠️  No patterns found in codebase")
            return {"patterns": [], "stats": self.stats}
        
        # Step 3: Generate embeddings for descriptions (or load from cache)
        if os.path.exists(embeddings_cache_path):
            print("🧠 Loading embeddings from cache...")
            with open(embeddings_cache_path, 'r') as f:
                embeddings_cache = json.load(f)
            embeddings = embeddings_cache["embeddings"]
        else:
            print("🧠 Generating semantic embeddings...")
            descriptions = [segment['description'] for segment in all_segments]
            embeddings = self.embedding_service.embed_descriptions(descriptions)
            
            # Save embeddings cache
            embeddings_cache = {
                "segments": all_segments,
                "embeddings": embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings,
                "config": {
                    "codebase_path": str(self.codebase_path),
                    "model_name": self.model_name,
                    "eps": self.eps
                }
            }
            with open(embeddings_cache_path, 'w') as f:
                json.dump(embeddings_cache, f, indent=2)
        
        # Step 4: Cluster patterns
        print("📊 Clustering patterns...")
        patterns = self.pattern_clusterer.cluster_patterns(all_segments, embeddings)
        self.stats['patterns_detected'] = len(patterns)
        
        # Step 5: Generate output
        result = {
            "patterns": patterns,
            "stats": self.stats,
            "config": {
                "codebase_path": str(self.codebase_path),
                "model_name": self.model_name,
                "eps": self.eps
            }
        }
        
        # Save results
        with open(self.output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
    
    def print_summary(self, result: Dict[str, Any]):
        """Print CLI summary of analysis results."""
        stats = result['stats']
        patterns = result['patterns']
        
        print(f"\n✅ Analyzed {stats['files_processed']} files")
        print(f"🔍 Extracted {stats['segments_found']} candidate segments")
        print(f"🧩 Detected {stats['patterns_detected']} repeating patterns")
        print(f"📁 Results saved to {self.output_path}")
        
        if patterns:
            print("\nTop patterns:")
            for i, pattern in enumerate(patterns[:5], 1):
                print(f"  {i}. {pattern['description']} ({pattern['occurrence_count']} occurrences)")


def main():
    parser = argparse.ArgumentParser(description='Code Pattern Analyzer - SLM-First Edition')
    parser.add_argument('path', help='Path to codebase directory')
    parser.add_argument('--output', '-o', default='patterns.json', 
                       help='Output JSON file path (default: patterns.json)')
    parser.add_argument('--model', '-m', default='default',
                       help='SLM model name in LM Studio (default: default)')
    parser.add_argument('--eps', type=float, default=0.35,
                       help='DBSCAN epsilon parameter (default: 0.35)')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear all cache files before running analysis')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.path):
        print(f"❌ Error: Path '{args.path}' does not exist")
        sys.exit(1)
    
    # Clear cache if requested
    if args.clear_cache:
        cache_files = [
            args.output.replace('.json', '_files_cache.json'),
            args.output.replace('.json', '_segments_cache.json'),
            args.output.replace('.json', '_embeddings_cache.json')
        ]
        for cache_file in cache_files:
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print(f"🗑️  Cleared cache: {cache_file}")
    
    analyzer = PatternAnalyzer(
        codebase_path=args.path,
        output_path=args.output,
        model_name=args.model,
        eps=args.eps
    )
    
    try:
        result = analyzer.analyze()
        analyzer.print_summary(result)
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()