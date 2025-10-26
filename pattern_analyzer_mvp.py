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
from typing import Dict, Any, Literal, Optional, Union

from file_processor import FileProcessor
from slm_extractor import SLMExtractor
from embedding_service import EmbeddingService
from pattern_clusterer import PatternClusterer


class LLMBackendConfig:
    """Configuration for LLM backend selection."""
    
    def __init__(self, backend: Literal["lmstudio", "llamacpp"] = "lmstudio", model_path: str = "qwen/qwen3-4b-2507", embedding_path: str = "text-embedding-qwen3-embedding-0.6b", use_http: bool = False, base_url: str = "http://localhost:1234"):
        self.backend = backend
        self.use_http = use_http
        self.model_path = model_path
        self.embedding_path = embedding_path
        self.base_url = base_url

class PatternAnalyzer:
    """Main orchestrator for code pattern analysis pipeline."""
    
    def __init__(self, codebase_path: str, output_path: str = "patterns.json", 
                 llm_backend_config: LLMBackendConfig = LLMBackendConfig()):
        self.codebase_path = Path(codebase_path)
        self.output_path = output_path
        self.model_name = llm_backend_config.model_path
        
        self.file_processor = FileProcessor()
        self.slm_extractor = SLMExtractor(llm_backend_config.model_path, use_http=llm_backend_config.use_http, base_url=llm_backend_config.base_url, backend=llm_backend_config.backend)
        self.embedding_service = EmbeddingService(llm_backend_config.embedding_path, backend=llm_backend_config.backend)
        self.pattern_clusterer = PatternClusterer()
        
        self.stats = {
            'files_processed': 0,
            'segments_found': 0,
            'patterns_detected': 0
        }
    
    def _get_cache_paths(self) -> tuple[str, str, str]:
        """Get cache file paths for different stages."""
        files_cache_path = self.output_path.replace('.json', '_files_cache.json')
        segments_cache_path = self.output_path.replace('.json', '_segments_cache.json')
        embeddings_cache_path = self.output_path.replace('.json', '_embeddings_cache.json')
        return files_cache_path, segments_cache_path, embeddings_cache_path
    
    def _process_files(self, files_cache_path: str):
        """Process codebase files or load from cache."""
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
                }
            }
            with open(files_cache_path, 'w') as f:
                json.dump(files_cache, f, indent=2)
        
        return files_data
    
    def _chunk_file_content(self, file_content: str, max_chars: int = 6000, overlap_ratio: float = 0.35) -> list:
        """Split file content into overlapping chunks with balanced nesting."""

        if len(file_content) <= max_chars:
            return [file_content]

        def is_nesting_depth_zero(text: str) -> bool:
            """Check if nesting depth is zero (safe to split)."""
            depth = 0
            for char in text:
                if char in '({[':
                    depth += 1
                elif char in ')}]':
                    depth -= 1
            return depth == 0
        
        lines = file_content.splitlines(keepends=True)
        chunks = []
        i = 0
        total_lines = len(lines)
        
        while i < total_lines:
            current_lines = []
            char_count = 0
            j = i
            
            while j < total_lines:
                line = lines[j]
                current_lines.append(line)
                char_count += len(line)
                
                if char_count >= max_chars * 0.9:
                    chunk_text = ''.join(current_lines)
                    if is_nesting_depth_zero(chunk_text):
                        break
                if char_count > max_chars * 1.1:
                    break
                j += 1
            
            chunk_text = ''.join(current_lines)
            chunks.append(chunk_text)
            
            # Slide forward with overlap
            chunk_lines = len(current_lines)
            overlap_lines = int(chunk_lines * overlap_ratio)
            i = i + (chunk_lines - overlap_lines)
        
        return chunks
    
    def _extract_segments(self, files_data, segments_cache_path: str) -> list:
        """Extract code segments using SLM or load from cache."""
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
                
                # Chunk large files
                file_content = file_data['content']
                chunks = self._chunk_file_content(file_content)
                print(f"   📦 Split into {len(chunks)} chunks")
                for chunk_idx, chunk_content in enumerate(chunks):
                    segments_result = self.slm_extractor.extract_patterns(chunk_content, file_data['language'])
                    for segment in segments_result.segments:
                        segment_dict = {
                            'code': segment.code,
                            'description': segment.description,
                            'file_path': str(file_path),
                            'file_name': file_data['file_name'],
                            'chunk_index': chunk_idx
                        }
                        all_segments.append(segment_dict)

            self.stats['segments_found'] = len(all_segments)
            
            segments_cache = {
                "segments": all_segments,
                "config": {
                    "codebase_path": str(self.codebase_path),
                    "model_name": self.model_name,
                }
            }
            with open(segments_cache_path, 'w') as f:
                json.dump(segments_cache, f, indent=2)
        
        return all_segments
    
    def _generate_embeddings(self, all_segments: list, embeddings_cache_path: str):
        """Generate embeddings for segment descriptions or load from cache."""
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
                }
            }
            with open(embeddings_cache_path, 'w') as f:
                json.dump(embeddings_cache, f, indent=2)
        
        return embeddings
    
    def _cluster_patterns(self, all_segments: list, embeddings) -> list:
        """Cluster patterns using DBSCAN algorithm."""
        print("📊 Clustering patterns...")
        patterns = self.pattern_clusterer.cluster_patterns(all_segments, embeddings)
        self.stats['patterns_detected'] = len(patterns)
        return patterns
    
    def _generate_output(self, patterns: list) -> Dict[str, Any]:
        """Generate final output structure and save results."""
        result = {
            "patterns": patterns,
            "stats": self.stats,
            "config": {
                "codebase_path": str(self.codebase_path),
                "model_name": self.model_name,
            }
        }
        
        # Save results
        with open(self.output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
    
    def analyze(self) -> Dict[str, Any]:
        """Run the complete analysis pipeline with cache support."""
        print("🔍 Starting code pattern analysis...")
        
        files_cache_path, segments_cache_path, embeddings_cache_path = self._get_cache_paths()
        
        files_data = self._process_files(files_cache_path)
        
        all_segments = self._extract_segments(files_data, segments_cache_path)
        
        if not all_segments:
            print("⚠️  No patterns found in codebase")
            return {"patterns": [], "stats": self.stats}
        
        self.embedding_service.load_model()
        embeddings = self._generate_embeddings(all_segments, embeddings_cache_path)
        
        patterns = self._cluster_patterns(all_segments, embeddings)
        
        result = self._generate_output(patterns)
        
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
    parser.add_argument('--model', '-m', default="qwen/qwen3-4b-2507",
                       help='SLM model name (default: qwen/qwen3-4b-2507). Use local path for llama-cpp-python models.')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear all cache files before running analysis')
    parser.add_argument('--backend', '-b', choices=['lmstudio', 'llamacpp'], default='lmstudio',
                       help='LLM backend to use (default: lmstudio)')
    parser.add_argument('--embedding-model', '-e', default='text-embedding-qwen3-embedding-0.6b',
                       help='Embedding model name/path (default: text-embedding-qwen3-embedding-0.6b). Use local path for llama-cpp-python models.')
    
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
        llm_backend_config=LLMBackendConfig(
            backend=args.backend,
            model_path=args.model,
            embedding_path=args.embedding_model,
        )
    )
    
    try:
        result = analyzer.analyze()
        analyzer.print_summary(result)
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()