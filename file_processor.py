"""
File processor for reading code files as whole units.
Files are processed as complete units for SLM analysis.
"""

import os
from pathlib import Path
from typing import Dict, Any


class FileProcessor:
    """Processes code files as complete units for SLM analysis."""
    
    SUPPORTED_EXTENSIONS = {
        '.py': 'python',
        '.js': 'javascript', 
        '.jsx': 'javascript', 
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.go': 'go',
        '.rs': 'rust'
    }
    
    def process_directory(self, directory_path: Path) -> Dict[Path, Dict[str, Any]]:
        """Process all supported files in directory recursively."""
        files_data = {}
        
        for ext in self.SUPPORTED_EXTENSIONS:
            pattern = f"**/*{ext}"
            for file_path in directory_path.glob(pattern):
                if self._should_skip_file(file_path):
                    continue
                
                file_data = self.process_file(file_path)
                if file_data:
                    files_data[str(file_path)] = file_data
        
        return files_data
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped (e.g., in venv, node_modules)."""
        skip_dirs = {'__pycache__', 'node_modules', 'venv', '.git', 'build', 'dist'}
        return any(skip_dir in str(file_path) for skip_dir in skip_dirs)
    
    def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Read a single file as a complete unit."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            ext = file_path.suffix.lower()
            language = self.SUPPORTED_EXTENSIONS.get(ext)
            
            return {
                'content': content,
                'language': language,
                'file_path': str(file_path),
                'file_name': file_path.name
            }
                
        except Exception as e:
            print(f"⚠️  Failed to process {file_path}: {e}")
            return {}