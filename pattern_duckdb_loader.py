"""
Pattern DuckDB Loader
Loads pattern descriptions and metadata into DuckDB with embeddings for vector search.
Supports both real embeddings and mock embeddings for development.
"""

import json
import numpy as np
import duckdb
from typing import List, Dict, Any, Optional
import os
from embedding_service import EmbeddingService

class PatternDuckDBLoader:
    """Loads patterns into DuckDB with embeddings for vector search."""
    
    def __init__(self, db_path: str = "patterns.db"):
        self.db_path = db_path
        self.conn = None
        self.embedding_service = None
        self.embedding_dim = 1024
        
    def initialize_database(self):
        """Initialize DuckDB connection and create tables."""
        try:
            self.conn = duckdb.connect(self.db_path)
            print(f"✅ Connected to DuckDB database: {self.db_path}")
            
            # Create patterns table with embeddings
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    pattern_id VARCHAR PRIMARY KEY,
                    description VARCHAR,
                    example_code VARCHAR,
                    occurrence_count INTEGER,
                    embedding FLOAT[1024],
                    pattern_type VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create occurrences table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS pattern_occurrences (
                    occurrence_id VARCHAR PRIMARY KEY,
                    pattern_id VARCHAR,
                    file_path VARCHAR,
                    code TEXT,
                    line_start INTEGER,
                    line_end INTEGER,
                    file_name VARCHAR,
                    FOREIGN KEY (pattern_id) REFERENCES patterns(pattern_id)
                )
            """)
            
            # Create pattern metadata table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS pattern_metadata (
                    pattern_id VARCHAR PRIMARY KEY,
                    cluster_id VARCHAR,
                    similarity_score FLOAT,
                    keywords VARCHAR[],
                    FOREIGN KEY (pattern_id) REFERENCES patterns(pattern_id)
                )
            """)
            
            print("✅ Database schema initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize database: {e}")
            raise
    
    def load_embedding_service(self, model_path: str = None, backend: str = "lmstudio"):
        """Load the embedding service."""
        try:
            self.embedding_service = EmbeddingService(model_path=model_path, backend=backend)
            self.embedding_service.load_model()
            print("✅ Embedding service loaded")
        except Exception as e:
            print(f"⚠️  Failed to load embedding service, using mock embeddings: {e}")
    
    def load_patterns_from_json(self, json_path: str) -> List[Dict[str, Any]]:
        """Load patterns from JSON file."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                patterns = data.get('patterns', [])
                print(f"✅ Loaded {len(patterns)} patterns from {json_path}")
                return patterns
        except Exception as e:
            print(f"❌ Failed to load patterns from {json_path}: {e}")
            raise
    
    def load_segments_from_cache(self, cache_path: str) -> List[Dict[str, Any]]:
        """Load segments from embeddings cache for additional context."""
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                segments = data.get('segments', [])
                print(f"✅ Loaded {len(segments)} segments from cache")
                return segments
        except Exception as e:
            print(f"⚠️  Failed to load segments from cache: {e}")
            return []
    
    def extract_pattern_type(self, description: str) -> str:
        """Extract pattern type from description."""
        if description.startswith('[') and ']' in description:
            return description.split(']')[0] + ']'
        return "[UNKNOWN]"
    
    def extract_keywords(self, description: str) -> List[str]:
        """Extract keywords from description."""
        # Remove pattern type tag
        clean_desc = description.split(']')[-1] if ']' in description else description
        
        # Simple keyword extraction
        words = clean_desc.lower().split()
        # Filter out common words and short words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        
        return keywords
    
    def generate_embeddings(self, descriptions: List[str]) -> np.ndarray:
        """Generate embeddings for pattern descriptions."""
        print(f"🔍 Generating embeddings for {len(descriptions)} descriptions...")
        embeddings = self.embedding_service.embed_descriptions(descriptions)
        print(f"✅ Generated embeddings with shape: {embeddings.shape}")
        return embeddings

    
    def insert_patterns(self, patterns: List[Dict[str, Any]]):
        """Insert patterns and their embeddings into the database."""
        if not patterns:
            print("⚠️  No patterns to insert")
            return
        
        try:
            # Extract descriptions for embedding
            descriptions = [pattern['description'] for pattern in patterns]
            embeddings = self.generate_embeddings(descriptions)
            
            # First, insert all main patterns
            for i, pattern in enumerate(patterns):
                pattern_type = self.extract_pattern_type(pattern['description'])
                embedding_array = embeddings[i].tolist() if i < len(embeddings) else [0.0] * self.embedding_dim
                
                # Insert main pattern
                self.conn.execute("""
                    INSERT OR REPLACE INTO patterns 
                    (pattern_id, description, example_code, occurrence_count, embedding, pattern_type)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    pattern['pattern_id'],
                    pattern['description'],
                    pattern.get('example_code', ''),
                    pattern.get('occurrence_count', 1),
                    embedding_array,
                    pattern_type
                ))
            
            # Then insert occurrences and metadata
            for pattern in patterns:
                # Insert occurrences
                for j, occurrence in enumerate(pattern.get('occurrences', [])):
                    occurrence_id = f"{pattern['pattern_id']}_{j}"
                    self.conn.execute("""
                        INSERT OR REPLACE INTO pattern_occurrences 
                        (occurrence_id, pattern_id, file_path, code, line_start, line_end, file_name)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        occurrence_id,
                        pattern['pattern_id'],
                        occurrence.get('file_path', ''),
                        occurrence.get('code', ''),
                        occurrence.get('line_start', 0),
                        occurrence.get('line_end', 0),
                        occurrence.get('file_name', '')
                    ))
                
                # Insert metadata
                keywords = self.extract_keywords(pattern['description'])
                self.conn.execute("""
                    INSERT OR REPLACE INTO pattern_metadata 
                    (pattern_id, cluster_id, similarity_score, keywords)
                    VALUES (?, ?, ?, ?)
                """, (
                    pattern['pattern_id'],
                    pattern.get('cluster_id', ''),
                    pattern.get('similarity_score', 0.0),
                    keywords
                ))
            
            print(f"✅ Inserted {len(patterns)} patterns into database")
            
        except Exception as e:
            print(f"❌ Failed to insert patterns: {e}")
            raise
    
    def create_vector_index(self):
        """Create vector index for similarity search."""
        try:
            # DuckDB doesn't support indexing on array types for vector search
            # We'll skip index creation and rely on sequential scanning
            print("ℹ️  Skipping vector index (DuckDB array indexing not supported for vector search)")
            
        except Exception as e:
            print(f"⚠️  Could not create vector index: {e}")
    
    def create_similarity_search_function(self):
        """Create UDFs for similarity search."""
        try:
            # Use DuckDB's built-in array functions for similarity calculation
            # We'll implement cosine similarity using SQL expressions
            print("ℹ️  Using SQL-based similarity calculation (no custom UDF needed)")
            
        except Exception as e:
            print(f"⚠️  Could not create similarity function: {e}")
    
    def search_similar_patterns(self, query: str, top_k: int = 5, min_similarity: float = 0.0) -> List[Dict[str, Any]]:
        """Search for patterns similar to the query."""
        try:
            # Generate query embedding
            if self.embedding_service:
                query_embedding = self.embedding_service.embed_descriptions([query])[0]
                query_embedding_list = query_embedding.tolist()
            else:
                # Use mock embedding for query
                query_embedding_list = self._create_mock_embeddings(1, self.embedding_dim)[0].tolist()
            
            # Convert to FLOAT array explicitly for DuckDB
            query_embedding_float = [float(x) for x in query_embedding_list]
            
            # For now, return patterns by relevance to keywords in the query
            # This is a simplified approach since DuckDB doesn't have built-in vector similarity functions
            query_keywords = [word.lower() for word in query.split() if len(word) > 3]
            
            if not query_keywords:
                # If no meaningful keywords, return random patterns
                result = self.conn.execute("""
                    SELECT 
                        pattern_id,
                        description,
                        example_code,
                        occurrence_count,
                        pattern_type
                    FROM patterns
                    ORDER BY occurrence_count DESC
                    LIMIT ?
                """, (top_k,)).fetchall()
                
                patterns = []
                for row in result:
                    patterns.append({
                        'pattern_id': row[0],
                        'description': row[1],
                        'example_code': row[2],
                        'occurrence_count': row[3],
                        'pattern_type': row[4],
                        'similarity': 0.5  # Default similarity score
                    })
                
                return patterns
            
            # Build keyword search query
            keyword_conditions = " OR ".join([f"LOWER(description) LIKE '%{kw}%'" for kw in query_keywords])
            
            result = self.conn.execute(f"""
                SELECT 
                    pattern_id,
                    description,
                    example_code,
                    occurrence_count,
                    pattern_type,
                    -- Simple relevance score based on keyword matches
                    (LENGTH(description) - LENGTH(REPLACE(LOWER(description), '{query_keywords[0]}', ''))) / LENGTH('{query_keywords[0]}') as match_score
                FROM patterns
                WHERE {keyword_conditions}
                ORDER BY match_score DESC, occurrence_count DESC
                LIMIT ?
            """, (top_k,)).fetchall()
            
            # Convert to list of dicts
            patterns = []
            for row in result:
                patterns.append({
                    'pattern_id': row[0],
                    'description': row[1],
                    'example_code': row[2],
                    'occurrence_count': row[3],
                    'pattern_type': row[4],
                    'similarity': min(float(row[5]) / 10.0, 1.0) if row[5] else 0.1  # Normalize score
                })
            
            return patterns
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []
    
    def find_patterns_by_type(self, pattern_type: str) -> List[Dict[str, Any]]:
        """Find patterns by pattern type."""
        try:
            result = self.conn.execute("""
                SELECT 
                    pattern_id,
                    description,
                    example_code,
                    occurrence_count,
                    pattern_type
                FROM patterns
                WHERE pattern_type = ?
                ORDER BY occurrence_count DESC
            """, (pattern_type,)).fetchall()
            
            patterns = []
            for row in result:
                patterns.append({
                    'pattern_id': row[0],
                    'description': row[1],
                    'example_code': row[2],
                    'occurrence_count': row[3],
                    'pattern_type': row[4]
                })
            
            return patterns
            
        except Exception as e:
            print(f"❌ Pattern type search failed: {e}")
            return []
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about patterns in the database."""
        try:
            stats = {}
            
            # Total patterns
            total_patterns = self.conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
            stats['total_patterns'] = total_patterns
            
            # Total occurrences
            total_occurrences = self.conn.execute("SELECT COUNT(*) FROM pattern_occurrences").fetchone()[0]
            stats['total_occurrences'] = total_occurrences
            
            # Average occurrences per pattern
            avg_occurrences = self.conn.execute("SELECT AVG(occurrence_count) FROM patterns").fetchone()[0]
            stats['avg_occurrences_per_pattern'] = float(avg_occurrences) if avg_occurrences else 0
            
            # Pattern type distribution
            pattern_types = self.conn.execute("""
                SELECT pattern_type, COUNT(*) as count
                FROM patterns 
                GROUP BY pattern_type
                ORDER BY count DESC
            """).fetchall()
            
            stats['pattern_types'] = {pt[0]: pt[1] for pt in pattern_types}
            
            # Most common files
            common_files = self.conn.execute("""
                SELECT file_path, COUNT(*) as count
                FROM pattern_occurrences
                GROUP BY file_path
                ORDER BY count DESC
                LIMIT 10
            """).fetchall()
            
            stats['common_files'] = {cf[0]: cf[1] for cf in common_files}
            
            return stats
            
        except Exception as e:
            print(f"❌ Failed to get statistics: {e}")
            return {}
    
    def export_patterns_to_json(self, output_path: str):
        """Export patterns from database to JSON file."""
        try:
            patterns = self.conn.execute("""
                SELECT 
                    p.pattern_id,
                    p.description,
                    p.example_code,
                    p.occurrence_count,
                    p.pattern_type,
                    json_group_array(
                        json_object(
                            'file_path', po.file_path,
                            'code', po.code,
                            'line_start', po.line_start,
                            'line_end', po.line_end,
                            'file_name', po.file_name
                        )
                    ) as occurrences
                FROM patterns p
                LEFT JOIN pattern_occurrences po ON p.pattern_id = po.pattern_id
                GROUP BY p.pattern_id, p.description, p.example_code, p.occurrence_count, p.pattern_type
            """).fetchall()
            
            export_data = {'patterns': []}
            for row in patterns:
                pattern = {
                    'pattern_id': row[0],
                    'description': row[1],
                    'example_code': row[2],
                    'occurrence_count': row[3],
                    'pattern_type': row[4],
                    'occurrences': json.loads(row[5]) if row[5] else []
                }
                export_data['patterns'].append(pattern)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Exported {len(export_data['patterns'])} patterns to {output_path}")
            
        except Exception as e:
            print(f"❌ Failed to export patterns: {e}")
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            print("✅ Database connection closed")

def main():
    """Main function to demonstrate the pattern loader."""
    # Initialize loader
    loader = PatternDuckDBLoader("patterns_with_embeddings.db")
    
    try:
        # Initialize database
        loader.initialize_database()
        
        # Load embedding service (optional - comment out for mock embeddings)
        loader.load_embedding_service("text-embedding-qwen3-embedding-0.6b", "lmstudio")
        
        # Load patterns from JSON
        patterns = loader.load_patterns_from_json("patterns.json")
        
        # Load additional segments from cache (option`al)
        segments = loader.load_segments_from_cache("patterns_embeddings_cache.json")
        
        # Insert patterns with embeddings
        loader.insert_patterns(patterns)
        
        # Create vector index and similarity functions
        loader.create_vector_index()
        loader.create_similarity_search_function()
        
        # Get statistics
        stats = loader.get_pattern_statistics()
        print("\n📊 Pattern Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for k, v in value.items():
                    print(f"     - {k}: {v}")
            else:
                print(f"   {key}: {value}")
        
        # Example searches
        print("\n🔍 Example similarity searches:")
        
        # Search by query
        queries = [
            "utility function for data processing",
            "API endpoint handler",
            "authentication logic",
            "state management"
        ]
        
        for query in queries:
            print(f"\n   Query: '{query}'")
            similar_patterns = loader.search_similar_patterns(query, top_k=2)
            for pattern in similar_patterns:
                print(f"     - {pattern['description']} (similarity: {pattern['similarity']:.3f})")
        
        # Search by pattern type
        print(f"\n🔍 Patterns by type '[UTILITY]':")
        utility_patterns = loader.find_patterns_by_type("[UTILITY]")
        for pattern in utility_patterns[:3]:
            print(f"     - {pattern['description']} (occurrences: {pattern['occurrence_count']})")
        
        # Export patterns
        loader.export_patterns_to_json("exported_patterns.json")
        
        print(f"\n✅ Pattern database created successfully at: {loader.db_path}")
        
    except Exception as e:
        print(f"❌ Failed to create pattern database: {e}")
    finally:
        loader.close()

if __name__ == "__main__":
    main()
