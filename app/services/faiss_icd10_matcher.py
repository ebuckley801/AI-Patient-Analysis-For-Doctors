#!/usr/bin/env python3
"""
Faiss ICD-10 Vector Matcher
High-performance vector similarity search optimized for 70K+ ICD codes using Faiss
"""

import logging
import numpy as np
import pickle
import ast
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import json

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("✅ Faiss successfully imported")
except ImportError as e:
    FAISS_AVAILABLE = False
    logger.warning(f"❌ Faiss not available: {e}")
    logger.warning("Install with: pip install faiss-cpu")

from app.services.supabase_service import SupabaseService


class FaissICD10VectorMatcher:
    """
    High-performance ICD-10 vector matching using Faiss
    Optimized for large datasets (70K+ entries) with advanced indexing strategies
    """
    
    def __init__(self, index_path: Optional[str] = None, force_rebuild: bool = False):
        """
        Initialize Faiss ICD-10 vector matcher
        
        Args:
            index_path: Path to save/load Faiss index files
            force_rebuild: Force rebuilding index from database
        """
        if not FAISS_AVAILABLE:
            raise ImportError("Faiss is required but not available. Install with: pip install faiss-cpu")
        
        self.dimension = 1536  # OpenAI text-embedding-ada-002 dimension
        self.index = None
        self.icd_metadata = []
        self.supabase_service = SupabaseService()
        
        # File paths for persistence
        self.index_path = index_path or "data/indexes/faiss_icd10_index.bin"
        self.metadata_path = self.index_path.replace('.bin', '_metadata.pkl')
        self.config_path = self.index_path.replace('.bin', '_config.json')
        
        # Ensure directory exists
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.build_time = None
        self.total_vectors = 0
        self.index_type = None
        
        # Initialize index
        if force_rebuild or not self._load_existing_index():
            logger.info("🔄 Building Faiss index from database...")
            self._build_index_from_database()
        else:
            logger.info("✅ Loaded existing Faiss index")
    
    def _load_existing_index(self) -> bool:
        """
        Load existing Faiss index and metadata from disk
        
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            if not (Path(self.index_path).exists() and 
                   Path(self.metadata_path).exists() and 
                   Path(self.config_path).exists()):
                return False
            
            # Load index
            self.index = faiss.read_index(self.index_path)
            
            # Load metadata
            with open(self.metadata_path, 'rb') as f:
                self.icd_metadata = pickle.load(f)
            
            # Load config
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                self.total_vectors = config.get('total_vectors', 0)
                self.index_type = config.get('index_type', 'unknown')
                self.build_time = config.get('build_time', None)
            
            logger.info(f"📊 Loaded Faiss index: {self.total_vectors} vectors, type: {self.index_type}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load existing index: {e}")
            return False
    
    def _build_index_from_database(self) -> None:
        """
        Build optimized Faiss index from Supabase ICD-10 data
        Handles large datasets (70K+ entries) with memory-efficient processing
        """
        start_time = time.time()
        
        try:
            logger.info("📊 Loading ICD-10 data from database...")
            
            # Load ALL data in batches - continue until no more records
            batch_size = 1000  # Process in chunks for memory efficiency
            all_embeddings = []
            all_metadata = []
            offset = 0
            batch_count = 0
            consecutive_errors = 0
            max_consecutive_errors = 3  # Stop only after 3 consecutive errors
            
            logger.info(f"🔄 Loading ALL ICD records from database (estimated ~74K records)...")
            
            while True:  # Continue until we've read all records
                try:
                    # Query batch of data
                    response = self.supabase_service.client.table('icd_10_codes')\
                        .select('icd_10_code,description,embedded_description')\
                        .range(offset, offset + batch_size - 1)\
                        .execute()
                    
                    if not response.data:
                        logger.info(f"📋 No more data at offset {offset}, finished loading")
                        break
                    
                    batch_embeddings, batch_metadata = self._process_batch(response.data)
                    all_embeddings.extend(batch_embeddings)
                    all_metadata.extend(batch_metadata)
                    
                    offset += batch_size
                    batch_count += 1
                    consecutive_errors = 0  # Reset error counter on success
                    
                    # Progress reporting with estimated total
                    estimated_total = 74000  # Estimated 74K records
                    current_total = len(all_embeddings)
                    percent = min((current_total / estimated_total) * 100, 100)
                    
                    # Progress bar
                    bar_length = 40
                    filled_length = int(bar_length * (current_total / estimated_total))
                    bar = '█' * filled_length + '-' * (bar_length - filled_length)
                    
                    print(f'\r📈 Loading: |{bar}| {current_total}/{estimated_total} ({percent:.1f}%) records', end='', flush=True)
                    
                    # Also log every 10 batches
                    if batch_count % 10 == 0:
                        logger.info(f"📈 Batch {batch_count}: Processed {len(batch_embeddings)} vectors (total: {len(all_embeddings)})")
                    
                    # Break if batch was smaller than batch_size (last batch)
                    if len(response.data) < batch_size:
                        logger.info(f"📋 Last batch detected ({len(response.data)} < {batch_size}), finished loading ALL records")
                        break
                        
                except Exception as batch_error:
                    consecutive_errors += 1
                    logger.error(f"❌ Error in batch {batch_count} at offset {offset}: {batch_error}")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"❌ {max_consecutive_errors} consecutive errors, stopping batch loading")
                        break
                    
                    # Try to continue with next batch
                    offset += batch_size
                    batch_count += 1
                    logger.info(f"🔄 Retrying with next batch (error {consecutive_errors}/{max_consecutive_errors})")
                    continue
            
            # Final progress update
            print()  # New line after progress bar
            logger.info(f"✅ Completed loading {len(all_embeddings)} records in {batch_count} batches")
            
            if not all_embeddings:
                raise ValueError(f"No valid embeddings found in database after processing {batch_count} batches")
            
            # Convert to numpy array
            logger.info(f"🔄 Converting {len(all_embeddings)} embeddings to numpy array...")
            embeddings_array = np.array(all_embeddings, dtype=np.float32)
            self.icd_metadata = all_metadata
            self.total_vectors = len(all_embeddings)
            
            logger.info(f"📊 Dataset summary: {self.total_vectors} vectors, dimension: {self.dimension}")
            
            # Build optimized index based on dataset size
            logger.info(f"🔄 Building Faiss index for {self.total_vectors} vectors...")
            self._build_optimized_index(embeddings_array)
            
            # Save index and metadata
            logger.info(f"💾 Saving index and metadata...")
            self._save_index_and_metadata(embeddings_array)
            
            self.build_time = time.time() - start_time
            logger.info(f"✅ Faiss index built successfully in {self.build_time:.2f}s with {self.total_vectors} vectors")
            
        except Exception as e:
            logger.error(f"❌ Failed to build Faiss index: {e}")
            raise
    
    def _process_batch(self, batch_data: List[Dict]) -> Tuple[List[List[float]], List[Dict]]:
        """
        Process a batch of ICD code data, extracting valid embeddings
        
        Args:
            batch_data: List of ICD code records from database
            
        Returns:
            Tuple of (embeddings_list, metadata_list)
        """
        embeddings = []
        metadata = []
        
        for record in batch_data:
            try:
                # Parse embedding - handle both string and list formats
                embedding_data = record.get('embedded_description', '[]')
                
                if isinstance(embedding_data, str):
                    try:
                        embedding = ast.literal_eval(embedding_data)
                    except (ValueError, SyntaxError):
                        # Try JSON parsing as fallback
                        import json
                        embedding = json.loads(embedding_data)
                elif isinstance(embedding_data, list):
                    embedding = embedding_data
                else:
                    logger.warning(f"⚠️ Unsupported embedding format for {record.get('icd_10_code')}")
                    continue
                
                # Validate embedding
                if (isinstance(embedding, list) and 
                    len(embedding) == self.dimension and 
                    all(isinstance(x, (int, float)) and not np.isnan(x) for x in embedding)):
                    
                    embeddings.append(embedding)
                    metadata.append({
                        'icd_code': record['icd_10_code'],
                        'description': record['description']
                    })
                else:
                    logger.warning(f"⚠️ Invalid embedding for {record.get('icd_10_code')}: len={len(embedding) if isinstance(embedding, list) else 'N/A'}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Error processing {record.get('icd_10_code')}: {e}")
                continue
        
        return embeddings, metadata
    
    def _build_optimized_index(self, embeddings: np.ndarray) -> None:
        """
        Build optimized Faiss index based on dataset size
        
        Args:
            embeddings: Numpy array of embeddings [n_vectors, dimension]
        """
        n_vectors = len(embeddings)
        
        if n_vectors < 1000:
            # Small dataset: Use flat L2 for exact search
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index_type = "FlatL2"
            logger.info("🔧 Using FlatL2 index for small dataset")
            
        elif n_vectors < 10000:
            # Medium dataset: Use HNSW for good accuracy/speed tradeoff
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 50
            self.index_type = "HNSWFlat"
            logger.info("🔧 Using HNSWFlat index for medium dataset")
            
        else:
            # Large dataset (70K+): Use IVF + Product Quantization for memory efficiency
            nlist = min(int(np.sqrt(n_vectors)), 4096)  # Number of clusters
            quantizer = faiss.IndexFlatL2(self.dimension)
            
            # Use Product Quantization for memory compression
            self.index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, 16, 8)
            self.index_type = f"IVFPQ_nlist{nlist}"
            
            # Train the index (required for IVF)
            logger.info(f"🔧 Training IVFPQ index with {nlist} clusters...")
            self.index.train(embeddings)
            
            logger.info(f"🔧 Using IVFPQ index for large dataset (nlist={nlist})")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add vectors to index
        logger.info("📥 Adding vectors to index...")
        self.index.add(embeddings)
        
        # Set optimal search parameters for large indices
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = min(32, max(1, nlist // 32))
            logger.info(f"🔧 Set nprobe to {self.index.nprobe} for optimal search")
    
    def _save_index_and_metadata(self, embeddings: np.ndarray) -> None:
        """
        Save Faiss index, metadata, and configuration to disk
        
        Args:
            embeddings: Original embeddings array for validation
        """
        try:
            # Save Faiss index
            faiss.write_index(self.index, self.index_path)
            
            # Save metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.icd_metadata, f)
            
            # Save configuration
            config = {
                'total_vectors': self.total_vectors,
                'dimension': self.dimension,
                'index_type': self.index_type,
                'build_time': time.time(),
                'embeddings_shape': embeddings.shape,
                'faiss_version': faiss.__version__ if hasattr(faiss, '__version__') else 'unknown'
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"💾 Saved index files: {self.index_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save index: {e}")
            raise
    
    def search_similar_codes(self, 
                           query_embedding: np.ndarray, 
                           top_k: int = 5, 
                           min_similarity: float = 0.1) -> List[Dict[str, Any]]:
        """
        High-performance vector similarity search using Faiss
        
        Args:
            query_embedding: Query vector [dimension,]
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold (cosine similarity)
            
        Returns:
            List of similar ICD codes with metadata and similarity scores
        """
        if self.index is None or not self.icd_metadata:
            logger.warning("⚠️ Index not initialized")
            return []
        
        try:
            # Prepare query vector
            query_vector = query_embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query_vector)
            
            # Perform search
            search_k = min(top_k * 2, len(self.icd_metadata))  # Search more, filter later
            similarities, indices = self.index.search(query_vector, search_k)
            
            # Format results
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx >= 0 and idx < len(self.icd_metadata):  # Valid index
                    # Convert L2 distance to cosine similarity (for normalized vectors)
                    cosine_sim = 1.0 - (similarity / 2.0)
                    
                    if cosine_sim >= min_similarity:
                        metadata = self.icd_metadata[idx]
                        results.append({
                            'icd_code': metadata['icd_code'],
                            'description': metadata['description'],
                            'similarity': float(cosine_sim),
                            'rank': len(results) + 1,
                            'search_method': 'faiss_vector',
                            'index_type': self.index_type
                        })
                        
                        if len(results) >= top_k:
                            break
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the Faiss index
        
        Returns:
            Dictionary with index statistics and performance metrics
        """
        stats = {
            'total_vectors': self.total_vectors,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'build_time_seconds': self.build_time,
            'faiss_available': FAISS_AVAILABLE,
            'index_loaded': self.index is not None,
            'metadata_loaded': len(self.icd_metadata) > 0
        }
        
        if self.index is not None:
            stats.update({
                'index_is_trained': self.index.is_trained,
                'index_ntotal': self.index.ntotal,
                'index_d': self.index.d
            })
            
            # Add index-specific stats
            if hasattr(self.index, 'nlist'):
                stats['nlist'] = self.index.nlist
            if hasattr(self.index, 'nprobe'):
                stats['nprobe'] = self.index.nprobe
        
        return stats
    
    def benchmark_search(self, num_queries: int = 100) -> Dict[str, float]:
        """
        Benchmark search performance
        
        Args:
            num_queries: Number of test queries to run
            
        Returns:
            Performance metrics
        """
        if not self.icd_metadata:
            return {'error': 'No data loaded'}
        
        # Generate random query vectors
        np.random.seed(42)
        query_vectors = np.random.randn(num_queries, self.dimension).astype(np.float32)
        faiss.normalize_L2(query_vectors)
        
        # Benchmark single queries
        start_time = time.time()
        for query in query_vectors:
            self.search_similar_codes(query, top_k=5)
        single_query_time = time.time() - start_time
        
        # Benchmark batch search if available
        batch_time = None
        if hasattr(self.index, 'search'):
            start_time = time.time()
            similarities, indices = self.index.search(query_vectors, 5)
            batch_time = time.time() - start_time
        
        return {
            'num_queries': num_queries,
            'total_single_query_time': single_query_time,
            'avg_single_query_ms': (single_query_time / num_queries) * 1000,
            'queries_per_second': num_queries / single_query_time,
            'total_batch_time': batch_time,
            'avg_batch_query_ms': (batch_time / num_queries * 1000) if batch_time else None,
            'batch_queries_per_second': (num_queries / batch_time) if batch_time else None
        }
    
    def rebuild_index(self) -> None:
        """Force rebuild of the index from database"""
        logger.info("🔄 Force rebuilding Faiss index...")
        self._build_index_from_database()
    
    def clear_cache(self) -> None:
        """Clear cached index files"""
        try:
            for file_path in [self.index_path, self.metadata_path, self.config_path]:
                if Path(file_path).exists():
                    Path(file_path).unlink()
                    logger.info(f"🗑️ Removed {file_path}")
        except Exception as e:
            logger.error(f"❌ Error clearing cache: {e}")


def create_faiss_icd10_matcher(index_path: Optional[str] = None, 
                             force_rebuild: bool = False) -> Optional[FaissICD10VectorMatcher]:
    """
    Factory function to create FaissICD10VectorMatcher with error handling
    
    Args:
        index_path: Path for index files
        force_rebuild: Force rebuild from database
        
    Returns:
        FaissICD10VectorMatcher instance or None if creation fails
    """
    try:
        return FaissICD10VectorMatcher(index_path=index_path, force_rebuild=force_rebuild)
    except Exception as e:
        logger.error(f"❌ Failed to create Faiss matcher: {e}")
        return None