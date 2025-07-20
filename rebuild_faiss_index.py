#!/usr/bin/env python3
"""
Rebuild Faiss index with all 74K records - with progress tracking
"""

import time
import logging
from pathlib import Path
from app.services.faiss_icd10_matcher import create_faiss_icd10_matcher

# Enable debug logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_progress_bar(current, total, bar_length=50, prefix="Progress"):
    """Print a progress bar"""
    if total == 0:
        return
    
    percent = current / total
    filled_length = int(bar_length * percent)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    print(f'\r{prefix}: |{bar}| {current}/{total} ({percent:.1%}) records', end='', flush=True)
    
    if current >= total:
        print()  # New line when complete

def main():
    """Rebuild Faiss index from scratch"""
    print("🔄 Rebuilding Faiss index with ALL records...")
    print("=" * 50)
    
    # Clear existing cache first
    index_dir = Path("data/indexes")
    if index_dir.exists():
        for file in index_dir.glob("faiss_icd10_index*"):
            try:
                file.unlink()
                print(f"🗑️ Removed old index file: {file}")
            except Exception as e:
                print(f"⚠️ Could not remove {file}: {e}")
    
    # Create new Faiss matcher with force rebuild
    start_time = time.time()
    
    try:
        print("🔄 Creating Faiss matcher with force_rebuild=True...")
        matcher = create_faiss_icd10_matcher(force_rebuild=True)
        
        if matcher:
            build_time = time.time() - start_time
            stats = matcher.get_index_stats()
            
            print(f"✅ Faiss index rebuilt successfully in {build_time:.2f}s")
            print(f"📊 Total vectors: {stats.get('total_vectors', 0)}")
            print(f"📊 Index type: {stats.get('index_type', 'unknown')}")
            print(f"📊 Dimension: {stats.get('dimension', 0)}")
            
            # Test a quick search
            print("\n🔍 Testing search functionality...")
            search_start = time.time()
            
            # We need to get an embedding for testing
            from app.services.icd10_vector_matcher import ICD10VectorMatcher
            numpy_matcher = ICD10VectorMatcher(force_numpy=True)
            test_embedding = numpy_matcher._get_entity_embedding('chest pain')
            
            results = matcher.search_similar_codes(test_embedding, top_k=3)
            search_time = (time.time() - search_start) * 1000
            
            print(f"⚡ Search time: {search_time:.1f}ms")
            print(f"📋 Results found: {len(results)}")
            
            if results:
                print(f"🎯 Best match: {results[0].get('icd_code')} - {results[0].get('description', '')[:60]}...")
            
            return True
            
        else:
            print("❌ Failed to create Faiss matcher")
            return False
            
    except Exception as e:
        print(f"❌ Error rebuilding Faiss index: {e}")
        return False

if __name__ == '__main__':
    success = main()
    if success:
        print("\n✅ Faiss index rebuild completed successfully!")
    else:
        print("\n❌ Faiss index rebuild failed!")