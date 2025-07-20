#!/usr/bin/env python3
"""
Diagnose why Faiss search is extremely slow (11+ seconds)
"""

import time
import numpy as np
from app.services.faiss_icd10_matcher import create_faiss_icd10_matcher
from app.services.icd10_vector_matcher import ICD10VectorMatcher

def test_faiss_search_bottleneck():
    """Test where the bottleneck is in Faiss search"""
    print("🔍 Diagnosing Faiss Search Performance Issues")
    print("=" * 50)
    
    # Create matchers
    print("Creating Faiss matcher...")
    faiss_matcher = create_faiss_icd10_matcher()
    
    print("Creating numpy matcher...")
    numpy_matcher = ICD10VectorMatcher(force_numpy=True)
    
    if not faiss_matcher:
        print("❌ Faiss matcher creation failed")
        return
    
    # Get stats
    faiss_stats = faiss_matcher.get_index_stats()
    print(f"\n📊 Faiss Index Stats:")
    print(f"   Total vectors: {faiss_stats.get('total_vectors')}")
    print(f"   Index type: {faiss_stats.get('index_type')}")
    print(f"   Dimension: {faiss_stats.get('dimension')}")
    print(f"   Is trained: {faiss_stats.get('index_is_trained')}")
    
    # Test embedding generation speed
    print(f"\n🧪 Testing Embedding Generation:")
    start_time = time.time()
    test_embedding = numpy_matcher._get_entity_embedding('chest pain')
    embedding_time = (time.time() - start_time) * 1000
    print(f"   Embedding generation: {embedding_time:.1f}ms")
    print(f"   Embedding shape: {test_embedding.shape}")
    print(f"   Embedding type: {type(test_embedding)}")
    
    # Test direct Faiss search (bypassing entity embedding)
    print(f"\n🔍 Testing Direct Faiss Search:")
    query_vector = test_embedding.reshape(1, -1).astype(np.float32)
    
    start_time = time.time()
    similarities, indices = faiss_matcher.index.search(query_vector, 5)
    direct_search_time = (time.time() - start_time) * 1000
    print(f"   Direct Faiss search: {direct_search_time:.1f}ms")
    
    # Test Faiss search through wrapper
    print(f"\n🔍 Testing Faiss Wrapper Search:")
    start_time = time.time()
    results = faiss_matcher.search_similar_codes(test_embedding, top_k=5)
    wrapper_search_time = (time.time() - start_time) * 1000
    print(f"   Wrapper search: {wrapper_search_time:.1f}ms")
    print(f"   Results found: {len(results)}")
    
    # Test ICD10VectorMatcher.find_similar_icd_codes (the full pipeline)
    print(f"\n🔍 Testing Full Pipeline Search:")
    start_time = time.time()
    pipeline_results = numpy_matcher.find_similar_icd_codes('chest pain', top_k=5)
    pipeline_time = (time.time() - start_time) * 1000
    print(f"   Full pipeline: {pipeline_time:.1f}ms")
    print(f"   Pipeline results: {len(pipeline_results)}")
    
    # Compare with numpy search
    print(f"\n📊 Performance Comparison:")
    print(f"   Embedding generation: {embedding_time:.1f}ms")
    print(f"   Direct Faiss search: {direct_search_time:.1f}ms")
    print(f"   Faiss wrapper: {wrapper_search_time:.1f}ms")
    print(f"   Full pipeline (numpy): {pipeline_time:.1f}ms")
    
    if wrapper_search_time > 1000:
        print(f"\n❌ PROBLEM: Faiss wrapper taking {wrapper_search_time:.0f}ms (should be <100ms)")
        print("   Possible issues:")
        print("   - Index not optimized for search")
        print("   - Wrong index type for dataset size")
        print("   - Search parameters not tuned")
    
    # Test if the problem is in ICD10VectorMatcher when using Faiss
    print(f"\n🔍 Testing ICD10VectorMatcher with Faiss:")
    faiss_icd_matcher = ICD10VectorMatcher(force_numpy=False)  # Allow Faiss
    
    start_time = time.time()
    faiss_icd_results = faiss_icd_matcher.find_similar_icd_codes('chest pain', top_k=5)
    faiss_icd_time = (time.time() - start_time) * 1000
    print(f"   ICD matcher with Faiss: {faiss_icd_time:.1f}ms")
    print(f"   Results: {len(faiss_icd_results)}")
    
    if faiss_icd_time > 5000:
        print(f"\n❌ MAJOR PROBLEM: ICD matcher with Faiss taking {faiss_icd_time:.0f}ms")
        print("   This suggests the issue is in ICD10VectorMatcher, not Faiss itself")

if __name__ == '__main__':
    test_faiss_search_bottleneck()