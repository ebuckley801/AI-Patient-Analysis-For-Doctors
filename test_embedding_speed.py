#!/usr/bin/env python3
"""
Test embedding generation speed after fixes
"""

import time
from app.services.icd10_vector_matcher import ICD10VectorMatcher

def test_embedding_speed():
    """Test embedding generation speed"""
    print("🚀 Testing Embedding Speed After Fixes")
    print("=" * 40)
    
    # Create matcher
    matcher = ICD10VectorMatcher(force_numpy=False)
    
    test_terms = [
        'chest pain',
        'diabetes mellitus',
        'hypertension',
        'pneumonia',
        'heart attack'
    ]
    
    print("Testing embedding generation speed...")
    
    total_start = time.time()
    
    for term in test_terms:
        start_time = time.time()
        embedding = matcher._get_entity_embedding(term)
        embedding_time = (time.time() - start_time) * 1000
        
        print(f"  {term}: {embedding_time:.1f}ms")
        
        # Second call should be cached and even faster
        start_time = time.time()
        embedding2 = matcher._get_entity_embedding(term)
        cached_time = (time.time() - start_time) * 1000
        
        print(f"  {term} (cached): {cached_time:.1f}ms")
    
    total_time = (time.time() - total_start) * 1000
    avg_time = total_time / (len(test_terms) * 2)  # 2 calls per term
    
    print(f"\nTotal time: {total_time:.1f}ms")
    print(f"Average per embedding: {avg_time:.1f}ms")
    
    if avg_time < 100:
        print("✅ EXCELLENT: Embedding generation is now fast!")
    elif avg_time < 1000:
        print("✅ GOOD: Embedding generation is much improved")
    else:
        print("❌ STILL SLOW: Embedding generation needs more work")

if __name__ == '__main__':
    test_embedding_speed()