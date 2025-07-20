#!/usr/bin/env python3
"""
Diagnostic test to identify Faiss integration issues
"""

import time
import logging
from app.services.faiss_icd10_matcher import create_faiss_icd10_matcher, FAISS_AVAILABLE
from app.services.icd10_vector_matcher import ICD10VectorMatcher
from app.services.enhanced_clinical_analysis import create_enhanced_clinical_analysis_service

# Enable debug logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_faiss_availability():
    """Test if Faiss is available and can be imported"""
    print(f"=== Faiss Availability ===")
    print(f"Faiss available: {FAISS_AVAILABLE}")
    
    if FAISS_AVAILABLE:
        try:
            import faiss
            print(f"Faiss version: {getattr(faiss, '__version__', 'unknown')}")
            return True
        except Exception as e:
            print(f"Faiss import error: {e}")
            return False
    return False

def test_faiss_matcher_creation():
    """Test creating Faiss matcher"""
    print(f"\n=== Faiss Matcher Creation ===")
    
    try:
        # Try to create without force rebuild first
        print("Attempting to create Faiss matcher (no rebuild)...")
        start_time = time.time()
        matcher = create_faiss_icd10_matcher(force_rebuild=False)
        creation_time = time.time() - start_time
        
        if matcher:
            print(f"✅ Faiss matcher created in {creation_time:.2f}s")
            stats = matcher.get_index_stats()
            print(f"Index stats: {stats}")
            return matcher
        else:
            print("❌ Faiss matcher creation returned None")
            return None
            
    except Exception as e:
        print(f"❌ Faiss matcher creation failed: {e}")
        return None

def test_numpy_vs_faiss_performance():
    """Compare numpy vs faiss performance"""
    print(f"\n=== Performance Comparison ===")
    
    # Test numpy matcher
    print("Testing numpy matcher...")
    numpy_start = time.time()
    numpy_matcher = ICD10VectorMatcher(force_numpy=True)
    numpy_creation_time = time.time() - numpy_start
    
    cache_info = numpy_matcher.get_cache_info()
    print(f"Numpy creation time: {numpy_creation_time:.2f}s")
    print(f"Numpy loaded {cache_info.get('total_icd_codes', 0)} codes")
    
    # Test search performance
    search_start = time.time()
    numpy_results = numpy_matcher.find_similar_icd_codes('chest pain', top_k=5)
    numpy_search_time = time.time() - search_start
    print(f"Numpy search time: {numpy_search_time*1000:.1f}ms")
    
    # Test faiss matcher if available
    faiss_matcher = test_faiss_matcher_creation()
    if faiss_matcher:
        try:
            # Get embedding for search test
            test_embedding = numpy_matcher._get_entity_embedding('chest pain')
            
            search_start = time.time()
            faiss_results = faiss_matcher.search_similar_codes(test_embedding, top_k=5)
            faiss_search_time = time.time() - search_start
            print(f"Faiss search time: {faiss_search_time*1000:.1f}ms")
            
            speedup = numpy_search_time / faiss_search_time if faiss_search_time > 0 else float('inf')
            print(f"Faiss speedup: {speedup:.1f}x faster than numpy")
            
        except Exception as e:
            print(f"❌ Faiss search test failed: {e}")
    
    return numpy_matcher, faiss_matcher

def test_enhanced_service_with_faiss():
    """Test enhanced service using Faiss"""
    print(f"\n=== Enhanced Service with Faiss ===")
    
    try:
        # Try to create enhanced service with Faiss (not forcing numpy)
        print("Creating enhanced service (allowing Faiss)...")
        start_time = time.time()
        service = create_enhanced_clinical_analysis_service(force_numpy_icd=False)
        creation_time = time.time() - start_time
        
        print(f"Service creation time: {creation_time:.2f}s")
        
        # Check which matcher it's using
        cache_info = service.icd_matcher.get_cache_info()
        print(f"Service using: {cache_info.get('search_method', 'unknown')}")
        print(f"ICD codes loaded: {cache_info.get('total_icd_codes', 0)}")
        
        # Test analysis performance
        test_note = "Patient has severe chest pain radiating to left arm and shortness of breath."
        
        print("Testing analysis with ICD mapping...")
        analysis_start = time.time()
        result = service.extract_clinical_entities_enhanced(
            test_note,
            include_icd_mapping=True,
            icd_top_k=3,
            enable_nlp_preprocessing=False
        )
        analysis_time = time.time() - analysis_start
        
        print(f"Analysis time: {analysis_time:.2f}s ({analysis_time*1000:.0f}ms)")
        
        if 'icd_mappings' in result:
            print(f"Found {len(result['icd_mappings'])} ICD mappings")
        
        return service, analysis_time
        
    except Exception as e:
        print(f"❌ Enhanced service test failed: {e}")
        return None, None

def main():
    """Run complete Faiss diagnostic"""
    print("🔍 Faiss Integration Diagnostic")
    print("=" * 50)
    
    # Test 1: Faiss availability
    faiss_available = test_faiss_availability()
    
    # Test 2: Performance comparison
    numpy_matcher, faiss_matcher = test_numpy_vs_faiss_performance()
    
    # Test 3: Enhanced service
    service, analysis_time = test_enhanced_service_with_faiss()
    
    # Summary
    print(f"\n=== DIAGNOSTIC SUMMARY ===")
    print(f"Faiss available: {faiss_available}")
    print(f"Faiss matcher created: {faiss_matcher is not None}")
    print(f"Enhanced service created: {service is not None}")
    if analysis_time:
        print(f"Analysis time: {analysis_time:.2f}s")
        print(f"Performance acceptable (<30s): {analysis_time < 30}")

if __name__ == '__main__':
    main()