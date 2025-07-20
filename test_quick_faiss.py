#!/usr/bin/env python3
"""
Quick test of Faiss performance after fixes
"""

import time
from app.services.enhanced_clinical_analysis import create_enhanced_clinical_analysis_service

def test_faiss_performance():
    """Test Faiss performance after embedding speed fix"""
    print("🚀 Testing Faiss Performance After Embedding Fix")
    print("=" * 50)
    
    # Create service allowing Faiss
    service = create_enhanced_clinical_analysis_service(force_numpy_icd=False)
    
    # Check what we're using
    cache_info = service.icd_matcher.get_cache_info()
    print(f"Search method: {cache_info.get('search_method')}")
    print(f"Total ICD codes: {cache_info.get('total_icd_codes')}")
    
    # Simple test
    test_note = "Patient has chest pain and shortness of breath."
    
    print(f"\nTesting analysis: '{test_note}'")
    
    start_time = time.time()
    result = service.extract_clinical_entities_enhanced(
        test_note,
        include_icd_mapping=True,
        icd_top_k=3,
        enable_nlp_preprocessing=False
    )
    analysis_time = (time.time() - start_time) * 1000
    
    print(f"\n📊 Results:")
    print(f"   Analysis time: {analysis_time:.0f}ms ({analysis_time/1000:.2f}s)")
    print(f"   Symptoms: {len(result.get('symptoms', []))}")
    print(f"   ICD mappings: {len(result.get('icd_mappings', []))}")
    
    # Show any errors
    if 'error' in result:
        print(f"   ❌ Error: {result['error']}")
    
    # Performance assessment
    if analysis_time < 5000:  # Under 5 seconds
        print(f"   ✅ EXCELLENT: Analysis completed in {analysis_time:.0f}ms!")
    elif analysis_time < 30000:  # Under 30 seconds
        print(f"   ✅ GOOD: Analysis completed in {analysis_time/1000:.1f}s")
    else:
        print(f"   ❌ STILL SLOW: {analysis_time/1000:.1f}s")
    
    # Show some ICD mappings if found
    if result.get('icd_mappings'):
        print(f"\n📋 ICD Mappings:")
        for i, mapping in enumerate(result['icd_mappings'][:3]):
            entity = mapping.get('entity', 'unknown')
            matches = mapping.get('icd_matches', [])
            if matches:
                best = matches[0]
                print(f"   {i+1}. {entity} → {best.get('code')} ({best.get('similarity', 0):.3f})")

if __name__ == '__main__':
    test_faiss_performance()