#!/usr/bin/env python3
"""
Quick test of parallel search after fixing search_time error
"""

import time
from app.services.enhanced_clinical_analysis import create_enhanced_clinical_analysis_service

def test_quick():
    """Quick test of the fixed parallel search"""
    print("🚀 Quick Test of Fixed Parallel Search")
    print("=" * 40)
    
    service = create_enhanced_clinical_analysis_service(force_numpy_icd=False)
    
    test_note = "Patient has chest pain and shortness of breath."
    
    print(f"Testing: '{test_note}'")
    
    start_time = time.time()
    result = service.extract_clinical_entities_enhanced(
        test_note,
        include_icd_mapping=True,
        icd_top_k=3,
        enable_nlp_preprocessing=False
    )
    analysis_time = (time.time() - start_time) * 1000
    
    print(f"Analysis time: {analysis_time:.0f}ms")
    print(f"Symptoms: {len(result.get('symptoms', []))}")
    print(f"ICD mappings: {len(result.get('icd_mappings', []))}")
    
    if result.get('icd_mappings'):
        print("✅ SUCCESS: Found ICD mappings!")
        for mapping in result['icd_mappings'][:2]:
            entity = mapping.get('entity')
            matches = mapping.get('icd_matches', [])
            if matches:
                best = matches[0]
                print(f"   {entity} → {best.get('code')} (sim: {best.get('similarity', 0):.3f})")
    else:
        print("❌ No ICD mappings found")

if __name__ == '__main__':
    test_quick()