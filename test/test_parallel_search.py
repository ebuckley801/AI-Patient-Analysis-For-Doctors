#!/usr/bin/env python3
"""
Test the new parallel search implementation
"""

import time
from app.services.enhanced_clinical_analysis import create_enhanced_clinical_analysis_service

def test_parallel_search():
    """Test parallel variant search implementation"""
    print("🚀 Testing Parallel Multi-Variant Search Implementation")
    print("=" * 60)
    
    # Create service with Faiss
    service = create_enhanced_clinical_analysis_service(force_numpy_icd=False)
    
    # Check setup
    cache_info = service.icd_matcher.get_cache_info()
    print(f"Search method: {cache_info.get('search_method')}")
    print(f"Total ICD codes: {cache_info.get('total_icd_codes')}")
    
    # Test cases with expected improvements
    test_cases = [
        {
            'text': "Patient has chest pain and shortness of breath.",
            'expected_entities': ['chest pain', 'shortness of breath'],
            'description': "Common cardiac symptoms with known synonyms"
        },
        {
            'text': "Patient presents with severe diabetes and hypertension.",
            'expected_entities': ['diabetes', 'hypertension'],
            'description': "Chronic conditions with medical synonyms"
        },
        {
            'text': "Acute myocardial infarction with chest discomfort.",
            'expected_entities': ['myocardial infarction', 'chest discomfort'],
            'description': "Specific medical terminology"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}: {test_case['description']}")
        print(f"Text: '{test_case['text']}'")
        
        start_time = time.time()
        
        result = service.extract_clinical_entities_enhanced(
            test_case['text'],
            include_icd_mapping=True,
            icd_top_k=3,
            enable_nlp_preprocessing=False
        )
        
        analysis_time = (time.time() - start_time) * 1000
        
        print(f"⏱️  Analysis time: {analysis_time:.0f}ms")
        print(f"📋 Symptoms found: {len(result.get('symptoms', []))}")
        print(f"🏥 Conditions found: {len(result.get('conditions', []))}")
        print(f"🔍 ICD mappings: {len(result.get('icd_mappings', []))}")
        
        # Show ICD mappings with variant info
        icd_mappings = result.get('icd_mappings', [])
        if icd_mappings:
            print(f"📊 ICD Mapping Results:")
            for j, mapping in enumerate(icd_mappings[:3], 1):
                entity = mapping.get('entity', 'unknown')
                matches = mapping.get('icd_matches', [])
                
                if matches:
                    best_match = matches[0]
                    code = best_match.get('code', best_match.get('icd_code', 'N/A'))
                    similarity = best_match.get('similarity', 0)
                    variant_count = best_match.get('variant_count', 1)
                    confidence_boost = best_match.get('confidence_boost', 0)
                    
                    print(f"   {j}. '{entity}' → {code}")
                    print(f"      Similarity: {similarity:.3f} (boost: +{confidence_boost:.3f})")
                    print(f"      Found via {variant_count} variant(s)")
                    
                    if 'variants_found' in best_match:
                        variants = best_match['variants_found'][:3]  # Show first 3
                        print(f"      Variants: {variants}")
        else:
            print("❌ No ICD mappings found")
        
        # Performance assessment
        if analysis_time < 5000:
            print(f"✅ EXCELLENT: Fast analysis ({analysis_time:.0f}ms)")
        elif analysis_time < 15000:
            print(f"✅ GOOD: Reasonable analysis time ({analysis_time/1000:.1f}s)")
        else:
            print(f"⚠️  SLOW: Analysis took {analysis_time/1000:.1f}s")

def test_variant_generation():
    """Test the variant generation specifically"""
    print(f"\n🧪 Testing Variant Generation")
    print("=" * 40)
    
    service = create_enhanced_clinical_analysis_service(force_numpy_icd=False)
    
    test_entities = [
        {'entity': 'chest pain', 'severity': 'severe', 'temporal': 'acute'},
        {'entity': 'shortness of breath', 'severity': 'moderate'},
        {'entity': 'diabetes', 'status': 'chronic'},
        {'entity': 'heart attack', 'temporal': 'sudden'}
    ]
    
    for entity_data in test_entities:
        entity_text = entity_data['entity']
        variants = service._generate_search_variants(entity_text, entity_data)
        
        print(f"\n🔍 '{entity_text}' → {len(variants)} variants:")
        for variant in variants:
            print(f"   • {variant}")

if __name__ == '__main__':
    test_parallel_search()
    test_variant_generation()