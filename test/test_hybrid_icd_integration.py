#!/usr/bin/env python3
"""
Test the hybrid ICD integration (Vector + Claude AI)
"""

import sys
import os
import time

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from app.services.enhanced_clinical_analysis import create_enhanced_clinical_analysis_service

def test_hybrid_integration():
    """Test the complete hybrid ICD matching integration"""
    print("🚀 Testing Hybrid ICD Integration (Vector + Claude AI)")
    print("=" * 60)
    
    # Create service with hybrid approach
    service = create_enhanced_clinical_analysis_service(force_numpy_icd=False)
    
    # Test cases that previously failed with vector search
    test_cases = [
        {
            'text': "Patient has hypertension and diabetes.",
            'expected_improvements': ['hypertension → I10', 'diabetes → E11.9'],
            'description': "Conditions that should trigger Claude AI"
        },
        {
            'text': "Patient presents with severe chest pain.",
            'expected_improvements': ['chest pain → R07.9'],
            'description': "Symptom that might work with vector search"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_case['description']}")
        print(f"Text: '{test_case['text']}'")
        print(f"Expected: {test_case['expected_improvements']}")
        
        start_time = time.time()
        
        result = service.extract_clinical_entities_enhanced(
            test_case['text'],
            include_icd_mapping=True,
            icd_top_k=3,
            enable_nlp_preprocessing=False
        )
        
        analysis_time = (time.time() - start_time) * 1000
        
        print(f"⏱️  Analysis time: {analysis_time:.0f}ms")
        print(f"📋 Symptoms: {len(result.get('symptoms', []))}")
        print(f"🏥 Conditions: {len(result.get('conditions', []))}")
        print(f"🎯 ICD mappings: {len(result.get('icd_mappings', []))}")
        
        # Show detailed ICD mapping results
        icd_mappings = result.get('icd_mappings', [])
        if icd_mappings:
            print(f"\n📊 ICD Mapping Results:")
            for j, mapping in enumerate(icd_mappings, 1):
                entity = mapping.get('entity', 'unknown')
                matches = mapping.get('icd_matches', [])
                
                if matches:
                    best_match = matches[0]
                    code = best_match.get('code', best_match.get('icd_code', 'N/A'))
                    similarity = best_match.get('similarity', 0)
                    method = best_match.get('search_method', 'unknown')
                    validated = best_match.get('validated', False)
                    
                    status = "✅" if validated else "⚠️"
                    print(f"   {j}. {status} '{entity}' → {code}")
                    print(f"      Method: {method} | Similarity: {similarity:.3f}")
                    
                    if method == 'claude_ai':
                        reasoning = best_match.get('reasoning', '')[:80]
                        print(f"      Reasoning: {reasoning}...")
                        
                    # Check if this matches expected improvements
                    for expected in test_case['expected_improvements']:
                        if entity.lower() in expected.lower() and code in expected:
                            print(f"      🎉 EXPECTED IMPROVEMENT ACHIEVED!")
                            
                else:
                    print(f"   {j}. ❌ '{entity}' → No matches found")
        else:
            print("❌ No ICD mappings found")
        
        # Performance assessment
        if analysis_time < 10000:
            print(f"✅ EXCELLENT: Fast analysis ({analysis_time:.0f}ms)")
        elif analysis_time < 20000:
            print(f"✅ GOOD: Reasonable analysis time ({analysis_time/1000:.1f}s)")
        else:
            print(f"⚠️  Analysis took {analysis_time/1000:.1f}s")

def test_claude_fallback_specifically():
    """Test that Claude AI is used when vector search fails"""
    print(f"\n🤖 Testing Claude AI Fallback Logic")
    print("=" * 40)
    
    service = create_enhanced_clinical_analysis_service(force_numpy_icd=False)
    
    # Test with a term that should definitely trigger Claude fallback
    test_note = "Patient diagnosed with essential hypertension."
    
    print(f"Testing: '{test_note}'")
    print("This should trigger Claude AI since 'essential hypertension' is specific medical terminology")
    
    start_time = time.time()
    result = service.extract_clinical_entities_enhanced(
        test_note,
        include_icd_mapping=True,
        icd_top_k=3,
        enable_nlp_preprocessing=False
    )
    analysis_time = (time.time() - start_time) * 1000
    
    print(f"\n📊 Results:")
    print(f"   Analysis time: {analysis_time:.0f}ms")
    
    icd_mappings = result.get('icd_mappings', [])
    if icd_mappings:
        for mapping in icd_mappings:
            entity = mapping.get('entity')
            matches = mapping.get('icd_matches', [])
            if matches:
                best = matches[0]
                method = best.get('search_method', 'unknown')
                code = best.get('code')
                
                print(f"   Entity: {entity}")
                print(f"   Method used: {method}")
                print(f"   Code: {code}")
                
                if method == 'claude_ai':
                    print(f"   🤖 SUCCESS: Claude AI was used!")
                elif method in ['faiss_batch', 'faiss', 'numpy']:
                    print(f"   🚀 Vector search succeeded")
                else:
                    print(f"   ❓ Unknown method: {method}")

if __name__ == '__main__':
    test_hybrid_integration()
    test_claude_fallback_specifically()