#!/usr/bin/env python3
"""
Basic functionality test - quick verification that everything works
"""

import unittest
import time
from app.services.enhanced_clinical_analysis import create_enhanced_clinical_analysis_service
from app.services.icd10_vector_matcher import ICD10VectorMatcher


class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality quickly"""
    
    def test_service_initialization(self):
        """Test that services initialize correctly"""
        print("\n=== Testing Service Initialization ===")
        
        # Test numpy matcher (we know this works)
        matcher = ICD10VectorMatcher(force_numpy=True)
        cache_info = matcher.get_cache_info()
        
        print(f"ICD codes loaded: {cache_info.get('total_icd_codes', 0)}")
        print(f"Search method: {cache_info.get('search_method', 'unknown')}")
        
        self.assertGreater(cache_info.get('total_icd_codes', 0), 0, "Should load ICD codes")
        self.assertEqual(cache_info.get('search_method'), 'numpy', "Should use numpy")
    
    def test_enhanced_service_creation(self):
        """Test enhanced service creation"""
        print("\n=== Testing Enhanced Service ===")
        
        service = create_enhanced_clinical_analysis_service(force_numpy_icd=True)
        self.assertIsNotNone(service, "Should create enhanced service")
        
        cache_info = service.icd_matcher.get_cache_info()
        print(f"Service ICD codes: {cache_info.get('total_icd_codes', 0)}")
        
        self.assertGreater(cache_info.get('total_icd_codes', 0), 0, "Service should have ICD codes")
    
    def test_basic_analysis_without_icd(self):
        """Test basic analysis without ICD mapping (should be fast)"""
        print("\n=== Testing Basic Analysis (No ICD) ===")
        
        service = create_enhanced_clinical_analysis_service(force_numpy_icd=True)
        
        start_time = time.time()
        result = service.extract_clinical_entities_enhanced(
            "Patient has chest pain and fever.",
            include_icd_mapping=False,
            enable_nlp_preprocessing=False
        )
        analysis_time = (time.time() - start_time) * 1000
        
        print(f"Analysis time: {analysis_time:.1f}ms")
        
        self.assertIsInstance(result, dict, "Should return dict")
        self.assertIn('symptoms', result, "Should have symptoms")
        self.assertNotIn('icd_mappings', result, "Should not have ICD mappings when disabled")
        self.assertLess(analysis_time, 15000, "Should complete in under 15 seconds")
        
        print(f"Found {len(result.get('symptoms', []))} symptoms")
    
    def test_single_icd_search(self):
        """Test a single ICD search to verify it works"""
        print("\n=== Testing Single ICD Search ===")
        
        matcher = ICD10VectorMatcher(force_numpy=True)
        
        start_time = time.time()
        results = matcher.find_similar_icd_codes('chest pain', top_k=3, min_similarity=0.01)
        search_time = (time.time() - start_time) * 1000
        
        print(f"Search time: {search_time:.1f}ms")
        print(f"Results found: {len(results)}")
        
        self.assertIsInstance(results, list, "Should return list")
        self.assertLess(search_time, 30000, "Should complete search in under 30 seconds")
        
        if results:
            print(f"Best match: {results[0].get('icd_code', 'N/A')} - {results[0].get('description', 'N/A')[:50]}...")


if __name__ == '__main__':
    unittest.main(verbosity=2)