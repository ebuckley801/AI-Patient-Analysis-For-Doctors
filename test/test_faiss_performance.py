#!/usr/bin/env python3
"""
Test the rebuilt Faiss index performance with all 77K records
"""

import time
import unittest
from app.services.enhanced_clinical_analysis import create_enhanced_clinical_analysis_service
from app.services.icd10_vector_matcher import ICD10VectorMatcher

class TestFaissPerformance(unittest.TestCase):
    """Test Faiss performance with full dataset"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        print("\n🔍 Testing Faiss Performance with 77K Records")
        print("=" * 50)
        
        # Create services
        cls.faiss_service = create_enhanced_clinical_analysis_service(force_numpy_icd=False)  # Allow Faiss
        cls.numpy_service = create_enhanced_clinical_analysis_service(force_numpy_icd=True)   # Force numpy
        
        # Test note for analysis
        cls.test_note = """Patient is a 55-year-old male presenting with severe chest pain radiating to left arm, 
                         shortness of breath, and diaphoresis. Patient has history of diabetes mellitus and hypertension. 
                         Vital signs: BP 180/100, HR 110, temp 98.6°F. EKG shows ST elevation in leads II, III, aVF."""
    
    def test_faiss_index_loaded(self):
        """Test that Faiss index is loaded with correct number of records"""
        print("\n=== Testing Faiss Index Loading ===")
        
        cache_info = self.faiss_service.icd_matcher.get_cache_info()
        print(f"Search method: {cache_info.get('search_method')}")
        print(f"Total ICD codes: {cache_info.get('total_icd_codes')}")
        print(f"Cache loaded: {cache_info.get('cache_loaded')}")
        
        self.assertEqual(cache_info.get('search_method'), 'faiss', "Should use Faiss")
        self.assertGreater(cache_info.get('total_icd_codes'), 75000, "Should have 77K+ records")
        self.assertTrue(cache_info.get('cache_loaded'), "Cache should be loaded")
        
        # Print Faiss stats
        faiss_stats = cache_info.get('faiss_stats', {})
        print(f"Faiss index type: {faiss_stats.get('index_type')}")
        print(f"Faiss vectors: {faiss_stats.get('total_vectors')}")
    
    def test_faiss_vs_numpy_performance(self):
        """Compare Faiss vs numpy performance on the same analysis"""
        print("\n=== Performance Comparison: Faiss vs Numpy ===")
        
        # Test Faiss performance
        print("🚀 Testing Faiss performance...")
        faiss_start = time.time()
        faiss_result = self.faiss_service.extract_clinical_entities_enhanced(
            self.test_note,
            include_icd_mapping=True,
            icd_top_k=5,
            enable_nlp_preprocessing=False
        )
        faiss_time = time.time() - faiss_start
        
        # Test numpy performance
        print("🐌 Testing numpy performance...")
        numpy_start = time.time()
        numpy_result = self.numpy_service.extract_clinical_entities_enhanced(
            self.test_note,
            include_icd_mapping=True,
            icd_top_k=5,
            enable_nlp_preprocessing=False
        )
        numpy_time = time.time() - numpy_start
        
        # Performance comparison
        speedup = numpy_time / faiss_time if faiss_time > 0 else float('inf')
        
        print(f"\n📊 Performance Results:")
        print(f"   Faiss time: {faiss_time:.2f}s")
        print(f"   Numpy time: {numpy_time:.2f}s")
        print(f"   Speedup: {speedup:.1f}x faster with Faiss")
        
        # Assertions
        self.assertLess(faiss_time, 30, "Faiss analysis should complete in under 30 seconds")
        self.assertLess(faiss_time, numpy_time, "Faiss should be faster than numpy")
        self.assertGreater(speedup, 2, "Faiss should be at least 2x faster")
        
        # Check results quality
        faiss_mappings = len(faiss_result.get('icd_mappings', []))
        numpy_mappings = len(numpy_result.get('icd_mappings', []))
        
        print(f"   Faiss ICD mappings: {faiss_mappings}")
        print(f"   Numpy ICD mappings: {numpy_mappings}")
        
        self.assertGreater(faiss_mappings, 0, "Faiss should find ICD mappings")
        self.assertEqual(faiss_mappings, numpy_mappings, "Both should find same number of mappings")
    
    def test_faiss_search_accuracy(self):
        """Test Faiss search accuracy for specific medical terms"""
        print("\n=== Testing Faiss Search Accuracy ===")
        
        test_terms = [
            'chest pain',
            'diabetes mellitus', 
            'hypertension',
            'myocardial infarction',
            'pneumonia'
        ]
        
        matcher = self.faiss_service.icd_matcher
        
        for term in test_terms:
            start_time = time.time()
            results = matcher.find_similar_icd_codes(term, top_k=3, min_similarity=0.1)
            search_time = (time.time() - start_time) * 1000
            
            print(f"\n🔍 Search: '{term}'")
            print(f"   Time: {search_time:.1f}ms")
            print(f"   Results: {len(results)}")
            
            self.assertLess(search_time, 1000, f"Search for '{term}' should be under 1 second")
            self.assertGreater(len(results), 0, f"Should find results for '{term}'")
            
            if results:
                best = results[0]
                print(f"   Best: {best.get('icd_code')} - {best.get('description', '')[:50]}...")
                print(f"   Similarity: {best.get('similarity', 0):.3f}")
                
                self.assertGreaterEqual(best.get('similarity', 0), 0.1, f"Best result should have decent similarity for '{term}'")
    
    def test_complex_clinical_analysis_performance(self):
        """Test the full complex clinical analysis that was previously failing"""
        print("\n=== Testing Complex Clinical Analysis (Previous Failure) ===")
        
        start_time = time.time()
        
        result = self.faiss_service.extract_clinical_entities_enhanced(
            self.test_note,
            patient_context={'age': 55, 'gender': 'male'},
            include_icd_mapping=True,
            icd_top_k=3,
            enable_nlp_preprocessing=True  # Enable full processing
        )
        
        analysis_time = (time.time() - start_time) * 1000
        
        print(f"📊 Complex analysis time: {analysis_time:.0f}ms ({analysis_time/1000:.2f}s)")
        
        # This was previously failing at 144s, should now be much faster
        self.assertLess(analysis_time, 30000, "Complex analysis should complete in under 30 seconds")
        
        # Check result quality
        self.assertIn('symptoms', result, "Should have symptoms")
        self.assertIn('icd_mappings', result, "Should have ICD mappings")
        
        icd_mappings = result.get('icd_mappings', [])
        print(f"📋 Found {len(icd_mappings)} ICD mappings")
        
        if icd_mappings:
            for i, mapping in enumerate(icd_mappings[:3]):  # Show first 3
                entity = mapping.get('entity', 'unknown')
                matches = mapping.get('icd_matches', [])
                if matches:
                    best_match = matches[0]
                    print(f"   {i+1}. {entity} → {best_match.get('code')} ({best_match.get('similarity', 0):.3f})")
    
    def test_batch_search_performance(self):
        """Test performance with multiple search terms"""
        print("\n=== Testing Batch Search Performance ===")
        
        terms = [
            'acute chest pain', 'shortness of breath', 'diabetes mellitus',
            'hypertension', 'fever', 'pneumonia', 'heart attack',
            'stroke', 'kidney disease', 'asthma'
        ]
        
        matcher = self.faiss_service.icd_matcher
        
        total_start = time.time()
        all_results = []
        
        for term in terms:
            start_time = time.time()
            results = matcher.find_similar_icd_codes(term, top_k=5)
            search_time = (time.time() - start_time) * 1000
            all_results.append((term, len(results), search_time))
        
        total_time = (time.time() - total_start) * 1000
        avg_time = total_time / len(terms)
        
        print(f"📊 Batch search results:")
        print(f"   Total terms: {len(terms)}")
        print(f"   Total time: {total_time:.0f}ms")
        print(f"   Average per search: {avg_time:.1f}ms")
        print(f"   Searches per second: {1000/avg_time:.1f}")
        
        # Performance assertions
        self.assertLess(avg_time, 500, "Average search should be under 500ms")
        self.assertLess(total_time, 5000, "Batch search should complete under 5 seconds")
        
        # Show some results
        for term, count, time_ms in all_results[:5]:
            print(f"   {term}: {count} results in {time_ms:.1f}ms")

if __name__ == '__main__':
    unittest.main(verbosity=2)