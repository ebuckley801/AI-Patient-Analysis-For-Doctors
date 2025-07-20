#!/usr/bin/env python3
"""
Performance benchmark tests for Enhanced Clinical Analysis
Tests optimization improvements and measures performance metrics
"""

import unittest
import time
import statistics
from app.services.enhanced_clinical_analysis import create_enhanced_clinical_analysis_service
from app.services.icd10_vector_matcher import ICD10VectorMatcher


class TestAnalysisPerformanceBenchmark(unittest.TestCase):
    """Benchmark performance of clinical analysis pipeline"""
    
    @classmethod
    def setUpClass(cls):
        """Set up benchmark test fixtures"""
        cls.service = create_enhanced_clinical_analysis_service()
        cls.icd_matcher = ICD10VectorMatcher(force_numpy=True)
        
        # Benchmark test cases of varying complexity
        cls.benchmark_notes = {
            'simple': "Patient has chest pain.",
            
            'medium': """45-year-old male with chest pain and shortness of breath. 
                        History of diabetes and hypertension.""",
            
            'complex': """67-year-old female presents to ED with acute onset severe chest pain 
                         radiating to left arm, associated with nausea, diaphoresis, and dyspnea. 
                         Past medical history significant for diabetes mellitus type 2, hypertension, 
                         hyperlipidemia, and family history of coronary artery disease. 
                         Current medications include metformin, lisinopril, and atorvastatin. 
                         Vital signs: BP 180/100, HR 110, RR 22, O2 sat 95% on room air. 
                         Physical exam notable for diaphoresis, mild distress.""",
            
            'very_complex': """82-year-old male with multiple comorbidities including COPD, CHF, 
                              diabetes mellitus, chronic kidney disease stage 3, and atrial fibrillation 
                              presents with acute exacerbation of dyspnea, productive cough with green sputum, 
                              lower extremity edema, and fatigue. Patient reports medication noncompliance. 
                              Current medications include albuterol inhaler, furosemide, metformin, warfarin, 
                              and lisinopril. Vital signs show hypotension (BP 90/60), tachycardia (HR 130), 
                              tachypnea (RR 28), fever (101.5°F), and hypoxemia (O2 sat 88% on 2L NC). 
                              Labs pending. Chest X-ray shows bilateral lower lobe infiltrates and cardiomegaly."""
        }
        
        # Medical terms for ICD search benchmarking
        cls.benchmark_terms = [
            'chest pain', 'shortness of breath', 'fever', 'pneumonia', 'diabetes',
            'hypertension', 'heart attack', 'stroke', 'sepsis', 'kidney disease',
            'COPD', 'asthma', 'migraine', 'depression', 'anxiety'
        ]
    
    def test_baseline_performance(self):
        """Establish baseline performance metrics"""
        print("\n=== BASELINE PERFORMANCE TEST ===")
        
        # Test analysis without ICD mapping
        times_no_icd = []
        for note_type, note_text in self.benchmark_notes.items():
            start_time = time.time()
            
            result = self.service.extract_clinical_entities_enhanced(
                note_text,
                include_icd_mapping=False,
                enable_nlp_preprocessing=False
            )
            
            analysis_time = (time.time() - start_time) * 1000
            times_no_icd.append(analysis_time)
            
            print(f"{note_type:12}: {analysis_time:6.1f}ms (no ICD)")
            
            # Verify successful analysis
            self.assertIsInstance(result, dict, f"Should complete analysis for {note_type}")
            self.assertNotIn('error', result, f"Should not have errors for {note_type}")
        
        # Calculate baseline statistics
        avg_no_icd = statistics.mean(times_no_icd)
        max_no_icd = max(times_no_icd)
        
        print(f"{'Baseline avg':12}: {avg_no_icd:6.1f}ms")
        print(f"{'Baseline max':12}: {max_no_icd:6.1f}ms")
        
        # Performance assertions for baseline
        self.assertLess(avg_no_icd, 8000, "Baseline average should be under 8 seconds")
        self.assertLess(max_no_icd, 15000, "Baseline maximum should be under 15 seconds")
    
    def test_icd_mapping_performance(self):
        """Test performance with ICD mapping enabled"""
        print("\n=== ICD MAPPING PERFORMANCE TEST ===")
        
        times_with_icd = []
        entity_counts = []
        mapping_counts = []
        
        for note_type, note_text in self.benchmark_notes.items():
            start_time = time.time()
            
            result = self.service.extract_clinical_entities_enhanced(
                note_text,
                include_icd_mapping=True,
                icd_top_k=3,
                enable_nlp_preprocessing=False
            )
            
            analysis_time = (time.time() - start_time) * 1000
            times_with_icd.append(analysis_time)
            
            # Count entities and mappings
            total_entities = (len(result.get('symptoms', [])) + 
                            len(result.get('conditions', [])))
            total_mappings = len(result.get('icd_mappings', []))
            
            entity_counts.append(total_entities)
            mapping_counts.append(total_mappings)
            
            print(f"{note_type:12}: {analysis_time:6.1f}ms ({total_entities} entities, {total_mappings} mappings)")
        
        # Calculate statistics
        avg_with_icd = statistics.mean(times_with_icd)
        max_with_icd = max(times_with_icd)
        avg_entities = statistics.mean(entity_counts)
        avg_mappings = statistics.mean(mapping_counts)
        
        print(f"{'ICD avg':12}: {avg_with_icd:6.1f}ms")
        print(f"{'ICD max':12}: {max_with_icd:6.1f}ms")
        print(f"{'Avg entities':12}: {avg_entities:6.1f}")
        print(f"{'Avg mappings':12}: {avg_mappings:6.1f}")
        
        # Performance assertions
        self.assertLess(avg_with_icd, 20000, "ICD mapping average should be under 20 seconds")
        self.assertLess(max_with_icd, 30000, "ICD mapping maximum should be under 30 seconds")
        self.assertGreater(avg_mappings, 0, "Should find some ICD mappings on average")
    
    def test_icd_search_benchmark(self):
        """Benchmark ICD search performance"""
        print("\n=== ICD SEARCH BENCHMARK ===")
        
        search_times = []
        result_counts = []
        
        for term in self.benchmark_terms:
            start_time = time.time()
            
            results = self.icd_matcher.find_similar_icd_codes(
                term, 
                top_k=5, 
                min_similarity=0.01
            )
            
            search_time = (time.time() - start_time) * 1000
            search_times.append(search_time)
            result_counts.append(len(results))
            
            print(f"'{term:20}': {search_time:6.1f}ms ({len(results)} results)")
        
        # Calculate search statistics
        avg_search_time = statistics.mean(search_times)
        max_search_time = max(search_times)
        avg_results = statistics.mean(result_counts)
        
        print(f"{'Search avg':20}: {avg_search_time:6.1f}ms")
        print(f"{'Search max':20}: {max_search_time:6.1f}ms")
        print(f"{'Avg results':20}: {avg_results:6.1f}")
        
        # Performance assertions
        self.assertLess(avg_search_time, 2000, "Average search time should be under 2 seconds")
        self.assertLess(max_search_time, 5000, "Maximum search time should be under 5 seconds")
        self.assertGreater(avg_results, 0, "Should find some results on average")
    
    def test_embedding_generation_performance(self):
        """Benchmark semantic embedding generation"""
        print("\n=== EMBEDDING GENERATION BENCHMARK ===")
        
        embedding_times = []
        
        for term in self.benchmark_terms:
            start_time = time.time()
            
            embedding = self.icd_matcher._get_entity_embedding(term)
            
            embedding_time = (time.time() - start_time) * 1000
            embedding_times.append(embedding_time)
            
            # Verify embedding quality
            self.assertEqual(len(embedding), 1536, f"Embedding should have correct dimensions for {term}")
            self.assertGreater(sum(abs(x) for x in embedding), 0, f"Embedding should not be zero for {term}")
        
        avg_embedding_time = statistics.mean(embedding_times)
        max_embedding_time = max(embedding_times)
        
        print(f"{'Embedding avg':20}: {avg_embedding_time:6.1f}ms")
        print(f"{'Embedding max':20}: {max_embedding_time:6.1f}ms")
        
        # Performance assertions
        self.assertLess(avg_embedding_time, 3000, "Average embedding time should be under 3 seconds")
        self.assertLess(max_embedding_time, 8000, "Maximum embedding time should be under 8 seconds")
    
    def test_cache_performance(self):
        """Test performance improvements from caching"""
        print("\n=== CACHE PERFORMANCE TEST ===")
        
        test_term = 'chest pain'
        
        # First search (cold cache)
        start_time = time.time()
        results1 = self.icd_matcher.find_similar_icd_codes(test_term, top_k=3)
        first_search_time = (time.time() - start_time) * 1000
        
        # Second search (should use any caching)
        start_time = time.time()
        results2 = self.icd_matcher.find_similar_icd_codes(test_term, top_k=3)
        second_search_time = (time.time() - start_time) * 1000
        
        print(f"First search:  {first_search_time:6.1f}ms")
        print(f"Second search: {second_search_time:6.1f}ms")
        
        # Results should be identical
        self.assertEqual(len(results1), len(results2), "Cached results should match")
        
        # Second search should not be significantly slower (within 50% variance is acceptable)
        max_acceptable_ratio = 1.5
        if first_search_time > 100:  # Only test if first search took reasonable time
            ratio = second_search_time / first_search_time
            self.assertLess(ratio, max_acceptable_ratio, 
                          f"Second search should not be much slower (ratio: {ratio:.2f})")
    
    def test_batch_analysis_performance(self):
        """Test performance with batch analysis"""
        print("\n=== BATCH ANALYSIS PERFORMANCE ===")
        
        # Create batch of notes
        batch_notes = []
        for i in range(5):
            for note_type, note_text in list(self.benchmark_notes.items())[:3]:  # Use first 3 note types
                batch_notes.append(note_text)
        
        # Time individual analyses
        start_time = time.time()
        individual_results = []
        for note in batch_notes:
            result = self.service.extract_clinical_entities_enhanced(
                note,
                include_icd_mapping=True,
                enable_nlp_preprocessing=False
            )
            individual_results.append(result)
        individual_time = (time.time() - start_time) * 1000
        
        avg_individual_time = individual_time / len(batch_notes)
        
        print(f"Individual analyses: {individual_time:6.1f}ms total ({avg_individual_time:6.1f}ms avg)")
        print(f"Batch size: {len(batch_notes)} notes")
        
        # Verify all analyses completed
        self.assertEqual(len(individual_results), len(batch_notes), "All analyses should complete")
        
        for i, result in enumerate(individual_results):
            self.assertIsInstance(result, dict, f"Result {i} should be valid")
            self.assertNotIn('error', result, f"Result {i} should not have errors")
        
        # Performance assertion
        self.assertLess(avg_individual_time, 15000, "Average individual analysis should be under 15 seconds")
    
    def test_optimization_verification(self):
        """Verify that optimizations are working as expected"""
        print("\n=== OPTIMIZATION VERIFICATION ===")
        
        # Test the smart search strategy
        print("Testing smart search strategy...")
        
        # Test with a term that should benefit from medical synonyms
        test_term = 'MI'  # Should expand to myocardial infarction
        
        start_time = time.time()
        results = self.icd_matcher.find_similar_icd_codes(test_term, top_k=3)
        search_time = (time.time() - start_time) * 1000
        
        print(f"Smart search for '{test_term}': {search_time:6.1f}ms ({len(results)} results)")
        
        # Verify that entity expansion is working
        expanded = self.icd_matcher._expand_entity_for_matching(test_term)
        self.assertGreater(len(expanded), len(test_term), "Entity should be expanded")
        print(f"Expansion: '{test_term}' -> '{expanded[:50]}...'")
        
        # Test embedding cache info
        cache_info = self.icd_matcher.get_cache_info()
        print(f"ICD cache: {cache_info.get('total_icd_codes', 0)} codes loaded")
        print(f"Search method: {cache_info.get('search_method', 'unknown')}")
        
        # Verify optimizations are active
        self.assertGreater(cache_info.get('total_icd_codes', 0), 0, "ICD codes should be loaded")
        self.assertIn(cache_info.get('search_method'), ['faiss', 'numpy'], "Should use valid search method")


if __name__ == '__main__':
    # Run benchmark tests with detailed output
    unittest.main(verbosity=2)