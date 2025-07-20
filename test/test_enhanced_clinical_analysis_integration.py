#!/usr/bin/env python3
"""
Integration test for Enhanced Clinical Analysis with ICD-10 mapping
Tests the complete pipeline from clinical text to ICD code mappings
"""

import unittest
import time
import json
from app.services.enhanced_clinical_analysis import create_enhanced_clinical_analysis_service
from app.services.icd10_vector_matcher import ICD10VectorMatcher


class TestEnhancedClinicalAnalysisIntegration(unittest.TestCase):
    """Test the complete enhanced clinical analysis pipeline"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.service = create_enhanced_clinical_analysis_service(force_numpy_icd=True)  # Use numpy (works reliably)
        cls.icd_matcher = ICD10VectorMatcher(force_numpy=True)  # Use numpy (works reliably)
        
        # Test clinical notes
        cls.test_notes = {
            'cardiac': """Patient is a 55-year-old male presenting with severe chest pain radiating to left arm, 
                         shortness of breath, and diaphoresis. Patient has history of diabetes mellitus and hypertension. 
                         Vital signs: BP 180/100, HR 110, temp 98.6°F. EKG shows ST elevation in leads II, III, aVF.""",
            
            'respiratory': """45-year-old female with 3-day history of fever (102°F), productive cough with yellow sputum, 
                            and shortness of breath. Chest X-ray shows right lower lobe consolidation consistent with pneumonia.""",
            
            'simple': "Patient has chest pain and fever.",
            
            'complex': """67-year-old male with COPD exacerbation. Patient reports increased dyspnea, productive cough 
                        with green sputum, and fatigue. Current medications include albuterol inhaler and prednisone. 
                        Oxygen saturation 88% on room air."""
        }
    
    def test_service_initialization(self):
        """Test that the enhanced clinical analysis service initializes correctly"""
        self.assertIsNotNone(self.service, "Enhanced clinical analysis service should initialize")
        
        # Check ICD matcher status
        cache_info = self.service.icd_matcher.get_cache_info()
        self.assertGreater(cache_info.get('total_icd_codes', 0), 0, "ICD codes should be loaded")
        self.assertIn(cache_info.get('search_method'), ['faiss', 'numpy'], "Search method should be valid")
    
    def test_icd_matcher_direct(self):
        """Test ICD matcher functionality directly"""
        # Test basic search functionality
        results = self.icd_matcher.find_similar_icd_codes('chest pain', top_k=3, min_similarity=0.01)
        
        self.assertIsInstance(results, list, "Results should be a list")
        self.assertLessEqual(len(results), 3, "Should return at most 3 results")
        
        # Check result structure
        if results:
            result = results[0]
            self.assertIn('icd_code', result, "Result should have icd_code")
            self.assertIn('description', result, "Result should have description")
            self.assertIn('similarity', result, "Result should have similarity score")
            self.assertIsInstance(result['similarity'], (int, float), "Similarity should be numeric")
    
    def test_semantic_feature_extraction(self):
        """Test semantic feature extraction for medical terms"""
        test_terms = ['chest pain', 'fever', 'pneumonia', 'diabetes', 'heart attack']
        
        for term in test_terms:
            embedding = self.icd_matcher._get_entity_embedding(term)
            
            self.assertIsNotNone(embedding, f"Embedding should not be None for {term}")
            self.assertEqual(len(embedding), 1536, f"Embedding should have 1536 dimensions for {term}")
            self.assertGreater(sum(abs(x) for x in embedding), 0, f"Embedding should not be zero vector for {term}")
    
    def test_simple_clinical_analysis(self):
        """Test analysis with simple clinical note"""
        result = self.service.extract_clinical_entities_enhanced(
            self.test_notes['simple'],
            include_icd_mapping=True,
            enable_nlp_preprocessing=False
        )
        
        # Check basic structure
        self.assertIsInstance(result, dict, "Result should be a dictionary")
        self.assertIn('symptoms', result, "Result should have symptoms")
        self.assertIn('conditions', result, "Result should have conditions")
        self.assertIn('icd_mappings', result, "Result should have ICD mappings")
        
        # Check that we found some entities
        total_entities = len(result.get('symptoms', [])) + len(result.get('conditions', []))
        self.assertGreater(total_entities, 0, "Should find at least some clinical entities")
    
    def test_complex_clinical_analysis(self):
        """Test analysis with complex clinical note"""
        start_time = time.time()
        
        result = self.service.extract_clinical_entities_enhanced(
            self.test_notes['cardiac'],
            patient_context={'age': 55, 'gender': 'male'},
            include_icd_mapping=True,
            icd_top_k=3,
            enable_nlp_preprocessing=True
        )
        
        analysis_time = (time.time() - start_time) * 1000
        
        # Performance check (numpy is slower but should complete in reasonable time)
        self.assertLess(analysis_time, 120000, "Analysis should complete within 2 minutes")
        
        # Check result structure
        self.assertIn('analysis_timestamp', result, "Result should have timestamp")
        self.assertIn('performance_metrics', result, "Result should have performance metrics")
        
        # Check entity extraction
        self.assertIsInstance(result.get('symptoms', []), list, "Symptoms should be a list")
        self.assertIsInstance(result.get('conditions', []), list, "Conditions should be a list")
        self.assertIsInstance(result.get('vital_signs', []), list, "Vital signs should be a list")
        
        # Check that we found relevant entities for cardiac case
        symptoms = result.get('symptoms', [])
        conditions = result.get('conditions', [])
        
        # Look for cardiac-related entities
        all_entities = []
        for entity_list in [symptoms, conditions]:
            for entity in entity_list:
                entity_text = entity.get('text', entity.get('entity', '')).lower()
                all_entities.append(entity_text)
        
        cardiac_terms = ['chest pain', 'pain', 'chest', 'shortness of breath', 'breath']
        found_cardiac = any(any(term in entity for term in cardiac_terms) for entity in all_entities)
        self.assertTrue(found_cardiac, "Should find cardiac-related entities")
    
    def test_icd_mapping_quality(self):
        """Test quality of ICD code mappings"""
        result = self.service.extract_clinical_entities_enhanced(
            self.test_notes['respiratory'],
            include_icd_mapping=True,
            icd_top_k=3
        )
        
        icd_mappings = result.get('icd_mappings', [])
        
        if icd_mappings:
            # Check mapping structure
            mapping = icd_mappings[0]
            self.assertIn('entity', mapping, "Mapping should have entity")
            self.assertIn('entity_type', mapping, "Mapping should have entity type")
            self.assertIn('icd_matches', mapping, "Mapping should have ICD matches")
            
            # Check ICD match structure
            if mapping.get('icd_matches'):
                icd_match = mapping['icd_matches'][0]
                self.assertIn('code', icd_match, "ICD match should have code")
                self.assertIn('description', icd_match, "ICD match should have description")
                self.assertIn('similarity', icd_match, "ICD match should have similarity")
                
                # Check similarity bounds
                similarity = icd_match['similarity']
                self.assertGreaterEqual(similarity, 0, "Similarity should be non-negative")
                self.assertLessEqual(similarity, 1, "Similarity should not exceed 1")
    
    def test_performance_benchmarking(self):
        """Test performance with multiple analyses"""
        times = []
        
        for note_type, note_text in self.test_notes.items():
            start_time = time.time()
            
            result = self.service.extract_clinical_entities_enhanced(
                note_text,
                include_icd_mapping=True,
                enable_nlp_preprocessing=False  # Skip for speed
            )
            
            analysis_time = (time.time() - start_time) * 1000
            times.append(analysis_time)
            
            # Check that analysis completed successfully
            self.assertIsInstance(result, dict, f"Analysis should complete for {note_type}")
            self.assertNotIn('error', result, f"Analysis should not have errors for {note_type}")
        
        # Performance assertions
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        self.assertLess(avg_time, 60000, "Average analysis time should be under 60 seconds")
        self.assertLess(max_time, 120000, "Maximum analysis time should be under 2 minutes")
        
        print(f"Performance results: avg={avg_time:.1f}ms, max={max_time:.1f}ms")
    
    def test_error_handling(self):
        """Test error handling with invalid inputs"""
        # Test empty input
        result = self.service.extract_clinical_entities_enhanced("")
        self.assertIsInstance(result, dict, "Should handle empty input gracefully")
        
        # Test very long input
        long_text = "Patient has chest pain. " * 1000
        result = self.service.extract_clinical_entities_enhanced(long_text[:5000])  # Truncate to reasonable size
        self.assertIsInstance(result, dict, "Should handle long input gracefully")
        
        # Test non-medical text
        result = self.service.extract_clinical_entities_enhanced("The weather is nice today.")
        self.assertIsInstance(result, dict, "Should handle non-medical text gracefully")
    
    def test_entity_standardization(self):
        """Test that entities have consistent structure"""
        result = self.service.extract_clinical_entities_enhanced(
            self.test_notes['cardiac'],
            include_icd_mapping=True
        )
        
        # Check all entity types for consistent structure
        entity_types = ['symptoms', 'conditions', 'medications', 'vital_signs', 'procedures', 'abnormal_findings']
        
        for entity_type in entity_types:
            entities = result.get(entity_type, [])
            
            for entity in entities:
                # Check required fields
                self.assertIn('text', entity, f"{entity_type} entity should have 'text' field")
                self.assertIn('confidence', entity, f"{entity_type} entity should have 'confidence' field")
                
                # Check field types
                self.assertIsInstance(entity.get('text'), str, f"{entity_type} text should be string")
                self.assertIsInstance(entity.get('confidence'), (int, float), f"{entity_type} confidence should be numeric")
                
                # Check confidence bounds
                confidence = entity.get('confidence', 0)
                self.assertGreaterEqual(confidence, 0, f"{entity_type} confidence should be non-negative")
                self.assertLessEqual(confidence, 1, f"{entity_type} confidence should not exceed 1")


class TestClinicalAnalysisPerformance(unittest.TestCase):
    """Performance-focused tests for clinical analysis"""
    
    def setUp(self):
        """Set up performance test fixtures"""
        self.service = create_enhanced_clinical_analysis_service()
        self.test_note = "Patient presents with chest pain, shortness of breath, and fever."
    
    def test_analysis_without_icd_mapping(self):
        """Test analysis performance without ICD mapping"""
        start_time = time.time()
        
        result = self.service.extract_clinical_entities_enhanced(
            self.test_note,
            include_icd_mapping=False,
            enable_nlp_preprocessing=False
        )
        
        analysis_time = (time.time() - start_time) * 1000
        
        self.assertLess(analysis_time, 15000, "Analysis without ICD mapping should be under 15 seconds")
        self.assertIsInstance(result, dict, "Should return valid result")
        self.assertNotIn('icd_mappings', result, "Should not include ICD mappings when disabled")
    
    def test_analysis_with_icd_mapping(self):
        """Test analysis performance with ICD mapping"""
        start_time = time.time()
        
        result = self.service.extract_clinical_entities_enhanced(
            self.test_note,
            include_icd_mapping=True,
            enable_nlp_preprocessing=False
        )
        
        analysis_time = (time.time() - start_time) * 1000
        
        self.assertLess(analysis_time, 90000, "Analysis with ICD mapping should be under 90 seconds")
        self.assertIsInstance(result, dict, "Should return valid result")
        self.assertIn('icd_mappings', result, "Should include ICD mappings when enabled")
    
    def test_icd_search_performance(self):
        """Test ICD search performance directly"""
        matcher = ICD10VectorMatcher(force_numpy=True)
        
        test_terms = ['chest pain', 'fever', 'pneumonia', 'diabetes', 'hypertension']
        total_start = time.time()
        
        for term in test_terms:
            start_time = time.time()
            results = matcher.find_similar_icd_codes(term, top_k=5)
            search_time = (time.time() - start_time) * 1000
            
            self.assertLess(search_time, 10000, f"ICD search for '{term}' should be under 10 seconds (numpy is slower)")
            self.assertIsInstance(results, list, f"Should return list for '{term}'")
        
        total_time = (time.time() - total_start) * 1000
        avg_time = total_time / len(test_terms)
        
        self.assertLess(avg_time, 5000, "Average ICD search time should be under 5 seconds (numpy)")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)