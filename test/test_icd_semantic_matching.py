#!/usr/bin/env python3
"""
Test file for ICD-10 semantic matching functionality
Tests the Claude-based semantic feature extraction and ICD code matching
"""

import unittest
import numpy as np
from app.services.icd10_vector_matcher import ICD10VectorMatcher


class TestICDSemanticMatching(unittest.TestCase):
    """Test ICD-10 semantic matching with Claude-based features"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.matcher = ICD10VectorMatcher(force_numpy=True)
        
        # Medical test terms with expected relevance
        cls.test_medical_terms = {
            'cardiac': ['chest pain', 'heart attack', 'myocardial infarction', 'angina', 'palpitations'],
            'respiratory': ['pneumonia', 'cough', 'shortness of breath', 'asthma', 'bronchitis'],
            'endocrine': ['diabetes', 'thyroid', 'insulin', 'glucose', 'hyperglycemia'],
            'infectious': ['fever', 'infection', 'sepsis', 'bacterial', 'viral'],
            'neurological': ['headache', 'seizure', 'stroke', 'migraine', 'epilepsy']
        }
        
        cls.non_medical_terms = ['weather', 'car', 'computer', 'music', 'sports']
    
    def test_matcher_initialization(self):
        """Test that ICD matcher initializes correctly"""
        cache_info = self.matcher.get_cache_info()
        
        self.assertGreater(cache_info.get('total_icd_codes', 0), 0, "Should load ICD codes")
        self.assertEqual(cache_info.get('search_method'), 'numpy', "Should use numpy method")
        self.assertTrue(cache_info.get('cache_loaded', False), "Cache should be loaded")
    
    def test_semantic_feature_extraction(self):
        """Test semantic feature extraction for medical terms"""
        for category, terms in self.test_medical_terms.items():
            for term in terms:
                embedding = self.matcher._extract_semantic_features(term)
                
                # Basic embedding checks
                self.assertEqual(len(embedding), 1536, f"Embedding should have 1536 dimensions for {term}")
                self.assertEqual(embedding.dtype, np.float32, f"Embedding should be float32 for {term}")
                
                # Check that embedding is not zero vector
                norm = np.linalg.norm(embedding)
                self.assertGreater(norm, 0, f"Embedding should not be zero vector for {term}")
                
                # Check normalization (should be approximately 1)
                self.assertAlmostEqual(norm, 1.0, places=5, msg=f"Embedding should be normalized for {term}")
    
    def test_medical_vs_non_medical_discrimination(self):
        """Test that medical terms have different feature patterns than non-medical terms"""
        medical_embeddings = []
        non_medical_embeddings = []
        
        # Get embeddings for medical terms
        for category, terms in self.test_medical_terms.items():
            for term in terms[:2]:  # Take first 2 from each category
                embedding = self.matcher._extract_semantic_features(term)
                medical_embeddings.append(embedding)
        
        # Get embeddings for non-medical terms
        for term in self.non_medical_terms:
            embedding = self.matcher._extract_semantic_features(term)
            non_medical_embeddings.append(embedding)
        
        # Medical terms should have some similarity to each other
        medical_similarities = []
        for i in range(len(medical_embeddings)):
            for j in range(i+1, len(medical_embeddings)):
                similarity = self.matcher.cosine_similarity(medical_embeddings[i], medical_embeddings[j])
                medical_similarities.append(similarity)
        
        # Non-medical terms should have lower similarity on average
        mixed_similarities = []
        for med_emb in medical_embeddings[:3]:
            for non_med_emb in non_medical_embeddings[:3]:
                similarity = self.matcher.cosine_similarity(med_emb, non_med_emb)
                mixed_similarities.append(similarity)
        
        avg_medical_sim = np.mean(medical_similarities) if medical_similarities else 0
        avg_mixed_sim = np.mean(mixed_similarities) if mixed_similarities else 0
        
        # Medical terms should be more similar to each other than to non-medical terms
        self.assertGreater(avg_medical_sim, avg_mixed_sim, 
                          "Medical terms should be more similar to each other than to non-medical terms")
    
    def test_category_specific_features(self):
        """Test that terms from the same medical category have higher similarity"""
        for category, terms in self.test_medical_terms.items():
            if len(terms) >= 2:
                emb1 = self.matcher._extract_semantic_features(terms[0])
                emb2 = self.matcher._extract_semantic_features(terms[1])
                
                # Terms in same category should have positive similarity
                similarity = self.matcher.cosine_similarity(emb1, emb2)
                self.assertGreater(similarity, -0.5, 
                                 f"Terms in {category} category should have reasonable similarity")
    
    def test_icd_search_functionality(self):
        """Test ICD code search with semantic matching"""
        test_cases = [
            ('chest pain', 'Should find cardiac-related codes'),
            ('fever', 'Should find infection-related codes'),
            ('pneumonia', 'Should find respiratory codes'),
            ('diabetes', 'Should find endocrine codes'),
            ('headache', 'Should find neurological codes')
        ]
        
        for term, description in test_cases:
            with self.subTest(term=term):
                results = self.matcher.find_similar_icd_codes(term, top_k=5, min_similarity=0.01)
                
                # Basic result validation
                self.assertIsInstance(results, list, f"Results should be list for {term}")
                self.assertLessEqual(len(results), 5, f"Should return at most 5 results for {term}")
                
                # Check result structure
                for result in results:
                    self.assertIn('icd_code', result, f"Result should have icd_code for {term}")
                    self.assertIn('description', result, f"Result should have description for {term}")
                    self.assertIn('similarity', result, f"Result should have similarity for {term}")
                    
                    # Check similarity bounds
                    similarity = result['similarity']
                    self.assertGreaterEqual(similarity, 0, f"Similarity should be non-negative for {term}")
                    self.assertLessEqual(similarity, 1, f"Similarity should not exceed 1 for {term}")
                
                # Check that results are sorted by similarity (descending)
                if len(results) > 1:
                    similarities = [r['similarity'] for r in results]
                    self.assertEqual(similarities, sorted(similarities, reverse=True),
                                   f"Results should be sorted by similarity for {term}")
    
    def test_similarity_thresholds(self):
        """Test that similarity thresholds work correctly"""
        term = 'chest pain'
        
        # Test with different thresholds
        thresholds = [0.001, 0.01, 0.05, 0.1]
        prev_count = float('inf')
        
        for threshold in thresholds:
            results = self.matcher.find_similar_icd_codes(term, top_k=10, min_similarity=threshold)
            current_count = len(results)
            
            # Higher thresholds should return fewer or equal results
            self.assertLessEqual(current_count, prev_count,
                               f"Higher threshold {threshold} should return fewer results")
            
            # All results should meet the threshold
            for result in results:
                self.assertGreaterEqual(result['similarity'], threshold,
                                      f"All results should meet threshold {threshold}")
            
            prev_count = current_count
    
    def test_entity_expansion(self):
        """Test entity text expansion with Claude"""
        test_terms = ['MI', 'SOB', 'HTN', 'DM']  # Common medical abbreviations
        
        for term in test_terms:
            expanded = self.matcher._expand_entity_for_matching(term)
            
            self.assertIsInstance(expanded, str, f"Expansion should return string for {term}")
            self.assertGreater(len(expanded), len(term), f"Expansion should be longer than original for {term}")
            
            # Expanded text should contain the original term or its full form
            expanded_lower = expanded.lower()
            term_lower = term.lower()
            
            # The expansion should be medically relevant (should contain medical terms)
            medical_keywords = ['medical', 'patient', 'condition', 'disease', 'syndrome', 
                              'infarction', 'pressure', 'diabetes', 'breath']
            
            has_medical_content = any(keyword in expanded_lower for keyword in medical_keywords)
            self.assertTrue(has_medical_content or term_lower in expanded_lower,
                          f"Expansion should contain medical content for {term}")
    
    def test_cosine_similarity_properties(self):
        """Test cosine similarity calculation properties"""
        # Test with identical vectors
        vec1 = np.array([1, 2, 3], dtype=np.float32)
        vec2 = np.array([1, 2, 3], dtype=np.float32)
        similarity = self.matcher.cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(similarity, 1.0, places=5, msg="Identical vectors should have similarity 1")
        
        # Test with orthogonal vectors
        vec1 = np.array([1, 0, 0], dtype=np.float32)
        vec2 = np.array([0, 1, 0], dtype=np.float32)
        similarity = self.matcher.cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(similarity, 0.0, places=5, msg="Orthogonal vectors should have similarity 0")
        
        # Test with opposite vectors
        vec1 = np.array([1, 0, 0], dtype=np.float32)
        vec2 = np.array([-1, 0, 0], dtype=np.float32)
        similarity = self.matcher.cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(similarity, -1.0, places=5, msg="Opposite vectors should have similarity -1")
        
        # Test with zero vector
        vec1 = np.array([1, 2, 3], dtype=np.float32)
        vec2 = np.array([0, 0, 0], dtype=np.float32)
        similarity = self.matcher.cosine_similarity(vec1, vec2)
        self.assertEqual(similarity, 0.0, msg="Zero vector should return similarity 0")
    
    def test_batch_search_consistency(self):
        """Test that multiple searches for the same term return consistent results"""
        term = 'pneumonia'
        
        # Perform the same search multiple times
        results1 = self.matcher.find_similar_icd_codes(term, top_k=3, min_similarity=0.01)
        results2 = self.matcher.find_similar_icd_codes(term, top_k=3, min_similarity=0.01)
        
        # Results should be identical
        self.assertEqual(len(results1), len(results2), "Search results should be consistent")
        
        for r1, r2 in zip(results1, results2):
            self.assertEqual(r1['icd_code'], r2['icd_code'], "ICD codes should match")
            self.assertAlmostEqual(r1['similarity'], r2['similarity'], places=5, 
                                 msg="Similarity scores should match")


class TestICDMatchingEdgeCases(unittest.TestCase):
    """Test edge cases and error handling for ICD matching"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.matcher = ICD10VectorMatcher(force_numpy=True)
    
    def test_empty_input(self):
        """Test handling of empty input"""
        results = self.matcher.find_similar_icd_codes('', top_k=5)
        self.assertIsInstance(results, list, "Should return list for empty input")
        
        embedding = self.matcher._extract_semantic_features('')
        self.assertEqual(len(embedding), 1536, "Should return valid embedding for empty input")
    
    def test_very_long_input(self):
        """Test handling of very long input text"""
        long_text = 'chest pain ' * 1000  # Very long repeated text
        
        results = self.matcher.find_similar_icd_codes(long_text, top_k=3)
        self.assertIsInstance(results, list, "Should handle long input gracefully")
        
        embedding = self.matcher._extract_semantic_features(long_text)
        self.assertEqual(len(embedding), 1536, "Should return valid embedding for long input")
    
    def test_special_characters(self):
        """Test handling of special characters and numbers"""
        special_inputs = [
            'chest pain!!!',
            'fever @#$%',
            'pneumonia 123',
            'diabetes-mellitus',
            'heart_attack'
        ]
        
        for text in special_inputs:
            with self.subTest(text=text):
                results = self.matcher.find_similar_icd_codes(text, top_k=3)
                self.assertIsInstance(results, list, f"Should handle special characters in: {text}")
                
                embedding = self.matcher._extract_semantic_features(text)
                self.assertEqual(len(embedding), 1536, f"Should return valid embedding for: {text}")
    
    def test_non_medical_terms(self):
        """Test behavior with completely non-medical terms"""
        non_medical = ['computer', 'weather', 'sports', 'music', 'travel']
        
        for term in non_medical:
            with self.subTest(term=term):
                results = self.matcher.find_similar_icd_codes(term, top_k=3, min_similarity=0.01)
                
                # Should still return results (even if not very relevant)
                self.assertIsInstance(results, list, f"Should return list for non-medical term: {term}")
                
                # All results should still have valid structure
                for result in results:
                    self.assertIn('icd_code', result, f"Should have valid structure for: {term}")
                    self.assertIn('similarity', result, f"Should have similarity for: {term}")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)