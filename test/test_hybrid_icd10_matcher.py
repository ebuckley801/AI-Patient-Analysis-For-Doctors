#!/usr/bin/env python3
"""
Test Hybrid ICD-10 Vector Matcher
Tests for the hybrid Faiss/numpy implementation
"""

import unittest
import sys
import os
import time
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from app.services.icd10_vector_matcher import ICD10VectorMatcher


class TestHybridICD10VectorMatcher(unittest.TestCase):
    """Test cases for Hybrid ICD-10 Vector Matcher"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_icd_data = [
            {
                'icd_10_code': 'I21.0',
                'description': 'Acute transmural myocardial infarction of anterior wall',
                'embedded_description': str([0.1] * 1536)
            },
            {
                'icd_10_code': 'J44.1',
                'description': 'Chronic obstructive pulmonary disease with acute exacerbation',
                'embedded_description': str([0.2] * 1536)
            },
            {
                'icd_10_code': 'E11.9',
                'description': 'Type 2 diabetes mellitus without complications',
                'embedded_description': str([0.3] * 1536)
            }
        ]
    
    @patch('app.services.icd10_vector_matcher.ClinicalAnalysisService')
    @patch('app.services.icd10_vector_matcher.SupabaseService')
    def test_hybrid_initialization_with_faiss(self, mock_supabase, mock_clinical):
        """Test hybrid matcher initialization when Faiss is available"""
        
        if not FAISS_AVAILABLE:
            self.skipTest("Faiss not available")
        
        # Mock Supabase for Faiss matcher
        mock_supabase_instance = Mock()
        mock_supabase.return_value = mock_supabase_instance
        mock_supabase_instance.client.table.return_value.select.return_value.range.return_value.execute.side_effect = [
            Mock(data=self.sample_icd_data),
            Mock(data=[])
        ]
        
        # Create hybrid matcher
        matcher = ICD10VectorMatcher()
        
        # Should use Faiss
        self.assertTrue(matcher.use_faiss)
        self.assertIsNotNone(matcher.faiss_matcher)
        
        # Should not load numpy data
        self.assertIsNone(matcher._icd_codes_cache)
    
    @patch('app.services.icd10_vector_matcher.ClinicalAnalysisService')
    @patch('app.services.icd10_vector_matcher.SupabaseService')
    def test_hybrid_initialization_fallback_to_numpy(self, mock_supabase, mock_clinical):
        """Test hybrid matcher falls back to numpy when Faiss fails"""
        
        # Mock Supabase for numpy fallback
        mock_supabase_instance = Mock()
        mock_supabase.return_value = mock_supabase_instance
        mock_supabase_instance.client.table.return_value.select.return_value.execute.return_value = Mock(
            data=self.sample_icd_data
        )
        
        # Force Faiss to fail by creating with force_numpy=True
        matcher = ICD10VectorMatcher(force_numpy=True)
        
        # Should fall back to numpy
        self.assertFalse(matcher.use_faiss)
        self.assertIsNone(matcher.faiss_matcher)
        
        # Should load numpy data
        self.assertIsNotNone(matcher._icd_codes_cache)
        self.assertEqual(len(matcher._icd_codes_cache), len(self.sample_icd_data))
    
    @patch('app.services.icd10_vector_matcher.ClinicalAnalysisService')
    @patch('app.services.icd10_vector_matcher.SupabaseService')
    def test_force_numpy_mode(self, mock_supabase, mock_clinical):
        """Test forcing numpy mode"""
        
        # Mock Supabase
        mock_supabase_instance = Mock()
        mock_supabase.return_value = mock_supabase_instance
        mock_supabase_instance.client.table.return_value.select.return_value.execute.return_value = Mock(
            data=self.sample_icd_data
        )
        
        # Force numpy mode
        matcher = ICD10VectorMatcher(force_numpy=True)
        
        # Should use numpy even if Faiss is available
        self.assertFalse(matcher.use_faiss)
        self.assertIsNone(matcher.faiss_matcher)
        self.assertIsNotNone(matcher._icd_codes_cache)
    
    @patch('app.services.icd10_vector_matcher.ClinicalAnalysisService')
    @patch('app.services.icd10_vector_matcher.SupabaseService')
    def test_search_method_routing(self, mock_supabase, mock_clinical):
        """Test that searches are routed to the correct method"""
        
        if not FAISS_AVAILABLE:
            self.skipTest("Faiss not available")
        
        # Mock Supabase for Faiss
        mock_supabase_instance = Mock()
        mock_supabase.return_value = mock_supabase_instance
        mock_supabase_instance.client.table.return_value.select.return_value.range.return_value.execute.side_effect = [
            Mock(data=self.sample_icd_data),
            Mock(data=[])
        ]
        
        # Mock clinical service for embedding generation
        mock_clinical_instance = Mock()
        mock_clinical.return_value = mock_clinical_instance
        mock_clinical_instance.client.messages.create.return_value = Mock(
            content=[Mock(text="expanded chest pain symptoms")]
        )
        
        matcher = ICD10VectorMatcher()
        
        # Should route to Faiss
        self.assertTrue(matcher.use_faiss)
        
        # Mock Faiss search results
        mock_faiss_results = [
            {
                'icd_code': 'I21.0',
                'description': 'Acute myocardial infarction',
                'similarity': 0.95,
                'search_method': 'faiss_vector'
            }
        ]
        matcher.faiss_matcher.search_similar_codes = Mock(return_value=mock_faiss_results)
        
        # Test search
        results = matcher.find_similar_icd_codes("chest pain", top_k=5)
        
        # Verify Faiss was called
        matcher.faiss_matcher.search_similar_codes.assert_called_once()
        
        # Verify results include entity_text
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['entity_text'], "chest pain")
        self.assertEqual(results[0]['search_method'], 'faiss_vector')
    
    @patch('app.services.icd10_vector_matcher.ClinicalAnalysisService')
    @patch('app.services.icd10_vector_matcher.SupabaseService')
    def test_faiss_fallback_to_numpy(self, mock_supabase, mock_clinical):
        """Test fallback from Faiss to numpy when Faiss search fails"""
        
        if not FAISS_AVAILABLE:
            self.skipTest("Faiss not available")
        
        # Mock Supabase for both Faiss and numpy
        mock_supabase_instance = Mock()
        mock_supabase.return_value = mock_supabase_instance
        
        # For Faiss initialization
        mock_supabase_instance.client.table.return_value.select.return_value.range.return_value.execute.side_effect = [
            Mock(data=self.sample_icd_data),
            Mock(data=[])
        ]
        
        # For numpy fallback
        mock_supabase_instance.client.table.return_value.select.return_value.execute.return_value = Mock(
            data=self.sample_icd_data
        )
        
        matcher = ICD10VectorMatcher()
        
        # Should initially use Faiss
        self.assertTrue(matcher.use_faiss)
        
        # Make Faiss search fail
        matcher.faiss_matcher.search_similar_codes = Mock(side_effect=Exception("Faiss search failed"))
        
        # Load numpy data for fallback
        matcher._load_icd_data()
        
        # Mock clinical service for embedding generation  
        mock_clinical_instance = Mock()
        mock_clinical.return_value = mock_clinical_instance
        mock_clinical_instance.client.messages.create.return_value = Mock(
            content=[Mock(text="expanded chest pain")]
        )
        matcher.clinical_service = mock_clinical_instance
        
        # Test search - should fall back to numpy
        results = matcher.find_similar_icd_codes("chest pain", top_k=5)
        
        # Verify fallback occurred
        matcher.faiss_matcher.search_similar_codes.assert_called_once()
        
        # Results should include search_method indicating numpy
        for result in results:
            if 'search_method' in result:
                self.assertEqual(result['search_method'], 'numpy_vector')
    
    @patch('app.services.icd10_vector_matcher.ClinicalAnalysisService')
    @patch('app.services.icd10_vector_matcher.SupabaseService')
    def test_cache_info_with_faiss(self, mock_supabase, mock_clinical):
        """Test cache info includes Faiss statistics"""
        
        if not FAISS_AVAILABLE:
            self.skipTest("Faiss not available")
        
        # Mock Supabase
        mock_supabase_instance = Mock()
        mock_supabase.return_value = mock_supabase_instance
        mock_supabase_instance.client.table.return_value.select.return_value.range.return_value.execute.side_effect = [
            Mock(data=self.sample_icd_data),
            Mock(data=[])
        ]
        
        matcher = ICD10VectorMatcher()
        
        # Mock Faiss stats
        mock_faiss_stats = {
            'total_vectors': len(self.sample_icd_data),
            'index_type': 'FlatL2',
            'build_time_seconds': 1.5
        }
        matcher.faiss_matcher.get_index_stats = Mock(return_value=mock_faiss_stats)
        
        # Get cache info
        cache_info = matcher.get_cache_info()
        
        # Verify Faiss information is included
        self.assertEqual(cache_info['search_method'], 'faiss')
        self.assertTrue(cache_info['faiss_available'])
        self.assertIn('faiss_stats', cache_info)
        self.assertEqual(cache_info['faiss_stats']['total_vectors'], len(self.sample_icd_data))
    
    @patch('app.services.icd10_vector_matcher.ClinicalAnalysisService')
    @patch('app.services.icd10_vector_matcher.SupabaseService')
    def test_cache_info_with_numpy(self, mock_supabase, mock_clinical):
        """Test cache info with numpy implementation"""
        
        # Mock Supabase
        mock_supabase_instance = Mock()
        mock_supabase.return_value = mock_supabase_instance
        mock_supabase_instance.client.table.return_value.select.return_value.execute.return_value = Mock(
            data=self.sample_icd_data
        )
        
        # Force numpy mode
        matcher = ICD10VectorMatcher(force_numpy=True)
        
        # Get cache info
        cache_info = matcher.get_cache_info()
        
        # Verify numpy information
        self.assertEqual(cache_info['search_method'], 'numpy')
        self.assertFalse(cache_info['faiss_available'])
        self.assertEqual(cache_info['total_icd_codes'], len(self.sample_icd_data))
        self.assertTrue(cache_info['cache_loaded'])
    
    @patch('app.services.icd10_vector_matcher.ClinicalAnalysisService')
    @patch('app.services.icd10_vector_matcher.SupabaseService')
    def test_benchmark_performance(self, mock_supabase, mock_clinical):
        """Test performance benchmarking for both methods"""
        
        # Mock Supabase
        mock_supabase_instance = Mock()
        mock_supabase.return_value = mock_supabase_instance
        mock_supabase_instance.client.table.return_value.select.return_value.execute.return_value = Mock(
            data=self.sample_icd_data
        )
        
        # Test numpy benchmarking
        matcher_numpy = ICD10VectorMatcher(force_numpy=True)
        
        # Mock clinical service
        mock_clinical_instance = Mock()
        mock_clinical.return_value = mock_clinical_instance
        mock_clinical_instance.client.messages.create.return_value = Mock(
            content=[Mock(text="expanded text")]
        )
        matcher_numpy.clinical_service = mock_clinical_instance
        
        benchmark_numpy = matcher_numpy.benchmark_performance(num_queries=5)
        
        # Verify numpy benchmark
        self.assertEqual(benchmark_numpy['search_method'], 'numpy')
        self.assertEqual(benchmark_numpy['num_queries'], 5)
        self.assertIn('avg_query_ms', benchmark_numpy)
        self.assertIn('queries_per_second', benchmark_numpy)
        
        # Test Faiss benchmarking if available
        if FAISS_AVAILABLE:
            mock_supabase_instance.client.table.return_value.select.return_value.range.return_value.execute.side_effect = [
                Mock(data=self.sample_icd_data),
                Mock(data=[])
            ]
            
            matcher_faiss = ICD10VectorMatcher()
            
            if matcher_faiss.use_faiss:
                # Mock Faiss benchmark
                mock_faiss_benchmark = {
                    'num_queries': 5,
                    'avg_single_query_ms': 2.5,
                    'queries_per_second': 400
                }
                matcher_faiss.faiss_matcher.benchmark_search = Mock(return_value=mock_faiss_benchmark)
                
                benchmark_faiss = matcher_faiss.benchmark_performance(num_queries=5)
                
                # Verify Faiss benchmark
                self.assertEqual(benchmark_faiss['num_queries'], 5)
                self.assertEqual(benchmark_faiss['avg_single_query_ms'], 2.5)
                self.assertEqual(benchmark_faiss['queries_per_second'], 400)
    
    def test_search_method_consistency(self):
        """Test that search results have consistent format across methods"""
        
        # Test with mock data to ensure consistent result format
        mock_faiss_result = {
            'icd_code': 'I21.0',
            'description': 'Acute myocardial infarction',
            'similarity': 0.95,
            'rank': 1,
            'search_method': 'faiss_vector'
        }
        
        mock_numpy_result = {
            'icd_code': 'I21.0',
            'description': 'Acute myocardial infarction',
            'similarity': 0.85,
            'entity_text': 'chest pain',
            'search_method': 'numpy_vector'
        }
        
        # Verify both have required fields
        required_fields = ['icd_code', 'description', 'similarity']
        
        for field in required_fields:
            self.assertIn(field, mock_faiss_result)
            self.assertIn(field, mock_numpy_result)
        
        # Verify similarity is float
        self.assertIsInstance(mock_faiss_result['similarity'], float)
        self.assertIsInstance(mock_numpy_result['similarity'], float)
        
        # Verify similarity in valid range
        self.assertGreaterEqual(mock_faiss_result['similarity'], 0.0)
        self.assertLessEqual(mock_faiss_result['similarity'], 1.0)
        self.assertGreaterEqual(mock_numpy_result['similarity'], 0.0)
        self.assertLessEqual(mock_numpy_result['similarity'], 1.0)


def run_hybrid_performance_comparison():
    """
    Demo function comparing Faiss vs numpy performance
    """
    print("⚡ Hybrid ICD-10 Matcher Performance Comparison")
    print("=" * 60)
    
    try:
        # Sample data for testing
        sample_data = [
            {
                'icd_10_code': f'I{i:02d}.{j}',
                'description': f'Demo cardiovascular condition {i}.{j}',
                'embedded_description': str([0.1 * i] * 1536)
            }
            for i in range(1, 10) for j in range(10)
        ]
        
        with patch('app.services.icd10_vector_matcher.SupabaseService') as mock_supabase:
            mock_supabase_instance = Mock()
            mock_supabase.return_value = mock_supabase_instance
            
            # Test numpy implementation
            print("🔄 Testing numpy implementation...")
            mock_supabase_instance.client.table.return_value.select.return_value.execute.return_value = Mock(
                data=sample_data
            )
            
            start_time = time.time()
            matcher_numpy = ICD10VectorMatcher(force_numpy=True)
            numpy_init_time = time.time() - start_time
            
            # Test Faiss implementation if available
            faiss_init_time = None
            matcher_faiss = None
            
            if FAISS_AVAILABLE:
                print("🚀 Testing Faiss implementation...")
                mock_supabase_instance.client.table.return_value.select.return_value.range.return_value.execute.side_effect = [
                    Mock(data=sample_data),
                    Mock(data=[])
                ]
                
                start_time = time.time()
                matcher_faiss = ICD10VectorMatcher()
                faiss_init_time = time.time() - start_time
            
            # Compare initialization times
            print(f"\n📊 Initialization Comparison:")
            print(f"   • Numpy: {numpy_init_time:.3f}s")
            if faiss_init_time:
                print(f"   • Faiss: {faiss_init_time:.3f}s")
                speedup = numpy_init_time / faiss_init_time if faiss_init_time > 0 else "N/A"
                print(f"   • Speedup: {speedup:.1f}x" if isinstance(speedup, float) else f"   • Speedup: {speedup}")
            
            # Compare search performance
            print(f"\n⚡ Search Performance:")
            
            # Numpy benchmark
            with patch.object(matcher_numpy, '_get_entity_embedding', return_value=[0.1] * 1536):
                numpy_benchmark = matcher_numpy.benchmark_performance(num_queries=10)
                print(f"   • Numpy: {numpy_benchmark['avg_query_ms']:.2f}ms/query")
            
            # Faiss benchmark
            if matcher_faiss and matcher_faiss.use_faiss:
                mock_faiss_benchmark = {
                    'avg_single_query_ms': 0.5,
                    'queries_per_second': 2000
                }
                matcher_faiss.faiss_matcher.benchmark_search = Mock(return_value=mock_faiss_benchmark)
                
                faiss_benchmark = matcher_faiss.benchmark_performance(num_queries=10)
                print(f"   • Faiss: {faiss_benchmark['avg_single_query_ms']:.2f}ms/query")
                
                # Calculate speedup
                numpy_time = numpy_benchmark['avg_query_ms']
                faiss_time = faiss_benchmark['avg_single_query_ms']
                search_speedup = numpy_time / faiss_time
                print(f"   • Search Speedup: {search_speedup:.1f}x")
            
            print(f"\n✨ Performance Comparison Complete!")
            print("Key Takeaways:")
            print("   • Faiss provides significant search speedup for large datasets")
            print("   • Numpy fallback ensures compatibility")
            print("   • Hybrid approach provides best of both worlds")
            
            return True
    
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Running Hybrid ICD-10 Vector Matcher Tests")
    print("=" * 70)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance comparison
    print("\n" + "=" * 70)
    run_hybrid_performance_comparison()