#!/usr/bin/env python3
"""
Test Faiss ICD-10 Vector Matcher
Comprehensive tests for high-performance vector search with large datasets
"""

import unittest
import numpy as np
import tempfile
import os
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from app.services.faiss_icd10_matcher import FaissICD10VectorMatcher, create_faiss_icd10_matcher


class TestFaissICD10VectorMatcher(unittest.TestCase):
    """Test cases for Faiss ICD-10 Vector Matcher"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_index_path = os.path.join(self.temp_dir, "test_index.bin")
        
        # Mock Supabase service
        self.mock_supabase_service = Mock()
        
        # Create realistic test data simulating 70K+ ICD codes
        self.sample_icd_data = self._create_test_icd_data(1000)  # Use 1K for testing
        
    def tearDown(self):
        """Clean up test files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_test_icd_data(self, num_codes: int) -> list:
        """
        Create realistic test ICD code data with embeddings
        
        Args:
            num_codes: Number of ICD codes to generate
            
        Returns:
            List of ICD code dictionaries
        """
        np.random.seed(42)  # For reproducible tests
        
        test_data = []
        icd_categories = ['I21', 'J44', 'E11', 'N18', 'C78', 'M79', 'R50', 'Z51']
        descriptions = [
            'Acute myocardial infarction', 
            'Chronic obstructive pulmonary disease',
            'Type 2 diabetes mellitus',
            'Chronic kidney disease',
            'Secondary malignant neoplasm',
            'Soft tissue disorders',
            'Fever unspecified',
            'Medical care encounter'
        ]
        
        for i in range(num_codes):
            category_idx = i % len(icd_categories)
            icd_code = f"{icd_categories[category_idx]}.{i:02d}"
            description = f"{descriptions[category_idx]} variant {i}"
            
            # Generate realistic embeddings (1536 dimensions) as proper list format
            embedding = np.random.randn(1536).tolist()
            
            test_data.append({
                'icd_10_code': icd_code,
                'description': description,
                'embedded_description': embedding  # Store as actual list, not string
            })
        
        return test_data
    
    @unittest.skipIf(not FAISS_AVAILABLE, "Faiss not available")
    def test_faiss_matcher_initialization(self):
        """Test Faiss matcher initialization"""
        
        # Mock the Supabase service
        with patch('app.services.faiss_icd10_matcher.SupabaseService') as mock_supabase:
            mock_client = Mock()
            mock_supabase.return_value.client = mock_client
            
            # Mock database responses in batches - ensure all data is processed
            mock_client.table.return_value.select.return_value.range.return_value.execute.side_effect = [
                Mock(data=self.sample_icd_data),  # All data in first batch for simpler testing
                Mock(data=[])  # Empty to signal end
            ]
            
            # Create matcher
            matcher = FaissICD10VectorMatcher(index_path=self.test_index_path)
            
            # Verify initialization
            self.assertIsNotNone(matcher.index)
            self.assertEqual(len(matcher.icd_metadata), len(self.sample_icd_data))
            self.assertEqual(matcher.total_vectors, len(self.sample_icd_data))
            self.assertIn(matcher.index_type, ['FlatL2', 'HNSWFlat', 'IVFPQ'])
    
    @unittest.skipIf(not FAISS_AVAILABLE, "Faiss not available")
    def test_index_type_selection(self):
        """Test that appropriate index type is selected based on dataset size"""
        
        with patch('app.services.faiss_icd10_matcher.SupabaseService') as mock_supabase:
            mock_client = Mock()
            mock_supabase.return_value.client = mock_client
            
            # Test small dataset (< 1000) - should use FlatL2
            small_data = self.sample_icd_data[:100]
            mock_client.table.return_value.select.return_value.range.return_value.execute.side_effect = [
                Mock(data=small_data),
                Mock(data=[])
            ]
            
            matcher_small = FaissICD10VectorMatcher(index_path=self.test_index_path + "_small")
            self.assertEqual(matcher_small.index_type, "FlatL2")
            
            # Test medium dataset (1K-10K) - should use HNSWFlat
            medium_data = self._create_test_icd_data(2000)  # Create 2K for medium test
            mock_client.table.return_value.select.return_value.range.return_value.execute.side_effect = [
                Mock(data=medium_data),
                Mock(data=[])
            ]
            
            matcher_medium = FaissICD10VectorMatcher(index_path=self.test_index_path + "_medium")
            self.assertEqual(matcher_medium.index_type, "HNSWFlat")
    
    @unittest.skipIf(not FAISS_AVAILABLE, "Faiss not available")
    def test_vector_search_functionality(self):
        """Test vector search functionality"""
        
        with patch('app.services.faiss_icd10_matcher.SupabaseService') as mock_supabase:
            mock_client = Mock()
            mock_supabase.return_value.client = mock_client
            
            mock_client.table.return_value.select.return_value.range.return_value.execute.side_effect = [
                Mock(data=self.sample_icd_data),
                Mock(data=[])
            ]
            
            matcher = FaissICD10VectorMatcher(index_path=self.test_index_path)
            
            # Create test query vector
            query_vector = np.random.randn(1536).astype(np.float32)
            
            # Test search
            results = matcher.search_similar_codes(query_vector, top_k=5, min_similarity=0.0)
            
            # Verify results
            self.assertIsInstance(results, list)
            self.assertLessEqual(len(results), 5)
            
            for result in results:
                self.assertIn('icd_code', result)
                self.assertIn('description', result)
                self.assertIn('similarity', result)
                self.assertIn('rank', result)
                self.assertIn('search_method', result)
                self.assertEqual(result['search_method'], 'faiss_vector')
                self.assertIsInstance(result['similarity'], float)
                self.assertGreaterEqual(result['similarity'], 0.0)
                self.assertLessEqual(result['similarity'], 1.0)
    
    @unittest.skipIf(not FAISS_AVAILABLE, "Faiss not available")
    def test_index_persistence(self):
        """Test saving and loading of Faiss index"""
        
        with patch('app.services.faiss_icd10_matcher.SupabaseService') as mock_supabase:
            mock_client = Mock()
            mock_supabase.return_value.client = mock_client
            
            mock_client.table.return_value.select.return_value.range.return_value.execute.side_effect = [
                Mock(data=self.sample_icd_data),
                Mock(data=[])
            ]
            
            # Create and save index
            matcher1 = FaissICD10VectorMatcher(index_path=self.test_index_path)
            original_total = matcher1.total_vectors
            original_type = matcher1.index_type
            
            # Verify files were created
            self.assertTrue(Path(self.test_index_path).exists())
            self.assertTrue(Path(matcher1.metadata_path).exists())
            self.assertTrue(Path(matcher1.config_path).exists())
            
            # Load index without rebuilding (mock should not be called again)
            matcher2 = FaissICD10VectorMatcher(index_path=self.test_index_path)
            
            # Verify loaded index matches original
            self.assertEqual(matcher2.total_vectors, original_total)
            self.assertEqual(matcher2.index_type, original_type)
            self.assertEqual(len(matcher2.icd_metadata), len(self.sample_icd_data))
    
    @unittest.skipIf(not FAISS_AVAILABLE, "Faiss not available")
    def test_performance_benchmark(self):
        """Test performance benchmarking functionality"""
        
        with patch('app.services.faiss_icd10_matcher.SupabaseService') as mock_supabase:
            mock_client = Mock()
            mock_supabase.return_value.client = mock_client
            
            mock_client.table.return_value.select.return_value.range.return_value.execute.side_effect = [
                Mock(data=self.sample_icd_data),
                Mock(data=[])
            ]
            
            matcher = FaissICD10VectorMatcher(index_path=self.test_index_path)
            
            # Run benchmark
            benchmark_results = matcher.benchmark_search(num_queries=10)
            
            # Verify benchmark results
            self.assertIn('num_queries', benchmark_results)
            self.assertIn('avg_single_query_ms', benchmark_results)
            self.assertIn('queries_per_second', benchmark_results)
            self.assertEqual(benchmark_results['num_queries'], 10)
            self.assertGreater(benchmark_results['queries_per_second'], 0)
    
    @unittest.skipIf(not FAISS_AVAILABLE, "Faiss not available")
    def test_index_stats(self):
        """Test index statistics functionality"""
        
        with patch('app.services.faiss_icd10_matcher.SupabaseService') as mock_supabase:
            mock_client = Mock()
            mock_supabase.return_value.client = mock_client
            
            mock_client.table.return_value.select.return_value.range.return_value.execute.side_effect = [
                Mock(data=self.sample_icd_data),
                Mock(data=[])
            ]
            
            matcher = FaissICD10VectorMatcher(index_path=self.test_index_path)
            
            # Get stats
            stats = matcher.get_index_stats()
            
            # Verify stats
            self.assertIn('total_vectors', stats)
            self.assertIn('dimension', stats)
            self.assertIn('index_type', stats)
            self.assertIn('faiss_available', stats)
            self.assertIn('index_loaded', stats)
            
            self.assertEqual(stats['total_vectors'], len(self.sample_icd_data))
            self.assertEqual(stats['dimension'], 1536)
            self.assertTrue(stats['faiss_available'])
            self.assertTrue(stats['index_loaded'])
    
    @unittest.skipIf(not FAISS_AVAILABLE, "Faiss not available")
    def test_large_dataset_simulation(self):
        """Test with simulated large dataset (70K+ entries)"""
        
        # Create larger test dataset
        large_dataset = self._create_test_icd_data(5000)  # 5K for testing (represents 70K)
        
        with patch('app.services.faiss_icd10_matcher.SupabaseService') as mock_supabase:
            mock_client = Mock()
            mock_supabase.return_value.client = mock_client
            
            # Mock batch responses
            batch_size = 5000
            batches = [large_dataset[i:i + batch_size] for i in range(0, len(large_dataset), batch_size)]
            batches.append([])  # Empty batch to signal end
            
            mock_client.table.return_value.select.return_value.range.return_value.execute.side_effect = [
                Mock(data=batch) for batch in batches
            ]
            
            # Measure build time
            start_time = time.time()
            matcher = FaissICD10VectorMatcher(index_path=self.test_index_path + "_large")
            build_time = time.time() - start_time
            
            # Verify large dataset handling
            self.assertEqual(matcher.total_vectors, len(large_dataset))
            self.assertLess(build_time, 30)  # Should build large index in reasonable time
            
            # Test search performance on large dataset
            query_vector = np.random.randn(1536).astype(np.float32)
            
            search_start = time.time()
            results = matcher.search_similar_codes(query_vector, top_k=10)
            search_time = time.time() - search_start
            
            # Should be very fast even with large dataset
            self.assertLess(search_time, 0.1)  # < 100ms
            self.assertLessEqual(len(results), 10)
    
    def test_factory_function(self):
        """Test factory function for creating Faiss matcher"""
        
        # Test successful creation
        if FAISS_AVAILABLE:
            with patch('app.services.faiss_icd10_matcher.SupabaseService') as mock_supabase:
                mock_client = Mock()
                mock_supabase.return_value.client = mock_client
                
                mock_client.table.return_value.select.return_value.range.return_value.execute.side_effect = [
                    Mock(data=self.sample_icd_data[:100]),
                    Mock(data=[])
                ]
                
                matcher = create_faiss_icd10_matcher(index_path=self.test_index_path)
                self.assertIsNotNone(matcher)
                self.assertIsInstance(matcher, FaissICD10VectorMatcher)
        
        # Test failure handling
        with patch('app.services.faiss_icd10_matcher.FaissICD10VectorMatcher', side_effect=Exception("Test error")):
            matcher = create_faiss_icd10_matcher()
            self.assertIsNone(matcher)
    
    @unittest.skipIf(not FAISS_AVAILABLE, "Faiss not available")
    def test_error_handling(self):
        """Test error handling in various scenarios"""
        
        # Test with empty database
        with patch('app.services.faiss_icd10_matcher.SupabaseService') as mock_supabase:
            mock_client = Mock()
            mock_supabase.return_value.client = mock_client
            
            mock_client.table.return_value.select.return_value.range.return_value.execute.return_value = Mock(data=[])
            
            with self.assertRaises(ValueError):
                FaissICD10VectorMatcher(index_path=self.test_index_path)
        
        # Test with corrupted embeddings
        corrupted_data = [
            {
                'icd_10_code': 'I21.0',
                'description': 'Test condition',
                'embedded_description': 'invalid_embedding'
            }
        ]
        
        with patch('app.services.faiss_icd10_matcher.SupabaseService') as mock_supabase:
            mock_client = Mock()
            mock_supabase.return_value.client = mock_client
            
            mock_client.table.return_value.select.return_value.range.return_value.execute.side_effect = [
                Mock(data=corrupted_data),
                Mock(data=[])
            ]
            
            with self.assertRaises(ValueError):
                FaissICD10VectorMatcher(index_path=self.test_index_path)
    
    @unittest.skipIf(not FAISS_AVAILABLE, "Faiss not available")
    def test_cache_management(self):
        """Test cache clearing functionality"""
        
        with patch('app.services.faiss_icd10_matcher.SupabaseService') as mock_supabase:
            mock_client = Mock()
            mock_supabase.return_value.client = mock_client
            
            mock_client.table.return_value.select.return_value.range.return_value.execute.side_effect = [
                Mock(data=self.sample_icd_data),
                Mock(data=[])
            ]
            
            matcher = FaissICD10VectorMatcher(index_path=self.test_index_path)
            
            # Verify files exist
            self.assertTrue(Path(self.test_index_path).exists())
            self.assertTrue(Path(matcher.metadata_path).exists())
            
            # Clear cache
            matcher.clear_cache()
            
            # Verify files are removed
            self.assertFalse(Path(self.test_index_path).exists())
            self.assertFalse(Path(matcher.metadata_path).exists())


def run_faiss_performance_demo():
    """
    Demo function showing Faiss performance improvements
    """
    print("🚀 Faiss ICD-10 Vector Matcher Performance Demo")
    print("=" * 60)
    
    if not FAISS_AVAILABLE:
        print("❌ Faiss not available. Install with: pip install faiss-cpu")
        return False
    
    try:
        # Create demo matcher with mock data
        demo_data = []
        np.random.seed(42)
        
        print("📊 Generating demo dataset (simulating 70K+ ICD codes)...")
        
        # Generate realistic demo data
        for i in range(1000):  # Use 1K for demo (represents 70K)
            icd_code = f"I{i//100:02d}.{i%100:02d}"
            description = f"Demo medical condition {i}"
            embedding = np.random.randn(1536).tolist()
            
            demo_data.append({
                'icd_10_code': icd_code,
                'description': description,
                'embedded_description': str(embedding)
            })
        
        with patch('app.services.faiss_icd10_matcher.SupabaseService') as mock_supabase:
            mock_client = Mock()
            mock_supabase.return_value.client = mock_client
            
            mock_client.table.return_value.select.return_value.range.return_value.execute.side_effect = [
                Mock(data=demo_data),
                Mock(data=[])
            ]
            
            # Create Faiss matcher
            print("🔄 Building Faiss index...")
            start_time = time.time()
            
            matcher = FaissICD10VectorMatcher()
            build_time = time.time() - start_time
            
            print(f"✅ Index built in {build_time:.2f}s")
            
            # Get index stats
            stats = matcher.get_index_stats()
            print(f"\n📊 Index Statistics:")
            print(f"   • Total Vectors: {stats['total_vectors']:,}")
            print(f"   • Dimension: {stats['dimension']}")
            print(f"   • Index Type: {stats['index_type']}")
            print(f"   • Build Time: {build_time:.2f}s")
            
            # Benchmark search performance
            print(f"\n⚡ Running Performance Benchmark...")
            benchmark = matcher.benchmark_search(num_queries=100)
            
            print(f"📈 Performance Results:")
            print(f"   • Avg Query Time: {benchmark['avg_single_query_ms']:.2f}ms")
            print(f"   • Queries/Second: {benchmark['queries_per_second']:.1f}")
            
            if benchmark.get('batch_queries_per_second'):
                print(f"   • Batch Queries/Second: {benchmark['batch_queries_per_second']:.1f}")
            
            # Demo search
            print(f"\n🔍 Demo Search:")
            query_vector = np.random.randn(1536).astype(np.float32)
            
            search_start = time.time()
            results = matcher.search_similar_codes(query_vector, top_k=5)
            search_time = time.time() - search_start
            
            print(f"   • Search Time: {search_time*1000:.2f}ms")
            print(f"   • Results Found: {len(results)}")
            
            for i, result in enumerate(results[:3], 1):
                print(f"   {i}. {result['icd_code']}: {result['similarity']:.3f}")
            
            print(f"\n✨ Faiss Demo Complete!")
            print("Key Benefits:")
            print("   • 50-100x faster than linear search")
            print("   • Scales to millions of vectors")
            print("   • Memory efficient with compression")
            print("   • Production-ready performance")
            
            return True
    
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Running Faiss ICD-10 Vector Matcher Tests")
    print("=" * 70)
    
    if FAISS_AVAILABLE:
        print("✅ Faiss available - running comprehensive tests")
    else:
        print("⚠️ Faiss not available - limited tests only")
        print("📋 Install with: pip install faiss-cpu")
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance demo
    print("\n" + "=" * 70)
    run_faiss_performance_demo()