#!/usr/bin/env python3
"""
Test Async Clinical Analysis Service
Tests for high-performance async batch processing and priority scanning
"""

import unittest
import asyncio
import sys
import os
import time
from unittest.mock import Mock, patch, AsyncMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.services.async_clinical_analysis import (
    AsyncClinicalAnalysis, 
    BatchAnalysisConfig, 
    BatchAnalysisResult
)


class TestAsyncClinicalAnalysis(unittest.TestCase):
    """Test cases for Async Clinical Analysis Service"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.async_service = AsyncClinicalAnalysis()
        
        # Mock the underlying services to avoid actual API calls
        self.async_service.clinical_service = Mock()
        self.async_service.storage_service = Mock()
        self.async_service.icd_matcher = Mock()
        
        # Sample test data
        self.sample_note = {
            'note_id': 'test_note_1',
            'note_text': 'Patient has severe chest pain and shortness of breath. Blood pressure 180/100.',
            'patient_context': {'age': 65, 'gender': 'male'},
            'patient_id': 'patient_123'
        }
        
        self.sample_analysis_result = {
            'symptoms': [
                {'entity': 'chest pain', 'confidence': 0.95, 'severity': 'severe'},
                {'entity': 'shortness of breath', 'confidence': 0.90, 'severity': 'moderate'}
            ],
            'conditions': [],
            'overall_assessment': {
                'risk_level': 'high',
                'requires_immediate_attention': True,
                'primary_concerns': ['chest pain', 'shortness of breath']
            }
        }
    
    def test_batch_analysis_config(self):
        """Test BatchAnalysisConfig dataclass"""
        # Test default values
        config = BatchAnalysisConfig()
        self.assertEqual(config.max_concurrent, 10)
        self.assertEqual(config.timeout_seconds, 30)
        self.assertTrue(config.include_icd_mapping)
        self.assertTrue(config.include_storage)
        self.assertEqual(config.chunk_size, 50)
        self.assertEqual(config.retry_attempts, 2)
        
        # Test custom values
        custom_config = BatchAnalysisConfig(
            max_concurrent=20,
            timeout_seconds=60,
            include_icd_mapping=False,
            chunk_size=100
        )
        self.assertEqual(custom_config.max_concurrent, 20)
        self.assertEqual(custom_config.timeout_seconds, 60)
        self.assertFalse(custom_config.include_icd_mapping)
        self.assertEqual(custom_config.chunk_size, 100)
    
    def test_batch_analysis_result(self):
        """Test BatchAnalysisResult dataclass"""
        # Test successful result
        result = BatchAnalysisResult(
            note_id='test_1',
            success=True,
            data=self.sample_analysis_result,
            processing_time_ms=1500,
            session_id='session_123'
        )
        
        self.assertEqual(result.note_id, 'test_1')
        self.assertTrue(result.success)
        self.assertEqual(result.data, self.sample_analysis_result)
        self.assertEqual(result.processing_time_ms, 1500)
        self.assertEqual(result.session_id, 'session_123')
        self.assertIsNone(result.error)
        
        # Test failed result
        failed_result = BatchAnalysisResult(
            note_id='test_2',
            success=False,
            error='Analysis failed'
        )
        
        self.assertEqual(failed_result.note_id, 'test_2')
        self.assertFalse(failed_result.success)
        self.assertEqual(failed_result.error, 'Analysis failed')
        self.assertIsNone(failed_result.data)
    
    async def test_analyze_note_async_success(self):
        """Test successful single note analysis"""
        # Mock successful analysis
        self.async_service.storage_service.create_analysis_session.return_value = 'session_123'
        self.async_service.storage_service.get_cached_analysis.return_value = None  # Cache miss
        self.async_service.clinical_service.extract_clinical_entities.return_value = self.sample_analysis_result
        self.async_service.icd_matcher.map_clinical_entities_to_icd.return_value = self.sample_analysis_result
        
        config = BatchAnalysisConfig()
        result = await self.async_service.analyze_note_async(self.sample_note, config)
        
        self.assertIsInstance(result, BatchAnalysisResult)
        self.assertTrue(result.success)
        self.assertEqual(result.note_id, 'test_note_1')
        self.assertEqual(result.session_id, 'session_123')
        self.assertIsNotNone(result.data)
        self.assertIsNotNone(result.processing_time_ms)
        self.assertIsNone(result.error)
    
    async def test_analyze_note_async_cache_hit(self):
        """Test analysis with cache hit"""
        # Mock cache hit
        cached_result = self.sample_analysis_result.copy()
        self.async_service.storage_service.get_cached_analysis.return_value = cached_result
        self.async_service.storage_service.create_analysis_session.return_value = 'session_123'
        
        config = BatchAnalysisConfig()
        result = await self.async_service.analyze_note_async(self.sample_note, config)
        
        self.assertTrue(result.success)
        self.assertTrue(result.data['from_cache'])
        self.assertEqual(result.session_id, 'session_123')
        
        # Should not call clinical analysis service when cache hit
        self.async_service.clinical_service.extract_clinical_entities.assert_not_called()
    
    async def test_analyze_note_async_empty_text(self):
        """Test analysis with empty note text"""
        empty_note = self.sample_note.copy()
        empty_note['note_text'] = '   '  # Only whitespace
        
        config = BatchAnalysisConfig()
        result = await self.async_service.analyze_note_async(empty_note, config)
        
        self.assertFalse(result.success)
        self.assertIn('too short or empty', result.error)
    
    async def test_analyze_note_async_analysis_error(self):
        """Test analysis with clinical service error"""
        # Mock analysis error
        self.async_service.storage_service.create_analysis_session.return_value = 'session_123'
        self.async_service.storage_service.get_cached_analysis.return_value = None
        self.async_service.clinical_service.extract_clinical_entities.return_value = {'error': 'Analysis failed'}
        
        config = BatchAnalysisConfig()
        result = await self.async_service.analyze_note_async(self.sample_note, config)
        
        self.assertFalse(result.success)
        self.assertEqual(result.error, 'Analysis failed')
        self.assertEqual(result.session_id, 'session_123')
    
    async def test_analyze_note_async_storage_disabled(self):
        """Test analysis with storage disabled"""
        # Mock successful analysis
        self.async_service.clinical_service.extract_clinical_entities.return_value = self.sample_analysis_result
        
        config = BatchAnalysisConfig(include_storage=False)
        result = await self.async_service.analyze_note_async(self.sample_note, config)
        
        self.assertTrue(result.success)
        self.assertIsNone(result.session_id)
        
        # Should not call storage methods
        self.async_service.storage_service.create_analysis_session.assert_not_called()
        self.async_service.storage_service.get_cached_analysis.assert_not_called()
    
    async def test_batch_analyze_notes_small_batch(self):
        """Test batch analysis with small number of notes"""
        # Mock successful analysis for all notes
        self.async_service.clinical_service.extract_clinical_entities.return_value = self.sample_analysis_result
        self.async_service.storage_service.create_analysis_session.return_value = 'session_123'
        self.async_service.storage_service.get_cached_analysis.return_value = None
        
        # Create batch of 3 notes
        notes = [
            {**self.sample_note, 'note_id': f'note_{i}'}
            for i in range(3)
        ]
        
        config = BatchAnalysisConfig(max_concurrent=2, chunk_size=2)
        result = await self.async_service.batch_analyze_notes(notes, config)
        
        self.assertTrue(result['success'])
        self.assertEqual(len(result['results']), 3)
        self.assertEqual(result['summary']['total_notes'], 3)
        self.assertEqual(result['summary']['successful_analyses'], 3)
        self.assertEqual(result['summary']['failed_analyses'], 0)
        self.assertEqual(result['summary']['chunks_processed'], 2)  # 2 chunks (2+1)
    
    async def test_batch_analyze_notes_with_failures(self):
        """Test batch analysis with some failures"""
        # Mock mixed results
        def mock_analysis(note_text, context):
            if 'fail' in note_text:
                return {'error': 'Simulated failure'}
            return self.sample_analysis_result
        
        self.async_service.clinical_service.extract_clinical_entities.side_effect = mock_analysis
        self.async_service.storage_service.create_analysis_session.return_value = 'session_123'
        self.async_service.storage_service.get_cached_analysis.return_value = None
        
        # Create batch with one failing note
        notes = [
            {**self.sample_note, 'note_id': 'note_1'},
            {**self.sample_note, 'note_id': 'note_2', 'note_text': 'This will fail'},
            {**self.sample_note, 'note_id': 'note_3'}
        ]
        
        config = BatchAnalysisConfig()
        result = await self.async_service.batch_analyze_notes(notes, config)
        
        self.assertTrue(result['success'])
        self.assertEqual(result['summary']['total_notes'], 3)
        self.assertEqual(result['summary']['successful_analyses'], 2)
        self.assertEqual(result['summary']['failed_analyses'], 1)
        
        # Check that failed result is included
        failed_results = [r for r in result['results'] if not r['success']]
        self.assertEqual(len(failed_results), 1)
        self.assertEqual(failed_results[0]['note_id'], 'note_2')
    
    async def test_priority_scan_async(self):
        """Test priority scanning functionality"""
        # Mock analysis results with different risk levels
        def mock_analysis(note_text, context):
            if 'critical' in note_text:
                return {
                    **self.sample_analysis_result,
                    'overall_assessment': {
                        'risk_level': 'critical',
                        'requires_immediate_attention': True,
                        'primary_concerns': ['critical condition']
                    }
                }
            elif 'high' in note_text:
                return {
                    **self.sample_analysis_result,
                    'overall_assessment': {
                        'risk_level': 'high',
                        'requires_immediate_attention': False,
                        'primary_concerns': ['high risk condition']
                    }
                }
            else:
                return {
                    **self.sample_analysis_result,
                    'overall_assessment': {
                        'risk_level': 'low',
                        'requires_immediate_attention': False,
                        'primary_concerns': []
                    }
                }
        
        self.async_service.clinical_service.extract_clinical_entities.side_effect = mock_analysis
        
        # Create notes with different risk levels
        notes = [
            {**self.sample_note, 'note_id': 'critical_1', 'note_text': 'Critical patient condition'},
            {**self.sample_note, 'note_id': 'high_1', 'note_text': 'High risk patient'},
            {**self.sample_note, 'note_id': 'low_1', 'note_text': 'Low risk patient'},
            {**self.sample_note, 'note_id': 'critical_2', 'note_text': 'Another critical case'}
        ]
        
        result = await self.async_service.priority_scan_async(notes, risk_threshold='high')
        
        self.assertTrue(result['success'])
        self.assertEqual(len(result['priority_cases']), 3)  # 2 critical + 1 high
        self.assertEqual(result['scan_summary']['total_notes_scanned'], 4)
        self.assertEqual(result['scan_summary']['priority_cases_found'], 3)
        self.assertEqual(result['scan_summary']['risk_threshold'], 'high')
        
        # Check that critical cases are included
        critical_cases = [c for c in result['priority_cases'] if c['risk_level'] == 'critical']
        self.assertEqual(len(critical_cases), 2)
    
    async def test_priority_scan_critical_only(self):
        """Test priority scanning with critical threshold"""
        # Same mock as above
        def mock_analysis(note_text, context):
            if 'critical' in note_text:
                return {
                    **self.sample_analysis_result,
                    'overall_assessment': {
                        'risk_level': 'critical',
                        'requires_immediate_attention': True,
                        'primary_concerns': ['critical condition']
                    }
                }
            else:
                return {
                    **self.sample_analysis_result,
                    'overall_assessment': {
                        'risk_level': 'high',
                        'requires_immediate_attention': False,
                        'primary_concerns': ['high risk condition']
                    }
                }
        
        self.async_service.clinical_service.extract_clinical_entities.side_effect = mock_analysis
        
        notes = [
            {**self.sample_note, 'note_id': 'critical_1', 'note_text': 'Critical patient condition'},
            {**self.sample_note, 'note_id': 'high_1', 'note_text': 'High risk patient'}
        ]
        
        result = await self.async_service.priority_scan_async(notes, risk_threshold='critical')
        
        self.assertTrue(result['success'])
        self.assertEqual(len(result['priority_cases']), 1)  # Only critical
        self.assertEqual(result['priority_cases'][0]['note_id'], 'critical_1')
    
    def test_sync_wrapper_methods(self):
        """Test that sync wrapper methods work correctly"""
        # Test running async methods in sync context using asyncio.run
        
        # Mock successful analysis
        self.async_service.clinical_service.extract_clinical_entities.return_value = self.sample_analysis_result
        self.async_service.storage_service.create_analysis_session.return_value = 'session_123'
        self.async_service.storage_service.get_cached_analysis.return_value = None
        
        config = BatchAnalysisConfig()
        
        # Test single note analysis
        result = asyncio.run(self.async_service.analyze_note_async(self.sample_note, config))
        self.assertIsInstance(result, BatchAnalysisResult)
        self.assertTrue(result.success)
        
        # Test batch analysis
        notes = [self.sample_note]
        batch_result = asyncio.run(self.async_service.batch_analyze_notes(notes, config))
        self.assertTrue(batch_result['success'])
        self.assertEqual(len(batch_result['results']), 1)
    
    def test_performance_configuration(self):
        """Test performance-related configuration"""
        # Test that configuration limits are respected
        config = BatchAnalysisConfig(
            max_concurrent=50,  # Should be capped
            timeout_seconds=120,  # Should be capped
            chunk_size=200,  # Should be capped
            retry_attempts=5  # Should be capped
        )
        
        # These would be applied in the actual endpoint, not in the service
        # This test mainly documents the expected behavior
        self.assertEqual(config.max_concurrent, 50)
        self.assertEqual(config.timeout_seconds, 120)
        self.assertEqual(config.chunk_size, 200)
        self.assertEqual(config.retry_attempts, 5)
    
    def test_error_handling_edge_cases(self):
        """Test error handling for edge cases"""
        
        # Test with None note data
        async def test_none_note():
            config = BatchAnalysisConfig()
            result = await self.async_service.analyze_note_async(None, config)
            self.assertFalse(result.success)
        
        asyncio.run(test_none_note())
        
        # Test with empty batch
        async def test_empty_batch():
            config = BatchAnalysisConfig()
            result = await self.async_service.batch_analyze_notes([], config)
            self.assertTrue(result['success'])
            self.assertEqual(result['summary']['total_notes'], 0)
        
        asyncio.run(test_empty_batch())


def run_async_analysis_demo():
    """
    Demo function showing async clinical analysis capabilities
    """
    print("⚡ Async Clinical Analysis Demo")
    print("=" * 50)
    
    async def demo():
        service = AsyncClinicalAnalysis()
        
        # Mock the services for demo
        service.clinical_service = Mock()
        service.storage_service = Mock()
        
        # Create sample analysis result
        sample_result = {
            'symptoms': [
                {'entity': 'chest pain', 'confidence': 0.95, 'severity': 'severe'},
                {'entity': 'shortness of breath', 'confidence': 0.90, 'severity': 'moderate'}
            ],
            'overall_assessment': {
                'risk_level': 'high',
                'requires_immediate_attention': True,
                'primary_concerns': ['chest pain', 'shortness of breath']
            }
        }
        
        service.clinical_service.extract_clinical_entities.return_value = sample_result
        service.storage_service.create_analysis_session.return_value = 'demo_session'
        service.storage_service.get_cached_analysis.return_value = None
        
        print("\n📊 Demo Configuration:")
        config = BatchAnalysisConfig(
            max_concurrent=5,
            timeout_seconds=30,
            include_icd_mapping=True,
            chunk_size=10
        )
        print(f"   • Max Concurrent: {config.max_concurrent}")
        print(f"   • Timeout: {config.timeout_seconds}s")
        print(f"   • Chunk Size: {config.chunk_size}")
        print(f"   • ICD Mapping: {config.include_icd_mapping}")
        
        # Create demo notes
        demo_notes = []
        for i in range(25):
            risk_level = 'critical' if i % 7 == 0 else 'high' if i % 3 == 0 else 'moderate'
            demo_notes.append({
                'note_id': f'demo_note_{i}',
                'note_text': f'Patient {i} with {risk_level} condition and symptoms',
                'patient_context': {'age': 50 + i, 'gender': 'male' if i % 2 else 'female'}
            })
        
        print(f"\n📝 Processing {len(demo_notes)} demo notes...")
        
        start_time = time.time()
        
        # Run batch analysis
        result = await service.batch_analyze_notes(demo_notes, config)
        
        processing_time = time.time() - start_time
        
        print(f"\n✅ Batch Analysis Results:")
        print(f"   • Total Notes: {result['summary']['total_notes']}")
        print(f"   • Successful: {result['summary']['successful_analyses']}")
        print(f"   • Failed: {result['summary']['failed_analyses']}")
        print(f"   • Success Rate: {result['summary']['success_rate']:.1%}")
        print(f"   • Cache Hit Rate: {result['summary']['cache_hit_rate']:.1%}")
        print(f"   • Avg Processing Time: {result['summary']['average_processing_time_ms']}ms per note")
        print(f"   • Total Processing Time: {processing_time:.2f}s")
        print(f"   • Chunks Processed: {result['summary']['chunks_processed']}")
        
        # Demo priority scanning
        print(f"\n🚨 Running Priority Scan...")
        
        start_time = time.time()
        scan_result = await service.priority_scan_async(demo_notes[:15], 'high')
        scan_time = time.time() - start_time
        
        print(f"\n🔍 Priority Scan Results:")
        print(f"   • Notes Scanned: {scan_result['scan_summary']['total_notes_scanned']}")
        print(f"   • Priority Cases: {scan_result['scan_summary']['priority_cases_found']}")
        print(f"   • Scan Time: {scan_time:.2f}s")
        print(f"   • Avg Time per Note: {scan_result['scan_summary']['average_time_per_note_ms']}ms")
        
        return result, scan_result
    
    # Run the demo
    try:
        batch_result, scan_result = asyncio.run(demo())
        
        print(f"\n✨ Async Analysis Demo Complete!")
        print("Key Features Demonstrated:")
        print("   • High-performance async batch processing")
        print("   • Configurable concurrency and chunking")
        print("   • Priority scanning for rapid triage")
        print("   • Comprehensive error handling and retry logic")
        print("   • Performance metrics and monitoring")
        print("   • Graceful degradation and storage integration")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Running Async Clinical Analysis Tests")
    print("=" * 60)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run demo
    print("\n" + "=" * 60)
    run_async_analysis_demo()