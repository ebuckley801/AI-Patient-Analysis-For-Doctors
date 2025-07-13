#!/usr/bin/env python3
"""
Test Analysis Storage Service for Intelligence Layer
Tests database persistence, caching, and retrieval functionality
"""

import unittest
import sys
import os
import json
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.services.analysis_storage_service import AnalysisStorageService


class TestAnalysisStorageService(unittest.TestCase):
    """Test cases for AnalysisStorageService"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.service = AnalysisStorageService()
        
        # Mock supabase client to avoid actual database calls during testing
        self.mock_supabase = Mock()
        self.service.supabase = self.mock_supabase
        
        # Sample test data
        self.sample_note_text = "Patient has chest pain and fever. Blood pressure 160/95."
        self.sample_patient_context = {"age": 45, "gender": "male"}
        self.sample_entities = [
            {
                "entity": "chest pain",
                "type": "symptom",
                "confidence": 0.95,
                "severity": "moderate",
                "status": "active"
            },
            {
                "entity": "fever",
                "type": "symptom", 
                "confidence": 0.90,
                "severity": "mild",
                "status": "active"
            }
        ]
        self.sample_analysis_result = {
            "symptoms": self.sample_entities,
            "conditions": [],
            "overall_assessment": {
                "risk_level": "moderate",
                "requires_immediate_attention": False
            }
        }
    
    def test_generate_session_id(self):
        """Test session ID generation"""
        session_id = self.service.generate_session_id()
        
        self.assertIsInstance(session_id, str)
        self.assertTrue(session_id.startswith('session_'))
        self.assertGreater(len(session_id), 20)  # Should be reasonably long
        
        # Generate another to ensure uniqueness
        session_id2 = self.service.generate_session_id()
        self.assertNotEqual(session_id, session_id2)
    
    def test_generate_cache_key(self):
        """Test cache key generation"""
        cache_key = self.service.generate_cache_key(
            self.sample_note_text, 
            self.sample_patient_context, 
            "extract"
        )
        
        self.assertIsInstance(cache_key, str)
        self.assertEqual(len(cache_key), 64)  # SHA256 hash length
        
        # Same inputs should produce same key
        cache_key2 = self.service.generate_cache_key(
            self.sample_note_text,
            self.sample_patient_context,
            "extract"
        )
        self.assertEqual(cache_key, cache_key2)
        
        # Different inputs should produce different keys
        cache_key3 = self.service.generate_cache_key(
            "Different note text",
            self.sample_patient_context,
            "extract"
        )
        self.assertNotEqual(cache_key, cache_key3)
    
    def test_generate_note_text_hash(self):
        """Test note text hash generation"""
        hash1 = self.service.generate_note_text_hash(self.sample_note_text)
        hash2 = self.service.generate_note_text_hash(self.sample_note_text)
        
        self.assertEqual(hash1, hash2)  # Same text should produce same hash
        self.assertEqual(len(hash1), 64)  # SHA256 length
        
        # Different text should produce different hash
        hash3 = self.service.generate_note_text_hash("Different text")
        self.assertNotEqual(hash1, hash3)
    
    @patch('app.services.analysis_storage_service.datetime')
    def test_create_analysis_session(self, mock_datetime):
        """Test creating analysis session"""
        # Mock datetime for consistent testing
        mock_now = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = mock_now
        
        # Mock successful database insert
        mock_result = Mock()
        mock_result.data = [{'id': 'test-uuid', 'session_id': 'test-session'}]
        self.mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_result
        
        session_id = self.service.create_analysis_session(
            note_id="test_note_123",
            patient_id="patient_456", 
            analysis_type="extract",
            request_data={"note_text": self.sample_note_text}
        )
        
        # Verify session ID was returned
        self.assertIsInstance(session_id, str)
        
        # Verify database was called correctly
        self.mock_supabase.table.assert_called_with('analysis_sessions')
        insert_call = self.mock_supabase.table.return_value.insert.call_args[0][0]
        
        self.assertEqual(insert_call['note_id'], 'test_note_123')
        self.assertEqual(insert_call['patient_id'], 'patient_456')
        self.assertEqual(insert_call['analysis_type'], 'extract')
        self.assertEqual(insert_call['status'], 'pending')
    
    def test_create_analysis_session_failure(self):
        """Test handling of session creation failure"""
        # Mock failed database insert
        mock_result = Mock()
        mock_result.data = None
        self.mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_result
        
        with self.assertRaises(Exception) as context:
            self.service.create_analysis_session(analysis_type="extract")
        
        self.assertIn("Failed to create analysis session", str(context.exception))
    
    @patch('app.services.analysis_storage_service.datetime')
    def test_update_analysis_session(self, mock_datetime):
        """Test updating analysis session"""
        mock_now = datetime(2023, 1, 1, 12, 30, 0)
        mock_datetime.now.return_value = mock_now
        
        # Mock successful update
        mock_result = Mock()
        mock_result.data = [{'session_id': 'test-session'}]
        self.mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_result
        
        success = self.service.update_analysis_session(
            'test-session',
            status='completed',
            risk_level='high'
        )
        
        self.assertTrue(success)
        
        # Verify database calls
        self.mock_supabase.table.assert_called_with('analysis_sessions')
        update_call = self.mock_supabase.table.return_value.update.call_args[0][0]
        
        self.assertEqual(update_call['status'], 'completed')
        self.assertEqual(update_call['risk_level'], 'high')
        self.assertIn('updated_at', update_call)
    
    def test_store_clinical_entities(self):
        """Test storing clinical entities"""
        # Mock successful insert
        mock_result = Mock()
        mock_result.data = [
            {'id': 'entity-1'}, 
            {'id': 'entity-2'}
        ]
        self.mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_result
        
        entity_ids = self.service.store_clinical_entities('test-session', self.sample_entities)
        
        self.assertEqual(len(entity_ids), 2)
        self.assertEqual(entity_ids, ['entity-1', 'entity-2'])
        
        # Verify database call
        self.mock_supabase.table.assert_called_with('clinical_entities')
        insert_call = self.mock_supabase.table.return_value.insert.call_args[0][0]
        
        self.assertEqual(len(insert_call), 2)
        self.assertEqual(insert_call[0]['session_id'], 'test-session')
        self.assertEqual(insert_call[0]['entity_type'], 'symptom')
        self.assertEqual(insert_call[0]['entity_text'], 'chest pain')
        self.assertEqual(insert_call[0]['confidence'], 0.95)
    
    def test_store_icd_mappings(self):
        """Test storing ICD-10 mappings"""
        sample_mappings = [
            {
                'icd_10_code': 'R50.9',
                'description': 'Fever',
                'similarity_score': 0.95,
                'confidence': 0.90,
                'method': 'vector_similarity'
            },
            {
                'icd_10_code': 'R06.02', 
                'description': 'Shortness of breath',
                'similarity_score': 0.80,
                'confidence': 0.75,
                'method': 'vector_similarity'
            }
        ]
        
        # Mock successful insert
        mock_result = Mock()
        mock_result.data = [
            {'id': 'mapping-1'},
            {'id': 'mapping-2'}
        ]
        self.mock_supabase.table.return_value.insert.return_value.execute.return_value = mock_result
        
        mapping_ids = self.service.store_icd_mappings('test-session', 'entity-1', sample_mappings)
        
        self.assertEqual(len(mapping_ids), 2)
        self.assertEqual(mapping_ids, ['mapping-1', 'mapping-2'])
        
        # Verify database call
        self.mock_supabase.table.assert_called_with('entity_icd_mappings')
        insert_call = self.mock_supabase.table.return_value.insert.call_args[0][0]
        
        self.assertEqual(len(insert_call), 2)
        self.assertEqual(insert_call[0]['entity_id'], 'entity-1')
        self.assertEqual(insert_call[0]['icd_10_code'], 'R50.9')
        self.assertTrue(insert_call[0]['is_primary_mapping'])  # First mapping is primary
        self.assertFalse(insert_call[1]['is_primary_mapping'])  # Second is not primary
        self.assertEqual(insert_call[1]['rank_order'], 2)
    
    @patch('app.services.analysis_storage_service.datetime')
    def test_cache_analysis_result(self, mock_datetime):
        """Test caching analysis results"""
        mock_now = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = mock_now
        
        # Mock successful upsert
        mock_result = Mock()
        mock_result.data = [{'cache_key': 'test-key'}]
        self.mock_supabase.table.return_value.upsert.return_value.execute.return_value = mock_result
        
        success = self.service.cache_analysis_result(
            self.sample_note_text,
            self.sample_patient_context,
            'extract',
            self.sample_analysis_result
        )
        
        self.assertTrue(success)
        
        # Verify database call
        self.mock_supabase.table.assert_called_with('analysis_cache')
        upsert_call = self.mock_supabase.table.return_value.upsert.call_args[0][0]
        
        self.assertIn('cache_key', upsert_call)
        self.assertIn('note_text_hash', upsert_call)
        self.assertEqual(upsert_call['analysis_type'], 'extract')
        self.assertEqual(upsert_call['cached_result'], self.sample_analysis_result)
    
    def test_get_cached_analysis_hit(self):
        """Test retrieving cached analysis (cache hit)"""
        # Mock cache hit
        mock_result = Mock()
        mock_result.data = [{
            'cache_key': 'test-key',
            'cached_result': self.sample_analysis_result,
            'hit_count': 5
        }]
        self.mock_supabase.table.return_value.select.return_value.eq.return_value.gt.return_value.execute.return_value = mock_result
        
        # Mock update hit count
        mock_update_result = Mock()
        self.mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_update_result
        
        cached_result = self.service.get_cached_analysis(
            self.sample_note_text,
            self.sample_patient_context,
            'extract'
        )
        
        self.assertEqual(cached_result, self.sample_analysis_result)
        
        # Verify hit count was updated
        update_call = self.mock_supabase.table.return_value.update.call_args[0][0]
        self.assertEqual(update_call['hit_count'], 6)
    
    def test_get_cached_analysis_miss(self):
        """Test retrieving cached analysis (cache miss)"""
        # Mock cache miss
        mock_result = Mock()
        mock_result.data = []
        self.mock_supabase.table.return_value.select.return_value.eq.return_value.gt.return_value.execute.return_value = mock_result
        
        cached_result = self.service.get_cached_analysis(
            self.sample_note_text,
            self.sample_patient_context,
            'extract'
        )
        
        self.assertIsNone(cached_result)
    
    def test_get_priority_findings(self):
        """Test retrieving priority findings"""
        # Mock priority findings query
        sample_sessions = [
            {
                'session_id': 'session-1',
                'risk_level': 'critical',
                'requires_immediate_attention': True,
                'created_at': '2023-01-01T12:00:00Z'
            },
            {
                'session_id': 'session-2', 
                'risk_level': 'high',
                'requires_immediate_attention': False,
                'created_at': '2023-01-01T11:00:00Z'
            }
        ]
        
        mock_result = Mock()
        mock_result.data = sample_sessions
        
        # Mock the query chain
        mock_query = Mock()
        mock_query.eq.return_value = mock_query
        mock_query.in_.return_value = mock_query
        mock_query.order.return_value = mock_query
        mock_query.execute.return_value = mock_result
        
        self.mock_supabase.table.return_value.select.return_value = mock_query
        
        findings = self.service.get_priority_findings(note_id='test-note', risk_threshold='high')
        
        self.assertEqual(len(findings), 2)
        self.assertEqual(findings[0]['session_id'], 'session-1')
        self.assertEqual(findings[0]['risk_level'], 'critical')
    
    def test_cleanup_expired_cache(self):
        """Test cleaning up expired cache entries"""
        # Mock RPC call
        mock_result = Mock()
        mock_result.data = 5  # 5 entries deleted
        self.mock_supabase.rpc.return_value.execute.return_value = mock_result
        
        deleted_count = self.service.cleanup_expired_cache()
        
        self.assertEqual(deleted_count, 5)
        self.mock_supabase.rpc.assert_called_with('cleanup_expired_cache')
    
    def test_get_cache_stats(self):
        """Test getting cache performance statistics"""
        # Mock various queries for statistics
        total_mock = Mock()
        total_mock.count = 100
        
        expired_mock = Mock() 
        expired_mock.count = 10
        
        hits_mock = Mock()
        hits_mock.data = [
            {'hit_count': 5, 'analysis_type': 'extract'},
            {'hit_count': 3, 'analysis_type': 'diagnose'},
            {'hit_count': 0, 'analysis_type': 'extract'}
        ]
        
        # Mock query chain
        self.mock_supabase.table.return_value.select.return_value.execute.side_effect = [
            total_mock,   # Total entries
            expired_mock, # Expired entries  
            hits_mock     # Hit statistics
        ]
        
        stats = self.service.get_cache_stats()
        
        self.assertEqual(stats['total_cache_entries'], 100)
        self.assertEqual(stats['active_cache_entries'], 90)
        self.assertEqual(stats['expired_cache_entries'], 10)
        self.assertEqual(stats['total_cache_hits'], 8)  # 5 + 3 + 0
        self.assertEqual(stats['cache_hit_rate'], 0.08)  # 8/100
        self.assertTrue(stats['cleanup_needed'])
    
    def test_error_handling(self):
        """Test error handling in various methods"""
        # Test session creation with database error
        self.mock_supabase.table.side_effect = Exception("Database connection failed")
        
        with self.assertRaises(Exception):
            self.service.create_analysis_session(analysis_type="extract")
        
        # Reset side effect for other tests
        self.mock_supabase.table.side_effect = None
        
        # Test update with database error
        self.mock_supabase.table.return_value.update.side_effect = Exception("Update failed")
        
        success = self.service.update_analysis_session('test-session', status='completed')
        self.assertFalse(success)


def run_storage_service_demo():
    """
    Demo function showing how the Analysis Storage Service works
    Note: This requires actual database connection and won't work in CI/testing
    """
    print("🧠 Analysis Storage Service Demo")
    print("=" * 50)
    
    try:
        # Initialize service (will use real database if .env configured)
        storage_service = AnalysisStorageService()
        
        # Generate test data
        note_text = "Patient presents with severe chest pain, shortness of breath, and diaphoresis. Blood pressure 180/100."
        patient_context = {"age": 65, "gender": "male", "medical_history": "hypertension, diabetes"}
        
        print(f"\n📝 Sample Note: {note_text[:50]}...")
        print(f"👤 Patient Context: {patient_context}")
        
        # Test cache key generation
        cache_key = storage_service.generate_cache_key(note_text, patient_context, 'extract')
        print(f"\n🔑 Generated Cache Key: {cache_key[:16]}...")
        
        # Test session creation (would create real session if DB available)
        print("\n📊 Testing Session Creation...")
        try:
            session_id = storage_service.create_analysis_session(
                note_id="demo_note_001",
                patient_id="demo_patient_001", 
                analysis_type="extract",
                request_data={"note_text": note_text, "patient_context": patient_context}
            )
            print(f"✅ Created session: {session_id}")
            
            # Test cache statistics
            cache_stats = storage_service.get_cache_stats()
            print(f"\n📈 Cache Statistics:")
            for key, value in cache_stats.items():
                print(f"   • {key}: {value}")
                
        except Exception as e:
            print(f"⚠️  Database not available: {str(e)}")
            print("   This is expected if Supabase is not configured")
        
        print("\n✨ Storage Service Demo Complete!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    print("🧪 Running Analysis Storage Service Tests")
    print("=" * 60)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run demo
    print("\n" + "=" * 60)
    run_storage_service_demo()