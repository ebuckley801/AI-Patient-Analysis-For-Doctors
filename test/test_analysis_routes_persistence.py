#!/usr/bin/env python3
"""
Test Analysis Routes with Persistence Features
Tests the updated API endpoints with database storage and caching
"""

import unittest
import json
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import Flask app for testing
from app import create_app


class TestAnalysisRoutesWithPersistence(unittest.TestCase):
    """Test cases for analysis routes with persistence features"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # Sample test data
        self.sample_request = {
            "note_text": "Patient presents with severe chest pain, shortness of breath. Blood pressure 180/100, heart rate 110.",
            "patient_context": {
                "age": 65,
                "gender": "male",
                "medical_history": "hypertension, diabetes"
            },
            "note_id": "test_note_123",
            "patient_id": "test_patient_456"
        }
        
        self.sample_analysis_result = {
            "symptoms": [
                {
                    "entity": "chest pain",
                    "confidence": 0.95,
                    "severity": "severe",
                    "status": "active"
                }
            ],
            "conditions": [],
            "overall_assessment": {
                "risk_level": "high",
                "requires_immediate_attention": True,
                "summary": "Patient presenting with acute symptoms"
            }
        }
    
    @patch('app.routes.analysis_routes.storage_service')
    @patch('app.routes.analysis_routes.clinical_service')
    def test_extract_endpoint_with_caching(self, mock_clinical_service, mock_storage_service):
        """Test extract endpoint with caching functionality"""
        
        # Mock cache miss (no cached result)
        mock_storage_service.get_cached_analysis.return_value = None
        
        # Mock session creation
        mock_storage_service.create_analysis_session.return_value = 'test-session-123'
        mock_storage_service.update_analysis_session.return_value = True
        mock_storage_service.store_clinical_entities.return_value = ['entity-1', 'entity-2']
        mock_storage_service.cache_analysis_result.return_value = True
        
        # Mock clinical analysis
        mock_clinical_service.extract_clinical_entities.return_value = self.sample_analysis_result
        
        # Make request
        response = self.client.post(
            '/api/analysis/extract',
            data=json.dumps(self.sample_request),
            content_type='application/json'
        )
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertTrue(response_data['success'])
        self.assertIn('session_id', response_data['data']['request_metadata'])
        self.assertFalse(response_data['data']['request_metadata']['from_cache'])
        
        # Verify storage service calls
        mock_storage_service.get_cached_analysis.assert_called_once()
        mock_storage_service.create_analysis_session.assert_called_once()
        mock_storage_service.update_analysis_session.assert_called()
        mock_storage_service.store_clinical_entities.assert_called_once()
        mock_storage_service.cache_analysis_result.assert_called_once()
    
    @patch('app.routes.analysis_routes.storage_service')
    def test_extract_endpoint_cache_hit(self, mock_storage_service):
        """Test extract endpoint with cache hit"""
        
        # Mock cache hit
        cached_result = self.sample_analysis_result.copy()
        mock_storage_service.get_cached_analysis.return_value = cached_result
        
        # Make request
        response = self.client.post(
            '/api/analysis/extract',
            data=json.dumps(self.sample_request),
            content_type='application/json'
        )
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertTrue(response_data['success'])
        self.assertTrue(response_data['data']['request_metadata']['from_cache'])
        
        # Verify only cache was checked (no session creation)
        mock_storage_service.get_cached_analysis.assert_called_once()
        mock_storage_service.create_analysis_session.assert_not_called()
    
    @patch('app.routes.analysis_routes.storage_service')
    @patch('app.routes.analysis_routes.clinical_service')
    def test_extract_endpoint_storage_failure(self, mock_clinical_service, mock_storage_service):
        """Test extract endpoint handling of storage failures"""
        
        # Mock cache miss and storage failures
        mock_storage_service.get_cached_analysis.return_value = None
        mock_storage_service.create_analysis_session.return_value = 'test-session-123'
        mock_storage_service.update_analysis_session.return_value = True
        mock_storage_service.store_clinical_entities.side_effect = Exception("Storage failed")
        mock_storage_service.cache_analysis_result.side_effect = Exception("Cache failed")
        
        # Mock successful clinical analysis
        mock_clinical_service.extract_clinical_entities.return_value = self.sample_analysis_result
        
        # Make request
        response = self.client.post(
            '/api/analysis/extract',
            data=json.dumps(self.sample_request),
            content_type='application/json'
        )
        
        # Should still succeed despite storage failures
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertTrue(response_data['success'])
        
        # Analysis should still complete
        self.assertIn('symptoms', response_data['data'])
    
    @patch('app.routes.analysis_routes.storage_service')
    def test_priority_endpoint_with_findings(self, mock_storage_service):
        """Test priority endpoint returning findings"""
        
        # Mock priority findings
        sample_sessions = [
            {
                'session_id': 'session-1',
                'analysis_type': 'extract',
                'risk_level': 'critical',
                'requires_immediate_attention': True,
                'created_at': '2023-01-01T12:00:00Z',
                'response_data': {
                    'overall_assessment': {
                        'summary': 'Critical patient condition'
                    }
                }
            }
        ]
        
        mock_storage_service.get_priority_findings.return_value = sample_sessions
        mock_storage_service.get_session_entities.return_value = []
        
        # Make request
        response = self.client.get('/api/analysis/priority/test_note_123')
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertTrue(response_data['success'])
        
        # Check findings
        findings = response_data['data']['priority_findings']
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]['session_id'], 'session-1')
        self.assertEqual(findings[0]['risk_level'], 'critical')
        self.assertTrue(findings[0]['requires_immediate_attention'])
        
        # Check summary
        summary = response_data['data']['summary']
        self.assertEqual(summary['total_findings'], 1)
        self.assertEqual(summary['critical_findings'], 1)
        self.assertTrue(summary['requires_immediate_attention'])
    
    @patch('app.routes.analysis_routes.storage_service')
    def test_priority_endpoint_no_findings(self, mock_storage_service):
        """Test priority endpoint with no findings"""
        
        # Mock no findings
        mock_storage_service.get_priority_findings.return_value = []
        
        # Make request
        response = self.client.get('/api/analysis/priority/test_note_123')
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertTrue(response_data['success'])
        
        # Check empty findings
        findings = response_data['data']['priority_findings']
        self.assertEqual(len(findings), 0)
        
        # Check summary
        summary = response_data['data']['summary']
        self.assertEqual(summary['total_findings'], 0)
        self.assertFalse(summary['requires_immediate_attention'])
        self.assertIn('message', response_data['data'])
    
    @patch('app.routes.analysis_routes.storage_service')
    def test_priority_endpoint_with_details(self, mock_storage_service):
        """Test priority endpoint with entity details"""
        
        # Mock priority findings
        sample_sessions = [
            {
                'session_id': 'session-1',
                'analysis_type': 'extract',
                'risk_level': 'high',
                'requires_immediate_attention': False,
                'created_at': '2023-01-01T12:00:00Z',
                'response_data': {'overall_assessment': {}}
            }
        ]
        
        # Mock entities
        sample_entities = [
            {
                'entity_type': 'symptom',
                'entity_text': 'chest pain',
                'confidence': 0.95,
                'severity': 'severe',
                'status': 'active'
            },
            {
                'entity_type': 'symptom',
                'entity_text': 'shortness of breath',
                'confidence': 0.90,
                'severity': 'moderate', 
                'status': 'active'
            }
        ]
        
        mock_storage_service.get_priority_findings.return_value = sample_sessions
        mock_storage_service.get_session_entities.return_value = sample_entities
        
        # Make request with details
        response = self.client.get('/api/analysis/priority/test_note_123?include_details=true')
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertTrue(response_data['success'])
        
        # Check findings with entities
        findings = response_data['data']['priority_findings']
        self.assertEqual(len(findings), 1)
        self.assertIn('entities', findings[0])
        self.assertIn('symptom', findings[0]['entities'])
        self.assertEqual(len(findings[0]['entities']['symptom']), 2)
    
    def test_priority_endpoint_invalid_threshold(self):
        """Test priority endpoint with invalid risk threshold"""
        
        # Make request with invalid threshold
        response = self.client.get('/api/analysis/priority/test_note_123?risk_threshold=invalid')
        
        # Verify error response
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.data)
        self.assertFalse(response_data['success'])
        self.assertIn('Invalid risk_threshold', response_data['error'])
    
    @patch('app.routes.analysis_routes.storage_service')
    def test_health_check_with_storage(self, mock_storage_service):
        """Test health check endpoint with storage statistics"""
        
        # Mock storage statistics
        mock_cache_stats = {
            'total_cache_entries': 100,
            'active_cache_entries': 90,
            'expired_cache_entries': 10,
            'total_cache_hits': 250,
            'cache_hit_rate': 0.75,
            'cleanup_needed': True
        }
        
        mock_storage_service.get_cache_stats.return_value = mock_cache_stats
        mock_storage_service.cleanup_expired_cache.return_value = 10
        
        # Make request
        response = self.client.get('/api/analysis/health')
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertEqual(response_data['status'], 'healthy')
        
        # Check services
        services = response_data['services']
        self.assertEqual(services['storage_service'], 'available')
        self.assertEqual(services['analysis_cache'], mock_cache_stats)
        
        # Check maintenance
        self.assertIn('maintenance', response_data)
        self.assertEqual(response_data['maintenance']['expired_cache_cleaned'], 10)
    
    @patch('app.routes.analysis_routes.storage_service')
    def test_health_check_storage_unavailable(self, mock_storage_service):
        """Test health check when storage is unavailable"""
        
        # Mock storage failure
        mock_storage_service.get_cache_stats.side_effect = Exception("Database unavailable")
        
        # Make request
        response = self.client.get('/api/analysis/health')
        
        # Verify degraded status
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertEqual(response_data['status'], 'degraded')
        
        # Check services
        services = response_data['services']
        self.assertEqual(services['storage_service'], 'unavailable')
        self.assertIn('error', services['analysis_cache'])
    
    def test_missing_note_text(self):
        """Test extract endpoint with missing note_text"""
        
        request_data = {"patient_context": {"age": 45}}
        
        response = self.client.post(
            '/api/analysis/extract',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.data)
        self.assertFalse(response_data['success'])
        self.assertIn('note_text is required', response_data['error'])
    
    def test_invalid_json(self):
        """Test extract endpoint with invalid JSON"""
        
        response = self.client.post(
            '/api/analysis/extract',
            data="invalid json",
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.data)
        self.assertFalse(response_data['success'])
    
    def test_empty_note_text(self):
        """Test extract endpoint with empty note text"""
        
        request_data = {"note_text": "   "}  # Only whitespace
        
        response = self.client.post(
            '/api/analysis/extract',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.data)
        self.assertFalse(response_data['success'])
        self.assertIn('at least 10 characters', response_data['error'])


def run_persistence_integration_demo():
    """
    Demo function showing the persistence features integration
    """
    print("💾 Analysis Routes Persistence Integration Demo")
    print("=" * 60)
    
    try:
        from app import create_app
        
        app = create_app()
        app.config['TESTING'] = True
        
        print("\n📋 New Persistence Features:")
        print("   • Analysis result caching for improved performance")
        print("   • Session tracking for analysis requests")
        print("   • Clinical entity storage in database")
        print("   • ICD-10 mapping persistence")
        print("   • Priority findings retrieval")
        print("   • Cache statistics and cleanup")
        
        print("\n🔄 API Flow with Persistence:")
        print("   1. Check cache for existing analysis")
        print("   2. Create analysis session if cache miss")
        print("   3. Perform clinical analysis")
        print("   4. Store entities and mappings to database")
        print("   5. Cache result for future requests")
        print("   6. Update session with completion status")
        
        print("\n📊 Enhanced Endpoints:")
        print("   • POST /api/analysis/extract - Now with caching")
        print("   • GET /api/analysis/priority/<note_id> - Fully implemented")
        print("   • GET /api/analysis/health - Includes storage stats")
        
        print("\n🔍 Priority Endpoint Features:")
        print("   • Query by risk threshold (moderate/high/critical)")
        print("   • Optional detailed entity information")
        print("   • Summary statistics")
        print("   • Flexible filtering by note_id or patient_id")
        
        print("\n⚡ Performance Benefits:")
        print("   • Cache hit rate tracking")
        print("   • Automatic cache cleanup")
        print("   • Session-based analysis tracking")
        print("   • Indexed database queries")
        
        print("\n✨ Persistence Integration Demo Complete!")
        print("\nTo test with real database:")
        print("   1. Run: python app/utils/create_intelligence_db.py")
        print("   2. Start server: python app.py")
        print("   3. Test endpoints: python test/test_api_endpoints.py")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    print("🧪 Running Analysis Routes Persistence Tests")
    print("=" * 60)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run demo
    print("\n" + "=" * 60)
    run_persistence_integration_demo()