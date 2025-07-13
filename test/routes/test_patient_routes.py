import unittest
from unittest.mock import patch, MagicMock
import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from app import create_app

class TestPatientRoutes(unittest.TestCase):
    
    def setUp(self):
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        self.sample_patient = {
            'patient_id': 1,
            'patient_uid': 'uid-001',
            'patient_note': 'Patient has mild symptoms',
            'age': 45,
            'gender': 'M'
        }

    @patch('app.routes.patient_routes.supabase_service')
    def test_get_patients_success(self, mock_service):
        mock_service.get_patient_notes.return_value = [self.sample_patient]
        
        response = self.client.get('/api/patients/')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['success'])
        self.assertEqual(len(data['data']), 1)

    @patch('app.routes.patient_routes.supabase_service')
    def test_get_patient_by_id_success(self, mock_service):
        mock_service.get_patient_notes_by_patient_id.return_value = [self.sample_patient]
        
        response = self.client.get('/api/patients/1')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['success'])

    @patch('app.routes.patient_routes.supabase_service')
    def test_get_patient_by_id_not_found(self, mock_service):
        mock_service.get_patient_notes_by_patient_id.return_value = []
        
        response = self.client.get('/api/patients/999')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 404)
        self.assertFalse(data['success'])

    @patch('app.routes.patient_routes.supabase_service')
    def test_create_patient_success(self, mock_service):
        mock_service.create_patient_note.return_value = self.sample_patient
        
        response = self.client.post('/api/patients/', 
                                  data=json.dumps(self.sample_patient),
                                  content_type='application/json')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 201)
        self.assertTrue(data['success'])

    def test_create_patient_missing_fields(self):
        incomplete_patient = {'patient_id': 1}
        
        response = self.client.post('/api/patients/',
                                  data=json.dumps(incomplete_patient),
                                  content_type='application/json')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 400)
        self.assertFalse(data['success'])

    @patch('app.routes.patient_routes.supabase_service')
    def test_search_patients_success(self, mock_service):
        mock_service.search_patient_notes.return_value = [self.sample_patient]
        
        response = self.client.get('/api/patients/search?q=symptoms')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['success'])

    def test_search_patients_missing_query(self):
        response = self.client.get('/api/patients/search')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 400)
        self.assertFalse(data['success'])

if __name__ == '__main__':
    unittest.main()