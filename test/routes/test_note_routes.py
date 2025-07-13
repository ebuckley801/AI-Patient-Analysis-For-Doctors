import unittest
from unittest.mock import patch, MagicMock
import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from app import create_app

class TestNoteRoutes(unittest.TestCase):
    
    def setUp(self):
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        self.sample_note = {
            'id': 1,
            'patient_id': 1,
            'patient_uid': 'uid-001',
            'patient_note': 'Patient has mild symptoms',
            'age': 45,
            'gender': 'M'
        }

    @patch('app.routes.note_routes.supabase_service')
    def test_get_notes_success(self, mock_service):
        mock_service.get_patient_notes.return_value = [self.sample_note]
        
        response = self.client.get('/api/notes/')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['success'])
        self.assertEqual(len(data['data']), 1)

    @patch('app.routes.note_routes.supabase_service')
    def test_get_note_by_id_success(self, mock_service):
        mock_service.get_patient_note_by_id.return_value = self.sample_note
        
        response = self.client.get('/api/notes/1')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['success'])

    @patch('app.routes.note_routes.supabase_service')
    def test_get_note_by_id_not_found(self, mock_service):
        mock_service.get_patient_note_by_id.return_value = None
        
        response = self.client.get('/api/notes/999')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 404)
        self.assertFalse(data['success'])

    @patch('app.routes.note_routes.supabase_service')
    def test_create_note_success(self, mock_service):
        mock_service.create_patient_note.return_value = self.sample_note
        
        response = self.client.post('/api/notes/', 
                                  data=json.dumps(self.sample_note),
                                  content_type='application/json')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 201)
        self.assertTrue(data['success'])

    @patch('app.routes.note_routes.supabase_service')
    def test_update_note_success(self, mock_service):
        mock_service.get_patient_note_by_id.return_value = self.sample_note
        mock_service.update_patient_note.return_value = self.sample_note
        
        update_data = {'patient_note': 'Updated note'}
        response = self.client.put('/api/notes/1',
                                 data=json.dumps(update_data),
                                 content_type='application/json')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['success'])

    @patch('app.routes.note_routes.supabase_service')
    def test_delete_note_success(self, mock_service):
        mock_service.get_patient_note_by_id.return_value = self.sample_note
        mock_service.delete_patient_note.return_value = self.sample_note
        
        response = self.client.delete('/api/notes/1')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['success'])

    @patch('app.routes.note_routes.supabase_service')
    def test_search_notes_success(self, mock_service):
        mock_service.search_patient_notes.return_value = [self.sample_note]
        
        response = self.client.get('/api/notes/search?q=symptoms')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['success'])

    def test_search_notes_invalid_field(self):
        response = self.client.get('/api/notes/search?q=test&field=invalid_field')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 400)
        self.assertFalse(data['success'])

if __name__ == '__main__':
    unittest.main()