from supabase import create_client, Client
from app.config.config import Config

class SupabaseService:
    def __init__(self):
        self.client: Client = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
    
    def get_patient_notes(self, limit=100, offset=0):
        """Get all patient notes with pagination"""
        response = self.client.table('patient_notes').select('*').range(offset, offset + limit - 1).execute()
        return response.data
    
    def get_patient_note_by_id(self, note_id):
        """Get a specific patient note by ID"""
        response = self.client.table('patient_notes').select('*').eq('id', note_id).execute()
        return response.data[0] if response.data else None
    
    def get_patient_notes_by_patient_id(self, patient_id):
        """Get all notes for a specific patient"""
        response = self.client.table('patient_notes').select('*').eq('patient_id', patient_id).execute()
        return response.data
    
    def create_patient_note(self, note_data):
        """Create a new patient note"""
        response = self.client.table('patient_notes').insert(note_data).execute()
        return response.data[0] if response.data else None
    
    def update_patient_note(self, note_id, note_data):
        """Update an existing patient note"""
        response = self.client.table('patient_notes').update(note_data).eq('id', note_id).execute()
        return response.data[0] if response.data else None
    
    def delete_patient_note(self, note_id):
        """Delete a patient note"""
        response = self.client.table('patient_notes').delete().eq('id', note_id).execute()
        return response.data[0] if response.data else None
    
    def search_patient_notes(self, query, field='patient_note'):
        """Search patient notes by text"""
        response = self.client.table('patient_notes').select('*').ilike(field, f'%{query}%').execute()
        return response.data