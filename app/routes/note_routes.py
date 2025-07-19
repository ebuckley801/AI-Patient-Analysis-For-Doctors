from flask import request, jsonify
from flask_restx import Namespace, Resource, fields
from app.services.supabase_service import SupabaseService
from app.utils.validation import Validator
from app.middleware.security import log_request, sanitize_middleware, require_content_type

note_ns = Namespace('notes', description='Note related operations')
supabase_service = SupabaseService()

# Define a model for patient notes for API documentation
note_model = note_ns.model('Note', {
    'id': fields.Integer(readOnly=True, description='The unique identifier of the note'),
    'patient_id': fields.Integer(required=True, description='The ID of the patient'),
    'patient_uid': fields.String(required=True, description='The unique identifier of the patient'),
    'patient_note': fields.String(required=True, description='The content of the patient note'),
    'age': fields.Integer(required=True, description='The age of the patient'),
    'gender': fields.String(required=True, description='The gender of the patient'),
    'created_at': fields.DateTime(readOnly=True, description='The timestamp when the note was created'),
    'updated_at': fields.DateTime(readOnly=True, description='The timestamp when the note was last updated')
})

note_create_model = note_ns.model('NoteCreate', {
    'patient_id': fields.Integer(required=True, description='The ID of the patient'),
    'patient_uid': fields.String(required=True, description='The unique identifier of the patient'),
    'patient_note': fields.String(required=True, description='The content of the patient note'),
    'age': fields.Integer(required=True, description='The age of the patient'),
    'gender': fields.String(required=True, description='The gender of the patient')
})

note_update_model = note_ns.model('NoteUpdate', {
    'patient_note': fields.String(description='The updated content of the patient note'),
    'age': fields.Integer(description='The updated age of the patient'),
    'gender': fields.String(description='The updated gender of the patient')
})

@note_ns.route('/')
class NoteList(Resource):
    @note_ns.doc('list_notes')
    @note_ns.expect(note_ns.parser().add_argument('limit', type=int, help='Limit the number of results', default=100, location='args').add_argument('offset', type=int, help='Offset the results', default=0, location='args'))
    @note_ns.marshal_list_with(note_model)
    def get(self):
        """Get all notes with pagination"""
        try:
            args = note_ns.parser().parse_args()
            limit = args['limit']
            offset = args['offset']
            
            notes = supabase_service.get_patient_notes(limit=limit, offset=offset)
            return notes, 200
        except Exception as e:
            note_ns.abort(500, message=str(e))

    @note_ns.doc('create_note')
    @note_ns.expect(note_create_model, validate=True)
    @note_ns.marshal_with(note_model, code=201)
    @log_request()
    @sanitize_middleware()
    @require_content_type('application/json')
    def post(self):
        """Create a new note"""
        try:
            validated_data = note_ns.payload
            result = supabase_service.create_patient_note(validated_data)
            return result, 201
        except Exception as e:
            note_ns.abort(500, message=str(e))

@note_ns.route('/<int:note_id>')
@note_ns.param('note_id', 'The note identifier')
class Note(Resource):
    @note_ns.doc('get_note')
    @note_ns.marshal_with(note_model)
    def get(self, note_id):
        """Get a specific note by ID"""
        try:
            note = supabase_service.get_patient_note_by_id(note_id)
            if not note:
                note_ns.abort(404, message='Note not found')
            return note, 200
        except Exception as e:
            note_ns.abort(500, message=str(e))

    @note_ns.doc('update_note')
    @note_ns.expect(note_update_model, validate=True)
    @note_ns.marshal_with(note_model)
    def put(self, note_id):
        """Update a specific note"""
        try:
            data = note_ns.payload
            existing_note = supabase_service.get_patient_note_by_id(note_id)
            if not existing_note:
                note_ns.abort(404, message='Note not found')
            
            result = supabase_service.update_patient_note(note_id, data)
            return result, 200
        except Exception as e:
            note_ns.abort(500, message=str(e))

    @note_ns.doc('delete_note')
    @note_ns.response(204, 'Note deleted')
    def delete(self, note_id):
        """Delete a specific note"""
        try:
            existing_note = supabase_service.get_patient_note_by_id(note_id)
            if not existing_note:
                note_ns.abort(404, message='Note not found')
            
            supabase_service.delete_patient_note(note_id)
            return '', 204
        except Exception as e:
            note_ns.abort(500, message=str(e))

@note_ns.route('/search')
class NoteSearch(Resource):
    @note_ns.doc('search_notes')
    @note_ns.expect(note_ns.parser().add_argument('q', type=str, help='Search query', required=True, location='args').add_argument('field', type=str, help='Field to search (patient_note, patient_uid, gender)', default='patient_note', location='args'))
    @note_ns.marshal_list_with(note_model)
    def get(self):
        """Search notes by content"""
        try:
            args = note_ns.parser().parse_args()
            query = args['q']
            field = args['field']
            
            if not query:
                note_ns.abort(400, message='Query parameter "q" is required')
            
            allowed_fields = ['patient_note', 'patient_uid', 'gender']
            if field not in allowed_fields:
                note_ns.abort(400, message=f'Invalid field. Allowed: {allowed_fields}')
            
            results = supabase_service.search_patient_notes(query, field)
            return results, 200
        except Exception as e:
            note_ns.abort(500, message=str(e))

@note_ns.route('/patient/<int:patient_id>')
@note_ns.param('patient_id', 'The patient identifier')
class NoteByPatient(Resource):
    @note_ns.doc('get_notes_by_patient')
    @note_ns.marshal_list_with(note_model)
    def get(self, patient_id):
        """Get all notes for a specific patient"""
        try:
            notes = supabase_service.get_patient_notes_by_patient_id(patient_id)
            return notes, 200
        except Exception as e:
            note_ns.abort(500, message=str(e))