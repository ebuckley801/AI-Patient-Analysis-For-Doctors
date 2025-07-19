from flask import request, jsonify
from flask_restx import Namespace, Resource, fields
from app.services.supabase_service import SupabaseService
from app.models.patient import Patient
from app.utils.validation import Validator
from app.middleware.security import log_request, sanitize_middleware, require_content_type

patient_ns = Namespace('patients', description='Patient related operations')
supabase_service = SupabaseService()

# Define a model for patient notes for API documentation
patient_note_model = patient_ns.model('PatientNote', {
    'id': fields.Integer(readOnly=True, description='The unique identifier of the patient note'),
    'patient_id': fields.Integer(required=True, description='The ID of the patient'),
    'patient_uid': fields.String(required=True, description='The unique identifier of the patient'),
    'patient_note': fields.String(required=True, description='The content of the patient note'),
    'age': fields.Integer(required=True, description='The age of the patient'),
    'gender': fields.String(required=True, description='The gender of the patient'),
    'created_at': fields.DateTime(readOnly=True, description='The timestamp when the note was created'),
    'updated_at': fields.DateTime(readOnly=True, description='The timestamp when the note was last updated')
})

patient_create_model = patient_ns.model('PatientCreate', {
    'patient_id': fields.Integer(required=True, description='The ID of the patient'),
    'patient_uid': fields.String(required=True, description='The unique identifier of the patient'),
    'patient_note': fields.String(required=True, description='The content of the patient note'),
    'age': fields.Integer(required=True, description='The age of the patient'),
    'gender': fields.String(required=True, description='The gender of the patient')
})

patient_update_model = patient_ns.model('PatientUpdate', {
    'patient_note': fields.String(description='The updated content of the patient note'),
    'age': fields.Integer(description='The updated age of the patient'),
    'gender': fields.String(description='The updated gender of the patient')
})

from flask import request, jsonify
from flask_restx import Namespace, Resource, fields
from app.services.supabase_service import SupabaseService
from app.models.patient import Patient
from app.utils.validation import Validator
from app.middleware.security import log_request, sanitize_middleware, require_content_type
from flask_jwt_extended import jwt_required, get_jwt_identity # Import jwt_required

patient_ns = Namespace('patients', description='Patient related operations')
supabase_service = SupabaseService()

# Define a model for patient notes for API documentation
patient_note_model = patient_ns.model('PatientNote', {
    'id': fields.Integer(readOnly=True, description='The unique identifier of the patient note'),
    'patient_id': fields.Integer(required=True, description='The ID of the patient'),
    'patient_uid': fields.String(required=True, description='The unique identifier of the patient'),
    'patient_note': fields.String(required=True, description='The content of the patient note'),
    'age': fields.Integer(required=True, description='The age of the patient'),
    'gender': fields.String(required=True, description='The gender of the patient'),
    'created_at': fields.DateTime(readOnly=True, description='The timestamp when the note was created'),
    'updated_at': fields.DateTime(readOnly=True, description='The timestamp when the note was last updated')
})

patient_create_model = patient_ns.model('PatientCreate', {
    'patient_id': fields.Integer(required=True, description='The ID of the patient'),
    'patient_uid': fields.String(required=True, description='The unique identifier of the patient'),
    'patient_note': fields.String(required=True, description='The content of the patient note'),
    'age': fields.Integer(required=True, description='The age of the patient'),
    'gender': fields.String(required=True, description='The gender of the patient')
})

patient_update_model = patient_ns.model('PatientUpdate', {
    'patient_note': fields.String(description='The updated content of the patient note'),
    'age': fields.Integer(description='The updated age of the patient'),
    'gender': fields.String(description='The updated gender of the patient')
})

@patient_ns.route('/')
class PatientList(Resource):
    @patient_ns.doc('list_patients')
    @patient_ns.expect(patient_ns.parser().add_argument('limit', type=int, help='Limit the number of results', default=100, location='args').add_argument('offset', type=int, help='Offset the results', default=0, location='args'))
    @patient_ns.marshal_list_with(patient_note_model)
    @log_request()
    @jwt_required() # Add JWT protection
    def get(self):
        """Get all patients with pagination"""
        try:
            args = patient_ns.parser().parse_args()
            limit = args['limit']
            offset = args['offset']
            
            patients = supabase_service.get_patient_notes(limit=limit, offset=offset)
            return patients, 200
        except Exception as e:
            patient_ns.abort(500, message=str(e))

    @patient_ns.doc('create_patient')
    @patient_ns.expect(patient_create_model, validate=True)
    @patient_ns.marshal_with(patient_note_model, code=201)
    @log_request()
    @sanitize_middleware()
    @require_content_type('application/json')
    @jwt_required() # Add JWT protection
    def post(self):
        """Create a new patient note"""
        try:
            validated_data = patient_ns.payload
            patient = Patient.from_dict(validated_data)
            result = supabase_service.create_patient_note(patient.to_dict())
            return result, 201
        except Exception as e:
            patient_ns.abort(500, message=str(e))

@patient_ns.route('/<int:patient_id>')
@patient_ns.param('patient_id', 'The patient identifier')
class Patient(Resource):
    @patient_ns.doc('get_patient')
    @patient_ns.marshal_list_with(patient_note_model)
    @log_request()
    @jwt_required() # Add JWT protection
    def get(self, patient_id):
        """Get all notes for a specific patient"""
        try:
            validated_id = Validator.validate_patient_id(patient_id)
            notes = supabase_service.get_patient_notes_by_patient_id(validated_id)
            if not notes:
                patient_ns.abort(404, message='Patient not found')
            return notes, 200
        except Exception as e:
            patient_ns.abort(500, message=str(e))

    @patient_ns.doc('update_patient')
    @patient_ns.expect(patient_update_model, validate=True)
    @patient_ns.marshal_with(patient_note_model)
    @jwt_required() # Add JWT protection
    def put(self, patient_id):
        """Update a patient note by patient_id"""
        try:
            data = patient_ns.payload
            existing_notes = supabase_service.get_patient_notes_by_patient_id(patient_id)
            if not existing_notes:
                patient_ns.abort(404, message='Patient not found')
            
            note_id = existing_notes[0]['id']
            result = supabase_service.update_patient_note(note_id, data)
            return result, 200
        except Exception as e:
            patient_ns.abort(500, message=str(e))

    @patient_ns.doc('delete_patient')
    @patient_ns.response(204, 'Patient notes deleted')
    @jwt_required() # Add JWT protection
    def delete(self, patient_id):
        """Delete all notes for a patient"""
        try:
            notes = supabase_service.get_patient_notes_by_patient_id(patient_id)
            if not notes:
                patient_ns.abort(404, message='Patient not found')
            
            for note in notes:
                supabase_service.delete_patient_note(note['id'])
            return '', 204
        except Exception as e:
            patient_ns.abort(500, message=str(e))

@patient_ns.route('/search')
class PatientSearch(Resource):
    @patient_ns.doc('search_patients')
    @patient_ns.expect(patient_ns.parser().add_argument('q', type=str, help='Search query', required=True, location='args'))
    @patient_ns.marshal_list_with(patient_note_model)
    @log_request()
    @jwt_required() # Add JWT protection
    def get(self):
        """Search patients by note content"""
        try:
            args = patient_ns.parser().parse_args()
            query = args['q']
            
            if not query:
                patient_ns.abort(400, message='Query parameter "q" is required')
            
            results = supabase_service.search_patient_notes(query)
            return results, 200
        except Exception as e:
            patient_ns.abort(500, message=str(e))