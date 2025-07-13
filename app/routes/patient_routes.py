from flask import Blueprint, request, jsonify
from app.services.supabase_service import SupabaseService
from app.models.patient import Patient
from app.utils.validation import (
    validate_json_request, validate_query_params, 
    PatientNoteSchema, Validator
)
from app.middleware.security import sanitize_middleware, require_content_type, log_request

patient_bp = Blueprint('patients', __name__)
supabase_service = SupabaseService()

@patient_bp.route('/', methods=['GET'])
@log_request()
@validate_query_params(
    limit=lambda x: Validator.validate_pagination(x, 0)[0],
    offset=lambda x: Validator.validate_pagination(100, x)[1]
)
def get_patients():
    """Get all patients with pagination"""
    try:
        limit = getattr(request, 'validated_params', {}).get('limit', 100)
        offset = getattr(request, 'validated_params', {}).get('offset', 0)
        
        patients = supabase_service.get_patient_notes(limit=limit, offset=offset)
        return jsonify({
            'success': True,
            'data': patients,
            'count': len(patients)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@patient_bp.route('/<int:patient_id>', methods=['GET'])
@log_request()
def get_patient(patient_id):
    """Get all notes for a specific patient"""
    try:
        # Validate patient_id
        validated_id = Validator.validate_patient_id(patient_id)
        notes = supabase_service.get_patient_notes_by_patient_id(validated_id)
        if not notes:
            return jsonify({'success': False, 'error': 'Patient not found'}), 404
        
        return jsonify({
            'success': True,
            'data': notes
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@patient_bp.route('/', methods=['POST'])
@log_request()
@require_content_type('application/json')
@sanitize_middleware()
@validate_json_request(PatientNoteSchema.validate_create_request)
def create_patient():
    """Create a new patient note"""
    try:
        validated_data = request.validated_data
        
        # Create patient object
        patient = Patient.from_dict(validated_data)
        
        # Save to database
        result = supabase_service.create_patient_note(patient.to_dict())
        
        return jsonify({
            'success': True,
            'data': result
        }), 201
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@patient_bp.route('/<int:patient_id>', methods=['PUT'])
def update_patient(patient_id):
    """Update a patient note by patient_id"""
    try:
        data = request.get_json()
        
        # Find the note to update (get the first note for this patient)
        existing_notes = supabase_service.get_patient_notes_by_patient_id(patient_id)
        if not existing_notes:
            return jsonify({'success': False, 'error': 'Patient not found'}), 404
        
        note_id = existing_notes[0]['id']
        result = supabase_service.update_patient_note(note_id, data)
        
        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@patient_bp.route('/<int:patient_id>', methods=['DELETE'])
def delete_patient(patient_id):
    """Delete all notes for a patient"""
    try:
        # Get all notes for this patient
        notes = supabase_service.get_patient_notes_by_patient_id(patient_id)
        if not notes:
            return jsonify({'success': False, 'error': 'Patient not found'}), 404
        
        # Delete all notes for this patient
        deleted_count = 0
        for note in notes:
            supabase_service.delete_patient_note(note['id'])
            deleted_count += 1
        
        return jsonify({
            'success': True,
            'message': f'Deleted {deleted_count} notes for patient {patient_id}'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@patient_bp.route('/search', methods=['GET'])
@log_request()
@validate_query_params(q=Validator.validate_search_query)
def search_patients():
    """Search patients by note content"""
    try:
        query = getattr(request, 'validated_params', {}).get('q')
        if not query:
            return jsonify({'success': False, 'error': 'Query parameter "q" is required'}), 400
        
        results = supabase_service.search_patient_notes(query)
        return jsonify({
            'success': True,
            'data': results,
            'count': len(results)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500