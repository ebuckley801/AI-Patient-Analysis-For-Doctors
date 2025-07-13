from flask import Blueprint, request, jsonify
from app.services.supabase_service import SupabaseService
from app.models.patient import Patient

patient_bp = Blueprint('patients', __name__)
supabase_service = SupabaseService()

@patient_bp.route('/', methods=['GET'])
def get_patients():
    """Get all patients with pagination"""
    try:
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        patients = supabase_service.get_patient_notes(limit=limit, offset=offset)
        return jsonify({
            'success': True,
            'data': patients,
            'count': len(patients)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@patient_bp.route('/<int:patient_id>', methods=['GET'])
def get_patient(patient_id):
    """Get all notes for a specific patient"""
    try:
        notes = supabase_service.get_patient_notes_by_patient_id(patient_id)
        if not notes:
            return jsonify({'success': False, 'error': 'Patient not found'}), 404
        
        return jsonify({
            'success': True,
            'data': notes
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@patient_bp.route('/', methods=['POST'])
def create_patient():
    """Create a new patient note"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['patient_id', 'patient_uid', 'patient_note', 'age', 'gender']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing field: {field}'}), 400
        
        # Create patient object
        patient = Patient.from_dict(data)
        
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
def search_patients():
    """Search patients by note content"""
    try:
        query = request.args.get('q', '')
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