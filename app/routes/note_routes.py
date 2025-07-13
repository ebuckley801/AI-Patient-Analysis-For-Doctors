from flask import Blueprint, request, jsonify
from app.services.supabase_service import SupabaseService

note_bp = Blueprint('notes', __name__)
supabase_service = SupabaseService()

@note_bp.route('/', methods=['GET'])
def get_notes():
    """Get all notes with pagination"""
    try:
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        notes = supabase_service.get_patient_notes(limit=limit, offset=offset)
        return jsonify({
            'success': True,
            'data': notes,
            'count': len(notes)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@note_bp.route('/<int:note_id>', methods=['GET'])
def get_note(note_id):
    """Get a specific note by ID"""
    try:
        note = supabase_service.get_patient_note_by_id(note_id)
        if not note:
            return jsonify({'success': False, 'error': 'Note not found'}), 404
        
        return jsonify({
            'success': True,
            'data': note
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@note_bp.route('/', methods=['POST'])
def create_note():
    """Create a new note"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['patient_id', 'patient_uid', 'patient_note', 'age', 'gender']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing field: {field}'}), 400
        
        result = supabase_service.create_patient_note(data)
        
        return jsonify({
            'success': True,
            'data': result
        }), 201
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@note_bp.route('/<int:note_id>', methods=['PUT'])
def update_note(note_id):
    """Update a specific note"""
    try:
        data = request.get_json()
        
        # Check if note exists
        existing_note = supabase_service.get_patient_note_by_id(note_id)
        if not existing_note:
            return jsonify({'success': False, 'error': 'Note not found'}), 404
        
        result = supabase_service.update_patient_note(note_id, data)
        
        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@note_bp.route('/<int:note_id>', methods=['DELETE'])
def delete_note(note_id):
    """Delete a specific note"""
    try:
        # Check if note exists
        existing_note = supabase_service.get_patient_note_by_id(note_id)
        if not existing_note:
            return jsonify({'success': False, 'error': 'Note not found'}), 404
        
        result = supabase_service.delete_patient_note(note_id)
        
        return jsonify({
            'success': True,
            'message': f'Note {note_id} deleted successfully'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@note_bp.route('/search', methods=['GET'])
def search_notes():
    """Search notes by content"""
    try:
        query = request.args.get('q', '')
        field = request.args.get('field', 'patient_note')
        
        if not query:
            return jsonify({'success': False, 'error': 'Query parameter "q" is required'}), 400
        
        # Validate field parameter
        allowed_fields = ['patient_note', 'patient_uid', 'gender']
        if field not in allowed_fields:
            return jsonify({'success': False, 'error': f'Invalid field. Allowed: {allowed_fields}'}), 400
        
        results = supabase_service.search_patient_notes(query, field)
        return jsonify({
            'success': True,
            'data': results,
            'count': len(results)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@note_bp.route('/patient/<int:patient_id>', methods=['GET'])
def get_notes_by_patient(patient_id):
    """Get all notes for a specific patient"""
    try:
        notes = supabase_service.get_patient_notes_by_patient_id(patient_id)
        return jsonify({
            'success': True,
            'data': notes,
            'count': len(notes)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500