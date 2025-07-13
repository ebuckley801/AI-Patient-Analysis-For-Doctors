import re
from typing import Dict, Any, List, Optional
from functools import wraps
from flask import request, jsonify

class ValidationError(Exception):
    def __init__(self, message: str, field: str = None):
        self.message = message
        self.field = field
        super().__init__(self.message)

class Validator:
    @staticmethod
    def validate_patient_id(patient_id: Any) -> int:
        """Validate patient_id is a positive integer"""
        if not isinstance(patient_id, (int, str)):
            raise ValidationError("Patient ID must be a number", "patient_id")
        
        try:
            patient_id = int(patient_id)
        except (ValueError, TypeError):
            raise ValidationError("Patient ID must be a valid integer", "patient_id")
        
        if patient_id <= 0:
            raise ValidationError("Patient ID must be positive", "patient_id")
        
        return patient_id
    
    @staticmethod
    def validate_patient_uid(patient_uid: Any) -> str:
        """Validate patient_uid format"""
        if not isinstance(patient_uid, str):
            raise ValidationError("Patient UID must be a string", "patient_uid")
        
        patient_uid = patient_uid.strip()
        
        if not patient_uid:
            raise ValidationError("Patient UID cannot be empty", "patient_uid")
        
        if len(patient_uid) > 100:
            raise ValidationError("Patient UID cannot exceed 100 characters", "patient_uid")
        
        # Check for valid UID format (alphanumeric, hyphens, underscores)
        if not re.match(r'^[a-zA-Z0-9_-]+$', patient_uid):
            raise ValidationError("Patient UID can only contain letters, numbers, hyphens, and underscores", "patient_uid")
        
        return patient_uid
    
    @staticmethod
    def validate_patient_note(patient_note: Any) -> str:
        """Validate patient note content"""
        if not isinstance(patient_note, str):
            raise ValidationError("Patient note must be a string", "patient_note")
        
        patient_note = patient_note.strip()
        
        if not patient_note:
            raise ValidationError("Patient note cannot be empty", "patient_note")
        
        if len(patient_note) > 10000:
            raise ValidationError("Patient note cannot exceed 10,000 characters", "patient_note")
        
        return patient_note
    
    @staticmethod
    def validate_age(age: Any) -> int:
        """Validate age is within reasonable range"""
        if not isinstance(age, (int, str)):
            raise ValidationError("Age must be a number", "age")
        
        try:
            age = int(age)
        except (ValueError, TypeError):
            raise ValidationError("Age must be a valid integer", "age")
        
        if age < 0 or age > 150:
            raise ValidationError("Age must be between 0 and 150", "age")
        
        return age
    
    @staticmethod
    def validate_gender(gender: Any) -> str:
        """Validate gender is valid option"""
        if not isinstance(gender, str):
            raise ValidationError("Gender must be a string", "gender")
        
        gender = gender.strip().upper()
        
        valid_genders = ['M', 'F', 'MALE', 'FEMALE', 'OTHER', 'NON-BINARY', 'PREFER_NOT_TO_SAY']
        
        if gender not in valid_genders:
            raise ValidationError(f"Gender must be one of: {', '.join(valid_genders)}", "gender")
        
        # Normalize to single letter for consistency
        if gender in ['MALE']:
            return 'M'
        elif gender in ['FEMALE']:
            return 'F'
        
        return gender
    
    @staticmethod
    def validate_search_query(query: Any) -> str:
        """Validate search query"""
        if not isinstance(query, str):
            raise ValidationError("Search query must be a string", "query")
        
        query = query.strip()
        
        if not query:
            raise ValidationError("Search query cannot be empty", "query")
        
        if len(query) < 2:
            raise ValidationError("Search query must be at least 2 characters", "query")
        
        if len(query) > 500:
            raise ValidationError("Search query cannot exceed 500 characters", "query")
        
        return query
    
    @staticmethod
    def validate_pagination(limit: Any, offset: Any) -> tuple[int, int]:
        """Validate pagination parameters"""
        try:
            limit = int(limit) if limit is not None else 100
            offset = int(offset) if offset is not None else 0
        except (ValueError, TypeError):
            raise ValidationError("Limit and offset must be valid integers")
        
        if limit <= 0 or limit > 1000:
            raise ValidationError("Limit must be between 1 and 1000")
        
        if offset < 0:
            raise ValidationError("Offset must be non-negative")
        
        return limit, offset

class PatientNoteSchema:
    """Schema validation for patient note data"""
    
    @staticmethod
    def validate_create_request(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate patient note creation request"""
        if not isinstance(data, dict):
            raise ValidationError("Request body must be a JSON object")
        
        required_fields = ['patient_id', 'patient_uid', 'patient_note', 'age', 'gender']
        
        # Check for required fields
        for field in required_fields:
            if field not in data:
                raise ValidationError(f"Missing required field: {field}", field)
        
        # Validate each field
        validated_data = {
            'patient_id': Validator.validate_patient_id(data['patient_id']),
            'patient_uid': Validator.validate_patient_uid(data['patient_uid']),
            'patient_note': Validator.validate_patient_note(data['patient_note']),
            'age': Validator.validate_age(data['age']),
            'gender': Validator.validate_gender(data['gender'])
        }
        
        return validated_data
    
    @staticmethod
    def validate_update_request(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate patient note update request"""
        if not isinstance(data, dict):
            raise ValidationError("Request body must be a JSON object")
        
        if not data:
            raise ValidationError("Update request cannot be empty")
        
        validated_data = {}
        
        # Validate only provided fields
        if 'patient_id' in data:
            validated_data['patient_id'] = Validator.validate_patient_id(data['patient_id'])
        
        if 'patient_uid' in data:
            validated_data['patient_uid'] = Validator.validate_patient_uid(data['patient_uid'])
        
        if 'patient_note' in data:
            validated_data['patient_note'] = Validator.validate_patient_note(data['patient_note'])
        
        if 'age' in data:
            validated_data['age'] = Validator.validate_age(data['age'])
        
        if 'gender' in data:
            validated_data['gender'] = Validator.validate_gender(data['gender'])
        
        return validated_data

def validate_json_request(schema_validator):
    """Decorator to validate JSON request body"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                data = request.get_json()
                if data is None:
                    return jsonify({
                        'success': False,
                        'error': 'Request body must be valid JSON'
                    }), 400
                
                validated_data = schema_validator(data)
                request.validated_data = validated_data
                return f(*args, **kwargs)
                
            except ValidationError as e:
                return jsonify({
                    'success': False,
                    'error': e.message,
                    'field': e.field
                }), 400
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': 'Invalid request data'
                }), 400
        
        return decorated_function
    return decorator

def validate_query_params(**param_validators):
    """Decorator to validate query parameters"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                validated_params = {}
                
                for param_name, validator in param_validators.items():
                    param_value = request.args.get(param_name)
                    if param_value is not None:
                        validated_params[param_name] = validator(param_value)
                
                request.validated_params = validated_params
                return f(*args, **kwargs)
                
            except ValidationError as e:
                return jsonify({
                    'success': False,
                    'error': e.message,
                    'field': e.field
                }), 400
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': 'Invalid query parameters'
                }), 400
        
        return decorated_function
    return decorator