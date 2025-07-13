import html
import re
import bleach
from typing import Any, Dict, List, Optional

class Sanitizer:
    """Data sanitization utilities for patient data"""
    
    # Allowed HTML tags for patient notes (very restricted)
    ALLOWED_TAGS = ['p', 'br', 'strong', 'em', 'ul', 'ol', 'li']
    ALLOWED_ATTRIBUTES = {}
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """Sanitize text input by removing harmful content"""
        if not isinstance(text, str):
            return str(text)
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        # HTML escape to prevent XSS
        text = html.escape(text)
        
        return text
    
    @staticmethod
    def sanitize_html(text: str) -> str:
        """Sanitize HTML content for patient notes"""
        if not isinstance(text, str):
            return str(text)
        
        # Use bleach to clean HTML
        cleaned = bleach.clean(
            text,
            tags=Sanitizer.ALLOWED_TAGS,
            attributes=Sanitizer.ALLOWED_ATTRIBUTES,
            strip=True
        )
        
        return cleaned.strip()
    
    @staticmethod
    def sanitize_patient_uid(uid: str) -> str:
        """Sanitize patient UID"""
        if not isinstance(uid, str):
            return str(uid)
        
        # Remove any non-alphanumeric characters except hyphens and underscores
        uid = re.sub(r'[^a-zA-Z0-9_-]', '', uid)
        
        # Limit length
        uid = uid[:100]
        
        return uid.strip()
    
    @staticmethod
    def sanitize_search_query(query: str) -> str:
        """Sanitize search query to prevent injection attacks"""
        if not isinstance(query, str):
            return str(query)
        
        # Remove special characters that could be used for injection
        query = re.sub(r'[<>"\';(){}[\]\\]', '', query)
        
        # Normalize whitespace
        query = re.sub(r'\s+', ' ', query)
        
        # Limit length
        query = query[:500]
        
        return query.strip()
    
    @staticmethod
    def sanitize_patient_note(note: str) -> str:
        """Sanitize patient note content"""
        if not isinstance(note, str):
            return str(note)
        
        # Allow some basic HTML formatting but sanitize it
        note = Sanitizer.sanitize_html(note)
        
        # Remove excessive newlines
        note = re.sub(r'\n{3,}', '\n\n', note)
        
        # Limit length
        note = note[:10000]
        
        return note.strip()
    
    @staticmethod
    def sanitize_integer(value: Any, min_val: int = None, max_val: int = None) -> Optional[int]:
        """Sanitize integer values with optional range validation"""
        if value is None:
            return None
        
        try:
            value = int(value)
        except (ValueError, TypeError):
            return None
        
        if min_val is not None and value < min_val:
            return min_val
        
        if max_val is not None and value > max_val:
            return max_val
        
        return value
    
    @staticmethod
    def sanitize_gender(gender: str) -> str:
        """Sanitize gender input"""
        if not isinstance(gender, str):
            return str(gender)
        
        # Remove any special characters
        gender = re.sub(r'[^a-zA-Z_-]', '', gender)
        
        # Convert to uppercase for consistency
        gender = gender.upper().strip()
        
        # Map common variations
        gender_mapping = {
            'MALE': 'M',
            'FEMALE': 'F',
            'MAN': 'M',
            'WOMAN': 'F'
        }
        
        return gender_mapping.get(gender, gender)
    
    @staticmethod
    def sanitize_patient_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize complete patient data object"""
        sanitized = {}
        
        if 'patient_id' in data:
            sanitized['patient_id'] = Sanitizer.sanitize_integer(
                data['patient_id'], min_val=1, max_val=999999999
            )
        
        if 'patient_uid' in data:
            sanitized['patient_uid'] = Sanitizer.sanitize_patient_uid(data['patient_uid'])
        
        if 'patient_note' in data:
            sanitized['patient_note'] = Sanitizer.sanitize_patient_note(data['patient_note'])
        
        if 'age' in data:
            sanitized['age'] = Sanitizer.sanitize_integer(
                data['age'], min_val=0, max_val=150
            )
        
        if 'gender' in data:
            sanitized['gender'] = Sanitizer.sanitize_gender(data['gender'])
        
        # Remove None values
        sanitized = {k: v for k, v in sanitized.items() if v is not None}
        
        return sanitized
    
    @staticmethod
    def sanitize_response_data(data: Any) -> Any:
        """Sanitize response data before sending to client"""
        if isinstance(data, dict):
            return {key: Sanitizer.sanitize_response_data(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [Sanitizer.sanitize_response_data(item) for item in data]
        elif isinstance(data, str):
            # Light sanitization for response data (preserve formatting)
            return html.escape(data)
        else:
            return data

class SQLInjectionPrevention:
    """Utilities to prevent SQL injection attacks"""
    
    # Common SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
        r"(--|/\*|\*/)",
        r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
        r"(\')(.*?)(\s+OR\s+.*?)(\s*--)",
        r"(\')(.*?)(\s+UNION\s+.*?)(\s*--)"
    ]
    
    @staticmethod
    def contains_sql_injection(text: str) -> bool:
        """Check if text contains potential SQL injection patterns"""
        if not isinstance(text, str):
            return False
        
        text_upper = text.upper()
        
        for pattern in SQLInjectionPrevention.SQL_INJECTION_PATTERNS:
            if re.search(pattern, text_upper, re.IGNORECASE):
                return True
        
        return False
    
    @staticmethod
    def sanitize_sql_input(text: str) -> str:
        """Remove potential SQL injection patterns"""
        if not isinstance(text, str):
            return str(text)
        
        # Remove common SQL keywords and patterns
        for pattern in SQLInjectionPrevention.SQL_INJECTION_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()

def sanitize_request_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Main function to sanitize all request data"""
    if not isinstance(data, dict):
        return {}
    
    # First check for SQL injection attempts
    for key, value in data.items():
        if isinstance(value, str) and SQLInjectionPrevention.contains_sql_injection(value):
            raise ValueError(f"Potential SQL injection detected in field: {key}")
    
    # Then sanitize the data
    return Sanitizer.sanitize_patient_data(data)