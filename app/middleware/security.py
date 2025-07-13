from flask import request, jsonify, g
from functools import wraps
import time
import hashlib
from collections import defaultdict
from app.utils.sanitization import sanitize_request_data, SQLInjectionPrevention

class SecurityMiddleware:
    """Security middleware for request/response handling"""
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.request_validator = RequestValidator()
    
    def before_request(self):
        """Execute before each request"""
        # Rate limiting
        if not self.rate_limiter.is_allowed(request.remote_addr):
            return jsonify({
                'success': False,
                'error': 'Rate limit exceeded. Please try again later.'
            }), 429
        
        # Request validation
        validation_result = self.request_validator.validate_request(request)
        if not validation_result['valid']:
            return jsonify({
                'success': False,
                'error': validation_result['error']
            }), 400
        
        # Store start time for response timing
        g.start_time = time.time()
    
    def after_request(self, response):
        """Execute after each request"""
        # Add security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        
        # Add response time header
        if hasattr(g, 'start_time'):
            response_time = time.time() - g.start_time
            response.headers['X-Response-Time'] = f"{response_time:.3f}s"
        
        return response

class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self, max_requests=100, window_seconds=3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_ip):
        """Check if request is allowed based on rate limits"""
        current_time = time.time()
        
        # Clean old requests
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if current_time - req_time < self.window_seconds
        ]
        
        # Check if under limit
        if len(self.requests[client_ip]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[client_ip].append(current_time)
        return True

class RequestValidator:
    """Validate incoming requests for security issues"""
    
    def validate_request(self, request):
        """Validate request for security issues"""
        
        # Check for excessively large requests
        if request.content_length and request.content_length > 10 * 1024 * 1024:  # 10MB
            return {'valid': False, 'error': 'Request too large'}
        
        # Check for suspicious user agents
        user_agent = request.headers.get('User-Agent', '')
        if self._is_suspicious_user_agent(user_agent):
            return {'valid': False, 'error': 'Suspicious user agent detected'}
        
        # Validate JSON requests
        if request.is_json:
            try:
                data = request.get_json()
                if data:
                    validation_result = self._validate_json_data(data)
                    if not validation_result['valid']:
                        return validation_result
            except Exception:
                return {'valid': False, 'error': 'Invalid JSON format'}
        
        # Check query parameters
        for key, value in request.args.items():
            if SQLInjectionPrevention.contains_sql_injection(value):
                return {'valid': False, 'error': f'Suspicious query parameter: {key}'}
        
        return {'valid': True}
    
    def _is_suspicious_user_agent(self, user_agent):
        """Check for suspicious user agents"""
        suspicious_patterns = [
            'sqlmap', 'nikto', 'nmap', 'masscan', 'burp',
            'scanner', 'bot', 'crawler', 'spider'
        ]
        
        user_agent_lower = user_agent.lower()
        return any(pattern in user_agent_lower for pattern in suspicious_patterns)
    
    def _validate_json_data(self, data):
        """Validate JSON data for security issues"""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str):
                    if SQLInjectionPrevention.contains_sql_injection(value):
                        return {'valid': False, 'error': f'Suspicious content in field: {key}'}
                    
                    # Check for excessively long strings
                    if len(value) > 50000:
                        return {'valid': False, 'error': f'Field too long: {key}'}
                
                elif isinstance(value, (dict, list)):
                    nested_result = self._validate_json_data(value)
                    if not nested_result['valid']:
                        return nested_result
        
        elif isinstance(data, list):
            for item in data:
                nested_result = self._validate_json_data(item)
                if not nested_result['valid']:
                    return nested_result
        
        return {'valid': True}

def sanitize_middleware():
    """Middleware decorator to sanitize request data"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if request.is_json:
                try:
                    data = request.get_json()
                    if data:
                        sanitized_data = sanitize_request_data(data)
                        request._cached_json = (sanitized_data, True)
                except ValueError as e:
                    return jsonify({
                        'success': False,
                        'error': str(e)
                    }), 400
                except Exception:
                    return jsonify({
                        'success': False,
                        'error': 'Invalid request data'
                    }), 400
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def require_content_type(*content_types):
    """Require specific content types"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if request.content_type not in content_types:
                return jsonify({
                    'success': False,
                    'error': f'Content-Type must be one of: {", ".join(content_types)}'
                }), 415
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def log_request():
    """Log request details for security monitoring"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # In production, this would log to a proper logging system
            request_hash = hashlib.md5(
                f"{request.remote_addr}{request.method}{request.path}".encode()
            ).hexdigest()[:8]
            
            print(f"[{request_hash}] {request.method} {request.path} from {request.remote_addr}")
            
            response = f(*args, **kwargs)
            
            if hasattr(response, 'status_code'):
                print(f"[{request_hash}] Response: {response.status_code}")
            
            return response
        return decorated_function
    return decorator