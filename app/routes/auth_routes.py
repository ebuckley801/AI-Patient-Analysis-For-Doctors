from flask import request
from flask_restx import Namespace, Resource, fields
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from app.services.supabase_service import SupabaseService
import logging

logger = logging.getLogger(__name__)

auth_ns = Namespace('auth', description='Authentication operations')
supabase_service = SupabaseService()

# Models for API documentation
user_model = auth_ns.model('User', {
    'id': fields.String(readOnly=True, description='The user unique identifier'),
    'email': fields.String(required=True, description='The user\'s email address'),
    'role': fields.String(readOnly=True, description='The user\'s role', default='user'),
    'created_at': fields.DateTime(readOnly=True, description='The timestamp of user creation')
})

register_model = auth_ns.model('Register', {
    'email': fields.String(required=True, description='The user\'s email address'),
    'password': fields.String(required=True, description='The user\'s password', min_length=6)
})

login_model = auth_ns.model('Login', {
    'email': fields.String(required=True, description='The user\'s email address'),
    'password': fields.String(required=True, description='The user\'s password')
})

token_model = auth_ns.model('Token', {
    'access_token': fields.String(required=True, description='JWT Access Token'),
    'user': fields.Nested(user_model, description='Authenticated user details')
})

@auth_ns.route('/register')
class UserRegister(Resource):
    def options(self):
        """Handle preflight OPTIONS request"""
        return {}, 200
    
    @auth_ns.doc('register_user')
    @auth_ns.expect(register_model, validate=True)
    @auth_ns.marshal_with(user_model, code=201)
    def post(self):
        """Register a new user"""
        data = auth_ns.payload
        email = data['email']
        password = data['password']

        # Check if user already exists
        existing_user = supabase_service.get_user_by_email(email)
        if existing_user:
            auth_ns.abort(409, message='User with that email already exists')

        # Hash password
        hashed_password = generate_password_hash(password)

        # Store user in Supabase
        user_data = {
            'email': email,
            'password_hash': hashed_password,
            'role': 'user' # Default role
        }
        new_user = supabase_service.create_user(user_data)

        if not new_user:
            auth_ns.abort(500, message='Failed to register user')

        return new_user, 201

@auth_ns.route('/login')
class UserLogin(Resource):
    def options(self):
        """Handle preflight OPTIONS request"""
        return {}, 200
    
    @auth_ns.doc('login_user')
    @auth_ns.expect(login_model, validate=True)
    @auth_ns.marshal_with(token_model)
    def post(self):
        """Login a user and return access token"""
        data = auth_ns.payload
        email = data['email']
        password = data['password']

        user = supabase_service.get_user_by_email(email)

        if not user or not check_password_hash(user['password_hash'], password):
            auth_ns.abort(401, message='Invalid credentials')

        access_token = create_access_token(identity=user['id'])
        return {'access_token': access_token, 'user': user}, 200

@auth_ns.route('/protected')
class ProtectedResource(Resource):
    @auth_ns.doc('protected_resource')
    @jwt_required()
    def get(self):
        """Access a protected resource (requires JWT)"""
        current_user_id = get_jwt_identity()
        user = supabase_service.get_user_by_id(current_user_id)
        if not user:
            auth_ns.abort(404, message='User not found')
        return {'message': f'Hello, {user['email']}! You have access to this protected resource.', 'user_id': current_user_id}, 200

# Response models for error documentation
error_model = auth_ns.model('Error', {
    'success': fields.Boolean(default=False),
    'error': fields.String(description='Error message'),
    'code': fields.String(description='Error code')
})