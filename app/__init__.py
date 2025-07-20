from flask import Flask
from flask_restx import Api
from flask_jwt_extended import JWTManager
from flask_cors import CORS
from app.config.config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    jwt = JWTManager(app)

    # Configure CORS with more permissive settings for development
    CORS(app, 
         origins=[
             'http://localhost:3000',  # Next.js development server
             'http://127.0.0.1:3000',
         ],
         supports_credentials=True,
         allow_headers=['Content-Type', 'Authorization'],
         methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
         expose_headers=['Authorization'])

    api = Api(app, 
              version='1.0', 
              title='Patient Analysis API', 
              description='A comprehensive API for patient data analysis and clinical decision support.',
              doc='/docs')
    
    # Import and add namespaces
    from app.routes.patient_routes import patient_ns
    from app.routes.note_routes import note_ns
    from app.routes.analysis_routes import analysis_ns
    from app.routes.explanation_routes import explanation_ns
    from app.routes.multimodal_routes import multimodal_ns
    from app.routes.unified_patient_routes import unified_ns
    from app.routes.auth_routes import auth_ns

    api.add_namespace(auth_ns, path='/api/auth')
    api.add_namespace(patient_ns, path='/api/patients')
    api.add_namespace(note_ns, path='/api/notes')
    api.add_namespace(analysis_ns, path='/api/analysis')
    api.add_namespace(explanation_ns, path='/api/explanation')
    api.add_namespace(multimodal_ns, path='/api/multimodal')
    api.add_namespace(unified_ns, path='/api/unified-patient')
    
    # Additional CORS handler for preflight requests
    @app.after_request
    def after_request(response):
        response.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
        response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        return response
    
    return app