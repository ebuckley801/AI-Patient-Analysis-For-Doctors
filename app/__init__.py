from flask import Flask
from app.config.config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Register blueprints
    from app.routes.patient_routes import patient_bp
    from app.routes.note_routes import note_bp
    
    app.register_blueprint(patient_bp, url_prefix='/api/patients')
    app.register_blueprint(note_bp, url_prefix='/api/notes')
    
    return app