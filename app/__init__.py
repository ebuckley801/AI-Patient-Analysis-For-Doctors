from flask import Flask
from app.config.config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Register blueprints
    from app.routes.patient_routes import patient_bp
    from app.routes.note_routes import note_bp
    from app.routes.analysis_routes import analysis_bp
    from app.routes.explanation_routes import explanation_bp
    from app.routes.multimodal_routes import multimodal_bp
    
    app.register_blueprint(patient_bp, url_prefix='/api/patients')
    app.register_blueprint(note_bp, url_prefix='/api/notes')
    app.register_blueprint(analysis_bp, url_prefix='/api/analysis')
    app.register_blueprint(explanation_bp, url_prefix='/api/explanation')
    app.register_blueprint(multimodal_bp, url_prefix='/api/multimodal')
    
    return app