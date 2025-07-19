"""
Multi-Modal Medical Data Integration API Routes

Provides REST endpoints for multi-modal healthcare data integration including:
- Cross-dataset patient similarity search
- Multi-modal data ingestion
- Patient identity resolution
- Clinical trial matching
- Comprehensive patient analysis across all data modalities

Built on Flask with comprehensive error handling and security middleware.
"""

from flask import Blueprint, request, jsonify
import logging
import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, List, Any

from app.services.multimodal_data_service import MultiModalDataService, DataIngestionResult
from app.services.multimodal_vector_service import MultiModalVectorService, ModalityType
from app.services.patient_identity_service import PatientIdentityService
from app.utils.validation import Validator, ValidationError
from app.utils.sanitization import Sanitizer
from app.middleware.security import log_request

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Blueprint for multi-modal routes
multimodal_bp = Blueprint('multimodal', __name__)

# Initialize services
try:
    multimodal_service = MultiModalDataService()
    vector_service = MultiModalVectorService()
    identity_service = PatientIdentityService()
    services_available = True
    logger.info("✅ Multi-modal services initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize multi-modal services: {e}")
    multimodal_service = None
    vector_service = None
    identity_service = None
    services_available = False

# ============================================================================
# DATA INGESTION ENDPOINTS
# ============================================================================

@multimodal_bp.route('/ingest/mimic', methods=['POST'])
@log_request()
def ingest_mimic_data():
    """
    Ingest MIMIC-IV critical care data
    
    Request body:
    {
        "data_type": "admissions|vitals|procedures",
        "data": [
            {
                "subject_id": 12345,
                "hadm_id": 67890,
                "age": 65,
                "gender": "M",
                "ethnicity": "WHITE",
                // ... additional MIMIC fields
            }
        ]
    }
    
    Returns:
    {
        "success": true,
        "data": {
            "records_processed": 100,
            "execution_time_ms": 5000,
            "errors": []
        }
    }
    """
    if not services_available:
        return jsonify({
            'success': False,
            'error': 'Multi-modal services not available',
            'code': 'SERVICE_UNAVAILABLE'
        }), 503
    
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Request must be JSON',
                'code': 'INVALID_CONTENT_TYPE'
            }), 400
        
        data = request.get_json()
        
        # Validate required fields
        if 'data_type' not in data or 'data' not in data:
            return jsonify({
                'success': False,
                'error': 'data_type and data fields are required',
                'code': 'MISSING_REQUIRED_FIELDS'
            }), 400
        
        data_type = data['data_type']
        mimic_data = data['data']
        
        # Validate data type
        valid_types = ['admissions', 'vitals', 'procedures']
        if data_type not in valid_types:
            return jsonify({
                'success': False,
                'error': f'Invalid data_type. Must be one of: {valid_types}',
                'code': 'INVALID_DATA_TYPE'
            }), 400
        
        if not isinstance(mimic_data, list) or len(mimic_data) == 0:
            return jsonify({
                'success': False,
                'error': 'data must be a non-empty array',
                'code': 'INVALID_DATA_FORMAT'
            }), 400
        
        logger.info(f"Ingesting {len(mimic_data)} MIMIC-IV {data_type} records")
        
        # Route to appropriate ingestion method
        async def ingest_data():
            if data_type == 'admissions':
                return await multimodal_service.ingest_mimic_admissions(mimic_data)
            elif data_type == 'vitals':
                return await multimodal_service.ingest_mimic_vitals(mimic_data)
            elif data_type == 'procedures':
                # Would implement procedure ingestion
                return DataIngestionResult(
                    success=False,
                    records_processed=0,
                    errors=['Procedure ingestion not yet implemented'],
                    execution_time_ms=0,
                    metadata={'data_type': data_type}
                )
        
        # Run async ingestion
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(lambda: asyncio.run(ingest_data()))
            result = future.result(timeout=300)  # 5 minute timeout
        
        return jsonify({
            'success': result.success,
            'data': {
                'records_processed': result.records_processed,
                'execution_time_ms': result.execution_time_ms,
                'errors': result.errors,
                'metadata': result.metadata
            }
        }), 200 if result.success else 500
        
    except concurrent.futures.TimeoutError:
        logger.error("MIMIC data ingestion timed out")
        return jsonify({
            'success': False,
            'error': 'Data ingestion timed out',
            'code': 'INGESTION_TIMEOUT'
        }), 504
    except Exception as e:
        logger.error(f"Error in MIMIC data ingestion: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error during data ingestion',
            'details': str(e),
            'code': 'INTERNAL_ERROR'
        }), 500

@multimodal_bp.route('/ingest/biobank', methods=['POST'])
@log_request()
def ingest_biobank_data():
    """
    Ingest UK Biobank genetic and lifestyle data
    
    Request body:
    {
        "data_type": "participants|genetics|lifestyle|diagnoses",
        "data": [
            {
                "eid": 123456,
                "birth_year": 1960,
                "sex": "Female",
                "ethnic_background": "British",
                // ... additional biobank fields
            }
        ]
    }
    """
    if not services_available:
        return jsonify({
            'success': False,
            'error': 'Multi-modal services not available',
            'code': 'SERVICE_UNAVAILABLE'
        }), 503
    
    try:
        data = request.get_json()
        
        if not data or 'data_type' not in data or 'data' not in data:
            return jsonify({
                'success': False,
                'error': 'data_type and data fields are required',
                'code': 'MISSING_REQUIRED_FIELDS'
            }), 400
        
        data_type = data['data_type']
        biobank_data = data['data']
        
        valid_types = ['participants', 'genetics', 'lifestyle', 'diagnoses']
        if data_type not in valid_types:
            return jsonify({
                'success': False,
                'error': f'Invalid data_type. Must be one of: {valid_types}',
                'code': 'INVALID_DATA_TYPE'
            }), 400
        
        logger.info(f"Ingesting {len(biobank_data)} UK Biobank {data_type} records")
        
        async def ingest_data():
            if data_type == 'participants':
                return await multimodal_service.ingest_biobank_participants(biobank_data)
            elif data_type == 'genetics':
                return await multimodal_service.ingest_biobank_genetics(biobank_data)
            else:
                # Placeholder for other biobank data types
                return DataIngestionResult(
                    success=False,
                    records_processed=0,
                    errors=[f'{data_type} ingestion not yet implemented'],
                    execution_time_ms=0,
                    metadata={'data_type': data_type}
                )
        
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(lambda: asyncio.run(ingest_data()))
            result = future.result(timeout=300)
        
        return jsonify({
            'success': result.success,
            'data': {
                'records_processed': result.records_processed,
                'execution_time_ms': result.execution_time_ms,
                'errors': result.errors,
                'metadata': result.metadata
            }
        }), 200 if result.success else 500
        
    except Exception as e:
        logger.error(f"Error in biobank data ingestion: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error during biobank ingestion',
            'details': str(e),
            'code': 'INTERNAL_ERROR'
        }), 500

@multimodal_bp.route('/ingest/faers', methods=['POST'])
@log_request()
def ingest_faers_data():
    """Ingest FDA FAERS adverse event data"""
    if not services_available:
        return jsonify({
            'success': False,
            'error': 'Multi-modal services not available',
            'code': 'SERVICE_UNAVAILABLE'
        }), 503
    
    try:
        data = request.get_json()
        faers_data = data.get('data', [])
        
        logger.info(f"Ingesting {len(faers_data)} FAERS case records")
        
        async def ingest_data():
            return await multimodal_service.ingest_faers_cases(faers_data)
        
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(lambda: asyncio.run(ingest_data()))
            result = future.result(timeout=300)
        
        return jsonify({
            'success': result.success,
            'data': {
                'records_processed': result.records_processed,
                'execution_time_ms': result.execution_time_ms,
                'errors': result.errors,
                'metadata': result.metadata
            }
        }), 200 if result.success else 500
        
    except Exception as e:
        logger.error(f"Error in FAERS data ingestion: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error during FAERS ingestion',
            'details': str(e),
            'code': 'INTERNAL_ERROR'
        }), 500

# ============================================================================
# PATIENT IDENTITY RESOLUTION
# ============================================================================

@multimodal_bp.route('/identity/resolve', methods=['POST'])
@log_request()
def resolve_patient_identity():
    """
    Resolve patient identity across datasets
    
    Request body:
    {
        "demographics": {
            "first_name": "John",
            "last_name": "Doe",
            "birth_date": "1970-01-15",
            "gender": "M"
        },
        "source_dataset": "mimic",
        "source_patient_id": "12345"
    }
    
    Returns:
    {
        "success": true,
        "data": {
            "unified_patient_id": "uuid",
            "confidence_score": 0.95,
            "matching_method": "exact",
            "conflicting_features": {}
        }
    }
    """
    if not services_available:
        return jsonify({
            'success': False,
            'error': 'Multi-modal services not available',
            'code': 'SERVICE_UNAVAILABLE'
        }), 503
    
    try:
        data = request.get_json()
        
        required_fields = ['demographics', 'source_dataset', 'source_patient_id']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'{field} is required',
                    'code': 'MISSING_REQUIRED_FIELD'
                }), 400
        
        demographics = data['demographics']
        source_dataset = Sanitizer.sanitize_text(data['source_dataset'])
        source_patient_id = Sanitizer.sanitize_text(data['source_patient_id'])
        
        # Sanitize demographics
        for key, value in demographics.items():
            if isinstance(value, str):
                demographics[key] = Sanitizer.sanitize_text(value)
        
        logger.info(f"Resolving identity for patient {source_patient_id} from {source_dataset}")
        
        # Resolve identity
        match_result = identity_service.resolve_patient_identity(
            demographics, source_dataset, source_patient_id
        )
        
        return jsonify({
            'success': True,
            'data': {
                'unified_patient_id': match_result.unified_patient_id,
                'confidence_score': match_result.confidence_score,
                'matching_method': match_result.matching_method,
                'matching_features': match_result.matching_features,
                'conflicting_features': match_result.conflicting_features
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error in patient identity resolution: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error during identity resolution',
            'details': str(e),
            'code': 'INTERNAL_ERROR'
        }), 500

@multimodal_bp.route('/identity/validate', methods=['POST'])
@log_request()
def validate_patient_identity():
    """
    Validate patient identity match
    
    Request body:
    {
        "unified_patient_id": "uuid",
        "new_demographics": {
            "first_name": "John",
            "last_name": "Doe",
            "birth_date": "1970-01-15"
        }
    }
    """
    if not services_available:
        return jsonify({
            'success': False,
            'error': 'Multi-modal services not available',
            'code': 'SERVICE_UNAVAILABLE'
        }), 503
    
    try:
        data = request.get_json()
        
        if 'unified_patient_id' not in data or 'new_demographics' not in data:
            return jsonify({
                'success': False,
                'error': 'unified_patient_id and new_demographics are required',
                'code': 'MISSING_REQUIRED_FIELDS'
            }), 400
        
        unified_patient_id = data['unified_patient_id']
        new_demographics = data['new_demographics']
        
        # Validate identity match
        is_valid, confidence_score, conflicts = identity_service.validate_identity_match(
            unified_patient_id, new_demographics
        )
        
        return jsonify({
            'success': True,
            'data': {
                'is_valid_match': is_valid,
                'confidence_score': confidence_score,
                'conflicts': conflicts,
                'validation_timestamp': datetime.now(timezone.utc).isoformat()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error in identity validation: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error during identity validation',
            'details': str(e),
            'code': 'INTERNAL_ERROR'
        }), 500

# ============================================================================
# CROSS-MODAL SIMILARITY SEARCH
# ============================================================================

@multimodal_bp.route('/similarity/patients', methods=['POST'])
@log_request()
def find_similar_patients():
    """
    Find patients similar to query patient across modalities
    
    Request body:
    {
        "query_patient_id": "uuid",
        "target_modality": "genetic_profile",
        "source_modalities": ["clinical_text", "vital_signs"],
        "top_k": 10,
        "min_similarity": 0.1
    }
    
    Returns:
    {
        "success": true,
        "data": {
            "similar_patients": [
                {
                    "patient_id": "uuid",
                    "similarity_score": 0.85,
                    "modality": "genetic_profile",
                    "data_source": "biobank",
                    "content_summary": "High cardiovascular risk profile"
                }
            ],
            "query_metadata": {
                "patient_id": "uuid",
                "modalities_searched": ["clinical_text", "vital_signs"],
                "total_results": 5
            }
        }
    }
    """
    if not services_available:
        return jsonify({
            'success': False,
            'error': 'Multi-modal services not available',
            'code': 'SERVICE_UNAVAILABLE'
        }), 503
    
    try:
        data = request.get_json()
        
        # Validate required fields
        if 'query_patient_id' not in data or 'target_modality' not in data:
            return jsonify({
                'success': False,
                'error': 'query_patient_id and target_modality are required',
                'code': 'MISSING_REQUIRED_FIELDS'
            }), 400
        
        query_patient_id = data['query_patient_id']
        target_modality_str = data['target_modality']
        source_modalities_str = data.get('source_modalities')
        top_k = min(data.get('top_k', 10), 50)  # Cap at 50
        min_similarity = max(data.get('min_similarity', 0.1), 0.0)  # Minimum 0.0
        
        # Convert string modalities to enum
        try:
            target_modality = ModalityType(target_modality_str)
        except ValueError:
            return jsonify({
                'success': False,
                'error': f'Invalid target_modality: {target_modality_str}',
                'code': 'INVALID_MODALITY'
            }), 400
        
        source_modalities = None
        if source_modalities_str:
            try:
                source_modalities = [ModalityType(mod) for mod in source_modalities_str]
            except ValueError as e:
                return jsonify({
                    'success': False,
                    'error': f'Invalid source modality: {str(e)}',
                    'code': 'INVALID_SOURCE_MODALITY'
                }), 400
        
        logger.info(f"Finding patients similar to {query_patient_id} in {target_modality_str}")
        
        # Run similarity search
        async def search_similar():
            return await vector_service.search_similar_patients(
                query_patient_id, target_modality, source_modalities, top_k, min_similarity
            )
        
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(lambda: asyncio.run(search_similar()))
            results = future.result(timeout=120)  # 2 minute timeout
        
        # Format results
        similar_patients = []
        for result in results:
            similar_patients.append({
                'patient_id': result.patient_id,
                'similarity_score': result.similarity_score,
                'modality': result.modality.value,
                'data_source': result.data_source,
                'content_summary': result.content_summary,
                'ranking': result.ranking
            })
        
        return jsonify({
            'success': True,
            'data': {
                'similar_patients': similar_patients,
                'query_metadata': {
                    'patient_id': query_patient_id,
                    'target_modality': target_modality_str,
                    'source_modalities': source_modalities_str or 'all',
                    'total_results': len(similar_patients),
                    'search_parameters': {
                        'top_k': top_k,
                        'min_similarity': min_similarity
                    },
                    'search_timestamp': datetime.now(timezone.utc).isoformat()
                }
            }
        }), 200
        
    except concurrent.futures.TimeoutError:
        return jsonify({
            'success': False,
            'error': 'Similarity search timed out',
            'code': 'SEARCH_TIMEOUT'
        }), 504
    except Exception as e:
        logger.error(f"Error in similarity search: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error during similarity search',
            'details': str(e),
            'code': 'INTERNAL_ERROR'
        }), 500

@multimodal_bp.route('/analysis/cross-modal/<patient_id>', methods=['GET'])
@log_request()
def cross_modal_patient_analysis(patient_id):
    """
    Comprehensive cross-modal analysis for a patient
    
    Query parameters:
    - include_cohorts: include similar patient cohorts (true/false) - default: true
    - include_insights: include AI-generated insights (true/false) - default: true
    
    Returns:
    {
        "success": true,
        "data": {
            "patient_id": "uuid",
            "modalities_available": ["clinical_text", "genetic_profile"],
            "cross_modal_similarities": {
                "genetic_profile": [
                    {
                        "patient_id": "uuid",
                        "similarity": 0.89,
                        "data_source": "biobank"
                    }
                ]
            },
            "risk_cohorts": {
                "genetic_clinical": ["uuid1", "uuid2"]
            },
            "clinical_insights": ["High genetic risk for cardiovascular disease"]
        }
    }
    """
    if not services_available:
        return jsonify({
            'success': False,
            'error': 'Multi-modal services not available',
            'code': 'SERVICE_UNAVAILABLE'
        }), 503
    
    try:
        # Sanitize patient ID
        patient_id = Sanitizer.sanitize_text(patient_id)
        
        # Get query parameters
        include_cohorts = request.args.get('include_cohorts', 'true').lower() == 'true'
        include_insights = request.args.get('include_insights', 'true').lower() == 'true'
        
        logger.info(f"Running cross-modal analysis for patient {patient_id}")
        
        # Run cross-modal analysis
        async def analyze_patient():
            return await vector_service.cross_modal_patient_analysis(patient_id)
        
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(lambda: asyncio.run(analyze_patient()))
            analysis = future.result(timeout=180)  # 3 minute timeout
        
        # Add query parameters to response
        analysis['query_parameters'] = {
            'include_cohorts': include_cohorts,
            'include_insights': include_insights
        }
        
        analysis['analysis_timestamp'] = datetime.now(timezone.utc).isoformat()
        
        return jsonify({
            'success': True,
            'data': analysis
        }), 200
        
    except concurrent.futures.TimeoutError:
        return jsonify({
            'success': False,
            'error': 'Cross-modal analysis timed out',
            'code': 'ANALYSIS_TIMEOUT'
        }), 504
    except Exception as e:
        logger.error(f"Error in cross-modal analysis: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error during cross-modal analysis',
            'details': str(e),
            'code': 'INTERNAL_ERROR'
        }), 500

# ============================================================================
# CLINICAL TRIALS MATCHING
# ============================================================================

@multimodal_bp.route('/trials/fetch', methods=['POST'])
@log_request()
def fetch_clinical_trials():
    """
    Fetch clinical trials from ClinicalTrials.gov API
    
    Request body:
    {
        "conditions": ["diabetes", "cardiovascular disease"],
        "status": "RECRUITING",
        "max_results": 100
    }
    """
    if not services_available:
        return jsonify({
            'success': False,
            'error': 'Multi-modal services not available',
            'code': 'SERVICE_UNAVAILABLE'
        }), 503
    
    try:
        data = request.get_json()
        
        conditions = data.get('conditions', [])
        status = data.get('status', 'RECRUITING')
        max_results = min(data.get('max_results', 100), 1000)  # Cap at 1000
        
        if not conditions:
            return jsonify({
                'success': False,
                'error': 'At least one condition is required',
                'code': 'MISSING_CONDITIONS'
            }), 400
        
        logger.info(f"Fetching clinical trials for conditions: {conditions}")
        
        # Fetch trials
        async def fetch_trials():
            trials = await multimodal_service.fetch_clinical_trials(conditions, status)
            # Ingest the trials data
            if trials:
                result = await multimodal_service.ingest_clinical_trials(trials[:max_results])
                return {
                    'trials_fetched': len(trials),
                    'trials_ingested': result.records_processed,
                    'ingestion_errors': result.errors
                }
            return {'trials_fetched': 0, 'trials_ingested': 0, 'ingestion_errors': []}
        
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(lambda: asyncio.run(fetch_trials()))
            result = future.result(timeout=300)  # 5 minute timeout
        
        return jsonify({
            'success': True,
            'data': result
        }), 200
        
    except concurrent.futures.TimeoutError:
        return jsonify({
            'success': False,
            'error': 'Clinical trials fetch timed out',
            'code': 'FETCH_TIMEOUT'
        }), 504
    except Exception as e:
        logger.error(f"Error fetching clinical trials: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error during trials fetch',
            'details': str(e),
            'code': 'INTERNAL_ERROR'
        }), 500

@multimodal_bp.route('/trials/match/<patient_id>', methods=['POST'])
@log_request()
def match_patient_to_trials(patient_id):
    """
    Match patient to relevant clinical trials
    
    Request body:
    {
        "session_id": "analysis_session_uuid",
        "trial_types": ["interventional", "observational"],
        "max_matches": 10
    }
    """
    if not services_available:
        return jsonify({
            'success': False,
            'error': 'Multi-modal services not available',
            'code': 'SERVICE_UNAVAILABLE'
        }), 503
    
    try:
        data = request.get_json()
        
        if 'session_id' not in data:
            return jsonify({
                'success': False,
                'error': 'session_id is required',
                'code': 'MISSING_SESSION_ID'
            }), 400
        
        session_id = data['session_id']
        trial_types = data.get('trial_types', ['interventional'])
        max_matches = min(data.get('max_matches', 10), 50)  # Cap at 50
        
        # Sanitize inputs
        patient_id = Sanitizer.sanitize_text(patient_id)
        session_id = Sanitizer.sanitize_text(session_id)
        
        logger.info(f"Matching patient {patient_id} to clinical trials")
        
        # Match patient to trials
        async def match_trials():
            return await multimodal_service.match_patient_to_trials(patient_id, session_id)
        
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(lambda: asyncio.run(match_trials()))
            matches = future.result(timeout=120)  # 2 minute timeout
        
        # Filter by trial types and limit results
        filtered_matches = [
            match for match in matches[:max_matches]
            # Would add trial type filtering here based on stored trial data
        ]
        
        return jsonify({
            'success': True,
            'data': {
                'patient_id': patient_id,
                'session_id': session_id,
                'trial_matches': filtered_matches,
                'total_matches': len(filtered_matches),
                'matching_timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except concurrent.futures.TimeoutError:
        return jsonify({
            'success': False,
            'error': 'Trial matching timed out',
            'code': 'MATCHING_TIMEOUT'
        }), 504
    except Exception as e:
        logger.error(f"Error matching trials: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error during trial matching',
            'details': str(e),
            'code': 'INTERNAL_ERROR'
        }), 500

# ============================================================================
# SERVICE STATUS AND STATISTICS
# ============================================================================

@multimodal_bp.route('/health', methods=['GET'])
@log_request()
def multimodal_health_check():
    """Health check for multi-modal services"""
    try:
        health_status = {
            'status': 'healthy' if services_available else 'degraded',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'services': {
                'multimodal_data_service': 'available' if multimodal_service else 'unavailable',
                'vector_service': 'available' if vector_service else 'unavailable',
                'identity_service': 'available' if identity_service else 'unavailable'
            }
        }
        
        if services_available:
            # Get service statistics
            try:
                vector_stats = vector_service.get_service_stats()
                health_status['service_stats'] = vector_stats
            except Exception as e:
                health_status['service_stats'] = {'error': str(e)}
        
        return jsonify(health_status), 200 if services_available else 503
        
    except Exception as e:
        logger.error(f"Multi-modal health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'error': str(e)
        }), 500

@multimodal_bp.route('/stats', methods=['GET'])
@log_request()
def get_multimodal_stats():
    """Get comprehensive multi-modal service statistics"""
    if not services_available:
        return jsonify({
            'success': False,
            'error': 'Multi-modal services not available',
            'code': 'SERVICE_UNAVAILABLE'
        }), 503
    
    try:
        stats = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'vector_service': vector_service.get_service_stats() if vector_service else {},
            'available_modalities': [modality.value for modality in ModalityType],
            'service_versions': {
                'multimodal_data_service': '1.0.0',
                'vector_service': '1.0.0',
                'identity_service': '1.0.0'
            }
        }
        
        return jsonify({
            'success': True,
            'data': stats
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting multi-modal stats: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error getting statistics',
            'details': str(e),
            'code': 'INTERNAL_ERROR'
        }), 500

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@multimodal_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Multi-modal endpoint not found',
        'code': 'NOT_FOUND'
    }), 404

@multimodal_bp.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'success': False,
        'error': 'Method not allowed',
        'code': 'METHOD_NOT_ALLOWED'
    }), 405

@multimodal_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error in multi-modal service',
        'code': 'INTERNAL_ERROR'
    }), 500