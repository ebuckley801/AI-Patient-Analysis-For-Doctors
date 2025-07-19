from flask import request
import logging
import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, List, Any
from flask_restx import Namespace, Resource, fields
from flask_jwt_extended import jwt_required, get_jwt_identity # Import jwt_required

from app.services.multimodal_data_service import MultiModalDataService, DataIngestionResult
from app.services.multimodal_vector_service import MultiModalVectorService, ModalityType
from app.services.patient_identity_service import PatientIdentityService
from app.utils.validation import Validator, ValidationError
from app.utils.sanitization import Sanitizer
from app.middleware.security import log_request

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Namespace for multi-modal routes
multimodal_ns = Namespace('multimodal', description='Multi-modal medical data integration operations')

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

# --- Models for API Documentation ---

# Data Ingestion Models
ingestion_data_item_model = multimodal_ns.model('IngestionDataItem', {
    'subject_id': fields.Integer(description='Subject ID'),
    'hadm_id': fields.Integer(description='Hospital Admission ID'),
    'age': fields.Integer(description='Age of patient'),
    'gender': fields.String(description='Gender of patient'),
    'ethnicity': fields.String(description='Ethnicity of patient'),
    fields.Wildcard: fields.Raw(description='Additional data fields')
})

ingest_request_model = multimodal_ns.model('IngestRequest', {
    'data_type': fields.String(required=True, description='Type of data to ingest (e.g., admissions, vitals, procedures)', enum=['admissions', 'vitals', 'procedures', 'participants', 'genetics', 'lifestyle', 'diagnoses']),
    'data': fields.List(fields.Nested(ingestion_data_item_model), required=True, description='List of data records to ingest')
})

ingestion_result_model = multimodal_ns.model('IngestionResult', {
    'records_processed': fields.Integer,
    'execution_time_ms': fields.Float,
    'errors': fields.List(fields.String),
    'metadata': fields.Raw
})

# Identity Resolution Models
demographics_model = multimodal_ns.model('Demographics', {
    'first_name': fields.String,
    'last_name': fields.String,
    'birth_date': fields.String(description='YYYY-MM-DD'),
    'gender': fields.String
})

resolve_identity_request_model = multimodal_ns.model('ResolveIdentityRequest', {
    'demographics': fields.Nested(demographics_model, required=True),
    'source_dataset': fields.String(required=True, description='Source dataset (e.g., mimic, biobank)'),
    'source_patient_id': fields.String(required=True, description='Patient ID in the source dataset')
})

identity_match_result_model = multimodal_ns.model('IdentityMatchResult', {
    'unified_patient_id': fields.String,
    'confidence_score': fields.Float,
    'matching_method': fields.String,
    'matching_features': fields.Raw,
    'conflicting_features': fields.Raw
})

validate_identity_request_model = multimodal_ns.model('ValidateIdentityRequest', {
    'unified_patient_id': fields.String(required=True),
    'new_demographics': fields.Nested(demographics_model, required=True)
})

validate_identity_response_model = multimodal_ns.model('ValidateIdentityResponse', {
    'is_valid_match': fields.Boolean,
    'confidence_score': fields.Float,
    'conflicts': fields.Raw,
    'validation_timestamp': fields.String
})

# Similarity Search Models
similar_patient_item_model = multimodal_ns.model('SimilarPatientItem', {
    'patient_id': fields.String,
    'similarity_score': fields.Float,
    'modality': fields.String,
    'data_source': fields.String,
    'content_summary': fields.String,
    'ranking': fields.Integer
})

similar_patients_query_metadata_model = multimodal_ns.model('SimilarPatientsQueryMetadata', {
    'patient_id': fields.String,
    'target_modality': fields.String,
    'source_modalities': fields.Raw,
    'total_results': fields.Integer,
    'search_parameters': fields.Raw,
    'search_timestamp': fields.String
})

find_similar_patients_request_model = multimodal_ns.model('FindSimilarPatientsRequest', {
    'query_patient_id': fields.String(required=True),
    'target_modality': fields.String(required=True, enum=[mod.value for mod in ModalityType])
    'source_modalities': fields.List(fields.String(enum=[mod.value for mod in ModalityType])),
    'top_k': fields.Integer(default=10),
    'min_similarity': fields.Float(default=0.1)
})

find_similar_patients_response_model = multimodal_ns.model('FindSimilarPatientsResponse', {
    'similar_patients': fields.List(fields.Nested(similar_patient_item_model)),
    'query_metadata': fields.Nested(similar_patients_query_metadata_model)
})

# Cross-Modal Analysis Models
cross_modal_analysis_response_model = multimodal_ns.model('CrossModalAnalysisResponse', {
    'patient_id': fields.String,
    'modalities_available': fields.List(fields.String),
    'cross_modal_similarities': fields.Raw,
    'risk_cohorts': fields.Raw,
    'clinical_insights': fields.List(fields.String),
    'query_parameters': fields.Raw,
    'analysis_timestamp': fields.String
})

# Clinical Trials Models
fetch_trials_request_model = multimodal_ns.model('FetchTrialsRequest', {
    'conditions': fields.List(fields.String, required=True),
    'status': fields.String(default='RECRUITING'),
    'max_results': fields.Integer(default=100)
})

fetch_trials_response_model = multimodal_ns.model('FetchTrialsResponse', {
    'trials_fetched': fields.Integer,
    'trials_ingested': fields.Integer,
    'ingestion_errors': fields.List(fields.String)
})

match_trials_request_model = multimodal_ns.model('MatchTrialsRequest', {
    'session_id': fields.String(required=True),
    'trial_types': fields.List(fields.String, default=['interventional']),
    'max_matches': fields.Integer(default=10)
})

trial_match_item_model = multimodal_ns.model('TrialMatchItem', {
    'nct_id': fields.String,
    'title': fields.String,
    'match_score': fields.Float,
    'eligibility_status': fields.String,
    'reasoning': fields.String
})

match_trials_response_model = multimodal_ns.model('MatchTrialsResponse', {
    'patient_id': fields.String,
    'session_id': fields.String,
    'trial_matches': fields.List(fields.Nested(trial_match_item_model)),
    'total_matches': fields.Integer,
    'matching_timestamp': fields.String
})

# Health and Stats Models
service_status_model = multimodal_ns.model('ServiceStatus', {
    'multimodal_data_service': fields.String,
    'vector_service': fields.String,
    'identity_service': fields.String
})

multimodal_health_response_model = multimodal_ns.model('MultimodalHealthResponse', {
    'status': fields.String,
    'timestamp': fields.String,
    'services': fields.Nested(service_status_model),
    'service_stats': fields.Raw(description='Detailed statistics for services if available')
})

multimodal_stats_response_model = multimodal_ns.model('MultimodalStatsResponse', {
    'timestamp': fields.String,
    'vector_service': fields.Raw,
    'available_modalities': fields.List(fields.String),
    'service_versions': fields.Raw
})


# ============================================================================
# DATA INGESTION ENDPOINTS
# ============================================================================

@multimodal_ns.route('/ingest/mimic')
class IngestMimicData(Resource):
    @multimodal_ns.doc('ingest_mimic_data')
    @multimodal_ns.expect(ingest_request_model, validate=True)
    @multimodal_ns.marshal_with(ingestion_result_model)
    @log_request()
    @jwt_required() # Add JWT protection
    def post(self):
        """
        Ingest MIMIC-IV critical care data.
        """
        if not services_available:
            multimodal_ns.abort(503, message='Multi-modal services not available', code='SERVICE_UNAVAILABLE')
        
        try:
            data = multimodal_ns.payload
            data_type = data['data_type']
            mimic_data = data['data']
            
            valid_types = ['admissions', 'vitals', 'procedures']
            if data_type not in valid_types:
                multimodal_ns.abort(400, message=f'Invalid data_type. Must be one of: {valid_types}', code='INVALID_DATA_TYPE')
            
            if not isinstance(mimic_data, list) or len(mimic_data) == 0:
                multimodal_ns.abort(400, message='data must be a non-empty array', code='INVALID_DATA_FORMAT')
            
            logger.info(f"Ingesting {len(mimic_data)} MIMIC-IV {data_type} records")
            
            async def ingest_data():
                if data_type == 'admissions':
                    return await multimodal_service.ingest_mimic_admissions(mimic_data)
                elif data_type == 'vitals':
                    return await multimodal_service.ingest_mimic_vitals(mimic_data)
                elif data_type == 'procedures':
                    return DataIngestionResult(
                        success=False,
                        records_processed=0,
                        errors=['Procedure ingestion not yet implemented'],
                        execution_time_ms=0,
                        metadata={'data_type': data_type}
                    )
            
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(lambda: asyncio.run(ingest_data()))
                result = future.result(timeout=300)
            
            return result, 200 if result.success else 500
            
        except concurrent.futures.TimeoutError:
            logger.error("MIMIC data ingestion timed out")
            multimodal_ns.abort(504, message='Data ingestion timed out', code='INGESTION_TIMEOUT')
        except Exception as e:
            logger.error(f"Error in MIMIC data ingestion: {str(e)}")
            multimodal_ns.abort(500, message='Internal server error during data ingestion', details=str(e), code='INTERNAL_ERROR')

@multimodal_ns.route('/ingest/biobank')
class IngestBiobankData(Resource):
    @multimodal_ns.doc('ingest_biobank_data')
    @multimodal_ns.expect(ingest_request_model, validate=True)
    @multimodal_ns.marshal_with(ingestion_result_model)
    @log_request()
    @jwt_required() # Add JWT protection
    def post(self):
        """
        Ingest UK Biobank genetic and lifestyle data.
        """
        if not services_available:
            multimodal_ns.abort(503, message='Multi-modal services not available', code='SERVICE_UNAVAILABLE')
        
        try:
            data = multimodal_ns.payload
            data_type = data['data_type']
            biobank_data = data['data']
            
            valid_types = ['participants', 'genetics', 'lifestyle', 'diagnoses']
            if data_type not in valid_types:
                multimodal_ns.abort(400, message=f'Invalid data_type. Must be one of: {valid_types}', code='INVALID_DATA_TYPE')
            
            logger.info(f"Ingesting {len(biobank_data)} UK Biobank {data_type} records")
            
            async def ingest_data():
                if data_type == 'participants':
                    return await multimodal_service.ingest_biobank_participants(biobank_data)
                elif data_type == 'genetics':
                    return await multimodal_service.ingest_biobank_genetics(biobank_data)
                else:
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
            
            return result, 200 if result.success else 500
            
        except Exception as e:
            logger.error(f"Error in biobank data ingestion: {str(e)}")
            multimodal_ns.abort(500, message='Internal server error during biobank ingestion', details=str(e), code='INTERNAL_ERROR')

@multimodal_ns.route('/ingest/faers')
class IngestFaersData(Resource):
    @multimodal_ns.doc('ingest_faers_data')
    @multimodal_ns.expect(ingest_request_model, validate=True)
    @multimodal_ns.marshal_with(ingestion_result_model)
    @log_request()
    @jwt_required() # Add JWT protection
    def post(self):
        """Ingest FDA FAERS adverse event data
        """
        if not services_available:
            multimodal_ns.abort(503, message='Multi-modal services not available', code='SERVICE_UNAVAILABLE')
        
        try:
            data = multimodal_ns.payload
            faers_data = data.get('data', [])
            
            logger.info(f"Ingesting {len(faers_data)} FAERS case records")
            
            async def ingest_data():
                return await multimodal_service.ingest_faers_cases(faers_data)
            
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(lambda: asyncio.run(ingest_data()))
                result = future.result(timeout=300)
            
            return result, 200 if result.success else 500
            
        except Exception as e:
            logger.error(f"Error in FAERS data ingestion: {str(e)}")
            multimodal_ns.abort(500, message='Internal server error during FAERS ingestion', details=str(e), code='INTERNAL_ERROR')

# ============================================================================
# PATIENT IDENTITY RESOLUTION
# ============================================================================

@multimodal_ns.route('/identity/resolve')
class ResolvePatientIdentity(Resource):
    @multimodal_ns.doc('resolve_patient_identity')
    @multimodal_ns.expect(resolve_identity_request_model, validate=True)
    @multimodal_ns.marshal_with(identity_match_result_model)
    @log_request()
    @jwt_required() # Add JWT protection
    def post(self):
        """
        Resolve patient identity across datasets.
        """
        if not services_available:
            multimodal_ns.abort(503, message='Multi-modal services not available', code='SERVICE_UNAVAILABLE')
        
        try:
            data = multimodal_ns.payload
            
            demographics = data['demographics']
            source_dataset = Sanitizer.sanitize_text(data['source_dataset'])
            source_patient_id = Sanitizer.sanitize_text(data['source_patient_id'])
            
            for key, value in demographics.items():
                if isinstance(value, str):
                    demographics[key] = Sanitizer.sanitize_text(value)
            
            logger.info(f"Resolving identity for patient {source_patient_id} from {source_dataset}")
            
            match_result = identity_service.resolve_patient_identity(
                demographics, source_dataset, source_patient_id
            )
            
            return match_result, 200
            
        except Exception as e:
            logger.error(f"Error in patient identity resolution: {str(e)}")
            multimodal_ns.abort(500, message='Internal server error during identity resolution', details=str(e), code='INTERNAL_ERROR')

@multimodal_ns.route('/identity/validate')
class ValidatePatientIdentity(Resource):
    @multimodal_ns.doc('validate_patient_identity')
    @multimodal_ns.expect(validate_identity_request_model, validate=True)
    @multimodal_ns.marshal_with(validate_identity_response_model)
    @log_request()
    @jwt_required() # Add JWT protection
    def post(self):
        """
        Validate patient identity match.
        """
        if not services_available:
            multimodal_ns.abort(503, message='Multi-modal services not available', code='SERVICE_UNAVAILABLE')
        
        try:
            data = multimodal_ns.payload
            
            unified_patient_id = data['unified_patient_id']
            new_demographics = data['new_demographics']
            
            is_valid, confidence_score, conflicts = identity_service.validate_identity_match(
                unified_patient_id, new_demographics
            )
            
            return {
                'is_valid_match': is_valid,
                'confidence_score': confidence_score,
                'conflicts': conflicts,
                'validation_timestamp': datetime.now(timezone.utc).isoformat()
            }, 200
            
        except Exception as e:
            logger.error(f"Error in identity validation: {str(e)}")
            multimodal_ns.abort(500, message='Internal server error during identity validation', details=str(e), code='INTERNAL_ERROR')

# ============================================================================
# CROSS-MODAL SIMILARITY SEARCH
# ============================================================================

@multimodal_ns.route('/similarity/patients')
class FindSimilarPatients(Resource):
    @multimodal_ns.doc('find_similar_patients')
    @multimodal_ns.expect(find_similar_patients_request_model, validate=True)
    @multimodal_ns.marshal_with(find_similar_patients_response_model)
    @log_request()
    @jwt_required() # Add JWT protection
    def post(self):
        """
        Find patients similar to query patient across modalities.
        """
        if not services_available:
            multimodal_ns.abort(503, message='Multi-modal services not available', code='SERVICE_UNAVAILABLE')
        
        try:
            data = multimodal_ns.payload
            
            query_patient_id = data['query_patient_id']
            target_modality_str = data['target_modality']
            source_modalities_str = data.get('source_modalities')
            top_k = min(data.get('top_k', 10), 50)
            min_similarity = max(data.get('min_similarity', 0.1), 0.0)
            
            try:
                target_modality = ModalityType(target_modality_str)
            except ValueError:
                multimodal_ns.abort(400, message=f'Invalid target_modality: {target_modality_str}', code='INVALID_MODALITY')
            
            source_modalities = None
            if source_modalities_str:
                try:
                    source_modalities = [ModalityType(mod) for mod in source_modalities_str]
                except ValueError as e:
                    multimodal_ns.abort(400, message=f'Invalid source modality: {str(e)}', code='INVALID_SOURCE_MODALITY')
            
            logger.info(f"Finding patients similar to {query_patient_id} in {target_modality_str}")
            
            async def search_similar():
                return await vector_service.search_similar_patients(
                    query_patient_id, target_modality, source_modalities, top_k, min_similarity
                )
            
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(lambda: asyncio.run(search_similar()))
                results = future.result(timeout=120)
            
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
            
            return {
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
            }, 200
            
        except concurrent.futures.TimeoutError:
            multimodal_ns.abort(504, message='Similarity search timed out', code='SEARCH_TIMEOUT')
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            multimodal_ns.abort(500, message='Internal server error during similarity search', details=str(e), code='INTERNAL_ERROR')

@multimodal_ns.route('/analysis/cross-modal/<string:patient_id>')
@multimodal_ns.param('patient_id', 'The ID of the patient')
class CrossModalPatientAnalysis(Resource):
    @multimodal_ns.doc('cross_modal_patient_analysis')
    @multimodal_ns.expect(multimodal_ns.parser()
                        .add_argument('include_cohorts', type=bool, help='Include similar patient cohorts', default=True, location='args')
                        .add_argument('include_insights', type=bool, help='Include AI-generated insights', default=True, location='args'))
    @multimodal_ns.marshal_with(cross_modal_analysis_response_model)
    @log_request()
    @jwt_required() # Add JWT protection
    def get(self, patient_id):
        """
        Comprehensive cross-modal analysis for a patient.
        """
        if not services_available:
            multimodal_ns.abort(503, message='Multi-modal services not available', code='SERVICE_UNAVAILABLE')
        
        try:
            patient_id = Sanitizer.sanitize_text(patient_id)
            args = multimodal_ns.parser().parse_args()
            include_cohorts = args['include_cohorts']
            include_insights = args['include_insights']
            
            logger.info(f"Running cross-modal analysis for patient {patient_id}")
            
            async def analyze_patient():
                return await vector_service.cross_modal_patient_analysis(patient_id)
            
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(lambda: asyncio.run(analyze_patient()))
                analysis = future.result(timeout=180)
            
            analysis['query_parameters'] = {
                'include_cohorts': include_cohorts,
                'include_insights': include_insights
            }
            
            analysis['analysis_timestamp'] = datetime.now(timezone.utc).isoformat()
            
            return analysis, 200
            
        except concurrent.futures.TimeoutError:
            multimodal_ns.abort(504, message='Cross-modal analysis timed out', code='ANALYSIS_TIMEOUT')
        except Exception as e:
            logger.error(f"Error in cross-modal analysis: {str(e)}")
            multimodal_ns.abort(500, message='Internal server error during cross-modal analysis', details=str(e), code='INTERNAL_ERROR')

# ============================================================================
# CLINICAL TRIALS MATCHING
# ============================================================================

@multimodal_ns.route('/trials/fetch')
class FetchClinicalTrials(Resource):
    @multimodal_ns.doc('fetch_clinical_trials')
    @multimodal_ns.expect(fetch_trials_request_model, validate=True)
    @multimodal_ns.marshal_with(fetch_trials_response_model)
    @log_request()
    @jwt_required() # Add JWT protection
    def post(self):
        """
        Fetch clinical trials from ClinicalTrials.gov API.
        """
        if not services_available:
            multimodal_ns.abort(503, message='Multi-modal services not available', code='SERVICE_UNAVAILABLE')
        
        try:
            data = multimodal_ns.payload
            
            conditions = data.get('conditions', [])
            status = data.get('status', 'RECRUITING')
            max_results = min(data.get('max_results', 100), 1000)
            
            if not conditions:
                multimodal_ns.abort(400, message='At least one condition is required', code='MISSING_CONDITIONS')
            
            logger.info(f"Fetching clinical trials for conditions: {conditions}")
            
            async def fetch_trials():
                trials = await multimodal_service.fetch_clinical_trials(conditions, status)
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
                result = future.result(timeout=300)
            
            return result, 200
            
        except concurrent.futures.TimeoutError:
            multimodal_ns.abort(504, message='Clinical trials fetch timed out', code='FETCH_TIMEOUT')
        except Exception as e:
            logger.error(f"Error fetching clinical trials: {str(e)}")
            multimodal_ns.abort(500, message='Internal server error during trials fetch', details=str(e), code='INTERNAL_ERROR')

@multimodal_ns.route('/trials/match/<string:patient_id>')
@multimodal_ns.param('patient_id', 'The ID of the patient')
class MatchPatientToTrials(Resource):
    @multimodal_ns.doc('match_patient_to_trials')
    @multimodal_ns.expect(match_trials_request_model, validate=True)
    @multimodal_ns.marshal_with(match_trials_response_model)
    @log_request()
    @jwt_required() # Add JWT protection
    def post(self):
        """
        Match patient to relevant clinical trials.
        """
        if not services_available:
            multimodal_ns.abort(503, message='Multi-modal services not available', code='SERVICE_UNAVAILABLE')
        
        try:
            data = multimodal_ns.payload
            
            session_id = data['session_id']
            trial_types = data.get('trial_types', ['interventional'])
            max_matches = min(data.get('max_matches', 10), 50)
            
            patient_id = Sanitizer.sanitize_text(patient_id)
            session_id = Sanitizer.sanitize_text(session_id)
            
            logger.info(f"Matching patient {patient_id} to clinical trials")
            
            async def match_trials():
                return await multimodal_service.match_patient_to_trials(patient_id, session_id)
            
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(lambda: asyncio.run(match_trials()))
                matches = future.result(timeout=120)
            
            filtered_matches = [
                match for match in matches[:max_matches]
            ]
            
            return {
                'patient_id': patient_id,
                'session_id': session_id,
                'trial_matches': filtered_matches,
                'total_matches': len(filtered_matches),
                'matching_timestamp': datetime.utcnow().isoformat()
            }, 200
            
        except concurrent.futures.TimeoutError:
            multimodal_ns.abort(504, message='Trial matching timed out', code='MATCHING_TIMEOUT')
        except Exception as e:
            logger.error(f"Error matching trials: {str(e)}")
            multimodal_ns.abort(500, message='Internal server error during trial matching', details=str(e), code='INTERNAL_ERROR')

# ============================================================================
# SERVICE STATUS AND STATISTICS
# ============================================================================

@multimodal_ns.route('/health')
class MultimodalHealthCheck(Resource):
    @multimodal_ns.doc('multimodal_health_check')
    @multimodal_ns.marshal_with(multimodal_health_response_model)
    @log_request()
    def get(self):
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
                try:
                    vector_stats = vector_service.get_service_stats()
                    health_status['service_stats'] = vector_stats
                except Exception as e:
                    health_status['service_stats'] = {'error': str(e)}
            
            return health_status, 200 if services_available else 503
            
        except Exception as e:
            logger.error(f"Multi-modal health check failed: {str(e)}")
            multimodal_ns.abort(500, message='Internal server error', details=str(e), code='INTERNAL_ERROR')

@multimodal_ns.route('/stats')
class GetMultimodalStats(Resource):
    @multimodal_ns.doc('get_multimodal_stats')
    @multimodal_ns.marshal_with(multimodal_stats_response_model)
    @log_request()
    def get(self):
        """
        Get comprehensive multi-modal service statistics.
        """
        if not services_available:
            multimodal_ns.abort(503, message='Multi-modal services not available', code='SERVICE_UNAVAILABLE')
        
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
            
            return stats, 200
            
        except Exception as e:
            logger.error(f"Error getting multi-modal stats: {str(e)}")
            multimodal_ns.abort(500, message='Internal server error getting statistics', details=str(e), code='INTERNAL_ERROR')
