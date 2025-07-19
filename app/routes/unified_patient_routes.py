from flask import request
import logging
import asyncio
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Any
from flask_restx import Namespace, Resource, fields
from flask_jwt_extended import jwt_required, get_jwt_identity # Import jwt_required

from app.services.unified_patient_service import (
    UnifiedPatientService, QueryContext, UnifiedPatientView
)
from app.utils.validation import Validator, ValidationError
from app.utils.sanitization import Sanitizer
from app.middleware.security import log_request

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Namespace
unified_ns = Namespace('unified-patient', description='Unified Patient Data Access')

# Initialize unified service
try:
    unified_service = UnifiedPatientService()
    service_available = True
    logger.info("✅ Unified patient service initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize unified patient service: {e}")
    unified_service = None
    service_available = False

# --- Models for API Documentation ---

# Unified Patient View Models
demographics_model = unified_ns.model('Demographics', {
    'age': fields.Integer,
    'gender': fields.String,
    'ethnicity': fields.String,
    'zip_code': fields.String,
    'marital_status': fields.String
})

data_availability_model = unified_ns.model('DataAvailability', {
    'demographics': fields.Boolean,
    'clinical_notes': fields.Boolean,
    'mimic_data': fields.Boolean,
    'genetic_data': fields.Boolean,
    'adverse_events': fields.Boolean,
    'trial_matches': fields.Boolean,
    'vector_embeddings': fields.Boolean
})

clinical_summary_model = unified_ns.model('ClinicalSummary', {
    'latest_note': fields.String,
    'active_conditions': fields.List(fields.String),
    'current_medications': fields.List(fields.String),
    'allergies': fields.List(fields.String)
})

risk_stratification_model = unified_ns.model('RiskStratification', {
    'overall': fields.String(enum=['low', 'moderate', 'high', 'critical']),
    'cardiovascular': fields.String(enum=['low', 'moderate', 'high', 'critical']),
    'oncology': fields.String(enum=['low', 'moderate', 'high', 'critical'])
})

mimic_profile_model = unified_ns.model('MimicProfile', {
    'admissions_count': fields.Integer,
    'avg_los_days': fields.Float,
    'icu_stays_count': fields.Integer
})

genetic_profile_model = unified_ns.model('GeneticProfile', {
    'genetic_markers': fields.List(fields.String),
    'polygenic_risk_scores': fields.Raw
})

adverse_event_profile_model = unified_ns.model('AdverseEventProfile', {
    'total_events': fields.Integer,
    'most_common_drugs': fields.List(fields.String),
    'serious_events_count': fields.Integer
})

fusion_insights_model = unified_ns.model('FusionInsights', {
    'key_findings': fields.List(fields.String),
    'discrepancies': fields.List(fields.String),
    'recommendations': fields.List(fields.String)
})

similar_patient_model = unified_ns.model('SimilarPatient', {
    'patient_id': fields.String,
    'similarity_score': fields.Float,
    'modality': fields.String,
    'data_source': fields.String
})

trial_match_model = unified_ns.model('TrialMatch', {
    'nct_id': fields.String,
    'title': fields.String,
    'match_score': fields.Float,
    'eligibility_status': fields.String
})

unified_patient_view_model = unified_ns.model('UnifiedPatientView', {
    'patient_id': fields.String(description='Unique identifier for the patient'),
    'query_context': fields.String(description='Context for which the view was generated'),
    'demographics': fields.Nested(demographics_model),
    'unified_identity': fields.String(description='Unified ID across datasets'),
    'data_availability': fields.Nested(data_availability_model),
    'clinical_summary': fields.Nested(clinical_summary_model),
    'recent_analyses': fields.List(fields.Raw),
    'risk_stratification': fields.Nested(risk_stratification_model),
    'mimic_profile': fields.Nested(mimic_profile_model),
    'genetic_profile': fields.Nested(genetic_profile_model),
    'adverse_event_profile': fields.Nested(adverse_event_profile_model),
    'fusion_insights': fields.Nested(fusion_insights_model),
    'similar_patients': fields.List(fields.Nested(similar_patient_model)),
    'trial_matches': fields.List(fields.Nested(trial_match_model)),
    'data_completeness_score': fields.Float,
    'query_performance_ms': fields.Float,
    'recommendations': fields.List(fields.String)
})

# Patient Summary Model
patient_summary_model = unified_ns.model('PatientSummary', {
    'patient_id': fields.String,
    'name': fields.String,
    'age': fields.Integer,
    'gender': fields.String,
    'latest_diagnosis': fields.String,
    'risk_level': fields.String,
    'last_updated': fields.String
})

# Refresh Patient Data Models
refresh_request_model = unified_ns.model('RefreshRequest', {
    'force_recompute': fields.Boolean(default=False, description='Force recomputation of expensive operations'),
    'components': fields.List(fields.String, description='Specific components to refresh (e.g., genetics, trials, fusion)')
})

refresh_response_model = unified_ns.model('RefreshResponse', {
    'patient_id': fields.String,
    'status': fields.String,
    'refreshed_components': fields.List(fields.String),
    'errors': fields.List(fields.String),
    'refresh_time_ms': fields.Float
})

# Batch Summary Models
batch_summary_request_model = unified_ns.model('BatchSummaryRequest', {
    'patient_ids': fields.List(fields.String, required=True, description='List of patient IDs to get summaries for'),
    'max_patients': fields.Integer(default=100, description='Maximum number of patients to process in batch')
})

batch_summary_response_model = unified_ns.model('BatchSummaryResponse', {
    'summaries': fields.List(fields.Nested(patient_summary_model)),
    'successful_queries': fields.Integer,
    'failed_queries': fields.Integer,
    'total_queries': fields.Integer,
    'errors': fields.List(fields.Raw, description='List of errors for failed queries')
})

# Search Patients Models
search_criteria_model = unified_ns.model('SearchCriteria', {
    'age_range': fields.List(fields.Integer, description='[min_age, max_age]'),
    'conditions': fields.List(fields.String),
    'genetic_risk': fields.List(fields.String),
    'has_mimic_data': fields.Boolean
})

search_patients_request_model = unified_ns.model('SearchPatientsRequest', {
    'criteria': fields.Nested(search_criteria_model, required=True, description='Criteria for patient search'),
    'context': fields.String(enum=[c.value for c in QueryContext], default='clinical_review', description='Context for the search'),
    'limit': fields.Integer(default=50, description='Maximum number of search results')
})

search_result_item_model = unified_ns.model('SearchResultItem', {
    'patient_id': fields.String,
    'relevance_score': fields.Float,
    'matched_data_sources': fields.List(fields.String),
    'summary': fields.String
})

search_patients_response_model = unified_ns.model('SearchPatientsResponse', {
    'matches': fields.List(fields.Nested(search_result_item_model)),
    'total_matches': fields.Integer,
    'search_criteria': fields.Nested(search_criteria_model),
    'context': fields.String,
    'search_timestamp': fields.String,
    'message': fields.String
})

# Health and Stats Models
unified_health_component_model = unified_ns.model('UnifiedHealthComponent', {
    'supabase_service': fields.String,
    'multimodal_service': fields.String,
    'vector_service': fields.String,
    'identity_service': fields.String,
    'fusion_service': fields.String,
    'trials_service': fields.String,
    'clinical_service': fields.String
})

unified_health_response_model = unified_ns.model('UnifiedHealthResponse', {
    'status': fields.String,
    'timestamp': fields.String,
    'service_available': fields.Boolean,
    'components': fields.Nested(unified_health_component_model)
})

unified_stats_response_model = unified_ns.model('UnifiedStatsResponse', {
    'timestamp': fields.String,
    'unified_patients': fields.Integer,
    'clinical_sessions': fields.Integer,
    'available_contexts': fields.List(fields.String),
    'service_version': fields.String
})


# ============================================================================
# MAIN PATIENT ACCESS ENDPOINTS
# ============================================================================

@unified_ns.route('/patient/<string:patient_id>')
@unified_ns.param('patient_id', 'The unique identifier of the patient')
class UnifiedPatientViewResource(Resource):
    @unified_ns.doc('get_unified_patient_view')
    @unified_ns.expect(unified_ns.parser()
                        .add_argument('context', type=str, help='Query context (clinical_review, emergency_triage, trial_matching, etc.)', default='clinical_review', choices=[c.value for c in QueryContext], location='args')
                        .add_argument('include_similar', type=bool, help='Include similar patients analysis', default=False, location='args')
                        .add_argument('include_trials', type=bool, help='Include trial matches', default=False, location='args')
                        .add_argument('max_time', type=int, help='Maximum response time in seconds', default=10, location='args'))
    @unified_ns.marshal_with(unified_patient_view_model)
    @log_request()
    @jwt_required() # Add JWT protection
    def get(self, patient_id):
        """
        Get comprehensive unified patient view with intelligent data loading.
        Returns unified patient view optimized for the specific context.
        """
        if not service_available:
            unified_ns.abort(503, message='Unified patient service not available', code='SERVICE_UNAVAILABLE')
        
        try:
            patient_id = Sanitizer.sanitize_text(patient_id)
            
            if len(patient_id) < 3:
                unified_ns.abort(400, message='Invalid patient ID', code='INVALID_PATIENT_ID')
            
            args = unified_ns.parser().parse_args()
            context_str = args['context']
            include_similar = args['include_similar']
            include_trials = args['include_trials']
            max_time = args['max_time']
            
            context = QueryContext(context_str)
            
            logger.info(f"Getting unified view for patient {patient_id} with context {context_str}")
            
            def run_unified_query():
                return asyncio.run(unified_service.get_unified_patient_view(
                    patient_identifier=patient_id,
                    query_context=context,
                    include_similar_patients=include_similar,
                    include_trial_matches=include_trials,
                    max_response_time_seconds=max_time
                ))
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_unified_query)
                unified_view = future.result(timeout=max_time + 5)
            
            response_data = _serialize_unified_view(unified_view)
            
            logger.info(f"Unified view generated in {unified_view.query_performance_ms}ms "
                       f"with {unified_view.data_completeness_score:.1%} completeness")
            
            return response_data, 200
            
        except concurrent.futures.TimeoutError:
            unified_ns.abort(504, message='Patient query timed out', code='QUERY_TIMEOUT')
        except ValueError as e:
            unified_ns.abort(404, message=str(e), code='PATIENT_NOT_FOUND')
        except Exception as e:
            logger.error(f"Error in unified patient view: {str(e)}")
            unified_ns.abort(500, message='Internal server error during patient query', details=str(e), code='INTERNAL_ERROR')

@unified_ns.route('/patient/<string:patient_id>/summary')
@unified_ns.param('patient_id', 'The unique identifier of the patient')
class PatientSummaryResource(Resource):
    @unified_ns.doc('get_patient_summary')
    @unified_ns.marshal_with(patient_summary_model)
    @log_request()
    @jwt_required() # Add JWT protection
    def get(self, patient_id):
        """
        Get quick patient summary for lists and searches.
        Optimized for speed - only essential data.
        """
        if not service_available:
            unified_ns.abort(503, message='Unified patient service not available', code='SERVICE_UNAVAILABLE')
        
        try:
            patient_id = Sanitizer.sanitize_text(patient_id)
            
            logger.info(f"Getting quick summary for patient {patient_id}")
            
            def run_summary_query():
                return asyncio.run(unified_service.get_patient_summary(patient_id))
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_summary_query)
                summary = future.result(timeout=5)
            
            if 'error' in summary:
                unified_ns.abort(404, message=summary['error'], code='SUMMARY_ERROR')
            
            return summary, 200
            
        except concurrent.futures.TimeoutError:
            unified_ns.abort(504, message='Summary query timed out', code='SUMMARY_TIMEOUT')
        except Exception as e:
            logger.error(f"Error in patient summary: {str(e)}")
            unified_ns.abort(500, message='Internal server error during summary generation', code='INTERNAL_ERROR')

@unified_ns.route('/patient/<string:patient_id>/refresh')
@unified_ns.param('patient_id', 'The unique identifier of the patient')
class RefreshPatientDataResource(Resource):
    @unified_ns.doc('refresh_patient_data')
    @unified_ns.expect(refresh_request_model, validate=True)
    @unified_ns.marshal_with(refresh_response_model)
    @log_request()
    @jwt_required() # Add JWT protection
    def post(self, patient_id):
        """
        Refresh patient data across all modalities.
        Useful when new data has been added or patient profile needs updating.
        """
        if not service_available:
            unified_ns.abort(503, message='Unified patient service not available', code='SERVICE_UNAVAILABLE')
        
        try:
            patient_id = Sanitizer.sanitize_text(patient_id)
            
            data = unified_ns.payload
            force_recompute = data.get('force_recompute', False)
            specific_components = data.get('components', [])
            
            logger.info(f"Refreshing data for patient {patient_id} (force={force_recompute})")
            
            def run_refresh():
                return asyncio.run(unified_service.refresh_patient_data(
                    patient_id, force_recompute=force_recompute
                ))
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_refresh)
                refresh_result = future.result(timeout=60)
            
            if 'error' in refresh_result:
                unified_ns.abort(500, message=refresh_result['error'], code='REFRESH_ERROR')
            
            return refresh_result, 200
            
        except concurrent.futures.TimeoutError:
            unified_ns.abort(504, message='Patient data refresh timed out', code='REFRESH_TIMEOUT')
        except Exception as e:
            logger.error(f"Error refreshing patient data: {str(e)}")
            unified_ns.abort(500, message='Internal server error during data refresh', code='INTERNAL_ERROR')

# ============================================================================
# CONTEXT-SPECIFIC ENDPOINTS
# ============================================================================

@unified_ns.route('/patient/<string:patient_id>/clinical-review')
@unified_ns.param('patient_id', 'The unique identifier of the patient')
class ClinicalReviewViewResource(Resource):
    @unified_ns.doc('get_clinical_review_view')
    @unified_ns.marshal_with(unified_patient_view_model)
    @log_request()
    @jwt_required() # Add JWT protection
    def get(self, patient_id):
        """
        Get patient view optimized for clinical review.
        Prioritizes clinical text, recent analyses, and risk indicators.
        """
        return _get_context_specific_view(patient_id, QueryContext.CLINICAL_REVIEW)

@unified_ns.route('/patient/<string:patient_id>/emergency-triage')
@unified_ns.param('patient_id', 'The unique identifier of the patient')
class EmergencyTriageViewResource(Resource):
    @unified_ns.doc('get_emergency_triage_view')
    @unified_ns.marshal_with(unified_patient_view_model)
    @log_request()
    @jwt_required() # Add JWT protection
    def get(self, patient_id):
        """
        Get patient view optimized for emergency triage.
        Prioritizes vital signs, MIMIC patterns, and immediate risk factors.
        """
        return _get_context_specific_view(patient_id, QueryContext.EMERGENCY_TRIAGE)

@unified_ns.route('/patient/<string:patient_id>/genetic-counseling')
@unified_ns.param('patient_id', 'The unique identifier of the patient')
class GeneticCounselingViewResource(Resource):
    @unified_ns.doc('get_genetic_counseling_view')
    @unified_ns.marshal_with(unified_patient_view_model)
    @log_request()
    @jwt_required() # Add JWT protection
    def get(self, patient_id):
        """
        Get patient view optimized for genetic counseling.
        Prioritizes genetic data, family history, and pharmacogenomics.
        """
        return _get_context_specific_view(patient_id, QueryContext.GENETIC_COUNSELING)

@unified_ns.route('/patient/<string:patient_id>/trial-matching')
@unified_ns.param('patient_id', 'The unique identifier of the patient')
class TrialMatchingViewResource(Resource):
    @unified_ns.doc('get_trial_matching_view')
    @unified_ns.marshal_with(unified_patient_view_model)
    @log_request()
    @jwt_required() # Add JWT protection
    def get(self, patient_id):
        """
        Get patient view optimized for clinical trial matching.
        Includes comprehensive trial eligibility analysis.
        """
        return _get_context_specific_view(patient_id, QueryContext.TRIAL_MATCHING, include_trials=True)

@unified_ns.route('/patient/<string:patient_id>/risk-assessment')
@unified_ns.param('patient_id', 'The unique identifier of the patient')
class RiskAssessmentViewResource(Resource):
    @unified_ns.doc('get_risk_assessment_view')
    @unified_ns.marshal_with(unified_patient_view_model)
    @log_request()
    @jwt_required() # Add JWT protection
    def get(self, patient_id):
        """
        Get patient view optimized for comprehensive risk assessment.
        Includes multi-modal risk stratification across all available data.
        """
        return _get_context_specific_view(patient_id, QueryContext.RISK_ASSESSMENT, include_similar=True)

# ============================================================================
# BATCH OPERATIONS
# ============================================================================

@unified_ns.route('/patients/batch-summary')
class BatchPatientSummariesResource(Resource):
    @unified_ns.doc('get_batch_patient_summaries')
    @unified_ns.expect(batch_summary_request_model, validate=True)
    @unified_ns.marshal_with(batch_summary_response_model)
    @log_request()
    @jwt_required() # Add JWT protection
    def post(self):
        """
        Get summaries for multiple patients efficiently.
        """
        if not service_available:
            unified_ns.abort(503, message='Unified patient service not available', code='SERVICE_UNAVAILABLE')
        
        try:
            data = unified_ns.payload
            
            patient_ids = data['patient_ids']
            max_patients = min(data.get('max_patients', 50), 100)
            
            if not isinstance(patient_ids, list) or len(patient_ids) == 0:
                unified_ns.abort(400, message='patient_ids must be a non-empty array', code='INVALID_PATIENT_IDS')
            
            if len(patient_ids) > max_patients:
                patient_ids = patient_ids[:max_patients]
            
            patient_ids = [Sanitizer.sanitize_text(pid) for pid in patient_ids]
            
            logger.info(f"Getting batch summaries for {len(patient_ids)} patients")
            
            async def run_batch_summaries():
                tasks = []
                for patient_id in patient_ids:
                    tasks.append(unified_service.get_patient_summary(patient_id))
                
                return await asyncio.gather(*tasks, return_exceptions=True)
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(lambda: asyncio.run(run_batch_summaries()))
                results = future.result(timeout=30)
            
            summaries = []
            errors = []
            
            for i, result in enumerate(results):
                patient_id = patient_ids[i]
                
                if isinstance(result, Exception):
                    errors.append({
                        'patient_id': patient_id,
                        'error': str(result)
                    })
                elif 'error' in result:
                    errors.append({
                        'patient_id': patient_id,
                        'error': result['error']
                    })
                else:
                    summaries.append(result)
            
            return {
                'summaries': summaries,
                'successful_queries': len(summaries),
                'failed_queries': len(errors),
                'total_queries': len(patient_ids),
                'errors': errors if errors else None
            }, 200
            
        except concurrent.futures.TimeoutError:
            unified_ns.abort(504, message='Batch summary operation timed out', code='BATCH_TIMEOUT')
        except Exception as e:
            logger.error(f"Error in batch summaries: {str(e)}")
            unified_ns.abort(500, message='Internal server error during batch operation', code='INTERNAL_ERROR')

# ============================================================================
# SEARCH AND DISCOVERY
# ============================================================================

@unified_ns.route('/patients/search')
class SearchPatientsResource(Resource):
    @unified_ns.doc('search_patients')
    @unified_ns.expect(search_patients_request_model, validate=True)
    @unified_ns.marshal_with(search_patients_response_model)
    @log_request()
    @jwt_required() # Add JWT protection
    def post(self):
        """
        Search for patients based on clinical criteria across all modalities.
        """
        if not service_available:
            unified_ns.abort(503, message='Unified patient service not available', code='SERVICE_UNAVAILABLE')
        
        try:
            data = unified_ns.payload
            
            criteria = data['criteria']
            context_str = data.get('context', 'clinical_review')
            limit = min(data.get('limit', 20), 100)
            
            try:
                context = QueryContext(context_str)
            except ValueError:
                context = QueryContext.CLINICAL_REVIEW
            
            logger.info(f"Searching patients with criteria: {criteria}")
            
            search_results = {
                'matches': [],
                'total_matches': 0,
                'search_criteria': criteria,
                'context': context_str,
                'search_timestamp': datetime.now().isoformat(),
                'message': 'Advanced patient search across modalities - implementation in progress'
            }
            
            return search_results, 200
            
        except Exception as e:
            logger.error(f"Error in patient search: {str(e)}")
            unified_ns.abort(500, message='Internal server error during patient search', code='INTERNAL_ERROR')

# ============================================================================
# SYSTEM STATUS AND ANALYTICS
# ============================================================================

@unified_ns.route('/health')
class UnifiedHealthCheckResource(Resource):
    @unified_ns.doc('unified_health_check')
    @unified_ns.marshal_with(unified_health_response_model)
    @log_request()
    @jwt_required() # Add JWT protection
    def get(self):
        """
        Health check for unified patient service.
        """
        try:
            health_status = {
                'status': 'healthy' if service_available else 'degraded',
                'timestamp': datetime.now().isoformat(),
                'service_available': service_available,
                'components': {}
            }
            
            if service_available and unified_service:
                health_status['components'] = {
                    'supabase_service': 'available',
                    'multimodal_service': 'available',
                    'vector_service': 'available',
                    'identity_service': 'available', 
                    'fusion_service': 'available',
                    'trials_service': 'available',
                    'clinical_service': 'available'
                }
            
            return health_status, 200 if service_available else 503
            
        except Exception as e:
            logger.error(f"Unified health check failed: {str(e)}")
            unified_ns.abort(500, message='Internal server error', code='INTERNAL_ERROR')

@unified_ns.route('/stats')
class GetUnifiedStatsResource(Resource):
    @unified_ns.doc('get_unified_stats')
    @unified_ns.marshal_with(unified_stats_response_model)
    @log_request()
    @jwt_required() # Add JWT protection
    def get(self):
        """
        Get unified patient service statistics.
        """
        if not service_available:
            unified_ns.abort(503, message='Unified patient service not available', code='SERVICE_UNAVAILABLE')
        
        try:
            def get_stats():
                return asyncio.run(_collect_unified_stats())
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(get_stats)
                stats = future.result(timeout=10)
            
            return stats, 200
            
        except Exception as e:
            logger.error(f"Error getting unified stats: {str(e)}")
            unified_ns.abort(500, message='Internal server error getting statistics', code='INTERNAL_ERROR')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _get_context_specific_view(patient_id: str, context: QueryContext, 
                              include_similar: bool = False,
                              include_trials: bool = False):
    """Helper function for context-specific views"""
    if not service_available:
        unified_ns.abort(503, message='Unified patient service not available', code='SERVICE_UNAVAILABLE')
    
    try:
        patient_id = Sanitizer.sanitize_text(patient_id)
        
        def run_context_query():
            return asyncio.run(unified_service.get_unified_patient_view(
                patient_identifier=patient_id,
                query_context=context,
                include_similar_patients=include_similar,
                include_trial_matches=include_trials,
                max_response_time_seconds=15
            ))
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_context_query)
            unified_view = future.result(timeout=20)
        
        response_data = _serialize_unified_view(unified_view)
        
        return response_data, 200
        
    except ValueError as e:
        unified_ns.abort(404, message=str(e), code='PATIENT_NOT_FOUND')
    except Exception as e:
        logger.error(f"Error in context-specific view: {str(e)}")
        unified_ns.abort(500, message='Internal server error', code='INTERNAL_ERROR')

def _serialize_unified_view(unified_view: UnifiedPatientView) -> Dict[str, Any]:
    """Convert UnifiedPatientView to JSON-serializable format"""
    return {
        'patient_id': unified_view.patient_id,
        'query_context': unified_view.query_context.value,
        'demographics': unified_view.demographics,
        'unified_identity': unified_view.unified_identity,
        'data_availability': {
            'demographics': unified_view.data_availability.demographics,
            'clinical_notes': unified_view.data_availability.clinical_notes,
            'mimic_data': unified_view.data_availability.mimic_data,
            'genetic_data': unified_view.data_availability.genetic_data,
            'adverse_events': unified_view.data_availability.adverse_events,
            'trial_matches': unified_view.data_availability.trial_matches,
            'vector_embeddings': unified_view.data_availability.vector_embeddings
        },
        'clinical_summary': unified_view.clinical_summary,
        'recent_analyses': unified_view.recent_analyses,
        'risk_stratification': {k: v.value if v else None for k, v in (unified_view.risk_stratification or {}).items()},
        'mimic_profile': unified_view.mimic_profile,
        'genetic_profile': unified_view.genetic_profile,
        'adverse_event_profile': unified_view.adverse_event_profile,
        'fusion_insights': unified_view.fusion_insights,
        'similar_patients': unified_view.similar_patients,
        'trial_matches': unified_view.trial_matches,
        'data_completeness_score': unified_view.data_completeness_score,
        'query_performance_ms': unified_view.query_performance_ms,
        'recommendations': unified_view.recommendations
    }

async def _collect_unified_stats() -> Dict[str, Any]:
    """Collect unified service statistics"""
    try:
        if not unified_service:
            return {'error': 'Service not available'}
        
        unified_patients = unified_service.supabase.client.table('unified_patients').select('unified_patient_id', count='exact').execute()
        clinical_sessions = unified_service.supabase.client.table('analysis_sessions').select('session_id', count='exact').execute()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'unified_patients': unified_patients.count if unified_patients else 0,
            'clinical_sessions': clinical_sessions.count if clinical_sessions else 0,
            'available_contexts': [context.value for context in QueryContext],
            'service_version': '1.0.0'
        }
        
    except Exception as e:
        logger.error(f"Error collecting stats: {e}")
        return {
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }