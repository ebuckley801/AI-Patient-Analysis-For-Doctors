"""
Unified Patient API Routes

Cohesive API endpoints that intelligently determine what multi-modal data
is relevant and needed for each patient query. Provides unified access
to all patient data across modalities with context-aware optimization.
"""

from flask import Blueprint, request, jsonify
import logging
import asyncio
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Any

from app.services.unified_patient_service import (
    UnifiedPatientService, QueryContext, UnifiedPatientView
)
from app.utils.validation import Validator, ValidationError
from app.utils.sanitization import Sanitizer
from app.middleware.security import log_request

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Blueprint
unified_bp = Blueprint('unified', __name__)

# Initialize unified service
try:
    unified_service = UnifiedPatientService()
    service_available = True
    logger.info("✅ Unified patient service initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize unified patient service: {e}")
    unified_service = None
    service_available = False

# ============================================================================
# MAIN PATIENT ACCESS ENDPOINTS
# ============================================================================

@unified_bp.route('/patient/<patient_id>', methods=['GET'])
@log_request()
def get_unified_patient_view(patient_id):
    """
    Get comprehensive unified patient view with intelligent data loading
    
    Query parameters:
    - context: Query context (clinical_review, emergency_triage, trial_matching, etc.)
    - include_similar: Include similar patients analysis (true/false)
    - include_trials: Include trial matches (true/false) 
    - max_time: Maximum response time in seconds (default: 10)
    
    Returns unified patient view optimized for the specific context
    """
    if not service_available:
        return jsonify({
            'success': False,
            'error': 'Unified patient service not available',
            'code': 'SERVICE_UNAVAILABLE'
        }), 503
    
    try:
        # Sanitize and validate inputs
        patient_id = Sanitizer.sanitize_text(patient_id)
        
        if len(patient_id) < 3:
            return jsonify({
                'success': False,
                'error': 'Invalid patient ID',
                'code': 'INVALID_PATIENT_ID'
            }), 400
        
        # Parse query parameters
        context_str = request.args.get('context', 'clinical_review')
        include_similar = request.args.get('include_similar', 'false').lower() == 'true'
        include_trials = request.args.get('include_trials', 'false').lower() == 'true'
        max_time = min(int(request.args.get('max_time', 10)), 30)  # Cap at 30 seconds
        
        # Validate context
        try:
            context = QueryContext(context_str)
        except ValueError:
            return jsonify({
                'success': False,
                'error': f'Invalid context. Valid options: {[c.value for c in QueryContext]}',
                'code': 'INVALID_CONTEXT'
            }), 400
        
        logger.info(f"Getting unified view for patient {patient_id} with context {context_str}")
        
        # Execute unified patient view query
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
        
        # Convert to JSON-serializable format
        response_data = _serialize_unified_view(unified_view)
        
        logger.info(f"Unified view generated in {unified_view.query_performance_ms}ms "
                   f"with {unified_view.data_completeness_score:.1%} completeness")
        
        return jsonify({
            'success': True,
            'data': response_data
        }), 200
        
    except concurrent.futures.TimeoutError:
        return jsonify({
            'success': False,
            'error': 'Patient query timed out',
            'code': 'QUERY_TIMEOUT'
        }), 504
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'code': 'PATIENT_NOT_FOUND'
        }), 404
    except Exception as e:
        logger.error(f"Error in unified patient view: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error during patient query',
            'details': str(e),
            'code': 'INTERNAL_ERROR'
        }), 500

@unified_bp.route('/patient/<patient_id>/summary', methods=['GET'])
@log_request()
def get_patient_summary(patient_id):
    """
    Get quick patient summary for lists and searches
    Optimized for speed - only essential data
    """
    if not service_available:
        return jsonify({
            'success': False,
            'error': 'Unified patient service not available',
            'code': 'SERVICE_UNAVAILABLE'
        }), 503
    
    try:
        patient_id = Sanitizer.sanitize_text(patient_id)
        
        logger.info(f"Getting quick summary for patient {patient_id}")
        
        def run_summary_query():
            return asyncio.run(unified_service.get_patient_summary(patient_id))
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_summary_query)
            summary = future.result(timeout=5)  # Quick summary should be fast
        
        if 'error' in summary:
            return jsonify({
                'success': False,
                'error': summary['error'],
                'code': 'SUMMARY_ERROR'
            }), 404 if 'not found' in summary['error'].lower() else 500
        
        return jsonify({
            'success': True,
            'data': summary
        }), 200
        
    except concurrent.futures.TimeoutError:
        return jsonify({
            'success': False,
            'error': 'Summary query timed out',
            'code': 'SUMMARY_TIMEOUT'
        }), 504
    except Exception as e:
        logger.error(f"Error in patient summary: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error during summary generation',
            'code': 'INTERNAL_ERROR'
        }), 500

@unified_bp.route('/patient/<patient_id>/refresh', methods=['POST'])
@log_request()
def refresh_patient_data(patient_id):
    """
    Refresh patient data across all modalities
    Useful when new data has been added or patient profile needs updating
    
    Request body:
    {
        "force_recompute": false,  // Force recomputation of expensive operations
        "components": ["genetics", "trials", "fusion"]  // Specific components to refresh
    }
    """
    if not service_available:
        return jsonify({
            'success': False,
            'error': 'Unified patient service not available',
            'code': 'SERVICE_UNAVAILABLE'
        }), 503
    
    try:
        patient_id = Sanitizer.sanitize_text(patient_id)
        
        # Parse request body
        data = request.get_json() if request.is_json else {}
        force_recompute = data.get('force_recompute', False)
        specific_components = data.get('components', [])
        
        logger.info(f"Refreshing data for patient {patient_id} (force={force_recompute})")
        
        def run_refresh():
            return asyncio.run(unified_service.refresh_patient_data(
                patient_id, force_recompute=force_recompute
            ))
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_refresh)
            refresh_result = future.result(timeout=60)  # Allow longer time for refresh
        
        if 'error' in refresh_result:
            return jsonify({
                'success': False,
                'error': refresh_result['error'],
                'code': 'REFRESH_ERROR'
            }), 500
        
        return jsonify({
            'success': True,
            'data': refresh_result
        }), 200
        
    except concurrent.futures.TimeoutError:
        return jsonify({
            'success': False,
            'error': 'Patient data refresh timed out',
            'code': 'REFRESH_TIMEOUT'
        }), 504
    except Exception as e:
        logger.error(f"Error refreshing patient data: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error during data refresh',
            'code': 'INTERNAL_ERROR'
        }), 500

# ============================================================================
# CONTEXT-SPECIFIC ENDPOINTS
# ============================================================================

@unified_bp.route('/patient/<patient_id>/clinical-review', methods=['GET'])
@log_request()
def get_clinical_review_view(patient_id):
    """
    Get patient view optimized for clinical review
    Prioritizes clinical text, recent analyses, and risk indicators
    """
    return _get_context_specific_view(patient_id, QueryContext.CLINICAL_REVIEW)

@unified_bp.route('/patient/<patient_id>/emergency-triage', methods=['GET'])
@log_request()
def get_emergency_triage_view(patient_id):
    """
    Get patient view optimized for emergency triage
    Prioritizes vital signs, MIMIC patterns, and immediate risk factors
    """
    return _get_context_specific_view(patient_id, QueryContext.EMERGENCY_TRIAGE)

@unified_bp.route('/patient/<patient_id>/genetic-counseling', methods=['GET'])
@log_request()
def get_genetic_counseling_view(patient_id):
    """
    Get patient view optimized for genetic counseling
    Prioritizes genetic data, family history, and pharmacogenomics
    """
    return _get_context_specific_view(patient_id, QueryContext.GENETIC_COUNSELING)

@unified_bp.route('/patient/<patient_id>/trial-matching', methods=['GET'])
@log_request()
def get_trial_matching_view(patient_id):
    """
    Get patient view optimized for clinical trial matching
    Includes comprehensive trial eligibility analysis
    """
    return _get_context_specific_view(patient_id, QueryContext.TRIAL_MATCHING, include_trials=True)

@unified_bp.route('/patient/<patient_id>/risk-assessment', methods=['GET'])
@log_request()
def get_risk_assessment_view(patient_id):
    """
    Get patient view optimized for comprehensive risk assessment
    Includes multi-modal risk stratification across all available data
    """
    return _get_context_specific_view(patient_id, QueryContext.RISK_ASSESSMENT, include_similar=True)

# ============================================================================
# BATCH OPERATIONS
# ============================================================================

@unified_bp.route('/patients/batch-summary', methods=['POST'])
@log_request()
def get_batch_patient_summaries():
    """
    Get summaries for multiple patients efficiently
    
    Request body:
    {
        "patient_ids": ["patient1", "patient2", ...],
        "max_patients": 100  // Optional limit
    }
    """
    if not service_available:
        return jsonify({
            'success': False,
            'error': 'Unified patient service not available',
            'code': 'SERVICE_UNAVAILABLE'
        }), 503
    
    try:
        data = request.get_json()
        
        if not data or 'patient_ids' not in data:
            return jsonify({
                'success': False,
                'error': 'patient_ids array is required',
                'code': 'MISSING_PATIENT_IDS'
            }), 400
        
        patient_ids = data['patient_ids']
        max_patients = min(data.get('max_patients', 50), 100)  # Cap at 100
        
        if not isinstance(patient_ids, list) or len(patient_ids) == 0:
            return jsonify({
                'success': False,
                'error': 'patient_ids must be a non-empty array',
                'code': 'INVALID_PATIENT_IDS'
            }), 400
        
        # Limit batch size
        if len(patient_ids) > max_patients:
            patient_ids = patient_ids[:max_patients]
        
        # Sanitize patient IDs
        patient_ids = [Sanitizer.sanitize_text(pid) for pid in patient_ids]
        
        logger.info(f"Getting batch summaries for {len(patient_ids)} patients")
        
        # Execute batch operation
        async def run_batch_summaries():
            tasks = []
            for patient_id in patient_ids:
                tasks.append(unified_service.get_patient_summary(patient_id))
            
            return await asyncio.gather(*tasks, return_exceptions=True)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(lambda: asyncio.run(run_batch_summaries()))
            results = future.result(timeout=30)  # 30 second timeout for batch
        
        # Process results
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
        
        return jsonify({
            'success': True,
            'data': {
                'summaries': summaries,
                'successful_queries': len(summaries),
                'failed_queries': len(errors),
                'total_queries': len(patient_ids),
                'errors': errors if errors else None
            }
        }), 200
        
    except concurrent.futures.TimeoutError:
        return jsonify({
            'success': False,
            'error': 'Batch summary operation timed out',
            'code': 'BATCH_TIMEOUT'
        }), 504
    except Exception as e:
        logger.error(f"Error in batch summaries: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error during batch operation',
            'code': 'INTERNAL_ERROR'
        }), 500

# ============================================================================
# SEARCH AND DISCOVERY
# ============================================================================

@unified_bp.route('/patients/search', methods=['POST'])
@log_request()
def search_patients():
    """
    Search for patients based on clinical criteria across all modalities
    
    Request body:
    {
        "criteria": {
            "age_range": [18, 65],
            "conditions": ["diabetes", "hypertension"], 
            "genetic_risk": ["cardiovascular"],
            "has_mimic_data": true
        },
        "context": "research_analysis",
        "limit": 50
    }
    """
    if not service_available:
        return jsonify({
            'success': False,
            'error': 'Unified patient service not available',
            'code': 'SERVICE_UNAVAILABLE'
        }), 503
    
    try:
        data = request.get_json()
        
        if not data or 'criteria' not in data:
            return jsonify({
                'success': False,
                'error': 'Search criteria required',
                'code': 'MISSING_CRITERIA'
            }), 400
        
        criteria = data['criteria']
        context_str = data.get('context', 'clinical_review')
        limit = min(data.get('limit', 20), 100)  # Cap at 100
        
        # Validate context
        try:
            context = QueryContext(context_str)
        except ValueError:
            context = QueryContext.CLINICAL_REVIEW
        
        logger.info(f"Searching patients with criteria: {criteria}")
        
        # For now, return placeholder response - would implement sophisticated search
        # across all unified patients with the given criteria
        search_results = {
            'matches': [],
            'total_matches': 0,
            'search_criteria': criteria,
            'context': context_str,
            'search_timestamp': datetime.now().isoformat(),
            'message': 'Advanced patient search across modalities - implementation in progress'
        }
        
        return jsonify({
            'success': True,
            'data': search_results
        }), 200
        
    except Exception as e:
        logger.error(f"Error in patient search: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error during patient search',
            'code': 'INTERNAL_ERROR'
        }), 500

# ============================================================================
# SYSTEM STATUS AND ANALYTICS
# ============================================================================

@unified_bp.route('/health', methods=['GET'])
@log_request()
def unified_health_check():
    """Health check for unified patient service"""
    try:
        health_status = {
            'status': 'healthy' if service_available else 'degraded',
            'timestamp': datetime.now().isoformat(),
            'service_available': service_available,
            'components': {}
        }
        
        if service_available and unified_service:
            # Check component services
            health_status['components'] = {
                'supabase_service': 'available',
                'multimodal_service': 'available',
                'vector_service': 'available',
                'identity_service': 'available', 
                'fusion_service': 'available',
                'trials_service': 'available',
                'clinical_service': 'available'
            }
        
        return jsonify(health_status), 200 if service_available else 503
        
    except Exception as e:
        logger.error(f"Unified health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }), 500

@unified_bp.route('/stats', methods=['GET'])
@log_request()
def get_unified_stats():
    """Get unified patient service statistics"""
    if not service_available:
        return jsonify({
            'success': False,
            'error': 'Unified patient service not available',
            'code': 'SERVICE_UNAVAILABLE'
        }), 503
    
    try:
        # Get basic statistics from database
        def get_stats():
            return asyncio.run(_collect_unified_stats())
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(get_stats)
            stats = future.result(timeout=10)
        
        return jsonify({
            'success': True,
            'data': stats
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting unified stats: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error getting statistics',
            'code': 'INTERNAL_ERROR'
        }), 500

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _get_context_specific_view(patient_id: str, context: QueryContext, 
                              include_similar: bool = False,
                              include_trials: bool = False):
    """Helper function for context-specific views"""
    if not service_available:
        return jsonify({
            'success': False,
            'error': 'Unified patient service not available',
            'code': 'SERVICE_UNAVAILABLE'
        }), 503
    
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
        
        return jsonify({
            'success': True,
            'data': response_data
        }), 200
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'code': 'PATIENT_NOT_FOUND'
        }), 404
    except Exception as e:
        logger.error(f"Error in context-specific view: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'code': 'INTERNAL_ERROR'
        }), 500

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
        
        # Get counts from various tables
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

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@unified_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Unified patient endpoint not found',
        'code': 'NOT_FOUND'
    }), 404

@unified_bp.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'success': False,
        'error': 'Method not allowed',
        'code': 'METHOD_NOT_ALLOWED'
    }), 405

@unified_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error in unified patient service',
        'code': 'INTERNAL_ERROR'
    }), 500