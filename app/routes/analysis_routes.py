from flask import Blueprint, request, jsonify
import logging
from datetime import datetime
from app.services.clinical_analysis_service import ClinicalAnalysisService
from app.services.icd10_vector_matcher import ICD10VectorMatcher
from app.services.analysis_storage_service import AnalysisStorageService
from app.utils.validation import Validator, ValidationError
from app.utils.sanitization import Sanitizer
from app.middleware.security import log_request

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Blueprint for analysis routes
analysis_bp = Blueprint('analysis', __name__)

# Initialize services
clinical_service = ClinicalAnalysisService()
icd_matcher = ICD10VectorMatcher()
storage_service = AnalysisStorageService()

@analysis_bp.route('/extract', methods=['POST'])
@log_request()
def extract_clinical_entities():
    """
    Extract clinical entities from patient note text
    
    Request body:
    {
        "note_text": "Patient note content",
        "patient_context": {  // Optional
            "age": 45,
            "gender": "M",
            "medical_history": "hypertension, diabetes"
        }
    }
    
    Returns:
    {
        "success": true,
        "data": {
            "symptoms": [...],
            "conditions": [...],
            "medications": [...],
            "vital_signs": [...],
            "procedures": [...],
            "abnormal_findings": [...],
            "overall_assessment": {...},
            "analysis_timestamp": "...",
            "model_version": "..."
        }
    }
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Request must be JSON',
                'code': 'INVALID_CONTENT_TYPE'
            }), 400
        
        data = request.get_json()
        
        # Validate required fields
        if 'note_text' not in data:
            return jsonify({
                'success': False,
                'error': 'note_text is required',
                'code': 'MISSING_REQUIRED_FIELD'
            }), 400
        
        # Sanitize input
        note_text = Sanitizer.sanitize_text(data['note_text'])
        patient_context = data.get('patient_context', {})
        
        # Validate note text
        if not note_text or len(note_text.strip()) < 10:
            return jsonify({
                'success': False,
                'error': 'note_text must be at least 10 characters',
                'code': 'INVALID_NOTE_TEXT'
            }), 400
        
        # Sanitize patient context
        if patient_context:
            for key, value in patient_context.items():
                if isinstance(value, str):
                    patient_context[key] = Sanitizer.sanitize_text(value)
        
        logger.info(f"Processing clinical entity extraction for note length: {len(note_text)}")
        
        # Check cache first
        cached_result = storage_service.get_cached_analysis(note_text, patient_context, 'extract')
        if cached_result:
            logger.info("Using cached analysis result")
            cached_result['request_metadata'] = {
                'note_length': len(note_text),
                'has_patient_context': bool(patient_context),
                'processed_at': datetime.utcnow().isoformat(),
                'from_cache': True
            }
            return jsonify({
                'success': True,
                'data': cached_result
            }), 200
        
        # Create analysis session
        session_id = storage_service.create_analysis_session(
            note_id=data.get('note_id'),
            patient_id=data.get('patient_id'),
            analysis_type='extract',
            request_data={'note_text': note_text, 'patient_context': patient_context}
        )
        
        # Update session status
        storage_service.update_analysis_session(session_id, status='processing')
        
        # Extract clinical entities
        result = clinical_service.extract_clinical_entities(note_text, patient_context)
        
        # Check for errors in extraction
        if 'error' in result:
            logger.error(f"Clinical analysis error: {result['error']}")
            # Update session with error
            storage_service.update_analysis_session(session_id, 
                status='failed', 
                error_message=result['error']
            )
            return jsonify({
                'success': False,
                'error': 'Clinical analysis failed',
                'details': result['error'],
                'code': 'ANALYSIS_FAILED'
            }), 500
        
        # Store extracted entities
        try:
            all_entities = []
            for entity_type in ['symptoms', 'conditions', 'medications', 'vital_signs', 'procedures', 'abnormal_findings']:
                for entity in result.get(entity_type, []):
                    entity_with_type = entity.copy()
                    entity_with_type['type'] = entity_type[:-1] if entity_type.endswith('s') else entity_type  # Remove plural
                    all_entities.append(entity_with_type)
            
            if all_entities:
                storage_service.store_clinical_entities(session_id, all_entities)
        except Exception as storage_error:
            logger.warning(f"Failed to store entities to database: {storage_error}")
        
        # Update session with success
        assessment = result.get('overall_assessment', {})
        storage_service.update_analysis_session(session_id,
            status='completed',
            response_data=result,
            risk_level=assessment.get('risk_level', 'low'),
            requires_immediate_attention=assessment.get('requires_immediate_attention', False)
        )
        
        # Cache the result
        try:
            storage_service.cache_analysis_result(note_text, patient_context, 'extract', result)
        except Exception as cache_error:
            logger.warning(f"Failed to cache analysis result: {cache_error}")
        
        # Add metadata
        result['request_metadata'] = {
            'note_length': len(note_text),
            'has_patient_context': bool(patient_context),
            'processed_at': datetime.utcnow().isoformat(),
            'session_id': session_id,
            'from_cache': False
        }
        
        logger.info(f"Successfully extracted {sum(len(result.get(k, [])) for k in ['symptoms', 'conditions', 'medications', 'vital_signs', 'procedures', 'abnormal_findings'])} entities")
        
        return jsonify({
            'success': True,
            'data': result
        }), 200
        
    except Exception as e:
        logger.error(f"Error in extract_clinical_entities: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error during clinical analysis',
            'code': 'INTERNAL_ERROR'
        }), 500

@analysis_bp.route('/diagnose', methods=['POST'])
@log_request()
def diagnose_with_icd_mapping():
    """
    Extract clinical entities and map them to ICD-10 codes
    
    Request body:
    {
        "note_text": "Patient note content",
        "patient_context": {  // Optional
            "age": 45,
            "gender": "M"
        },
        "options": {  // Optional
            "include_low_confidence": false,
            "max_icd_matches": 5
        }
    }
    
    Returns:
    {
        "success": true,
        "data": {
            // All clinical entity data
            "icd_mappings": {
                "conditions": [...],
                "symptoms": [...],
                "procedures": [...],
                "summary": {...}
            }
        }
    }
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Request must be JSON',
                'code': 'INVALID_CONTENT_TYPE'
            }), 400
        
        data = request.get_json()
        
        # Validate required fields
        if 'note_text' not in data:
            return jsonify({
                'success': False,
                'error': 'note_text is required',
                'code': 'MISSING_REQUIRED_FIELD'
            }), 400
        
        # Sanitize input
        note_text = Sanitizer.sanitize_text(data['note_text'])
        patient_context = data.get('patient_context', {})
        options = data.get('options', {})
        
        # Validate note text
        if not note_text or len(note_text.strip()) < 10:
            return jsonify({
                'success': False,
                'error': 'note_text must be at least 10 characters',
                'code': 'INVALID_NOTE_TEXT'
            }), 400
        
        logger.info(f"Processing diagnosis with ICD mapping for note length: {len(note_text)}")
        
        # Step 1: Extract clinical entities
        clinical_result = clinical_service.extract_clinical_entities(note_text, patient_context)
        
        if 'error' in clinical_result:
            logger.error(f"Clinical analysis error: {clinical_result['error']}")
            return jsonify({
                'success': False,
                'error': 'Clinical analysis failed',
                'details': clinical_result['error'],
                'code': 'ANALYSIS_FAILED'
            }), 500
        
        # Step 2: Map to ICD-10 codes
        enhanced_result = icd_matcher.map_clinical_entities_to_icd(clinical_result)
        
        # Add processing metadata
        enhanced_result['request_metadata'] = {
            'note_length': len(note_text),
            'has_patient_context': bool(patient_context),
            'processed_at': datetime.utcnow().isoformat(),
            'total_entities_extracted': sum(len(clinical_result.get(k, [])) for k in ['symptoms', 'conditions', 'medications', 'vital_signs', 'procedures', 'abnormal_findings'])
        }
        
        # Get ICD cache info for metadata
        cache_info = icd_matcher.get_cache_info()
        enhanced_result['icd_cache_info'] = cache_info
        
        mappings_summary = enhanced_result.get('icd_mappings', {}).get('summary', {})
        logger.info(f"Successfully mapped {mappings_summary.get('total_mappings', 0)} entities to ICD codes")
        
        return jsonify({
            'success': True,
            'data': enhanced_result
        }), 200
        
    except Exception as e:
        logger.error(f"Error in diagnose_with_icd_mapping: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error during diagnosis',
            'code': 'INTERNAL_ERROR'
        }), 500

@analysis_bp.route('/priority/<note_id>', methods=['GET'])
@log_request()
def get_high_priority_findings(note_id):
    """
    Get high-priority findings for a specific note
    
    Query parameters:
    - risk_threshold: minimum risk level ('moderate', 'high', 'critical') - default: 'high'
    - include_details: include full entity details (true/false) - default: false
    
    Returns:
    {
        "success": true,
        "data": {
            "note_id": "note_123",
            "priority_findings": [...],
            "summary": {
                "total_findings": 3,
                "critical_findings": 1,
                "high_risk_findings": 2,
                "requires_immediate_attention": true
            }
        }
    }
    """
    try:
        # Sanitize note_id
        note_id = Sanitizer.sanitize_text(note_id)
        
        # Get query parameters
        risk_threshold = request.args.get('risk_threshold', 'high')
        include_details = request.args.get('include_details', 'false').lower() == 'true'
        
        # Validate risk threshold
        valid_thresholds = ['moderate', 'high', 'critical']
        if risk_threshold not in valid_thresholds:
            return jsonify({
                'success': False,
                'error': f'Invalid risk_threshold. Must be one of: {valid_thresholds}',
                'code': 'INVALID_PARAMETER'
            }), 400
        
        logger.info(f"Retrieving priority findings for note {note_id} with threshold {risk_threshold}")
        
        # Get priority findings from storage
        priority_sessions = storage_service.get_priority_findings(
            note_id=note_id, 
            risk_threshold=risk_threshold
        )
        
        if not priority_sessions:
            return jsonify({
                'success': True,
                'data': {
                    'note_id': note_id,
                    'priority_findings': [],
                    'summary': {
                        'total_findings': 0,
                        'critical_findings': 0,
                        'high_risk_findings': 0,
                        'requires_immediate_attention': False
                    },
                    'message': 'No priority findings found for this note'
                }
            }), 200
        
        # Process findings
        priority_findings = []
        summary = {
            'total_findings': 0,
            'critical_findings': 0,
            'high_risk_findings': 0,
            'requires_immediate_attention': False
        }
        
        for session in priority_sessions:
            # Get entities for this session if details requested
            entities = []
            if include_details:
                entities = storage_service.get_session_entities(session['session_id'])
            
            finding = {
                'session_id': session['session_id'],
                'analysis_type': session['analysis_type'],
                'risk_level': session['risk_level'],
                'requires_immediate_attention': session['requires_immediate_attention'],
                'created_at': session['created_at'],
                'overall_assessment': session.get('response_data', {}).get('overall_assessment', {}),
                'entity_count': len(entities) if include_details else None
            }
            
            if include_details:
                # Group entities by type for easier reading
                grouped_entities = {}
                for entity in entities:
                    entity_type = entity['entity_type']
                    if entity_type not in grouped_entities:
                        grouped_entities[entity_type] = []
                    grouped_entities[entity_type].append({
                        'entity_text': entity['entity_text'],
                        'confidence': float(entity['confidence']),
                        'severity': entity['severity'],
                        'status': entity['status']
                    })
                finding['entities'] = grouped_entities
            
            priority_findings.append(finding)
            
            # Update summary
            summary['total_findings'] += 1
            if session['risk_level'] == 'critical':
                summary['critical_findings'] += 1
            elif session['risk_level'] == 'high':
                summary['high_risk_findings'] += 1
            
            if session['requires_immediate_attention']:
                summary['requires_immediate_attention'] = True
        
        logger.info(f"Found {len(priority_findings)} priority findings for note {note_id}")
        
        return jsonify({
            'success': True,
            'data': {
                'note_id': note_id,
                'priority_findings': priority_findings,
                'summary': summary,
                'query_parameters': {
                    'risk_threshold': risk_threshold,
                    'include_details': include_details
                },
                'retrieved_at': datetime.utcnow().isoformat()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error in get_high_priority_findings: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error while retrieving priority findings',
            'code': 'INTERNAL_ERROR'
        }), 500

@analysis_bp.route('/batch', methods=['POST'])
@log_request()
def batch_analysis():
    """
    Process multiple patient notes for clinical analysis
    
    Request body:
    {
        "notes": [
            {
                "note_id": "optional_id_1",
                "note_text": "Patient note content 1",
                "patient_context": {"age": 45, "gender": "M"}
            },
            {
                "note_id": "optional_id_2", 
                "note_text": "Patient note content 2",
                "patient_context": {"age": 62, "gender": "F"}
            }
        ],
        "options": {
            "include_icd_mapping": true,
            "include_priority_analysis": true
        }
    }
    
    Returns:
    {
        "success": true,
        "data": {
            "results": [...],
            "summary": {
                "total_notes": 2,
                "successful_analyses": 2,
                "failed_analyses": 0,
                "total_entities": 45,
                "high_priority_cases": 1
            }
        }
    }
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Request must be JSON',
                'code': 'INVALID_CONTENT_TYPE'
            }), 400
        
        data = request.get_json()
        
        # Validate required fields
        if 'notes' not in data or not isinstance(data['notes'], list):
            return jsonify({
                'success': False,
                'error': 'notes array is required',
                'code': 'MISSING_REQUIRED_FIELD'
            }), 400
        
        notes = data['notes']
        options = data.get('options', {})
        
        # Validate batch size
        if len(notes) == 0:
            return jsonify({
                'success': False,
                'error': 'At least one note is required',
                'code': 'EMPTY_BATCH'
            }), 400
        
        if len(notes) > 50:  # Limit batch size
            return jsonify({
                'success': False,
                'error': 'Maximum 50 notes per batch',
                'code': 'BATCH_TOO_LARGE'
            }), 400
        
        logger.info(f"Processing batch analysis for {len(notes)} notes")
        
        results = []
        summary = {
            'total_notes': len(notes),
            'successful_analyses': 0,
            'failed_analyses': 0,
            'total_entities': 0,
            'high_priority_cases': 0
        }
        
        # Process each note
        for i, note_data in enumerate(notes):
            try:
                # Validate individual note
                if 'note_text' not in note_data:
                    results.append({
                        'note_id': note_data.get('note_id', f'note_{i}'),
                        'success': False,
                        'error': 'note_text is required',
                        'code': 'MISSING_NOTE_TEXT'
                    })
                    summary['failed_analyses'] += 1
                    continue
                
                note_text = Sanitizer.sanitize_text(note_data['note_text'])
                patient_context = note_data.get('patient_context', {})
                
                # Extract clinical entities
                clinical_result = clinical_service.extract_clinical_entities(note_text, patient_context)
                
                if 'error' in clinical_result:
                    results.append({
                        'note_id': note_data.get('note_id', f'note_{i}'),
                        'success': False,
                        'error': 'Clinical analysis failed',
                        'details': clinical_result['error'],
                        'code': 'ANALYSIS_FAILED'
                    })
                    summary['failed_analyses'] += 1
                    continue
                
                # Optional ICD mapping
                if options.get('include_icd_mapping', True):
                    clinical_result = icd_matcher.map_clinical_entities_to_icd(clinical_result)
                
                # Optional priority analysis
                high_priority_findings = []
                if options.get('include_priority_analysis', True):
                    high_priority_findings = clinical_service.get_high_priority_findings(clinical_result)
                    if high_priority_findings:
                        summary['high_priority_cases'] += 1
                
                # Count entities
                entity_count = sum(len(clinical_result.get(k, [])) for k in ['symptoms', 'conditions', 'medications', 'vital_signs', 'procedures', 'abnormal_findings'])
                summary['total_entities'] += entity_count
                
                # Add to results
                result_entry = {
                    'note_id': note_data.get('note_id', f'note_{i}'),
                    'success': True,
                    'data': clinical_result,
                    'metadata': {
                        'entity_count': entity_count,
                        'high_priority_findings_count': len(high_priority_findings),
                        'requires_immediate_attention': clinical_result.get('overall_assessment', {}).get('requires_immediate_attention', False)
                    }
                }
                
                results.append(result_entry)
                summary['successful_analyses'] += 1
                
            except Exception as e:
                logger.error(f"Error processing note {i}: {str(e)}")
                results.append({
                    'note_id': note_data.get('note_id', f'note_{i}'),
                    'success': False,
                    'error': 'Processing error',
                    'details': str(e),
                    'code': 'PROCESSING_ERROR'
                })
                summary['failed_analyses'] += 1
        
        logger.info(f"Batch analysis completed: {summary['successful_analyses']}/{summary['total_notes']} successful")
        
        return jsonify({
            'success': True,
            'data': {
                'results': results,
                'summary': summary,
                'processed_at': datetime.utcnow().isoformat()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error in batch_analysis: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error during batch analysis',
            'code': 'INTERNAL_ERROR'
        }), 500

@analysis_bp.route('/health', methods=['GET'])
@log_request()
def health_check():
    """
    Health check endpoint for the analysis service
    """
    try:
        # Check service availability
        cache_info = icd_matcher.get_cache_info()
        
        # Get storage and cache statistics
        try:
            analysis_cache_stats = storage_service.get_cache_stats()
            storage_available = True
        except Exception as storage_error:
            logger.warning(f"Storage service unavailable: {storage_error}")
            analysis_cache_stats = {
                'error': str(storage_error),
                'available': False
            }
            storage_available = False
        
        health_status = {
            'status': 'healthy' if storage_available else 'degraded',
            'timestamp': datetime.utcnow().isoformat(),
            'services': {
                'clinical_analysis': 'available',
                'icd_matcher': 'available',
                'storage_service': 'available' if storage_available else 'unavailable',
                'icd_cache': {
                    'loaded': cache_info['cache_loaded'],
                    'total_codes': cache_info['total_icd_codes']
                },
                'analysis_cache': analysis_cache_stats
            }
        }
        
        # Clean up expired cache if storage is available
        if storage_available:
            try:
                deleted_count = storage_service.cleanup_expired_cache()
                if deleted_count > 0:
                    health_status['maintenance'] = {
                        'expired_cache_cleaned': deleted_count
                    }
            except Exception as cleanup_error:
                logger.warning(f"Cache cleanup failed: {cleanup_error}")
        
        return jsonify(health_status), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e)
        }), 500

# Error handlers for the blueprint
@analysis_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'code': 'NOT_FOUND'
    }), 404

@analysis_bp.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'success': False,
        'error': 'Method not allowed',
        'code': 'METHOD_NOT_ALLOWED'
    }), 405

@analysis_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'code': 'INTERNAL_ERROR'
    }), 500