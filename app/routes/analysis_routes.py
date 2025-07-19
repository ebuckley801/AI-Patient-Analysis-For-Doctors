from flask import Blueprint, request, jsonify
import logging
import asyncio
from datetime import datetime
from app.services.clinical_analysis_service import ClinicalAnalysisService
from app.services.enhanced_clinical_analysis import EnhancedClinicalAnalysisService, create_enhanced_clinical_analysis_service
from app.services.icd10_vector_matcher import ICD10VectorMatcher
from app.services.analysis_storage_service import AnalysisStorageService
from app.services.async_clinical_analysis import AsyncClinicalAnalysis, BatchAnalysisConfig
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
enhanced_service = create_enhanced_clinical_analysis_service()
icd_matcher = ICD10VectorMatcher()
storage_service = AnalysisStorageService()
async_clinical_service = AsyncClinicalAnalysis(use_enhanced_analysis=True)

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

@analysis_bp.route('/batch-async', methods=['POST'])
@log_request()
def batch_analysis_async():
    """
    High-performance async batch processing for large-scale clinical analysis
    
    Request body:
    {
        "notes": [
            {
                "note_id": "optional_id_1",
                "note_text": "Patient note content 1", 
                "patient_context": {"age": 45, "gender": "M"},
                "patient_id": "patient_123"
            }
        ],
        "config": {
            "max_concurrent": 10,
            "timeout_seconds": 30,
            "include_icd_mapping": true,
            "include_storage": true,
            "chunk_size": 50,
            "retry_attempts": 2
        }
    }
    
    Returns:
    {
        "success": true,
        "results": [...],
        "summary": {
            "total_notes": 100,
            "successful_analyses": 98,
            "cache_hit_rate": 0.15,
            "average_processing_time_ms": 1200
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
        config_data = data.get('config', {})
        
        # Validate batch size
        if len(notes) == 0:
            return jsonify({
                'success': False,
                'error': 'At least one note is required',
                'code': 'EMPTY_BATCH'
            }), 400
        
        if len(notes) > 1000:  # Higher limit for async processing
            return jsonify({
                'success': False,
                'error': 'Maximum 1000 notes per async batch',
                'code': 'BATCH_TOO_LARGE'
            }), 400
        
        # Sanitize input notes
        sanitized_notes = []
        for i, note_data in enumerate(notes):
            if 'note_text' not in note_data:
                return jsonify({
                    'success': False,
                    'error': f'note_text is required for note at index {i}',
                    'code': 'MISSING_NOTE_TEXT'
                }), 400
            
            sanitized_note = {
                'note_id': note_data.get('note_id', f'async_note_{i}'),
                'note_text': Sanitizer.sanitize_text(note_data['note_text']),
                'patient_context': note_data.get('patient_context', {}),
                'patient_id': note_data.get('patient_id')
            }
            
            # Sanitize patient context
            if sanitized_note['patient_context']:
                for key, value in sanitized_note['patient_context'].items():
                    if isinstance(value, str):
                        sanitized_note['patient_context'][key] = Sanitizer.sanitize_text(value)
            
            sanitized_notes.append(sanitized_note)
        
        # Create batch analysis configuration
        config = BatchAnalysisConfig(
            max_concurrent=min(config_data.get('max_concurrent', 10), 20),  # Cap at 20
            timeout_seconds=min(config_data.get('timeout_seconds', 30), 60),  # Cap at 60s
            include_icd_mapping=config_data.get('include_icd_mapping', True),
            include_storage=config_data.get('include_storage', True),
            chunk_size=min(config_data.get('chunk_size', 50), 100),  # Cap at 100
            retry_attempts=min(config_data.get('retry_attempts', 2), 3)  # Cap at 3
        )
        
        logger.info(f"Starting async batch analysis for {len(sanitized_notes)} notes")
        
        # Run async batch analysis
        def run_async_batch():
            return asyncio.run(async_clinical_service.batch_analyze_notes(sanitized_notes, config))
        
        # Execute in thread to avoid blocking Flask
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_async_batch)
            result = future.result(timeout=300)  # 5 minute total timeout
        
        logger.info(f"Async batch analysis completed: {result['summary']['successful_analyses']}/{len(sanitized_notes)} successful")
        
        return jsonify(result), 200
        
    except concurrent.futures.TimeoutError:
        logger.error("Async batch analysis timed out")
        return jsonify({
            'success': False,
            'error': 'Batch analysis timed out after 5 minutes',
            'code': 'BATCH_TIMEOUT'
        }), 504
    except Exception as e:
        logger.error(f"Error in async batch analysis: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error during async batch analysis',
            'details': str(e),
            'code': 'INTERNAL_ERROR'
        }), 500

@analysis_bp.route('/priority-scan', methods=['POST'])
@log_request()
def priority_scan():
    """
    High-speed priority scanning for identifying high-risk cases
    Optimized for rapid triage of large note volumes
    
    Request body:
    {
        "notes": [
            {
                "note_id": "scan_1",
                "note_text": "Patient note content",
                "patient_context": {"age": 45, "gender": "M"}
            }
        ],
        "risk_threshold": "high"  // "moderate", "high", "critical"
    }
    
    Returns:
    {
        "success": true,
        "priority_cases": [
            {
                "note_id": "scan_1",
                "risk_level": "critical",
                "requires_immediate_attention": true,
                "primary_concerns": ["chest pain", "shortness of breath"]
            }
        ],
        "scan_summary": {
            "total_notes_scanned": 500,
            "priority_cases_found": 23,
            "scan_time_ms": 15000
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
        risk_threshold = data.get('risk_threshold', 'high')
        
        # Validate risk threshold
        valid_thresholds = ['moderate', 'high', 'critical']
        if risk_threshold not in valid_thresholds:
            return jsonify({
                'success': False,
                'error': f'Invalid risk_threshold. Must be one of: {valid_thresholds}',
                'code': 'INVALID_PARAMETER'
            }), 400
        
        # Validate batch size for scanning
        if len(notes) == 0:
            return jsonify({
                'success': False,
                'error': 'At least one note is required',
                'code': 'EMPTY_BATCH'
            }), 400
        
        if len(notes) > 2000:  # Higher limit for priority scanning
            return jsonify({
                'success': False,
                'error': 'Maximum 2000 notes per priority scan',
                'code': 'SCAN_BATCH_TOO_LARGE'
            }), 400
        
        # Sanitize input notes
        sanitized_notes = []
        for i, note_data in enumerate(notes):
            if 'note_text' not in note_data:
                return jsonify({
                    'success': False,
                    'error': f'note_text is required for note at index {i}',
                    'code': 'MISSING_NOTE_TEXT'
                }), 400
            
            sanitized_note = {
                'note_id': note_data.get('note_id', f'scan_note_{i}'),
                'note_text': Sanitizer.sanitize_text(note_data['note_text']),
                'patient_context': note_data.get('patient_context', {})
            }
            
            # Sanitize patient context
            if sanitized_note['patient_context']:
                for key, value in sanitized_note['patient_context'].items():
                    if isinstance(value, str):
                        sanitized_note['patient_context'][key] = Sanitizer.sanitize_text(value)
            
            sanitized_notes.append(sanitized_note)
        
        logger.info(f"Starting priority scan for {len(sanitized_notes)} notes with {risk_threshold} threshold")
        
        # Run async priority scan
        def run_async_scan():
            return asyncio.run(async_clinical_service.priority_scan_async(sanitized_notes, risk_threshold))
        
        # Execute in thread to avoid blocking Flask
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_async_scan)
            result = future.result(timeout=180)  # 3 minute timeout for scanning
        
        logger.info(f"Priority scan completed: {result['scan_summary']['priority_cases_found']} priority cases found")
        
        return jsonify(result), 200
        
    except concurrent.futures.TimeoutError:
        logger.error("Priority scan timed out")
        return jsonify({
            'success': False,
            'error': 'Priority scan timed out after 3 minutes',
            'code': 'SCAN_TIMEOUT'
        }), 504
    except Exception as e:
        logger.error(f"Error in priority scan: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error during priority scan',
            'details': str(e),
            'code': 'INTERNAL_ERROR'
        }), 500

@analysis_bp.route('/extract-enhanced', methods=['POST'])
@log_request()
def extract_clinical_entities_enhanced():
    """
    Enhanced clinical entity extraction with Faiss + NLP integration
    
    Request body:
    {
        "note_text": "Patient note content",
        "patient_context": {  // Optional
            "age": 45,
            "gender": "M",
            "medical_history": "hypertension, diabetes"
        },
        "include_icd_mapping": true,  // Optional, default true
        "icd_top_k": 5,  // Optional, default 5
        "enable_nlp_preprocessing": true  // Optional, default true
    }
    
    Returns enhanced analysis with performance metrics and NLP analysis
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Invalid JSON in request body',
                'code': 'INVALID_JSON'
            }), 400
        
        # Validate required fields
        if 'note_text' not in data:
            return jsonify({
                'success': False,
                'error': 'note_text is required',
                'code': 'MISSING_NOTE_TEXT'
            }), 400
        
        note_text = Sanitizer.sanitize_text(data['note_text'])
        patient_context = data.get('patient_context', {})
        include_icd_mapping = data.get('include_icd_mapping', True)
        icd_top_k = data.get('icd_top_k', 5)
        enable_nlp_preprocessing = data.get('enable_nlp_preprocessing', True)
        
        if len(note_text.strip()) < 5:
            return jsonify({
                'success': False,
                'error': 'Note text is too short (minimum 5 characters)',
                'code': 'NOTE_TOO_SHORT'
            }), 400
        
        logger.info(f"Enhanced extraction request: {len(note_text)} chars, ICD mapping: {include_icd_mapping}")
        
        # Use enhanced service if available, fallback to standard
        if enhanced_service:
            result = enhanced_service.extract_clinical_entities_enhanced(
                note_text,
                patient_context=patient_context,
                include_icd_mapping=include_icd_mapping,
                icd_top_k=icd_top_k,
                enable_nlp_preprocessing=enable_nlp_preprocessing
            )
        else:
            # Fallback to standard analysis
            result = clinical_service.extract_clinical_entities(note_text, patient_context)
            if include_icd_mapping:
                result = icd_matcher.map_clinical_entities_to_icd(result)
            result['enhanced_service_available'] = False
        
        if 'error' in result:
            return jsonify({
                'success': False,
                'error': 'Enhanced clinical analysis failed',
                'details': result['error'],
                'code': 'ENHANCED_ANALYSIS_FAILED'
            }), 500
        
        # Store in database if storage is enabled
        try:
            if storage_service and data.get('note_id') and data.get('patient_id'):
                session_id = storage_service.create_analysis_session(
                    note_id=data['note_id'],
                    patient_id=data['patient_id'],
                    analysis_type='enhanced_extract',
                    request_data=data
                )
                
                # Store entities if found
                all_entities = []
                for entity_type in ['symptoms', 'conditions', 'medications', 'vital_signs', 'procedures', 'abnormal_findings']:
                    for entity in result.get(entity_type, []):
                        entity_with_type = entity.copy()
                        entity_with_type['type'] = entity_type[:-1] if entity_type.endswith('s') else entity_type
                        all_entities.append(entity_with_type)
                
                if all_entities:
                    storage_service.store_clinical_entities(session_id, all_entities)
                
                # Update session
                assessment = result.get('overall_assessment', {})
                storage_service.update_analysis_session(session_id, {
                    'status': 'completed',
                    'response_data': result,
                    'risk_level': assessment.get('risk_level', 'low'),
                    'requires_immediate_attention': assessment.get('requires_immediate_attention', False)
                })
                
                result['session_id'] = session_id
                
        except Exception as storage_error:
            logger.warning(f"Storage failed: {storage_error}")
            # Continue without storage
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except ValidationError as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'code': 'VALIDATION_ERROR'
        }), 400
    except Exception as e:
        logger.error(f"Error in enhanced extraction: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error during enhanced analysis',
            'details': str(e),
            'code': 'INTERNAL_ERROR'
        }), 500


@analysis_bp.route('/performance-stats', methods=['GET'])
@log_request()
def get_performance_stats():
    """
    Get comprehensive performance statistics for all analysis services
    
    Returns performance metrics including Faiss/numpy usage, timing, and service status
    """
    try:
        stats = {
            'timestamp': datetime.utcnow().isoformat(),
            'services': {}
        }
        
        # Enhanced service stats
        if enhanced_service:
            enhanced_stats = enhanced_service.get_performance_stats()
            stats['services']['enhanced_analysis'] = {
                **enhanced_stats,
                'available': True
            }
        else:
            stats['services']['enhanced_analysis'] = {
                'available': False,
                'reason': 'Enhanced service initialization failed'
            }
        
        # ICD matcher stats
        try:
            icd_stats = icd_matcher.get_cache_info()
            if hasattr(icd_matcher, 'benchmark_performance'):
                benchmark = icd_matcher.benchmark_performance(num_queries=5)
                icd_stats['benchmark'] = benchmark
            stats['services']['icd_matcher'] = icd_stats
        except Exception as e:
            stats['services']['icd_matcher'] = {'error': str(e)}
        
        # Async service stats
        try:
            async_stats = {
                'total_batches': async_clinical_service.batch_stats['total_batches'],
                'enhanced_analyses': async_clinical_service.batch_stats['enhanced_analyses'],
                'standard_analyses': async_clinical_service.batch_stats['standard_analyses'],
                'enhanced_service_active': async_clinical_service.use_enhanced
            }
            stats['services']['async_analysis'] = async_stats
        except Exception as e:
            stats['services']['async_analysis'] = {'error': str(e)}
        
        # Storage service stats
        try:
            storage_stats = storage_service.get_cache_statistics()
            stats['services']['storage'] = storage_stats
        except Exception as e:
            stats['services']['storage'] = {'error': str(e)}
        
        return jsonify({
            'success': True,
            'data': stats
        })
        
    except Exception as e:
        logger.error(f"Error getting performance stats: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve performance statistics',
            'details': str(e),
            'code': 'STATS_ERROR'
        }), 500


@analysis_bp.route('/benchmark', methods=['POST'])
@log_request()
def run_performance_benchmark():
    """
    Run performance benchmark on enhanced analysis service
    
    Request body:
    {
        "num_tests": 10,  // Optional, default 10
        "include_enhanced": true,  // Optional, default true
        "include_standard": true   // Optional, default true
    }
    
    Returns comprehensive benchmark results
    """
    try:
        data = request.get_json() if request.get_json() else {}
        
        num_tests = min(data.get('num_tests', 10), 50)  # Cap at 50 for safety
        include_enhanced = data.get('include_enhanced', True)
        include_standard = data.get('include_standard', True)
        
        benchmark_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'num_tests': num_tests,
            'results': {}
        }
        
        # Enhanced service benchmark
        if include_enhanced and enhanced_service:
            logger.info(f"Running enhanced analysis benchmark ({num_tests} tests)")
            enhanced_benchmark = enhanced_service.benchmark_enhanced_analysis(num_tests)
            benchmark_results['results']['enhanced_analysis'] = enhanced_benchmark
        
        # Standard service benchmark
        if include_standard:
            logger.info(f"Running standard analysis benchmark ({num_tests} tests)")
            
            import time
            test_note = "Patient presents with chest pain and shortness of breath. BP 160/90, HR 110."
            test_context = {'age': 55, 'gender': 'male'}
            
            standard_times = []
            total_start = time.time()
            
            for i in range(num_tests):
                start = time.time()
                result = clinical_service.extract_clinical_entities(test_note, test_context)
                duration = (time.time() - start) * 1000
                standard_times.append(duration)
            
            total_time = time.time() - total_start
            
            benchmark_results['results']['standard_analysis'] = {
                'num_tests': num_tests,
                'total_time_seconds': total_time,
                'avg_time_per_analysis_ms': sum(standard_times) / len(standard_times),
                'min_time_ms': min(standard_times),
                'max_time_ms': max(standard_times),
                'analyses_per_second': num_tests / total_time,
                'individual_times_ms': standard_times
            }
        
        # Performance comparison
        if include_enhanced and include_standard and enhanced_service:
            enhanced_avg = benchmark_results['results']['enhanced_analysis']['avg_time_per_analysis_ms']
            standard_avg = benchmark_results['results']['standard_analysis']['avg_time_per_analysis_ms']
            
            benchmark_results['performance_comparison'] = {
                'enhanced_faster': enhanced_avg < standard_avg,
                'speedup_factor': standard_avg / enhanced_avg if enhanced_avg > 0 else 'N/A',
                'time_difference_ms': standard_avg - enhanced_avg
            }
        
        return jsonify({
            'success': True,
            'data': benchmark_results
        })
        
    except Exception as e:
        logger.error(f"Error running benchmark: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Benchmark failed',
            'details': str(e),
            'code': 'BENCHMARK_ERROR'
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