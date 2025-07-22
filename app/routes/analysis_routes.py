from flask import request
from flask_restx import Namespace, Resource, fields
import logging
import asyncio
from datetime import datetime
import concurrent.futures
from flask_jwt_extended import jwt_required, get_jwt_identity

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

# Create Namespace for analysis routes
analysis_ns = Namespace('analysis', description='Clinical analysis operations')

# Initialize services
clinical_service = ClinicalAnalysisService()
enhanced_service = create_enhanced_clinical_analysis_service()
icd_matcher = ICD10VectorMatcher()
storage_service = AnalysisStorageService()
async_clinical_service = AsyncClinicalAnalysis(use_enhanced_analysis=True)

# --- Models for API Documentation ---

# Patient Context Model
patient_context_model = analysis_ns.model('PatientContext', {
    'age': fields.Integer(description='Age of the patient'),
    'gender': fields.String(description='Gender of the patient'),
    'medical_history': fields.String(description='Patient medical history')
})

# Base Note Model for requests
base_note_request_model = analysis_ns.model('BaseNoteRequest', {
    'note_text': fields.String(required=True, description='The patient note content'),
    'patient_context': fields.Nested(patient_context_model, description='Optional patient demographic and medical context')
})

# Clinical Entity Model
clinical_entity_model = analysis_ns.model('ClinicalEntity', {
    'text': fields.String(description='Extracted entity text'),
    'type': fields.String(description='Type of the entity (e.g., symptom, condition)'),
    'confidence': fields.Float(description='Confidence score of the extraction'),
    'severity': fields.String(description='Severity of the entity (e.g., mild, moderate, severe)'),
    'status': fields.String(description='Status of the entity (e.g., present, absent, historical)'),
    'icd_codes': fields.List(fields.String, description='Associated ICD-10 codes')
})

# Overall Assessment Model
overall_assessment_model = analysis_ns.model('OverallAssessment', {
    'risk_level': fields.String(description='Overall risk level (low, moderate, high, critical)'),
    'requires_immediate_attention': fields.Boolean(description='Indicates if immediate attention is required'),
    'summary': fields.String(description='Summary of the overall assessment')
})

# Base Analysis Result Model
base_analysis_result_model = analysis_ns.model('BaseAnalysisResult', {
    'symptoms': fields.List(fields.Nested(clinical_entity_model)),
    'conditions': fields.List(fields.Nested(clinical_entity_model)),
    'medications': fields.List(fields.Nested(clinical_entity_model)),
    'vital_signs': fields.List(fields.Nested(clinical_entity_model)),
    'procedures': fields.List(fields.Nested(clinical_entity_model)),
    'abnormal_findings': fields.List(fields.Nested(clinical_entity_model)),
    'overall_assessment': fields.Nested(overall_assessment_model),
    'analysis_timestamp': fields.String(description='Timestamp of the analysis'),
    'model_version': fields.String(description='Version of the AI model used')
})

# ICD Mapping Model
icd_mapping_detail_model = analysis_ns.model('ICDMappingDetail', {
    'code': fields.String(description='ICD-10 code'),
    'description': fields.String(description='Description of the ICD-10 code'),
    'similarity_score': fields.Float(description='Similarity score to the entity'),
    'entity_text': fields.String(description='Original entity text'),
    'entity_type': fields.String(description='Type of the entity')
})

icd_mappings_summary_model = analysis_ns.model('ICDMappingsSummary', {
    'total_mappings': fields.Integer,
    'unique_icd_codes': fields.Integer,
    'top_conditions': fields.List(fields.String),
    'top_symptoms': fields.List(fields.String)
})

icd_mappings_model = analysis_ns.model('ICDMappings', {
    'conditions': fields.List(fields.Nested(icd_mapping_detail_model)),
    'symptoms': fields.List(fields.Nested(icd_mapping_detail_model)),
    'procedures': fields.List(fields.Nested(icd_mapping_detail_model)),
    'summary': fields.Nested(icd_mappings_summary_model)
})

# Diagnose Response Model
diagnose_response_model = analysis_ns.inherit('DiagnoseResponse', base_analysis_result_model, {
    'icd_mappings': fields.Nested(icd_mappings_model),
    'request_metadata': fields.Raw(description='Metadata about the request and processing'),
    'icd_cache_info': fields.Raw(description='Information about the ICD cache')
})

# Priority Finding Model
priority_entity_detail_model = analysis_ns.model('PriorityEntityDetail', {
    'entity_text': fields.String,
    'confidence': fields.Float,
    'severity': fields.String,
    'status': fields.String
})

priority_finding_model = analysis_ns.model('PriorityFinding', {
    'session_id': fields.String,
    'analysis_type': fields.String,
    'risk_level': fields.String,
    'requires_immediate_attention': fields.Boolean,
    'created_at': fields.String,
    'overall_assessment': fields.Nested(overall_assessment_model),
    'entity_count': fields.Integer,
    'entities': fields.Raw(description='Grouped entities if include_details is true')
})

priority_summary_model = analysis_ns.model('PrioritySummary', {
    'total_findings': fields.Integer,
    'critical_findings': fields.Integer,
    'high_risk_findings': fields.Integer,
    'requires_immediate_attention': fields.Boolean
})

priority_response_model = analysis_ns.model('PriorityResponse', {
    'note_id': fields.String,
    'priority_findings': fields.List(fields.Nested(priority_finding_model)),
    'summary': fields.Nested(priority_summary_model),
    'query_parameters': fields.Raw,
    'retrieved_at': fields.String
})

# Batch Analysis Models
batch_note_request_model = analysis_ns.model('BatchNoteRequest', {
    'note_id': fields.String(description='Optional ID for the note'),
    'note_text': fields.String(required=True, description='The patient note content'),
    'patient_context': fields.Nested(patient_context_model, description='Optional patient demographic and medical context')
})

batch_options_model = analysis_ns.model('BatchOptions', {
    'include_icd_mapping': fields.Boolean(description='Whether to include ICD mapping', default=True),
    'include_priority_analysis': fields.Boolean(description='Whether to include priority analysis', default=True)
})

batch_analysis_request_model = analysis_ns.model('BatchAnalysisRequest', {
    'notes': fields.List(fields.Nested(batch_note_request_model), required=True, description='List of patient notes to analyze'),
    'options': fields.Nested(batch_options_model, description='Optional analysis options')
})

batch_result_metadata_model = analysis_ns.model('BatchResultMetadata', {
    'entity_count': fields.Integer,
    'high_priority_findings_count': fields.Integer,
    'requires_immediate_attention': fields.Boolean
})

batch_individual_result_model = analysis_ns.model('BatchIndividualResult', {
    'note_id': fields.String,
    'success': fields.Boolean,
    'data': fields.Raw(description='Clinical analysis result'),
    'metadata': fields.Nested(batch_result_metadata_model),
    'error': fields.String,
    'details': fields.String,
    'code': fields.String
})

batch_summary_model = analysis_ns.model('BatchSummary', {
    'total_notes': fields.Integer,
    'successful_analyses': fields.Integer,
    'failed_analyses': fields.Integer,
    'total_entities': fields.Integer,
    'high_priority_cases': fields.Integer
})

batch_analysis_response_model = analysis_ns.model('BatchAnalysisResponse', {
    'results': fields.List(fields.Nested(batch_individual_result_model)),
    'summary': fields.Nested(batch_summary_model),
    'processed_at': fields.String
})

# Health Check Models
icd_cache_info_model = analysis_ns.model('ICDCacheInfo', {
    'loaded': fields.Boolean,
    'total_codes': fields.Integer
})

analysis_cache_stats_model = analysis_ns.model('AnalysisCacheStats', {
    'total_entries': fields.Integer,
    'cache_hits': fields.Integer,
    'cache_misses': fields.Integer,
    'hit_rate': fields.Float,
    'available': fields.Boolean,
    'error': fields.String(description='Error message if cache is unavailable')
})

health_services_model = analysis_ns.model('HealthServices', {
    'clinical_analysis': fields.String,
    'icd_matcher': fields.String,
    'storage_service': fields.String,
    'icd_cache': fields.Nested(icd_cache_info_model),
    'analysis_cache': fields.Nested(analysis_cache_stats_model)
})

health_response_model = analysis_ns.model('HealthResponse', {
    'status': fields.String,
    'timestamp': fields.String,
    'services': fields.Nested(health_services_model),
    'maintenance': fields.Raw(description='Maintenance information, e.g., expired cache cleaned')
})

# Async Batch Analysis Models
batch_async_config_model = analysis_ns.model('BatchAsyncConfig', {
    'max_concurrent': fields.Integer(description='Maximum concurrent analyses', default=10),
    'timeout_seconds': fields.Integer(description='Timeout for each analysis in seconds', default=30),
    'include_icd_mapping': fields.Boolean(description='Whether to include ICD mapping', default=True),
    'include_storage': fields.Boolean(description='Whether to store results', default=True),
    'chunk_size': fields.Integer(description='Number of notes per processing chunk', default=50),
    'retry_attempts': fields.Integer(description='Number of retry attempts for failed analyses', default=2)
})

batch_async_request_model = analysis_ns.model('BatchAsyncRequest', {
    'notes': fields.List(fields.Nested(batch_note_request_model), required=True, description='List of patient notes for async analysis'),
    'config': fields.Nested(batch_async_config_model, description='Configuration for async batch processing')
})

batch_async_summary_model = analysis_ns.model('BatchAsyncSummary', {
    'total_notes': fields.Integer,
    'successful_analyses': fields.Integer,
    'failed_analyses': fields.Integer,
    'cache_hit_rate': fields.Float,
    'average_processing_time_ms': fields.Float
})

batch_async_response_model = analysis_ns.model('BatchAsyncResponse', {
    'results': fields.List(fields.Raw(description='Individual analysis results')),
    'summary': fields.Nested(batch_async_summary_model)
})

# Priority Scan Models
priority_scan_request_model = analysis_ns.model('PriorityScanRequest', {
    'notes': fields.List(fields.Nested(batch_note_request_model), required=True, description='List of patient notes for priority scanning'),
    'risk_threshold': fields.String(description='Minimum risk level to consider (moderate, high, critical)', enum=['moderate', 'high', 'critical'], default='high')
})

priority_case_model = analysis_ns.model('PriorityCase', {
    'note_id': fields.String,
    'risk_level': fields.String,
    'requires_immediate_attention': fields.Boolean,
    'primary_concerns': fields.List(fields.String)
})

priority_scan_summary_model = analysis_ns.model('PriorityScanSummary', {
    'total_notes_scanned': fields.Integer,
    'priority_cases_found': fields.Integer,
    'scan_time_ms': fields.Float
})

priority_scan_response_model = analysis_ns.model('PriorityScanResponse', {
    'priority_cases': fields.List(fields.Nested(priority_case_model)),
    'scan_summary': fields.Nested(priority_scan_summary_model)
})

# Enhanced Extract Models
enhanced_extract_request_model = analysis_ns.model('EnhancedExtractRequest', {
    'note_text': fields.String(required=True, description='The patient note content'),
    'patient_context': fields.Nested(patient_context_model, description='Optional patient demographic and medical context'),
    'include_icd_mapping': fields.Boolean(description='Whether to include ICD mapping', default=True),
    'icd_top_k': fields.Integer(description='Number of top ICD matches to return', default=5),
    'enable_nlp_preprocessing': fields.Boolean(description='Whether to enable advanced NLP preprocessing', default=True)
})

icd_match_model = analysis_ns.model('ICDMatch', {
    'code': fields.String,
    'description': fields.String,
    'similarity': fields.Float
})

icd_mapping_model = analysis_ns.model('ICDMapping', {
    'entity': fields.String,
    'entity_type': fields.String,
    'best_match': fields.Nested(icd_match_model),
    'icd_matches': fields.List(fields.Nested(icd_match_model))
})

performance_metrics_model = analysis_ns.model('PerformanceMetrics', {
    'total_time_ms': fields.Float,
    'preprocessing_time_ms': fields.Float,
    'extraction_time_ms': fields.Float,
    'nlp_enhancement_time_ms': fields.Float,
    'icd_mapping_time_ms': fields.Float,
    'chars_processed': fields.Integer,
    'chars_preprocessed': fields.Integer
})

enhanced_analysis_response_model = analysis_ns.inherit('EnhancedAnalysisResponse', base_analysis_result_model, {
    'icd_mappings': fields.List(fields.Nested(icd_mapping_model)),
    'performance_metrics': fields.Nested(performance_metrics_model),
    'icd_search_method': fields.String
})


# Performance Stats Models
service_performance_model = analysis_ns.model('ServicePerformance', {
    'available': fields.Boolean,
    'reason': fields.String(description='Reason if service is unavailable'),
    'total_calls': fields.Integer,
    'average_time_ms': fields.Float,
    'errors': fields.Integer,
    'cache_hits': fields.Integer,
    'cache_misses': fields.Integer,
    'hit_rate': fields.Float,
    'benchmark': fields.Raw(description='Benchmark results if available'),
    'total_batches': fields.Integer,
    'enhanced_analyses': fields.Integer,
    'standard_analyses': fields.Integer,
    'enhanced_service_active': fields.Boolean
})

performance_stats_response_model = analysis_ns.model('PerformanceStatsResponse', {
    'timestamp': fields.String,
    'services': fields.Raw(description='Performance statistics for each service')
})

# Benchmark Models
benchmark_request_model = analysis_ns.model('BenchmarkRequest', {
    'num_tests': fields.Integer(description='Number of tests to run', default=10),
    'include_enhanced': fields.Boolean(description='Whether to benchmark enhanced analysis', default=True),
    'include_standard': fields.Boolean(description='Whether to benchmark standard analysis', default=True)
})

benchmark_result_detail_model = analysis_ns.model('BenchmarkResultDetail', {
    'num_tests': fields.Integer,
    'total_time_seconds': fields.Float,
    'avg_time_per_analysis_ms': fields.Float,
    'min_time_ms': fields.Float,
    'max_time_ms': fields.Float,
    'analyses_per_second': fields.Float,
    'individual_times_ms': fields.List(fields.Float)
})

benchmark_comparison_model = analysis_ns.model('BenchmarkComparison', {
    'enhanced_faster': fields.Boolean,
    'speedup_factor': fields.Raw,
    'time_difference_ms': fields.Float
})

benchmark_response_model = analysis_ns.model('BenchmarkResponse', {
    'timestamp': fields.String,
    'num_tests': fields.Integer,
    'results': fields.Raw(description='Benchmark results for enhanced and standard analysis'),
    'performance_comparison': fields.Nested(benchmark_comparison_model, description='Comparison between enhanced and standard analysis')
})


@analysis_ns.route('/extract')
class ExtractClinicalEntities(Resource):
    @analysis_ns.doc('extract_clinical_entities')
    @analysis_ns.expect(base_note_request_model, validate=True)
    @analysis_ns.marshal_with(base_analysis_result_model)
    @log_request()
    @jwt_required() # Add JWT protection
    def post(self):
        """
        Extract clinical entities from patient note text.
        """
        try:
            data = analysis_ns.payload
            note_text = Sanitizer.sanitize_text(data['note_text'])
            patient_context = data.get('patient_context', {})

            if not note_text or len(note_text.strip()) < 10:
                analysis_ns.abort(400, message='note_text must be at least 10 characters', code='INVALID_NOTE_TEXT')

            if patient_context:
                for key, value in patient_context.items():
                    if isinstance(value, str):
                        patient_context[key] = Sanitizer.sanitize_text(value)

            logger.info(f"Processing clinical entity extraction for note length: {len(note_text)}")

            cached_result = storage_service.get_cached_analysis(note_text, patient_context, 'extract')
            if cached_result:
                logger.info("Using cached analysis result")
                cached_result['request_metadata'] = {
                    'note_length': len(note_text),
                    'has_patient_context': bool(patient_context),
                    'processed_at': datetime.utcnow().isoformat(),
                    'from_cache': True
                }
                return cached_result, 200

            session_id = storage_service.create_analysis_session(
                note_id=data.get('note_id'),
                patient_id=data.get('patient_id'),
                analysis_type='extract',
                request_data={'note_text': note_text, 'patient_context': patient_context}
            )
            storage_service.update_analysis_session(session_id, status='processing')

            result = clinical_service.extract_clinical_entities(note_text, patient_context)

            if 'error' in result:
                logger.error(f"Clinical analysis error: {result['error']}")
                storage_service.update_analysis_session(session_id, 
                    status='failed', 
                    error_message=result['error']
                )
                analysis_ns.abort(500, message='Clinical analysis failed', details=result['error'], code='ANALYSIS_FAILED')

            try:
                all_entities = []
                for entity_type in ['symptoms', 'conditions', 'medications', 'vital_signs', 'procedures', 'abnormal_findings']:
                    for entity in result.get(entity_type, []):
                        entity_with_type = entity.copy()
                        entity_with_type['type'] = entity_type[:-1] if entity_type.endswith('s') else entity_type
                        all_entities.append(entity_with_type)
                
                if all_entities:
                    storage_service.store_clinical_entities(session_id, all_entities)
            except Exception as storage_error:
                logger.warning(f"Failed to store entities to database: {storage_error}")
            
            assessment = result.get('overall_assessment', {})
            storage_service.update_analysis_session(session_id,
                status='completed',
                response_data=result,
                risk_level=assessment.get('risk_level', 'low'),
                requires_immediate_attention=assessment.get('requires_immediate_attention', False)
            )
            
            try:
                storage_service.cache_analysis_result(note_text, patient_context, 'extract', result)
            except Exception as cache_error:
                logger.warning(f"Failed to cache analysis result: {cache_error}")
            
            result['request_metadata'] = {
                'note_length': len(note_text),
                'has_patient_context': bool(patient_context),
                'processed_at': datetime.utcnow().isoformat(),
                'session_id': session_id,
                'from_cache': False
            }
            
            logger.info(f"Successfully extracted {sum(len(result.get(k, [])) for k in ['symptoms', 'conditions', 'medications', 'vital_signs', 'procedures', 'abnormal_findings'])} entities")
            
            return result, 200
            
        except Exception as e:
            logger.error(f"Error in extract_clinical_entities: {str(e)}")
            analysis_ns.abort(500, message='Internal server error during clinical analysis', code='INTERNAL_ERROR')

@analysis_ns.route('/diagnose')
class DiagnoseWithICDMappings(Resource):
    @analysis_ns.doc('diagnose_with_icd_mapping')
    @analysis_ns.expect(analysis_ns.model('DiagnoseRequest', {
        'note_text': fields.String(required=True, description='The patient note content'),
        'patient_context': fields.Nested(patient_context_model, description='Optional patient demographic and medical context'),
        'options': fields.Nested(analysis_ns.model('DiagnoseOptions', {
            'include_low_confidence': fields.Boolean(description='Whether to include low confidence ICD matches', default=False),
            'max_icd_matches': fields.Integer(description='Maximum number of ICD matches to return', default=5)
        }), description='Optional diagnosis options')
    }), validate=True)
    @analysis_ns.marshal_with(diagnose_response_model)
    @log_request()
    @jwt_required() # Add JWT protection
    def post(self):
        """
        Extract clinical entities and map them to ICD-10 codes.
        """
        try:
            data = analysis_ns.payload
            note_text = Sanitizer.sanitize_text(data['note_text'])
            patient_context = data.get('patient_context', {})
            options = data.get('options', {})

            if not note_text or len(note_text.strip()) < 10:
                analysis_ns.abort(400, message='note_text must be at least 10 characters', code='INVALID_NOTE_TEXT')

            logger.info(f"Processing diagnosis with ICD mapping for note length: {len(note_text)}")

            clinical_result = clinical_service.extract_clinical_entities(note_text, patient_context)

            if 'error' in clinical_result:
                logger.error(f"Clinical analysis error: {clinical_result['error']}")
                analysis_ns.abort(500, message='Clinical analysis failed', details=clinical_result['error'], code='ANALYSIS_FAILED')

            enhanced_result = icd_matcher.map_clinical_entities_to_icd(clinical_result)

            enhanced_result['request_metadata'] = {
                'note_length': len(note_text),
                'has_patient_context': bool(patient_context),
                'processed_at': datetime.utcnow().isoformat(),
                'total_entities_extracted': sum(len(clinical_result.get(k, [])) for k in ['symptoms', 'conditions', 'medications', 'vital_signs', 'procedures', 'abnormal_findings'])
            }

            cache_info = icd_matcher.get_cache_info()
            enhanced_result['icd_cache_info'] = cache_info

            mappings_summary = enhanced_result.get('icd_mappings', {}).get('summary', {})
            logger.info(f"Successfully mapped {mappings_summary.get('total_mappings', 0)} entities to ICD codes")

            return enhanced_result, 200

        except Exception as e:
            logger.error(f"Error in diagnose_with_icd_mapping: {str(e)}")
            analysis_ns.abort(500, message='Internal server error during diagnosis', code='INTERNAL_ERROR')

@analysis_ns.route('/priority/<string:note_id>')
@analysis_ns.param('note_id', 'The ID of the patient note')
class GetHighPriorityFindings(Resource):
    @analysis_ns.doc('get_high_priority_findings')
    @analysis_ns.expect(analysis_ns.parser()
                        .add_argument('risk_threshold', type=str, help='Minimum risk level (moderate, high, critical)', default='high', choices=('moderate', 'high', 'critical'), location='args')
                        .add_argument('include_details', type=bool, help='Include full entity details', default=False, location='args'))
    @analysis_ns.marshal_with(priority_response_model)
    @log_request()
    @jwt_required() # Add JWT protection
    def get(self, note_id):
        """
        Get high-priority findings for a specific note.
        """
        try:
            note_id = Sanitizer.sanitize_text(note_id)
            args = analysis_ns.parser().parse_args()
            risk_threshold = args['risk_threshold']
            include_details = args['include_details']

            logger.info(f"Retrieving priority findings for note {note_id} with threshold {risk_threshold}")

            priority_sessions = storage_service.get_priority_findings(
                note_id=note_id, 
                risk_threshold=risk_threshold
            )

            if not priority_sessions:
                return {
                    'note_id': note_id,
                    'priority_findings': [],
                    'summary': {
                        'total_findings': 0,
                        'critical_findings': 0,
                        'high_risk_findings': 0,
                        'requires_immediate_attention': False
                    },
                    'message': 'No priority findings found for this note'
                }, 200

            priority_findings = []
            summary = {
                'total_findings': 0,
                'critical_findings': 0,
                'high_risk_findings': 0,
                'requires_immediate_attention': False
            }

            for session in priority_sessions:
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
                
                summary['total_findings'] += 1
                if session['risk_level'] == 'critical':
                    summary['critical_findings'] += 1
                elif session['risk_level'] == 'high':
                    summary['high_risk_findings'] += 1
                
                if session['requires_immediate_attention']:
                    summary['requires_immediate_attention'] = True
            
            logger.info(f"Found {len(priority_findings)} priority findings for note {note_id}")
            
            return {
                'note_id': note_id,
                'priority_findings': priority_findings,
                'summary': summary,
                'query_parameters': {
                    'risk_threshold': risk_threshold,
                    'include_details': include_details
                },
                'retrieved_at': datetime.utcnow().isoformat()
            }, 200
            
        except Exception as e:
            logger.error(f"Error in get_high_priority_findings: {str(e)}")
            analysis_ns.abort(500, message='Internal server error while retrieving priority findings', code='INTERNAL_ERROR')

@analysis_ns.route('/batch')
class BatchAnalysis(Resource):
    @analysis_ns.doc('batch_analysis')
    @analysis_ns.expect(batch_analysis_request_model, validate=True)
    @analysis_ns.marshal_with(batch_analysis_response_model)
    @log_request()
    @jwt_required() # Add JWT protection
    def post(self):
        """
        Process multiple patient notes for clinical analysis.
        """
        try:
            data = analysis_ns.payload
            notes = data['notes']
            options = data.get('options', {})

            if len(notes) == 0:
                analysis_ns.abort(400, message='At least one note is required', code='EMPTY_BATCH')
            
            if len(notes) > 50:
                analysis_ns.abort(400, message='Maximum 50 notes per batch', code='BATCH_TOO_LARGE')

            logger.info(f"Processing batch analysis for {len(notes)} notes")

            results = []
            summary = {
                'total_notes': len(notes),
                'successful_analyses': 0,
                'failed_analyses': 0,
                'total_entities': 0,
                'high_priority_cases': 0
            }

            for i, note_data in enumerate(notes):
                try:
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
                    
                    if options.get('include_icd_mapping', True):
                        clinical_result = icd_matcher.map_clinical_entities_to_icd(clinical_result)
                    
                    high_priority_findings = []
                    if options.get('include_priority_analysis', True):
                        high_priority_findings = clinical_service.get_high_priority_findings(clinical_result)
                        if high_priority_findings:
                            summary['high_priority_cases'] += 1
                    
                    entity_count = sum(len(clinical_result.get(k, [])) for k in ['symptoms', 'conditions', 'medications', 'vital_signs', 'procedures', 'abnormal_findings'])
                    summary['total_entities'] += entity_count
                    
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
            
            return {
                'results': results,
                'summary': summary,
                'processed_at': datetime.utcnow().isoformat()
            }, 200
            
        except Exception as e:
            logger.error(f"Error in batch_analysis: {str(e)}")
            analysis_ns.abort(500, message='Internal server error during batch analysis', code='INTERNAL_ERROR')

@analysis_ns.route('/health')
class HealthCheck(Resource):
    @analysis_ns.doc('health_check')
    @analysis_ns.marshal_with(health_response_model)
    @log_request()
    def get(self):
        """
        Health check endpoint for the analysis service.
        """
        try:
            cache_info = icd_matcher.get_cache_info()
            
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
            
            if storage_available:
                try:
                    deleted_count = storage_service.cleanup_expired_cache()
                    if deleted_count > 0:
                        health_status['maintenance'] = {
                            'expired_cache_cleaned': deleted_count
                        }
                except Exception as cleanup_error:
                    logger.warning(f"Cache cleanup failed: {cleanup_error}")
            
            return health_status, 200
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            analysis_ns.abort(500, message='Internal server error', code='INTERNAL_ERROR')

@analysis_ns.route('/batch-async')
class BatchAnalysisAsync(Resource):
    @analysis_ns.doc('batch_analysis_async')
    @analysis_ns.expect(batch_async_request_model, validate=True)
    @analysis_ns.marshal_with(batch_async_response_model)
    @log_request()
    @jwt_required() # Add JWT protection
    def post(self):
        """
        High-performance async batch processing for large-scale clinical analysis.
        """
        try:
            data = analysis_ns.payload
            notes = data['notes']
            config_data = data.get('config', {})

            if len(notes) == 0:
                analysis_ns.abort(400, message='At least one note is required', code='EMPTY_BATCH')
            
            if len(notes) > 1000:
                analysis_ns.abort(400, message='Maximum 1000 notes per async batch', code='BATCH_TOO_LARGE')

            sanitized_notes = []
            for i, note_data in enumerate(notes):
                if 'note_text' not in note_data:
                    analysis_ns.abort(400, message=f'note_text is required for note at index {i}', code='MISSING_NOTE_TEXT')
                
                sanitized_note = {
                    'note_id': note_data.get('note_id', f'async_note_{i}'),
                    'note_text': Sanitizer.sanitize_text(note_data['note_text']),
                    'patient_context': note_data.get('patient_context', {}),
                    'patient_id': note_data.get('patient_id')
                }
                
                if sanitized_note['patient_context']:
                    for key, value in sanitized_note['patient_context'].items():
                        if isinstance(value, str):
                            sanitized_note['patient_context'][key] = Sanitizer.sanitize_text(value)
                
                sanitized_notes.append(sanitized_note)
            
            config = BatchAnalysisConfig(
                max_concurrent=min(config_data.get('max_concurrent', 10), 20),
                timeout_seconds=min(config_data.get('timeout_seconds', 30), 60),
                include_icd_mapping=config_data.get('include_icd_mapping', True),
                include_storage=config_data.get('include_storage', True),
                chunk_size=min(config_data.get('chunk_size', 50), 100),
                retry_attempts=min(config_data.get('retry_attempts', 2), 3)
            )
            
            logger.info(f"Starting async batch analysis for {len(sanitized_notes)} notes")
            
            def run_async_batch():
                return asyncio.run(async_clinical_service.batch_analyze_notes(sanitized_notes, config))
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_async_batch)
                result = future.result(timeout=300)
            
            logger.info(f"Async batch analysis completed: {result['summary']['successful_analyses']}/{len(sanitized_notes)} successful")
            
            return result, 200
            
        except concurrent.futures.TimeoutError:
            logger.error("Async batch analysis timed out")
            analysis_ns.abort(504, message='Batch analysis timed out after 5 minutes', code='BATCH_TIMEOUT')
        except Exception as e:
            logger.error(f"Error in async batch analysis: {str(e)}")
            analysis_ns.abort(500, message='Internal server error during async batch analysis', details=str(e), code='INTERNAL_ERROR')

@analysis_ns.route('/priority-scan')
class PriorityScan(Resource):
    @analysis_ns.doc('priority_scan')
    @analysis_ns.expect(priority_scan_request_model, validate=True)
    @analysis_ns.marshal_with(priority_scan_response_model)
    @log_request()
    @jwt_required() # Add JWT protection
    def post(self):
        """
        High-speed priority scanning for identifying high-risk cases.
        Optimized for rapid triage of large note volumes.
        """
        try:
            data = analysis_ns.payload
            notes = data['notes']
            risk_threshold = data.get('risk_threshold', 'high')

            if len(notes) == 0:
                analysis_ns.abort(400, message='At least one note is required', code='EMPTY_BATCH')
            
            if len(notes) > 2000:
                analysis_ns.abort(400, message='Maximum 2000 notes per priority scan', code='SCAN_BATCH_TOO_LARGE')

            sanitized_notes = []
            for i, note_data in enumerate(notes):
                if 'note_text' not in note_data:
                    analysis_ns.abort(400, message=f'note_text is required for note at index {i}', code='MISSING_NOTE_TEXT')
                
                sanitized_note = {
                    'note_id': note_data.get('note_id', f'scan_note_{i}'),
                    'note_text': Sanitizer.sanitize_text(note_data['note_text']),
                    'patient_context': note_data.get('patient_context', {})
                }
                
                if sanitized_note['patient_context']:
                    for key, value in sanitized_note['patient_context'].items():
                        if isinstance(value, str):
                            sanitized_note['patient_context'][key] = Sanitizer.sanitize_text(value)
                
                sanitized_notes.append(sanitized_note)
            
            logger.info(f"Starting priority scan for {len(sanitized_notes)} notes with {risk_threshold} threshold")
            
            def run_async_scan():
                return asyncio.run(async_clinical_service.priority_scan_async(sanitized_notes, risk_threshold))
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_async_scan)
                result = future.result(timeout=180)
            
            logger.info(f"Priority scan completed: {result['scan_summary']['priority_cases_found']} priority cases found")
            
            return result, 200
            
        except concurrent.futures.TimeoutError:
            logger.error("Priority scan timed out")
            analysis_ns.abort(504, message='Priority scan timed out after 3 minutes', code='SCAN_TIMEOUT')
        except Exception as e:
            logger.error(f"Error in priority scan: {str(e)}")
            analysis_ns.abort(500, message='Internal server error during priority scan', details=str(e), code='INTERNAL_ERROR')

@analysis_ns.route('/extract-enhanced')
class ExtractClinicalEntitiesEnhanced(Resource):
    def options(self):
        """Handle preflight OPTIONS request"""
        return {}, 200
    
    @analysis_ns.doc('extract_clinical_entities_enhanced')
    @analysis_ns.expect(enhanced_extract_request_model, validate=True)
    @analysis_ns.marshal_with(enhanced_analysis_response_model)
    @log_request()
    @jwt_required() # Add JWT protection
    def post(self):
        """
        Enhanced clinical entity extraction with Faiss + NLP integration.
        """
        try:
            data = analysis_ns.payload
            
            note_text = Sanitizer.sanitize_text(data['note_text'])
            patient_context = data.get('patient_context', {})
            include_icd_mapping = data.get('include_icd_mapping', True)
            icd_top_k = data.get('icd_top_k', 5)
            enable_nlp_preprocessing = data.get('enable_nlp_preprocessing', True)
            
            if len(note_text.strip()) < 5:
                analysis_ns.abort(400, message='Note text is too short (minimum 5 characters)', code='NOTE_TOO_SHORT')
            
            logger.info(f"Enhanced extraction request: {len(note_text)} chars, ICD mapping: {include_icd_mapping}")
            
            if enhanced_service:
                result = enhanced_service.extract_clinical_entities_enhanced(
                    note_text,
                    patient_context=patient_context,
                    include_icd_mapping=include_icd_mapping,
                    icd_top_k=icd_top_k,
                    enable_nlp_preprocessing=enable_nlp_preprocessing
                )
            else:
                result = clinical_service.extract_clinical_entities(note_text, patient_context)
                if include_icd_mapping:
                    result = icd_matcher.map_clinical_entities_to_icd(result)
                result['enhanced_service_available'] = False
            
            if 'error' in result:
                analysis_ns.abort(500, message='Enhanced clinical analysis failed', details=result['error'], code='ENHANCED_ANALYSIS_FAILED')
            
            try:
                if storage_service and data.get('note_id') and data.get('patient_id'):
                    session_id = storage_service.create_analysis_session(
                        note_id=data['note_id'],
                        patient_id=data['patient_id'],
                        analysis_type='enhanced_extract',
                        request_data=data
                    )
                    
                    all_entities = []
                    for entity_type in ['symptoms', 'conditions', 'medications', 'vital_signs', 'procedures', 'abnormal_findings']:
                        for entity in result.get(entity_type, []):
                            entity_with_type = entity.copy()
                            entity_with_type['type'] = entity_type[:-1] if entity_type.endswith('s') else entity_type
                            all_entities.append(entity_with_type)
                    
                    if all_entities:
                        storage_service.store_clinical_entities(session_id, all_entities)
                    
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
            
            return result, 200
            
        except ValidationError as e:
            analysis_ns.abort(400, message=str(e), code='VALIDATION_ERROR')
        except Exception as e:
            logger.error(f"Error in enhanced extraction: {str(e)}")
            analysis_ns.abort(500, message='Internal server error during enhanced analysis', details=str(e), code='INTERNAL_ERROR')

@analysis_ns.route('/performance-stats')
class GetPerformanceStats(Resource):
    @analysis_ns.doc('get_performance_stats')
    @analysis_ns.marshal_with(performance_stats_response_model)
    @log_request()
    @jwt_required() # Add JWT protection
    def get(self):
        """
        Get comprehensive performance statistics for all analysis services.
        """
        try:
            stats = {
                'timestamp': datetime.utcnow().isoformat(),
                'services': {}
            }
            
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
            
            try:
                icd_stats = icd_matcher.get_cache_info()
                if hasattr(icd_matcher, 'benchmark_performance'):
                    benchmark = icd_matcher.benchmark_performance(num_queries=5)
                    icd_stats['benchmark'] = benchmark
                stats['services']['icd_matcher'] = icd_stats
            except Exception as e:
                stats['services']['icd_matcher'] = {'error': str(e)}
            
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
            
            try:
                storage_stats = storage_service.get_cache_statistics()
                stats['services']['storage'] = storage_stats
            except Exception as e:
                stats['services']['storage'] = {'error': str(e)}
            
            return stats, 200
            
        except Exception as e:
            logger.error(f"Error getting performance stats: {str(e)}")
            analysis_ns.abort(500, message='Failed to retrieve performance statistics', details=str(e), code='STATS_ERROR')

@analysis_ns.route('/benchmark')
class RunPerformanceBenchmark(Resource):
    @analysis_ns.doc('run_performance_benchmark')
    @analysis_ns.expect(benchmark_request_model, validate=True)
    @analysis_ns.marshal_with(benchmark_response_model)
    @log_request()
    @jwt_required() # Add JWT protection
    def post(self):
        """
        Run performance benchmark on enhanced analysis service.
        """
        try:
            data = analysis_ns.payload
            
            num_tests = min(data.get('num_tests', 10), 50)
            include_enhanced = data.get('include_enhanced', True)
            include_standard = data.get('include_standard', True)
            
            benchmark_results = {
                'timestamp': datetime.utcnow().isoformat(),
                'num_tests': num_tests,
                'results': {}
            }
            
            if include_enhanced and enhanced_service:
                logger.info(f"Running enhanced analysis benchmark ({num_tests} tests)")
                enhanced_benchmark = enhanced_service.benchmark_enhanced_analysis(num_tests)
                benchmark_results['results']['enhanced_analysis'] = enhanced_benchmark
            
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
            
            if include_enhanced and include_standard and enhanced_service:
                enhanced_avg = benchmark_results['results']['enhanced_analysis']['avg_time_per_analysis_ms']
                standard_avg = benchmark_results['results']['standard_analysis']['avg_time_per_analysis_ms']
                
                benchmark_results['performance_comparison'] = {
                    'enhanced_faster': enhanced_avg < standard_avg,
                    'speedup_factor': standard_avg / enhanced_avg if enhanced_avg > 0 else 'N/A',
                    'time_difference_ms': standard_avg - enhanced_avg
                }
            
            return benchmark_results, 200
            
        except Exception as e:
            logger.error(f"Error running benchmark: {str(e)}")
            analysis_ns.abort(500, message='Benchmark failed', details=str(e), code='BENCHMARK_ERROR')