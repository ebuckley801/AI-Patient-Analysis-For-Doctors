import logging
from flask import request
from flask_restx import Namespace, Resource, fields
from flask_jwt_extended import jwt_required, get_jwt_identity # Import jwt_required

from app.services.explainable_clinical_service import ExplainableClinicalService
from app.services.pubmed_service import PubMedService
from app.services.pubmed_cache_service import PubMedCacheService
from app.services.uncertainty_service import UncertaintyCalculator
from app.services.pathway_explorer import TreatmentPathwayExplorer
from app.utils.sanitization import Sanitizer

logger = logging.getLogger(__name__)
explanation_ns = Namespace('explanation', description='Explainable AI features and insights')

# Initialize services
explainable_service = ExplainableClinicalService()
pubmed_service = PubMedService()
cache_service = PubMedCacheService()
uncertainty_calculator = UncertaintyCalculator()
pathway_explorer = TreatmentPathwayExplorer()

# --- Models for API Documentation ---

# Patient Context Model (re-used from analysis_routes if available, or defined here)
patient_context_model = explanation_ns.model('PatientContext', {
    'age': fields.Integer(description='Age of the patient'),
    'gender': fields.String(description='Gender of the patient'),
    'medical_history': fields.String(description='Patient medical history')
})

# Explain Clinical Analysis Request Model
explain_analysis_request_model = explanation_ns.model('ExplainAnalysisRequest', {
    'note_text': fields.String(required=True, description='Patient note content'),
    'patient_context': fields.Nested(patient_context_model, description='Optional patient demographic and medical context'),
    'explanation_depth': fields.String(enum=['detailed', 'summary'], default='detailed', description='Depth of explanation'),
    'include_literature': fields.Boolean(default=True, description='Include relevant literature evidence'),
    'include_alternatives': fields.Boolean(default=True, description='Include alternative treatment pathways')
})

# Explain Clinical Analysis Response Models
analysis_result_model = explanation_ns.model('AnalysisResult', {
    'symptoms': fields.List(fields.Raw),
    'conditions': fields.List(fields.Raw),
    'overall_assessment': fields.Raw
})

explanation_model = explanation_ns.model('Explanation', {
    'reasoning_chain': fields.List(fields.String, description='Step-by-step reasoning for the analysis'),
    'evidence_sources': fields.List(fields.String, description='Sources of evidence used in the explanation'),
    'uncertainty_analysis': fields.Raw(description='Detailed uncertainty metrics'),
    'alternative_pathways': fields.List(fields.Raw, description='Alternative treatment pathways')
})

explain_analysis_response_model = explanation_ns.model('ExplainAnalysisResponse', {
    'status': fields.String(description='Status of the operation'),
    'analysis': fields.Nested(analysis_result_model, description='The clinical analysis result'),
    'explanation': fields.Nested(explanation_model, description='The explanation for the analysis'),
    'metadata': fields.Raw(description='Metadata about the explanation request')
})

# Literature Evidence Models
literature_evidence_item_model = explanation_ns.model('LiteratureEvidenceItem', {
    'title': fields.String,
    'authors': fields.List(fields.String),
    'journal': fields.String,
    'year': fields.Integer,
    'pubmed_id': fields.String,
    'abstract': fields.String,
    'relevance_score': fields.Float
})

literature_evidence_response_model = explanation_ns.model('LiteratureEvidenceResponse', {
    'status': fields.String,
    'entity_id': fields.String,
    'entity': fields.Raw,
    'literature_evidence': fields.List(fields.Nested(literature_evidence_item_model)),
    'total_found': fields.Integer,
    'search_parameters': fields.Raw
})

# Treatment Pathways Models
primary_diagnosis_model = explanation_ns.model('PrimaryDiagnosis', {
    'condition': fields.String(required=True, description='The primary medical condition'),
    'icd_code': fields.String(description='Optional ICD-10 code for the condition')
})

treatment_pathway_request_model = explanation_ns.model('TreatmentPathwayRequest', {
    'primary_diagnosis': fields.Nested(primary_diagnosis_model, required=True, description='The primary diagnosis for pathway exploration'),
    'patient_context': fields.Nested(patient_context_model, description='Optional patient context for personalized pathways'),
    'max_pathways': fields.Integer(default=5, description='Maximum number of alternative pathways to generate'),
    'include_contraindications': fields.Boolean(default=True, description='Whether to check for contraindications')
})

treatment_pathway_item_model = explanation_ns.model('TreatmentPathwayItem', {
    'name': fields.String,
    'description': fields.String,
    'steps': fields.List(fields.String),
    'evidence_strength': fields.Float,
    'safety_profile': fields.String,
    'contraindications': fields.List(fields.String, description='List of contraindications for this pathway')
})

treatment_pathway_response_model = explanation_ns.model('TreatmentPathwayResponse', {
    'status': fields.String,
    'pathways': fields.List(fields.Nested(treatment_pathway_item_model)),
    'total_generated': fields.Integer,
    'ranking_criteria': fields.List(fields.String),
    'patient_specific_notes': fields.List(fields.String)
})

# Uncertainty Analysis Models
uncertainty_analysis_response_model = explanation_ns.model('UncertaintyAnalysisResponse', {
    'status': fields.String,
    'analysis_id': fields.String,
    'uncertainty_analysis': fields.Raw(description='Detailed uncertainty metrics'),
    'confidence_breakdown': fields.Raw(description='Breakdown of confidence levels'),
    'recommendation': fields.String,
    'timestamp': fields.String,
    'visualization_data': fields.Raw(description='Data for visualizing uncertainty')
})

# Health Check Models
health_service_status_model = explanation_ns.model('HealthServiceStatus', {
    'explainable_service': fields.Boolean,
    'pubmed_service': fields.Boolean,
    'cache_service': fields.Boolean,
    'uncertainty_calculator': fields.Boolean,
    'pathway_explorer': fields.Boolean
})

cache_stats_model = explanation_ns.model('CacheStats', {
    'total_entries': fields.Integer,
    'cache_hits': fields.Integer,
    'cache_misses': fields.Integer,
    'hit_ratio': fields.Float,
    'expired_entries': fields.Integer,
    'last_cleanup': fields.String,
    'last_updated': fields.String
})

explanation_health_response_model = explanation_ns.model('ExplanationHealthResponse', {
    'status': fields.String,
    'services': fields.Nested(health_service_status_model),
    'cache_stats': fields.Nested(cache_stats_model),
    'timestamp': fields.String
})

# Cache Statistics Models
cache_performance_model = explanation_ns.model('CachePerformance', {
    'total_entries': fields.Integer,
    'cache_hits': fields.Integer,
    'cache_misses': fields.Integer,
    'hit_ratio': fields.Float,
    'expired_entries': fields.Integer,
    'last_cleanup': fields.String,
    'last_updated': fields.String
})

cache_recommendations_model = explanation_ns.model('CacheRecommendations', {
    'cache_healthy': fields.Boolean,
    'cleanup_needed': fields.Boolean
})

cache_stats_response_model = explanation_ns.model('CacheStatsResponse', {
    'status': fields.String,
    'cache_performance': fields.Nested(cache_performance_model),
    'recommendations': fields.Nested(cache_recommendations_model),
    'cleanup_performed': fields.Boolean,
    'cleaned_entries': fields.Integer
})


@explanation_ns.route('/analyze')
class ExplainClinicalAnalysis(Resource):
    @explanation_ns.doc('explain_clinical_analysis')
    @explanation_ns.expect(explain_analysis_request_model, validate=True)
    @explanation_ns.marshal_with(explain_analysis_response_model)
    @jwt_required() # Add JWT protection
    def post(self):
        """
        Perform explainable clinical analysis.
        """
        try:
            data = explanation_ns.payload
            
            note_text = Sanitizer.sanitize_patient_note(data['note_text'])
            patient_context = data.get('patient_context', {})
            explanation_depth = data.get('explanation_depth', 'detailed')
            include_literature = data.get('include_literature', True)
            include_alternatives = data.get('include_alternatives', True)
            
            result = explainable_service.analyze_with_explanation(
                patient_note=note_text,
                patient_context=patient_context,
                depth=explanation_depth,
                include_literature=include_literature,
                include_alternatives=include_alternatives
            )
            
            return {
                'status': 'success',
                'analysis': result['analysis'],
                'explanation': result['explanation'],
                'metadata': {
                    'explanation_depth': explanation_depth,
                    'included_literature': include_literature,
                    'included_alternatives': include_alternatives,
                    'processing_time': result.get('processing_time', 0)
                }
            }, 200
            
        except Exception as e:
            logger.error(f"Error in explainable analysis: {str(e)}")
            explanation_ns.abort(500, status='error', message=str(e))

@explanation_ns.route('/literature/<string:entity_id>')
@explanation_ns.param('entity_id', 'The ID of the clinical entity')
class GetLiteratureEvidence(Resource):
    @explanation_ns.doc('get_literature_evidence')
    @explanation_ns.expect(explanation_ns.parser()
                        .add_argument('max_results', type=int, help='Maximum number of articles', default=10, location='args')
                        .add_argument('study_type', type=str, help='Filter by study type', location='args'))
    @explanation_ns.marshal_with(literature_evidence_response_model)
    @jwt_required() # Add JWT protection
    def get(self, entity_id):
        """
        Get literature evidence for specific clinical entity.
        """
        try:
            args = explanation_ns.parser().parse_args()
            max_results = args['max_results']
            study_type = args['study_type']
            
            entity = explainable_service.get_entity_by_id(entity_id)
            if not entity:
                explanation_ns.abort(404, status='error', message='Entity not found')
            
            literature = pubmed_service.find_evidence_for_condition(
                condition=entity['entity'],
                treatment=entity.get('related_treatment'),
                max_results=max_results,
                study_type=study_type
            )
            
            return {
                'status': 'success',
                'entity_id': entity_id,
                'entity': entity,
                'literature_evidence': literature,
                'total_found': len(literature),
                'search_parameters': {
                    'max_results': max_results,
                    'study_type': study_type
                }
            }, 200
            
        except Exception as e:
            logger.error(f"Error getting literature evidence: {str(e)}")
            explanation_ns.abort(500, status='error', message=str(e))

@explanation_ns.route('/pathways')
class ExploreTreatmentPathways(Resource):
    @explanation_ns.doc('explore_treatment_pathways')
    @explanation_ns.expect(treatment_pathway_request_model, validate=True)
    @explanation_ns.marshal_with(treatment_pathway_response_model)
    @jwt_required() # Add JWT protection
    def post(self):
        """
        Explore alternative treatment pathways for a condition.
        """
        try:
            data = explanation_ns.payload
            
            primary_diagnosis = data['primary_diagnosis']
            patient_context = data.get('patient_context', {})
            max_pathways = data.get('max_pathways', 5)
            include_contraindications = data.get('include_contraindications', True)
            
            pathways = pathway_explorer.generate_alternative_pathways(
                primary_diagnosis=primary_diagnosis,
                patient_context=patient_context,
                max_pathways=max_pathways
            )
            
            if include_contraindications:
                for pathway in pathways:
                    pathway['contraindications'] = pathway_explorer.check_contraindications(
                        pathway=pathway,
                        patient_context=patient_context
                    )
            
            ranked_pathways = pathway_explorer.rank_pathways_by_evidence(pathways)
            
            return {
                'status': 'success',
                'pathways': ranked_pathways[:max_pathways],
                'total_generated': len(pathways),
                'ranking_criteria': [
                    'evidence_strength',
                    'safety_profile',
                    'patient_compatibility',
                    'treatment_outcomes'
                ],
                'patient_specific_notes': pathway_explorer.get_patient_specific_notes(
                    pathways=ranked_pathways,
                    patient_context=patient_context
                )
            }, 200
            
        except Exception as e:
            logger.error(f"Error exploring pathways: {str(e)}")
            explanation_ns.abort(500, status='error', message=str(e))

@explanation_ns.route('/uncertainty/<string:analysis_id>')
@explanation_ns.param('analysis_id', 'The ID of the analysis for uncertainty calculation')
class GetUncertaintyAnalysis(Resource):
    @explanation_ns.doc('get_uncertainty_analysis')
    @explanation_ns.expect(explanation_ns.parser()
                        .add_argument('include_visualization', type=bool, help='Include visualization data', default=False, location='args'))
    @explanation_ns.marshal_with(uncertainty_analysis_response_model)
    @jwt_required() # Add JWT protection
    def get(self, analysis_id):
        """
        Get detailed uncertainty analysis for a previous analysis.
        """
        try:
            args = explanation_ns.parser().parse_args()
            include_visualization = args['include_visualization']
            
            analysis = explainable_service.get_analysis_by_id(analysis_id)
            if not analysis:
                explanation_ns.abort(404, status='error', message='Analysis not found')
            
            uncertainty_analysis = uncertainty_calculator.assess_diagnostic_uncertainty(
                entities=analysis['entities'],
                include_visualization=include_visualization
            )
            
            confidence_breakdown = uncertainty_calculator.get_confidence_breakdown(
                analysis=analysis
            )
            
            response_data = {
                'status': 'success',
                'analysis_id': analysis_id,
                'uncertainty_analysis': uncertainty_analysis,
                'confidence_breakdown': confidence_breakdown,
                'recommendation': uncertainty_analysis.get('recommendation', ''),
                'timestamp': analysis.get('timestamp')
            }
            
            if include_visualization:
                response_data['visualization_data'] = uncertainty_calculator.create_uncertainty_visualization(
                    analysis=uncertainty_analysis
                )
            
            return response_data, 200
            
        except Exception as e:
            logger.error(f"Error getting uncertainty analysis: {str(e)}")
            explanation_ns.abort(500, status='error', message=str(e))

@explanation_ns.route('/health')
class ExplanationHealthCheck(Resource):
    @explanation_ns.doc('health_check')
    @explanation_ns.marshal_with(explanation_health_response_model)
    def get(self):
        """
        Health check for explanation services.
        """
        try:
            services_status = {
                'explainable_service': explainable_service.health_check(),
                'pubmed_service': pubmed_service.health_check() if hasattr(pubmed_service, 'health_check') else True,
                'cache_service': True,
                'uncertainty_calculator': True,
                'pathway_explorer': True
            }
            
            cache_stats = cache_service.get_cache_statistics()
            
            overall_healthy = all(services_status.values())
            
            return {
                'status': 'healthy' if overall_healthy else 'degraded',
                'services': services_status,
                'cache_stats': cache_stats,
                'timestamp': cache_stats.get('last_updated')
            }, 200
            
        except Exception as e:
            logger.error(f"Error in health check: {str(e)}")
            explanation_ns.abort(500, status='error', message=str(e))

@explanation_ns.route('/cache/stats')
class GetCacheStatistics(Resource):
    @explanation_ns.doc('get_cache_statistics')
    @explanation_ns.expect(explanation_ns.parser()
                        .add_argument('cleanup', type=bool, help='Perform cache cleanup', default=False, location='args'))
    @explanation_ns.marshal_with(cache_stats_response_model)
    def get(self):
        """
        Get detailed cache performance statistics.
        """
        try:
            args = explanation_ns.parser().parse_args()
            cleanup_requested = args['cleanup']
            
            stats = cache_service.get_cache_statistics()
            
            if cleanup_requested:
                cleanup_count = cache_service.invalidate_outdated_cache()
                stats['cleanup_performed'] = True
                stats['cleaned_entries'] = cleanup_count
            
            return {
                'status': 'success',
                'cache_performance': stats,
                'recommendations': {
                    'cache_healthy': stats.get('cache_hit_ratio', 0) > 0.6,
                    'cleanup_needed': stats.get('expired_entries', 0) > 100
                },
                'cleanup_performed': cleanup_requested,
                'cleaned_entries': stats.get('cleaned_entries', 0) if cleanup_requested else 0
            }, 200
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            explanation_ns.abort(500, status='error', message=str(e))