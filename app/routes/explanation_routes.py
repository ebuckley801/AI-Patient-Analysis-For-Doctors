"""Flask routes for explainable AI features"""
import logging
from flask import Blueprint, request, jsonify
from app.services.explainable_clinical_service import ExplainableClinicalService
from app.services.pubmed_service import PubMedService
from app.services.pubmed_cache_service import PubMedCacheService
from app.services.uncertainty_service import UncertaintyCalculator
from app.services.pathway_explorer import TreatmentPathwayExplorer
from app.utils.validation import validate_request_data, require_fields
from app.utils.sanitization import sanitize_input

logger = logging.getLogger(__name__)
explanation_bp = Blueprint('explanation', __name__, url_prefix='/api/explanation')

# Initialize services
explainable_service = ExplainableClinicalService()
pubmed_service = PubMedService()
cache_service = PubMedCacheService()
uncertainty_calculator = UncertaintyCalculator()
pathway_explorer = TreatmentPathwayExplorer()

@explanation_bp.route('/analyze', methods=['POST'])
@validate_request_data(['note_text'])
def explain_clinical_analysis():
    """
    Perform explainable clinical analysis
    
    Request JSON:
    {
        "note_text": "Patient note content",
        "patient_context": {"age": 45, "gender": "M"},
        "explanation_depth": "detailed|summary",
        "include_literature": true,
        "include_alternatives": true
    }
    
    Response JSON:
    {
        "analysis": {...},
        "explanation": {
            "reasoning_chain": [...],
            "evidence_sources": [...],
            "uncertainty_analysis": {...},
            "alternative_pathways": [...]
        }
    }
    """
    try:
        data = request.get_json()
        
        # Sanitize input
        note_text = sanitize_input(data['note_text'])
        patient_context = data.get('patient_context', {})
        explanation_depth = data.get('explanation_depth', 'detailed')
        include_literature = data.get('include_literature', True)
        include_alternatives = data.get('include_alternatives', True)
        
        # Perform explainable analysis
        result = explainable_service.analyze_with_explanation(
            patient_note=note_text,
            patient_context=patient_context,
            depth=explanation_depth,
            include_literature=include_literature,
            include_alternatives=include_alternatives
        )
        
        return jsonify({
            'status': 'success',
            'analysis': result['analysis'],
            'explanation': result['explanation'],
            'metadata': {
                'explanation_depth': explanation_depth,
                'included_literature': include_literature,
                'included_alternatives': include_alternatives,
                'processing_time': result.get('processing_time', 0)
            }
        })
        
    except Exception as e:
        logger.error(f"Error in explainable analysis: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@explanation_bp.route('/literature/<entity_id>', methods=['GET'])
def get_literature_evidence(entity_id):
    """
    Get literature evidence for specific clinical entity
    
    Query parameters:
    - max_results: Maximum number of articles (default: 10)
    - study_type: Filter by study type
    
    Response JSON:
    {
        "entity_id": "...",
        "literature_evidence": [...],
        "total_found": 15,
        "cache_hit": true
    }
    """
    try:
        max_results = int(request.args.get('max_results', 10))
        study_type = request.args.get('study_type')
        
        # Get entity details first
        entity = explainable_service.get_entity_by_id(entity_id)
        if not entity:
            return jsonify({
                'status': 'error',
                'error': 'Entity not found'
            }), 404
        
        # Search for literature evidence
        literature = pubmed_service.find_evidence_for_condition(
            condition=entity['entity'],
            treatment=entity.get('related_treatment'),
            max_results=max_results,
            study_type=study_type
        )
        
        return jsonify({
            'status': 'success',
            'entity_id': entity_id,
            'entity': entity,
            'literature_evidence': literature,
            'total_found': len(literature),
            'search_parameters': {
                'max_results': max_results,
                'study_type': study_type
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting literature evidence: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@explanation_bp.route('/pathways', methods=['POST'])
@validate_request_data(['primary_diagnosis'])
def explore_treatment_pathways():
    """
    Explore alternative treatment pathways for condition
    
    Request JSON:
    {
        "primary_diagnosis": {...},
        "patient_context": {...},
        "max_pathways": 5,
        "include_contraindications": true
    }
    
    Response JSON:
    {
        "pathways": [...],
        "ranking_criteria": [...],
        "patient_specific_notes": [...]
    }
    """
    try:
        data = request.get_json()
        
        primary_diagnosis = data['primary_diagnosis']
        patient_context = data.get('patient_context', {})
        max_pathways = data.get('max_pathways', 5)
        include_contraindications = data.get('include_contraindications', True)
        
        # Generate treatment pathways
        pathways = pathway_explorer.generate_alternative_pathways(
            primary_diagnosis=primary_diagnosis,
            patient_context=patient_context,
            max_pathways=max_pathways
        )
        
        # Check contraindications if requested
        if include_contraindications:
            for pathway in pathways:
                pathway['contraindications'] = pathway_explorer.check_contraindications(
                    pathway=pathway,
                    patient_context=patient_context
                )
        
        # Rank pathways by evidence
        ranked_pathways = pathway_explorer.rank_pathways_by_evidence(pathways)
        
        return jsonify({
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
        })
        
    except Exception as e:
        logger.error(f"Error exploring pathways: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@explanation_bp.route('/uncertainty/<analysis_id>', methods=['GET'])
def get_uncertainty_analysis(analysis_id):
    """
    Get detailed uncertainty analysis for previous analysis
    
    Query parameters:
    - include_visualization: Include visualization data (default: false)
    
    Response JSON:
    {
        "uncertainty_analysis": {...},
        "confidence_breakdown": {...},
        "recommendation": "..."
    }
    """
    try:
        include_visualization = request.args.get('include_visualization', 'false').lower() == 'true'
        
        # Get analysis from database
        analysis = explainable_service.get_analysis_by_id(analysis_id)
        if not analysis:
            return jsonify({
                'status': 'error',
                'error': 'Analysis not found'
            }), 404
        
        # Calculate detailed uncertainty metrics
        uncertainty_analysis = uncertainty_calculator.assess_diagnostic_uncertainty(
            entities=analysis['entities'],
            include_visualization=include_visualization
        )
        
        # Get confidence breakdown
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
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error getting uncertainty analysis: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@explanation_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check for explanation services
    
    Response JSON:
    {
        "status": "healthy",
        "services": {...},
        "cache_stats": {...}
    }
    """
    try:
        # Check service health
        services_status = {
            'explainable_service': explainable_service.health_check(),
            'pubmed_service': pubmed_service.health_check() if hasattr(pubmed_service, 'health_check') else True,
            'cache_service': True,
            'uncertainty_calculator': True,
            'pathway_explorer': True
        }
        
        # Get cache statistics
        cache_stats = cache_service.get_cache_statistics()
        
        # Overall health status
        overall_healthy = all(services_status.values())
        
        return jsonify({
            'status': 'healthy' if overall_healthy else 'degraded',
            'services': services_status,
            'cache_stats': cache_stats,
            'timestamp': cache_stats.get('last_updated')
        })
        
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@explanation_bp.route('/cache/stats', methods=['GET'])
def get_cache_statistics():
    """
    Get detailed cache performance statistics
    
    Response JSON:
    {
        "cache_performance": {...},
        "recent_queries": [...],
        "cleanup_stats": {...}
    }
    """
    try:
        stats = cache_service.get_cache_statistics()
        
        # Perform cache cleanup if requested
        if request.args.get('cleanup', 'false').lower() == 'true':
            cleanup_count = cache_service.invalidate_outdated_cache()
            stats['cleanup_performed'] = True
            stats['cleaned_entries'] = cleanup_count
        
        return jsonify({
            'status': 'success',
            'cache_performance': stats,
            'recommendations': {
                'cache_healthy': stats.get('cache_hit_ratio', 0) > 0.6,
                'cleanup_needed': stats.get('expired_entries', 0) > 100
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500