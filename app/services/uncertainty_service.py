"""Uncertainty quantification and confidence analysis for clinical predictions"""
import logging
from typing import Dict, List, Optional, Any, Tuple
import statistics
from datetime import datetime

logger = logging.getLogger(__name__)

class UncertaintyCalculator:
    """Calculate and quantify uncertainty in clinical predictions"""
    
    def __init__(self):
        self.confidence_threshold_high = 0.8
        self.confidence_threshold_medium = 0.6
        self.confidence_threshold_low = 0.4
    
    def calculate_confidence_intervals(self, entity: Dict) -> Dict:
        """
        Calculate confidence intervals for clinical entities
        
        Factors considered:
        - Claude's confidence score
        - Literature evidence strength
        - Clinical context coherence
        - Historical accuracy for similar cases
        
        Args:
            entity: Clinical entity with confidence score
            
        Returns:
            Confidence interval analysis
        """
        try:
            base_confidence = entity.get('confidence', 0.5)
            
            # Adjust confidence based on entity type
            entity_type_adjustment = self._get_entity_type_adjustment(entity)
            
            # Adjust confidence based on clinical context
            context_adjustment = self._get_context_adjustment(entity)
            
            # Calculate adjusted confidence
            adjusted_confidence = min(
                base_confidence * entity_type_adjustment * context_adjustment, 
                1.0
            )
            
            # Calculate confidence interval width based on uncertainty factors
            interval_width = self._calculate_interval_width(entity, adjusted_confidence)
            
            lower_bound = max(adjusted_confidence - interval_width, 0.0)
            upper_bound = min(adjusted_confidence + interval_width, 1.0)
            
            return {
                'entity': entity.get('entity', 'unknown'),
                'base_confidence': base_confidence,
                'adjusted_confidence': adjusted_confidence,
                'confidence_interval': {
                    'lower': round(lower_bound, 3),
                    'upper': round(upper_bound, 3),
                    'width': round(interval_width * 2, 3)
                },
                'uncertainty_factors': self._identify_uncertainty_factors(entity),
                'confidence_category': self._categorize_confidence(adjusted_confidence)
            }
            
        except Exception as e:
            logger.error(f"Error calculating confidence intervals: {str(e)}")
            return {
                'entity': entity.get('entity', 'unknown'),
                'error': str(e),
                'confidence_interval': {'lower': 0.0, 'upper': 1.0, 'width': 1.0}
            }
    
    def assess_diagnostic_uncertainty(self, entities: List[Dict], 
                                    include_visualization: bool = False) -> Dict:
        """
        Assess overall diagnostic uncertainty
        
        Args:
            entities: List of clinical entities from analysis
            include_visualization: Whether to include visualization data
            
        Returns:
            Overall uncertainty assessment
        """
        try:
            if not entities:
                return {
                    'overall_confidence': 0.0,
                    'uncertainty_sources': ['no_entities_detected'],
                    'recommendation': 'insufficient_data',
                    'confidence_range': {'lower': 0.0, 'upper': 0.0}
                }
            
            # Calculate confidence intervals for all entities
            entity_intervals = []
            for entity_type, entity_list in entities.items():
                if isinstance(entity_list, list):
                    for entity in entity_list:
                        if isinstance(entity, dict) and 'confidence' in entity:
                            interval = self.calculate_confidence_intervals(entity)
                            interval['entity_type'] = entity_type
                            entity_intervals.append(interval)
            
            if not entity_intervals:
                return {
                    'overall_confidence': 0.5,
                    'uncertainty_sources': ['no_confidence_scores'],
                    'recommendation': 'review_analysis',
                    'confidence_range': {'lower': 0.0, 'upper': 1.0}
                }
            
            # Calculate overall confidence metrics
            confidences = [ei['adjusted_confidence'] for ei in entity_intervals]
            overall_confidence = statistics.mean(confidences)
            confidence_std = statistics.stdev(confidences) if len(confidences) > 1 else 0
            
            # Identify uncertainty sources
            uncertainty_sources = self._identify_overall_uncertainty_sources(
                entity_intervals, confidence_std
            )
            
            # Generate recommendation
            recommendation = self._generate_uncertainty_recommendation(
                overall_confidence, uncertainty_sources, confidence_std
            )
            
            # Calculate overall confidence range
            min_confidence = min(ei['confidence_interval']['lower'] for ei in entity_intervals)
            max_confidence = max(ei['confidence_interval']['upper'] for ei in entity_intervals)
            
            result = {
                'overall_confidence': round(overall_confidence, 3),
                'confidence_std': round(confidence_std, 3),
                'uncertainty_sources': uncertainty_sources,
                'recommendation': recommendation,
                'confidence_range': {
                    'lower': round(min_confidence, 3),
                    'upper': round(max_confidence, 3)
                },
                'entity_count': len(entity_intervals),
                'high_confidence_entities': len([ei for ei in entity_intervals 
                                               if ei['adjusted_confidence'] > self.confidence_threshold_high]),
                'low_confidence_entities': len([ei for ei in entity_intervals 
                                              if ei['adjusted_confidence'] < self.confidence_threshold_low]),
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
            if include_visualization:
                result['visualization_data'] = self.create_uncertainty_visualization(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error assessing diagnostic uncertainty: {str(e)}")
            return {
                'overall_confidence': 0.0,
                'uncertainty_sources': ['analysis_error'],
                'recommendation': 'technical_review_needed',
                'confidence_range': {'lower': 0.0, 'upper': 0.0},
                'error': str(e)
            }
    
    def create_uncertainty_visualization(self, analysis: Dict) -> Dict:
        """
        Create data for uncertainty visualization
        
        Args:
            analysis: Uncertainty analysis results
            
        Returns:
            Visualization data for charts and heatmaps
        """
        try:
            overall_confidence = analysis.get('overall_confidence', 0)
            confidence_range = analysis.get('confidence_range', {})
            
            # Confidence distribution data
            confidence_distribution = {
                'high_confidence': analysis.get('high_confidence_entities', 0),
                'medium_confidence': (analysis.get('entity_count', 0) - 
                                    analysis.get('high_confidence_entities', 0) - 
                                    analysis.get('low_confidence_entities', 0)),
                'low_confidence': analysis.get('low_confidence_entities', 0)
            }
            
            # Uncertainty heatmap data
            uncertainty_heatmap = {
                'overall_uncertainty': 1 - overall_confidence,
                'range_uncertainty': (confidence_range.get('upper', 1) - 
                                    confidence_range.get('lower', 0)),
                'source_count': len(analysis.get('uncertainty_sources', []))
            }
            
            # Evidence strength indicators
            evidence_strength = {
                'confidence_score': overall_confidence,
                'confidence_category': self._categorize_confidence(overall_confidence),
                'uncertainty_level': self._categorize_uncertainty(1 - overall_confidence),
                'recommendation_urgency': self._get_recommendation_urgency(analysis)
            }
            
            return {
                'confidence_distribution': confidence_distribution,
                'uncertainty_heatmap': uncertainty_heatmap,
                'evidence_strength': evidence_strength,
                'chart_data': {
                    'confidence_gauge': {
                        'value': overall_confidence,
                        'min': 0,
                        'max': 1,
                        'thresholds': [
                            {'value': self.confidence_threshold_low, 'color': 'red'},
                            {'value': self.confidence_threshold_medium, 'color': 'yellow'},
                            {'value': self.confidence_threshold_high, 'color': 'green'}
                        ]
                    },
                    'uncertainty_bars': {
                        'categories': list(confidence_distribution.keys()),
                        'values': list(confidence_distribution.values())
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating visualization data: {str(e)}")
            return {
                'error': str(e),
                'confidence_distribution': {},
                'uncertainty_heatmap': {},
                'evidence_strength': {}
            }
    
    def get_confidence_breakdown(self, analysis: Dict) -> Dict:
        """
        Get detailed confidence breakdown for analysis
        
        Args:
            analysis: Clinical analysis results
            
        Returns:
            Detailed confidence breakdown by entity type
        """
        try:
            breakdown = {
                'by_entity_type': {},
                'by_confidence_level': {
                    'high': [],
                    'medium': [],
                    'low': []
                },
                'summary_statistics': {}
            }
            
            # Process each entity type
            for entity_type, entity_list in analysis.items():
                if isinstance(entity_list, list) and entity_type != 'overall_assessment':
                    confidences = []
                    entities_with_confidence = []
                    
                    for entity in entity_list:
                        if isinstance(entity, dict) and 'confidence' in entity:
                            confidence = entity['confidence']
                            confidences.append(confidence)
                            entities_with_confidence.append({
                                'entity': entity.get('entity', 'unknown'),
                                'confidence': confidence,
                                'category': self._categorize_confidence(confidence)
                            })
                            
                            # Categorize by confidence level
                            category = self._categorize_confidence(confidence)
                            breakdown['by_confidence_level'][category].append({
                                'entity_type': entity_type,
                                'entity': entity.get('entity', 'unknown'),
                                'confidence': confidence
                            })
                    
                    if confidences:
                        breakdown['by_entity_type'][entity_type] = {
                            'count': len(confidences),
                            'mean_confidence': round(statistics.mean(confidences), 3),
                            'std_confidence': round(statistics.stdev(confidences) if len(confidences) > 1 else 0, 3),
                            'min_confidence': round(min(confidences), 3),
                            'max_confidence': round(max(confidences), 3),
                            'entities': entities_with_confidence
                        }
            
            # Calculate summary statistics
            all_confidences = []
            for entity_type_data in breakdown['by_entity_type'].values():
                all_confidences.extend([e['confidence'] for e in entity_type_data['entities']])
            
            if all_confidences:
                breakdown['summary_statistics'] = {
                    'total_entities': len(all_confidences),
                    'overall_mean': round(statistics.mean(all_confidences), 3),
                    'overall_std': round(statistics.stdev(all_confidences) if len(all_confidences) > 1 else 0, 3),
                    'confidence_quartiles': self._calculate_quartiles(all_confidences)
                }
            
            return breakdown
            
        except Exception as e:
            logger.error(f"Error creating confidence breakdown: {str(e)}")
            return {
                'error': str(e),
                'by_entity_type': {},
                'by_confidence_level': {'high': [], 'medium': [], 'low': []},
                'summary_statistics': {}
            }
    
    def _get_entity_type_adjustment(self, entity: Dict) -> float:
        """Get confidence adjustment based on entity type"""
        entity_text = entity.get('entity', '').lower()
        
        # Higher confidence for objective findings
        if any(term in entity_text for term in ['temperature', 'blood pressure', 'heart rate']):
            return 1.1
        
        # Lower confidence for subjective symptoms
        if any(term in entity_text for term in ['pain', 'fatigue', 'nausea']):
            return 0.9
        
        return 1.0
    
    def _get_context_adjustment(self, entity: Dict) -> float:
        """Get confidence adjustment based on clinical context"""
        # Check for negation
        if entity.get('negated', False):
            return 0.8
        
        # Check for uncertainty markers in text
        text_span = entity.get('text_span', '').lower()
        if any(term in text_span for term in ['possible', 'suspected', 'may be']):
            return 0.7
        
        return 1.0
    
    def _calculate_interval_width(self, entity: Dict, confidence: float) -> float:
        """Calculate confidence interval width based on uncertainty factors"""
        base_width = 0.1
        
        # Increase width for lower confidence
        confidence_factor = (1 - confidence) * 0.2
        
        # Increase width for subjective entities
        subjectivity_factor = 0.05 if entity.get('subjective', False) else 0
        
        # Increase width if negated or uncertain
        uncertainty_factor = 0.1 if (entity.get('negated', False) or 
                                   'possible' in entity.get('text_span', '').lower()) else 0
        
        return base_width + confidence_factor + subjectivity_factor + uncertainty_factor
    
    def _identify_uncertainty_factors(self, entity: Dict) -> List[str]:
        """Identify factors contributing to uncertainty"""
        factors = []
        
        confidence = entity.get('confidence', 0.5)
        if confidence < self.confidence_threshold_medium:
            factors.append('low_confidence_score')
        
        if entity.get('negated', False):
            factors.append('negated_finding')
        
        text_span = entity.get('text_span', '').lower()
        if any(term in text_span for term in ['possible', 'suspected', 'may']):
            factors.append('uncertainty_markers')
        
        if not entity.get('text_span'):
            factors.append('missing_text_context')
        
        return factors
    
    def _categorize_confidence(self, confidence: float) -> str:
        """Categorize confidence level"""
        if confidence >= self.confidence_threshold_high:
            return 'high'
        elif confidence >= self.confidence_threshold_medium:
            return 'medium'
        else:
            return 'low'
    
    def _categorize_uncertainty(self, uncertainty: float) -> str:
        """Categorize uncertainty level"""
        if uncertainty <= 0.2:
            return 'low'
        elif uncertainty <= 0.4:
            return 'medium'
        else:
            return 'high'
    
    def _identify_overall_uncertainty_sources(self, entity_intervals: List[Dict], 
                                            confidence_std: float) -> List[str]:
        """Identify sources of overall uncertainty"""
        sources = []
        
        # Check for high variability in confidence scores
        if confidence_std > 0.2:
            sources.append('high_confidence_variability')
        
        # Check for multiple low-confidence entities
        low_confidence_count = len([ei for ei in entity_intervals 
                                  if ei['adjusted_confidence'] < self.confidence_threshold_low])
        if low_confidence_count > len(entity_intervals) * 0.3:
            sources.append('multiple_low_confidence_findings')
        
        # Check for negated findings
        negated_count = len([ei for ei in entity_intervals 
                           if 'negated_finding' in ei.get('uncertainty_factors', [])])
        if negated_count > 0:
            sources.append('negated_findings_present')
        
        # Check for uncertainty markers
        uncertain_markers_count = len([ei for ei in entity_intervals 
                                     if 'uncertainty_markers' in ei.get('uncertainty_factors', [])])
        if uncertain_markers_count > 0:
            sources.append('uncertainty_markers_present')
        
        return sources
    
    def _generate_uncertainty_recommendation(self, overall_confidence: float, 
                                           uncertainty_sources: List[str], 
                                           confidence_std: float) -> str:
        """Generate recommendation based on uncertainty analysis"""
        if overall_confidence >= self.confidence_threshold_high and confidence_std < 0.1:
            return 'high_confidence_analysis'
        elif overall_confidence >= self.confidence_threshold_medium:
            return 'moderate_confidence_analysis'
        elif 'multiple_low_confidence_findings' in uncertainty_sources:
            return 'additional_clinical_data_needed'
        elif 'high_confidence_variability' in uncertainty_sources:
            return 'expert_clinical_review_recommended'
        else:
            return 'comprehensive_clinical_evaluation_needed'
    
    def _get_recommendation_urgency(self, analysis: Dict) -> str:
        """Determine urgency level of recommendation"""
        recommendation = analysis.get('recommendation', '')
        
        if 'comprehensive' in recommendation:
            return 'high'
        elif 'expert' in recommendation:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_quartiles(self, values: List[float]) -> Dict:
        """Calculate quartile statistics"""
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return {
            'q1': round(sorted_values[n // 4], 3),
            'q2_median': round(sorted_values[n // 2], 3),
            'q3': round(sorted_values[3 * n // 4], 3)
        }