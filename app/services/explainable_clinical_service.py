"""Enhanced clinical analysis service with explainable reasoning"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from app.services.clinical_analysis_service import ClinicalAnalysisService
from app.services.pubmed_service import PubMedService
from app.services.pubmed_cache_service import PubMedCacheService
from app.services.uncertainty_service import UncertaintyCalculator
from app.services.analysis_storage_service import AnalysisStorageService

logger = logging.getLogger(__name__)

class ExplainableClinicalService(ClinicalAnalysisService):
    """Extended clinical analysis with explainable reasoning"""
    
    def __init__(self):
        super().__init__()
        self.pubmed_service = PubMedService()
        self.cache_service = PubMedCacheService()
        self.uncertainty_calculator = UncertaintyCalculator()
        self.storage_service = AnalysisStorageService()
    
    def analyze_with_explanation(self, patient_note: str, patient_context: Dict,
                               depth: str = 'detailed', include_literature: bool = True,
                               include_alternatives: bool = True) -> Dict[str, Any]:
        """
        Perform clinical analysis with detailed explanations
        
        Args:
            patient_note: Patient note text
            patient_context: Patient demographic and medical context
            depth: Explanation depth ('summary' or 'detailed')
            include_literature: Whether to include literature evidence
            include_alternatives: Whether to include alternative pathways
            
        Returns:
            Complete analysis with explanations
        """
        start_time = datetime.utcnow()
        
        try:
            # Perform base clinical analysis
            base_analysis = self.extract_clinical_entities(patient_note, patient_context)
            
            # Generate reasoning chain
            reasoning_chain = self.generate_reasoning_chain(
                entities=base_analysis,
                patient_context=patient_context,
                depth=depth
            )
            
            # Get literature evidence if requested
            evidence_sources = []
            if include_literature:
                evidence_sources = self.gather_literature_evidence(
                    entities=base_analysis,
                    reasoning_chain=reasoning_chain
                )
            
            # Calculate uncertainty analysis
            uncertainty_analysis = self.uncertainty_calculator.assess_diagnostic_uncertainty(
                entities=base_analysis
            )
            
            # Generate alternative pathways if requested
            alternative_pathways = []
            if include_alternatives:
                alternative_pathways = self.generate_alternative_pathways(
                    entities=base_analysis,
                    patient_context=patient_context
                )
            
            # Store analysis session
            session_id = self.storage_service.create_analysis_session(
                patient_note=patient_note,
                patient_context=patient_context,
                analysis_result=base_analysis
            )
            
            # Store reasoning chain
            self.storage_service.store_reasoning_chain(
                session_id=session_id,
                reasoning_chain=reasoning_chain
            )
            
            # Store uncertainty analysis
            self.storage_service.store_uncertainty_analysis(
                session_id=session_id,
                uncertainty_analysis=uncertainty_analysis
            )
            
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            
            return {
                'analysis': base_analysis,
                'explanation': {
                    'reasoning_chain': reasoning_chain,
                    'evidence_sources': evidence_sources,
                    'uncertainty_analysis': uncertainty_analysis,
                    'alternative_pathways': alternative_pathways
                },
                'session_id': session_id,
                'processing_time': processing_time,
                'metadata': {
                    'depth': depth,
                    'literature_included': include_literature,
                    'alternatives_included': include_alternatives,
                    'timestamp': end_time.isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error in explainable analysis: {str(e)}")
            return {
                'analysis': self._empty_extraction_result(error=str(e)),
                'explanation': {
                    'reasoning_chain': [],
                    'evidence_sources': [],
                    'uncertainty_analysis': {},
                    'alternative_pathways': []
                },
                'error': str(e)
            }
    
    def generate_reasoning_chain(self, entities: Dict[str, Any], patient_context: Dict,
                               depth: str = 'detailed') -> List[Dict]:
        """
        Generate step-by-step clinical reasoning chain
        
        Args:
            entities: Extracted clinical entities
            patient_context: Patient context information
            depth: Level of detail for reasoning
            
        Returns:
            List of reasoning steps with evidence
        """
        try:
            reasoning_steps = []
            step_number = 1
            
            # Step 1: Initial presentation analysis
            primary_symptoms = [s for s in entities.get('symptoms', []) 
                              if s.get('confidence', 0) > 0.7]
            
            if primary_symptoms:
                step = {
                    'step': step_number,
                    'reasoning': self._generate_symptom_reasoning(primary_symptoms, patient_context),
                    'evidence_type': 'clinical_presentation',
                    'confidence': self._calculate_step_confidence(primary_symptoms),
                    'supporting_entities': [s['entity'] for s in primary_symptoms],
                    'clinical_significance': 'primary_assessment'
                }
                reasoning_steps.append(step)
                step_number += 1
            
            # Step 2: Vital signs and objective findings
            abnormal_vitals = [v for v in entities.get('vital_signs', []) 
                             if v.get('abnormal', False)]
            
            if abnormal_vitals:
                step = {
                    'step': step_number,
                    'reasoning': self._generate_vitals_reasoning(abnormal_vitals, patient_context),
                    'evidence_type': 'objective_findings',
                    'confidence': self._calculate_step_confidence(abnormal_vitals),
                    'supporting_entities': [v['entity'] for v in abnormal_vitals],
                    'clinical_significance': 'objective_assessment'
                }
                reasoning_steps.append(step)
                step_number += 1
            
            # Step 3: Diagnostic considerations
            conditions = [c for c in entities.get('conditions', []) 
                         if c.get('confidence', 0) > 0.6]
            
            if conditions:
                step = {
                    'step': step_number,
                    'reasoning': self._generate_diagnostic_reasoning(conditions, entities, patient_context),
                    'evidence_type': 'diagnostic_assessment',
                    'confidence': self._calculate_step_confidence(conditions),
                    'supporting_entities': [c['entity'] for c in conditions],
                    'clinical_significance': 'diagnostic_consideration'
                }
                reasoning_steps.append(step)
                step_number += 1
            
            # Step 4: Risk stratification
            overall_assessment = entities.get('overall_assessment', {})
            if overall_assessment:
                step = {
                    'step': step_number,
                    'reasoning': self._generate_risk_reasoning(overall_assessment, entities),
                    'evidence_type': 'risk_stratification',
                    'confidence': 0.85,  # Risk assessment confidence
                    'supporting_entities': overall_assessment.get('primary_concerns', []),
                    'clinical_significance': 'risk_assessment'
                }
                reasoning_steps.append(step)
                step_number += 1
            
            # Add detailed steps for comprehensive analysis
            if depth == 'detailed':
                reasoning_steps.extend(self._generate_detailed_reasoning_steps(
                    entities, patient_context, step_number
                ))
            
            return reasoning_steps
            
        except Exception as e:
            logger.error(f"Error generating reasoning chain: {str(e)}")
            return []
    
    def gather_literature_evidence(self, entities: Dict[str, Any], 
                                 reasoning_chain: List[Dict]) -> List[Dict]:
        """
        Gather supporting literature evidence for clinical findings
        
        Args:
            entities: Clinical entities from analysis
            reasoning_chain: Generated reasoning steps
            
        Returns:
            List of literature evidence with relevance scores
        """
        try:
            evidence_sources = []
            
            # Get evidence for high-confidence conditions
            conditions = [c for c in entities.get('conditions', []) 
                         if c.get('confidence', 0) > 0.7]
            
            for condition in conditions:
                query_hash = self.cache_service.generate_query_hash(
                    f"evidence_{condition['entity']}", max_results=5
                )
                
                # Check cache first
                cached_evidence = self.cache_service.get_cached_search(query_hash)
                
                if cached_evidence:
                    evidence_sources.extend(cached_evidence)
                else:
                    # Search PubMed for evidence
                    literature = self.pubmed_service.find_evidence_for_condition(
                        condition=condition['entity'],
                        treatment=self._extract_related_treatment(entities, condition)
                    )
                    
                    # Cache results
                    self.cache_service.cache_search_results(
                        query_hash=query_hash,
                        query=f"evidence_{condition['entity']}",
                        results=literature
                    )
                    
                    # Add relevance scoring
                    for article in literature:
                        article['relevance_score'] = self._calculate_relevance_score(
                            article, condition, entities
                        )
                        article['related_entity'] = condition['entity']
                    
                    evidence_sources.extend(literature)
            
            # Sort by relevance and return top results
            evidence_sources.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            return evidence_sources[:10]  # Top 10 most relevant
            
        except Exception as e:
            logger.error(f"Error gathering literature evidence: {str(e)}")
            return []
    
    def generate_alternative_pathways(self, entities: Dict[str, Any], 
                                    patient_context: Dict) -> List[Dict]:
        """
        Generate alternative diagnostic or treatment pathways
        
        Args:
            entities: Clinical entities from analysis
            patient_context: Patient context information
            
        Returns:
            List of alternative pathways with confidence scores
        """
        try:
            pathways = []
            
            # Get primary conditions for pathway exploration
            primary_conditions = [c for c in entities.get('conditions', []) 
                                if c.get('confidence', 0) > 0.6]
            
            for condition in primary_conditions:
                # Generate alternative diagnostic considerations
                alternative_diagnoses = self._generate_alternative_diagnoses(
                    primary_condition=condition,
                    symptoms=entities.get('symptoms', []),
                    patient_context=patient_context
                )
                
                # Generate treatment alternatives
                treatment_alternatives = self._generate_treatment_alternatives(
                    condition=condition,
                    patient_context=patient_context
                )
                
                pathway = {
                    'primary_condition': condition['entity'],
                    'confidence': condition.get('confidence', 0),
                    'alternative_diagnoses': alternative_diagnoses,
                    'treatment_alternatives': treatment_alternatives,
                    'clinical_rationale': self._generate_pathway_rationale(
                        condition, entities, patient_context
                    )
                }
                
                pathways.append(pathway)
            
            return pathways
            
        except Exception as e:
            logger.error(f"Error generating alternative pathways: {str(e)}")
            return []
    
    def _generate_symptom_reasoning(self, symptoms: List[Dict], patient_context: Dict) -> str:
        """Generate reasoning text for symptom analysis"""
        if not symptoms:
            return "No significant symptoms identified."
        
        symptom_names = [s['entity'] for s in symptoms]
        age = patient_context.get('age', 'unknown age')
        gender = patient_context.get('gender', 'unknown gender')
        
        reasoning = f"Patient presents with {', '.join(symptom_names)}. "
        reasoning += f"Given patient demographics ({age} years old, {gender}), "
        reasoning += "these symptoms suggest several possible clinical scenarios requiring further evaluation."
        
        return reasoning
    
    def _generate_vitals_reasoning(self, vitals: List[Dict], patient_context: Dict) -> str:
        """Generate reasoning text for vital signs analysis"""
        if not vitals:
            return "Vital signs within normal parameters."
        
        vital_findings = [f"{v['entity']}: {v.get('value', 'abnormal')}" for v in vitals]
        
        reasoning = f"Objective findings include abnormal vital signs: {', '.join(vital_findings)}. "
        reasoning += "These abnormalities support clinical concern and may indicate underlying pathophysiology."
        
        return reasoning
    
    def _generate_diagnostic_reasoning(self, conditions: List[Dict], entities: Dict, 
                                     patient_context: Dict) -> str:
        """Generate reasoning text for diagnostic considerations"""
        if not conditions:
            return "No specific diagnostic considerations identified."
        
        primary_condition = max(conditions, key=lambda x: x.get('confidence', 0))
        
        reasoning = f"Primary diagnostic consideration is {primary_condition['entity']} "
        reasoning += f"(confidence: {primary_condition.get('confidence', 0):.2f}). "
        
        supporting_symptoms = [s['entity'] for s in entities.get('symptoms', []) 
                              if s.get('confidence', 0) > 0.6]
        
        if supporting_symptoms:
            reasoning += f"This is supported by presenting symptoms: {', '.join(supporting_symptoms)}."
        
        return reasoning
    
    def _generate_risk_reasoning(self, assessment: Dict, entities: Dict) -> str:
        """Generate reasoning text for risk assessment"""
        risk_level = assessment.get('risk_level', 'unknown')
        requires_attention = assessment.get('requires_immediate_attention', False)
        
        reasoning = f"Overall risk assessment: {risk_level} risk. "
        
        if requires_attention:
            reasoning += "Immediate medical attention recommended. "
        
        primary_concerns = assessment.get('primary_concerns', [])
        if primary_concerns:
            reasoning += f"Primary concerns include: {', '.join(primary_concerns)}."
        
        return reasoning
    
    def _generate_detailed_reasoning_steps(self, entities: Dict, patient_context: Dict, 
                                         start_step: int) -> List[Dict]:
        """Generate additional detailed reasoning steps"""
        detailed_steps = []
        step_number = start_step
        
        # Medication considerations
        medications = entities.get('medications', [])
        if medications:
            step = {
                'step': step_number,
                'reasoning': f"Current medications include: {', '.join([m['entity'] for m in medications])}. "
                           "Medication history may influence diagnosis and treatment planning.",
                'evidence_type': 'medication_analysis',
                'confidence': 0.8,
                'supporting_entities': [m['entity'] for m in medications],
                'clinical_significance': 'treatment_context'
            }
            detailed_steps.append(step)
            step_number += 1
        
        # Procedure considerations
        procedures = entities.get('procedures', [])
        if procedures:
            step = {
                'step': step_number,
                'reasoning': f"Relevant procedures: {', '.join([p['entity'] for p in procedures])}. "
                           "These findings provide additional diagnostic information.",
                'evidence_type': 'procedure_analysis',
                'confidence': 0.75,
                'supporting_entities': [p['entity'] for p in procedures],
                'clinical_significance': 'diagnostic_support'
            }
            detailed_steps.append(step)
        
        return detailed_steps
    
    def _calculate_step_confidence(self, entities: List[Dict]) -> float:
        """Calculate confidence score for a reasoning step"""
        if not entities:
            return 0.5
        
        confidences = [e.get('confidence', 0.5) for e in entities]
        return sum(confidences) / len(confidences)
    
    def _extract_related_treatment(self, entities: Dict, condition: Dict) -> Optional[str]:
        """Extract related treatment for a condition from entities"""
        medications = entities.get('medications', [])
        procedures = entities.get('procedures', [])
        
        # Simple heuristic: return first medication or procedure
        if medications:
            return medications[0]['entity']
        elif procedures:
            return procedures[0]['entity']
        
        return None
    
    def _calculate_relevance_score(self, article: Dict, condition: Dict, entities: Dict) -> float:
        """Calculate relevance score for literature article"""
        score = 0.5  # Base score
        
        # Boost score if title contains condition
        if condition['entity'].lower() in article.get('title', '').lower():
            score += 0.3
        
        # Boost score if abstract contains condition
        if condition['entity'].lower() in article.get('abstract', '').lower():
            score += 0.2
        
        # Boost score for study type
        study_type = article.get('study_type', '').lower()
        if 'clinical trial' in study_type:
            score += 0.2
        elif 'systematic review' in study_type:
            score += 0.15
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _generate_alternative_diagnoses(self, primary_condition: Dict, symptoms: List[Dict],
                                      patient_context: Dict) -> List[Dict]:
        """Generate alternative diagnostic considerations"""
        # This would typically use medical knowledge bases or ML models
        # For now, return placeholder structure
        return [
            {
                'diagnosis': f"Alternative to {primary_condition['entity']}",
                'confidence': 0.6,
                'reasoning': "Based on similar symptom profile",
                'supporting_symptoms': [s['entity'] for s in symptoms[:2]]
            }
        ]
    
    def _generate_treatment_alternatives(self, condition: Dict, patient_context: Dict) -> List[Dict]:
        """Generate alternative treatment options"""
        # This would typically use treatment guidelines or databases
        # For now, return placeholder structure
        return [
            {
                'treatment': f"Standard treatment for {condition['entity']}",
                'evidence_level': 'high',
                'contraindications': [],
                'patient_factors': list(patient_context.keys())
            }
        ]
    
    def _generate_pathway_rationale(self, condition: Dict, entities: Dict, 
                                  patient_context: Dict) -> str:
        """Generate clinical rationale for pathway selection"""
        return (f"Pathway selection based on {condition['entity']} presentation with "
                f"{condition.get('confidence', 0):.2f} confidence. Patient factors "
                f"and clinical context support this approach.")
    
    def health_check(self) -> bool:
        """Check if the service is healthy"""
        try:
            # Test basic functionality
            test_result = self.extract_clinical_entities("test note", {})
            return 'error' not in test_result
        except Exception:
            return False
    
    def get_entity_by_id(self, entity_id: str) -> Optional[Dict]:
        """Get entity details by ID from storage"""
        try:
            return self.storage_service.get_entity_by_id(entity_id)
        except Exception as e:
            logger.error(f"Error getting entity: {str(e)}")
            return None
    
    def get_analysis_by_id(self, analysis_id: str) -> Optional[Dict]:
        """Get analysis details by ID from storage"""
        try:
            return self.storage_service.get_analysis_session(analysis_id)
        except Exception as e:
            logger.error(f"Error getting analysis: {str(e)}")
            return None