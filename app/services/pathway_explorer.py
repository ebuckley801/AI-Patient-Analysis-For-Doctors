"""Treatment pathway exploration and ranking service"""
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class TreatmentPathwayExplorer:
    """Explore and rank alternative treatment pathways"""
    
    def __init__(self):
        self.evidence_weights = {
            'clinical_trial': 0.9,
            'systematic_review': 0.85,
            'case_series': 0.7,
            'case_report': 0.6,
            'expert_opinion': 0.5
        }
        
        self.contraindication_categories = {
            'absolute': ['allergy', 'severe_reaction', 'drug_interaction'],
            'relative': ['age_related', 'comorbidity', 'pregnancy'],
            'monitoring': ['renal_function', 'liver_function', 'cardiac_function']
        }
    
    def generate_alternative_pathways(self, primary_diagnosis: Dict, patient_context: Dict,
                                    max_pathways: int = 5) -> List[Dict]:
        """
        Generate ranked alternative treatment pathways
        
        Args:
            primary_diagnosis: Primary diagnostic entity
            patient_context: Patient demographic and medical context
            max_pathways: Maximum number of pathways to generate
            
        Returns:
            List of treatment pathway dictionaries
        """
        try:
            condition = primary_diagnosis.get('entity', '')
            confidence = primary_diagnosis.get('confidence', 0.5)
            
            pathways = []
            
            # Generate standard treatment pathway
            standard_pathway = self._generate_standard_pathway(condition, patient_context)
            if standard_pathway:
                pathways.append(standard_pathway)
            
            # Generate alternative pathways based on condition
            alternative_pathways = self._generate_condition_specific_pathways(
                condition, patient_context
            )
            pathways.extend(alternative_pathways)
            
            # Generate conservative pathway
            conservative_pathway = self._generate_conservative_pathway(condition, patient_context)
            if conservative_pathway:
                pathways.append(conservative_pathway)
            
            # Generate aggressive pathway for severe cases
            if confidence > 0.8 and self._is_severe_condition(condition):
                aggressive_pathway = self._generate_aggressive_pathway(condition, patient_context)
                if aggressive_pathway:
                    pathways.append(aggressive_pathway)
            
            # Add pathway metadata
            for i, pathway in enumerate(pathways):
                pathway['pathway_id'] = f"pathway_{i+1}"
                pathway['primary_diagnosis'] = condition
                pathway['generation_timestamp'] = datetime.utcnow().isoformat()
            
            return pathways[:max_pathways]
            
        except Exception as e:
            logger.error(f"Error generating pathways: {str(e)}")
            return []
    
    def rank_pathways_by_evidence(self, pathways: List[Dict]) -> List[Dict]:
        """
        Rank pathways by strength of literature evidence
        
        Args:
            pathways: List of treatment pathways
            
        Returns:
            Pathways sorted by evidence strength
        """
        try:
            for pathway in pathways:
                evidence_score = self._calculate_evidence_score(pathway)
                pathway['evidence_score'] = evidence_score
                pathway['evidence_ranking'] = self._categorize_evidence_strength(evidence_score)
            
            # Sort by evidence score (descending)
            ranked_pathways = sorted(pathways, key=lambda x: x.get('evidence_score', 0), reverse=True)
            
            # Add ranking metadata
            for i, pathway in enumerate(ranked_pathways):
                pathway['rank'] = i + 1
                pathway['ranking_criteria'] = [
                    'evidence_strength',
                    'safety_profile',
                    'treatment_outcomes',
                    'patient_compatibility'
                ]
            
            return ranked_pathways
            
        except Exception as e:
            logger.error(f"Error ranking pathways: {str(e)}")
            return pathways
    
    def check_contraindications(self, pathway: Dict, patient_context: Dict) -> List[str]:
        """
        Check for contraindications based on patient context
        
        Args:
            pathway: Treatment pathway to check
            patient_context: Patient context information
            
        Returns:
            List of contraindications found
        """
        try:
            contraindications = []
            treatment_sequence = pathway.get('treatment_sequence', [])
            
            # Check age-related contraindications
            age = patient_context.get('age', 0)
            contraindications.extend(self._check_age_contraindications(treatment_sequence, age))
            
            # Check allergy contraindications
            allergies = patient_context.get('allergies', [])
            contraindications.extend(self._check_allergy_contraindications(treatment_sequence, allergies))
            
            # Check comorbidity contraindications
            comorbidities = patient_context.get('medical_history', [])
            contraindications.extend(self._check_comorbidity_contraindications(treatment_sequence, comorbidities))
            
            # Check drug interaction contraindications
            current_medications = patient_context.get('current_medications', [])
            contraindications.extend(self._check_drug_interactions(treatment_sequence, current_medications))
            
            return list(set(contraindications))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error checking contraindications: {str(e)}")
            return []
    
    def get_patient_specific_notes(self, pathways: List[Dict], patient_context: Dict) -> List[str]:
        """
        Generate patient-specific notes for pathway selection
        
        Args:
            pathways: List of treatment pathways
            patient_context: Patient context information
            
        Returns:
            List of patient-specific considerations
        """
        try:
            notes = []
            
            age = patient_context.get('age', 0)
            gender = patient_context.get('gender', '')
            comorbidities = patient_context.get('medical_history', [])
            
            # Age-specific considerations
            if age < 18:
                notes.append("Pediatric dosing and safety considerations apply")
            elif age > 65:
                notes.append("Geriatric considerations: reduced clearance, increased sensitivity")
            
            # Gender-specific considerations
            if gender.lower() == 'f':
                notes.append("Consider pregnancy status and reproductive health implications")
            
            # Comorbidity considerations
            if comorbidities:
                notes.append(f"Monitor for interactions with existing conditions: {', '.join(comorbidities[:3])}")
            
            # Pathway-specific notes
            high_risk_pathways = [p for p in pathways if p.get('evidence_score', 0) < 0.6]
            if high_risk_pathways:
                notes.append("Some pathways have limited evidence - consider consultation")
            
            # Monitoring requirements
            monitoring_required = any('monitoring' in p.get('contraindications', []) for p in pathways)
            if monitoring_required:
                notes.append("Regular monitoring of organ function required")
            
            return notes
            
        except Exception as e:
            logger.error(f"Error generating patient notes: {str(e)}")
            return []
    
    def _generate_standard_pathway(self, condition: str, patient_context: Dict) -> Optional[Dict]:
        """Generate standard/guideline-based treatment pathway"""
        # This would typically query treatment guidelines database
        # For now, generate based on common patterns
        
        if 'hypertension' in condition.lower():
            return {
                'pathway_name': 'Standard Hypertension Management',
                'treatment_sequence': [
                    {'step': 1, 'intervention': 'lifestyle_modifications', 'duration': '3_months'},
                    {'step': 2, 'intervention': 'ace_inhibitor', 'duration': 'ongoing'},
                    {'step': 3, 'intervention': 'diuretic_addition', 'duration': 'if_needed'}
                ],
                'evidence_strength': 0.9,
                'contraindications': [],
                'estimated_outcomes': {
                    'success_rate': 0.85,
                    'time_to_control': '2-4 weeks',
                    'side_effect_rate': 0.15
                },
                'supporting_studies': ['ACC/AHA_2017', 'ESC/ESH_2018']
            }
        
        elif 'diabetes' in condition.lower():
            return {
                'pathway_name': 'Standard Diabetes Management',
                'treatment_sequence': [
                    {'step': 1, 'intervention': 'metformin', 'duration': 'ongoing'},
                    {'step': 2, 'intervention': 'lifestyle_counseling', 'duration': 'ongoing'},
                    {'step': 3, 'intervention': 'glucose_monitoring', 'duration': 'ongoing'}
                ],
                'evidence_strength': 0.95,
                'contraindications': [],
                'estimated_outcomes': {
                    'success_rate': 0.8,
                    'time_to_control': '3-6 months',
                    'side_effect_rate': 0.1
                },
                'supporting_studies': ['ADA_2023', 'EASD_2023']
            }
        
        return None
    
    def _generate_condition_specific_pathways(self, condition: str, patient_context: Dict) -> List[Dict]:
        """Generate condition-specific alternative pathways"""
        pathways = []
        
        # Age-adjusted pathway
        age = patient_context.get('age', 0)
        if age > 65:
            pathways.append({
                'pathway_name': f'Geriatric-Optimized {condition} Management',
                'treatment_sequence': [
                    {'step': 1, 'intervention': 'low_dose_initiation', 'duration': '2_weeks'},
                    {'step': 2, 'intervention': 'gradual_titration', 'duration': '4_weeks'},
                    {'step': 3, 'intervention': 'regular_monitoring', 'duration': 'ongoing'}
                ],
                'evidence_strength': 0.75,
                'contraindications': [],
                'estimated_outcomes': {
                    'success_rate': 0.7,
                    'time_to_control': '4-8 weeks',
                    'side_effect_rate': 0.08
                },
                'supporting_studies': ['Geriatric_Guidelines_2022']
            })
        
        # Comorbidity-adjusted pathway
        comorbidities = patient_context.get('medical_history', [])
        if comorbidities:
            pathways.append({
                'pathway_name': f'Comorbidity-Adjusted {condition} Management',
                'treatment_sequence': [
                    {'step': 1, 'intervention': 'multidisciplinary_assessment', 'duration': '1_week'},
                    {'step': 2, 'intervention': 'coordinated_treatment', 'duration': 'ongoing'},
                    {'step': 3, 'intervention': 'integrated_monitoring', 'duration': 'ongoing'}
                ],
                'evidence_strength': 0.8,
                'contraindications': [],
                'estimated_outcomes': {
                    'success_rate': 0.75,
                    'time_to_control': '2-6 weeks',
                    'side_effect_rate': 0.12
                },
                'supporting_studies': ['Multimorbidity_Guidelines_2023']
            })
        
        return pathways
    
    def _generate_conservative_pathway(self, condition: str, patient_context: Dict) -> Optional[Dict]:
        """Generate conservative/watchful waiting pathway"""
        return {
            'pathway_name': f'Conservative {condition} Management',
            'treatment_sequence': [
                {'step': 1, 'intervention': 'lifestyle_modifications', 'duration': '6_weeks'},
                {'step': 2, 'intervention': 'regular_monitoring', 'duration': 'ongoing'},
                {'step': 3, 'intervention': 'symptom_tracking', 'duration': 'ongoing'}
            ],
            'evidence_strength': 0.6,
            'contraindications': [],
            'estimated_outcomes': {
                'success_rate': 0.6,
                'time_to_control': '6-12 weeks',
                'side_effect_rate': 0.05
            },
            'supporting_studies': ['Conservative_Management_2022']
        }
    
    def _generate_aggressive_pathway(self, condition: str, patient_context: Dict) -> Optional[Dict]:
        """Generate aggressive/intensive treatment pathway"""
        return {
            'pathway_name': f'Intensive {condition} Management',
            'treatment_sequence': [
                {'step': 1, 'intervention': 'combination_therapy', 'duration': '2_weeks'},
                {'step': 2, 'intervention': 'frequent_monitoring', 'duration': '4_weeks'},
                {'step': 3, 'intervention': 'specialist_consultation', 'duration': 'as_needed'}
            ],
            'evidence_strength': 0.85,
            'contraindications': ['elderly', 'frail', 'multiple_comorbidities'],
            'estimated_outcomes': {
                'success_rate': 0.9,
                'time_to_control': '1-2 weeks',
                'side_effect_rate': 0.25
            },
            'supporting_studies': ['Intensive_Treatment_2023']
        }
    
    def _is_severe_condition(self, condition: str) -> bool:
        """Check if condition is considered severe"""
        severe_conditions = [
            'acute myocardial infarction',
            'stroke',
            'sepsis',
            'pulmonary embolism',
            'diabetic ketoacidosis'
        ]
        return any(severe in condition.lower() for severe in severe_conditions)
    
    def _calculate_evidence_score(self, pathway: Dict) -> float:
        """Calculate evidence strength score for pathway"""
        base_evidence = pathway.get('evidence_strength', 0.5)
        
        # Boost score based on supporting studies
        studies = pathway.get('supporting_studies', [])
        study_boost = min(len(studies) * 0.1, 0.3)
        
        # Adjust based on estimated outcomes
        outcomes = pathway.get('estimated_outcomes', {})
        success_rate = outcomes.get('success_rate', 0.5)
        side_effect_rate = outcomes.get('side_effect_rate', 0.2)
        
        outcome_adjustment = (success_rate - side_effect_rate) * 0.2
        
        return min(base_evidence + study_boost + outcome_adjustment, 1.0)
    
    def _categorize_evidence_strength(self, score: float) -> str:
        """Categorize evidence strength score"""
        if score >= 0.8:
            return 'strong'
        elif score >= 0.6:
            return 'moderate'
        else:
            return 'limited'
    
    def _check_age_contraindications(self, treatments: List[Dict], age: int) -> List[str]:
        """Check for age-related contraindications"""
        contraindications = []
        
        for treatment in treatments:
            intervention = treatment.get('intervention', '').lower()
            
            # Pediatric contraindications
            if age < 18:
                if 'aspirin' in intervention:
                    contraindications.append('aspirin_pediatric_contraindication')
                
            # Geriatric contraindications
            elif age > 75:
                if 'anticholinergic' in intervention:
                    contraindications.append('anticholinergic_elderly_risk')
        
        return contraindications
    
    def _check_allergy_contraindications(self, treatments: List[Dict], allergies: List[str]) -> List[str]:
        """Check for allergy-related contraindications"""
        contraindications = []
        
        for treatment in treatments:
            intervention = treatment.get('intervention', '').lower()
            
            for allergy in allergies:
                if allergy.lower() in intervention:
                    contraindications.append(f'allergy_to_{allergy}')
        
        return contraindications
    
    def _check_comorbidity_contraindications(self, treatments: List[Dict], comorbidities: List[str]) -> List[str]:
        """Check for comorbidity-related contraindications"""
        contraindications = []
        
        high_risk_combinations = {
            'kidney_disease': ['ace_inhibitor', 'nsaid'],
            'liver_disease': ['acetaminophen', 'statin'],
            'heart_failure': ['calcium_channel_blocker']
        }
        
        for treatment in treatments:
            intervention = treatment.get('intervention', '').lower()
            
            for comorbidity in comorbidities:
                comorbidity_lower = comorbidity.lower()
                if comorbidity_lower in high_risk_combinations:
                    risky_drugs = high_risk_combinations[comorbidity_lower]
                    if any(drug in intervention for drug in risky_drugs):
                        contraindications.append(f'{comorbidity}_drug_interaction')
        
        return contraindications
    
    def _check_drug_interactions(self, treatments: List[Dict], current_meds: List[str]) -> List[str]:
        """Check for drug-drug interactions"""
        contraindications = []
        
        # Major drug interactions (simplified)
        interactions = {
            'warfarin': ['aspirin', 'nsaid'],
            'digoxin': ['quinidine', 'verapamil'],
            'lithium': ['ace_inhibitor', 'diuretic']
        }
        
        for treatment in treatments:
            intervention = treatment.get('intervention', '').lower()
            
            for current_med in current_meds:
                current_med_lower = current_med.lower()
                if current_med_lower in interactions:
                    interacting_drugs = interactions[current_med_lower]
                    if any(drug in intervention for drug in interacting_drugs):
                        contraindications.append(f'{current_med}_interaction')
        
        return contraindications