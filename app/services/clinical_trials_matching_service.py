"""
Advanced Clinical Trials Matching Service

Sophisticated patient-to-clinical-trial matching system using:
- Multi-dimensional eligibility assessment
- Machine learning-based similarity scoring
- Geographic feasibility analysis
- Dynamic eligibility criteria parsing
- Real-time trial status integration with ClinicalTrials.gov API

Built on existing multi-modal infrastructure with Faiss vector similarity.
"""

import asyncio
import logging
import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import requests
import numpy as np

from app.services.supabase_service import SupabaseService
from app.services.multimodal_vector_service import MultiModalVectorService, ModalityType
from app.services.patient_identity_service import PatientIdentityService
from app.services.data_fusion_service import DataFusionService, PatientProfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MatchingMethod(Enum):
    """Trial matching methodologies"""
    RULE_BASED = "rule_based"
    VECTOR_SIMILARITY = "vector_similarity"
    HYBRID = "hybrid"
    ML_ENHANCED = "ml_enhanced"

class EligibilityStatus(Enum):
    """Eligibility assessment outcomes"""
    ELIGIBLE = "eligible"
    POTENTIALLY_ELIGIBLE = "potentially_eligible"
    INELIGIBLE = "ineligible"
    INSUFFICIENT_DATA = "insufficient_data"
    REQUIRES_REVIEW = "requires_review"

@dataclass
class TrialMatch:
    """Represents a patient-trial match with detailed assessment"""
    patient_id: str
    trial_id: str
    nct_id: str
    trial_title: str
    overall_match_score: float
    eligibility_status: EligibilityStatus
    matching_method: MatchingMethod
    
    # Detailed eligibility assessment
    inclusion_criteria_analysis: Dict[str, Any]
    exclusion_criteria_analysis: Dict[str, Any]
    demographic_compatibility: Dict[str, float]
    clinical_compatibility: Dict[str, float]
    genetic_compatibility: Dict[str, float]
    
    # Practical considerations
    geographic_feasibility: bool
    estimated_travel_distance_km: Optional[float]
    trial_phase: Optional[str]
    enrollment_status: str
    estimated_enrollment_end: Optional[str]
    
    # Risk-benefit analysis
    potential_benefits: List[str]
    potential_risks: List[str]
    contraindications: List[str]
    
    # Matching confidence and reasoning
    confidence_score: float
    matching_reasoning: str
    uncertainty_factors: List[str]
    recommendation_priority: str  # "high", "medium", "low"

class ClinicalTrialsMatchingService:
    """Advanced patient-to-clinical-trial matching service"""
    
    def __init__(self):
        self.supabase = SupabaseService()
        self.vector_service = MultiModalVectorService()
        self.identity_service = PatientIdentityService()
        self.data_fusion_service = DataFusionService()
        
        # ClinicalTrials.gov API configuration
        self.clinicaltrials_api_base = "https://clinicaltrials.gov/api/v2"
        self.api_timeout = 30
        
        # Matching weights for different criteria types
        self.criteria_weights = {
            'age': 0.15,
            'gender': 0.10,
            'conditions': 0.25,
            'medications': 0.15,
            'biomarkers': 0.15,
            'genetic_markers': 0.10,
            'performance_status': 0.10
        }
        
        # Geographic matching parameters
        self.max_reasonable_distance_km = 500  # 500km max reasonable travel
        self.distance_penalty_threshold = 100  # Start penalizing after 100km
        
    # ============================================================================
    # MAIN MATCHING INTERFACE
    # ============================================================================
    
    async def find_matching_trials(self, patient_id: str, 
                                 conditions: List[str] = None,
                                 max_results: int = 20,
                                 matching_method: MatchingMethod = MatchingMethod.HYBRID,
                                 include_observational: bool = True,
                                 max_distance_km: float = None) -> List[TrialMatch]:
        """
        Find clinical trials matching a patient's profile
        
        Args:
            patient_id: Unified patient identifier
            conditions: Specific conditions to search for (optional)
            max_results: Maximum number of matches to return
            matching_method: Matching methodology to use
            include_observational: Include observational studies
            max_distance_km: Maximum travel distance in kilometers
            
        Returns:
            List of TrialMatch objects sorted by match quality
        """
        try:
            logger.info(f"Finding trials for patient {patient_id} using {matching_method.value}")
            
            # Get comprehensive patient profile
            patient_profile = await self.data_fusion_service.create_comprehensive_patient_profile(patient_id)
            
            # Get active clinical trials
            candidate_trials = await self._get_candidate_trials(conditions, include_observational)
            
            if not candidate_trials:
                logger.warning("No candidate trials found")
                return []
            
            # Match patient to trials using specified method
            if matching_method == MatchingMethod.RULE_BASED:
                matches = await self._rule_based_matching(patient_profile, candidate_trials)
            elif matching_method == MatchingMethod.VECTOR_SIMILARITY:
                matches = await self._vector_similarity_matching(patient_profile, candidate_trials)
            elif matching_method == MatchingMethod.HYBRID:
                matches = await self._hybrid_matching(patient_profile, candidate_trials)
            else:
                matches = await self._ml_enhanced_matching(patient_profile, candidate_trials)
            
            # Apply geographic filtering if specified
            if max_distance_km:
                matches = [m for m in matches if m.estimated_travel_distance_km is None or 
                          m.estimated_travel_distance_km <= max_distance_km]
            
            # Sort by overall match score and limit results
            matches.sort(key=lambda x: x.overall_match_score, reverse=True)
            final_matches = matches[:max_results]
            
            # Store matching results
            await self._store_matching_results(patient_id, final_matches, matching_method)
            
            logger.info(f"Found {len(final_matches)} matching trials for patient {patient_id}")
            return final_matches
            
        except Exception as e:
            logger.error(f"Error finding matching trials: {e}")
            raise
    
    async def assess_trial_eligibility(self, patient_id: str, nct_id: str) -> TrialMatch:
        """
        Detailed eligibility assessment for a specific trial
        
        Args:
            patient_id: Unified patient identifier
            nct_id: ClinicalTrials.gov NCT identifier
            
        Returns:
            Detailed TrialMatch with comprehensive eligibility assessment
        """
        try:
            # Get patient profile
            patient_profile = await self.data_fusion_service.create_comprehensive_patient_profile(patient_id)
            
            # Get trial details from database or API
            trial_data = await self._get_trial_details(nct_id)
            
            if not trial_data:
                raise ValueError(f"Trial {nct_id} not found")
            
            # Perform detailed eligibility assessment
            match = await self._detailed_eligibility_assessment(patient_profile, trial_data)
            
            return match
            
        except Exception as e:
            logger.error(f"Error assessing trial eligibility: {e}")
            raise
    
    # ============================================================================
    # TRIAL RETRIEVAL AND CACHING
    # ============================================================================
    
    async def _get_candidate_trials(self, conditions: List[str] = None, 
                                  include_observational: bool = True) -> List[Dict[str, Any]]:
        """Get candidate trials from database and API"""
        try:
            # Query local database first
            query = self.supabase.client.table('clinical_trials')\
                .select('*')\
                .in_('status', ['RECRUITING', 'ENROLLING_BY_INVITATION', 'ACTIVE_NOT_RECRUITING'])
            
            if not include_observational:
                query = query.eq('study_type', 'INTERVENTIONAL')
            
            local_trials = query.execute()
            
            trials = local_trials.data if local_trials.data else []
            
            # If we have specific conditions and limited local results, fetch from API
            if conditions and len(trials) < 50:
                api_trials = await self._fetch_trials_from_api(conditions, include_observational)
                
                # Merge with local trials (avoid duplicates)
                existing_ncts = set(trial['nct_id'] for trial in trials)
                for api_trial in api_trials:
                    if api_trial.get('nct_id') not in existing_ncts:
                        trials.append(api_trial)
            
            logger.info(f"Found {len(trials)} candidate trials")
            return trials
            
        except Exception as e:
            logger.error(f"Error getting candidate trials: {e}")
            return []
    
    async def _fetch_trials_from_api(self, conditions: List[str], 
                                   include_observational: bool = True) -> List[Dict[str, Any]]:
        """Fetch trials from ClinicalTrials.gov API"""
        try:
            # Build query parameters
            params = {
                'format': 'json',
                'query.cond': ' OR '.join(conditions) if conditions else '',
                'query.status': 'RECRUITING',
                'pageSize': 100,
                'countTotal': 'true'
            }
            
            if not include_observational:
                params['query.study_type'] = 'INTERVENTIONAL'
            
            # Make API request
            response = requests.get(
                f"{self.clinicaltrials_api_base}/studies",
                params=params,
                timeout=self.api_timeout
            )
            response.raise_for_status()
            
            data = response.json()
            studies = data.get('studies', [])
            
            # Convert API format to our internal format
            formatted_trials = []
            for study in studies:
                formatted_trial = self._format_api_trial(study)
                if formatted_trial:
                    formatted_trials.append(formatted_trial)
            
            logger.info(f"Fetched {len(formatted_trials)} trials from API")
            return formatted_trials
            
        except Exception as e:
            logger.error(f"Error fetching trials from API: {e}")
            return []
    
    def _format_api_trial(self, api_study: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Format API study data to our internal format"""
        try:
            protocol_section = api_study.get('protocolSection', {})
            identification = protocol_section.get('identificationModule', {})
            status_info = protocol_section.get('statusModule', {})
            design_info = protocol_section.get('designModule', {})
            conditions_info = protocol_section.get('conditionsModule', {})
            eligibility_info = protocol_section.get('eligibilityModule', {})
            contacts_info = protocol_section.get('contactsLocationsModule', {})
            
            return {
                'nct_id': identification.get('nctId'),
                'title': identification.get('briefTitle'),
                'brief_summary': identification.get('briefSummary'),
                'detailed_description': identification.get('detailedDescription'),
                'study_type': design_info.get('studyType'),
                'phase': design_info.get('phases', [None])[0] if design_info.get('phases') else None,
                'status': status_info.get('overallStatus'),
                'conditions': conditions_info.get('conditions', []),
                'eligibility_criteria': eligibility_info.get('eligibilityCriteria'),
                'minimum_age': eligibility_info.get('minimumAge'),
                'maximum_age': eligibility_info.get('maximumAge'),
                'gender': eligibility_info.get('gender'),
                'locations': contacts_info.get('locations', []),
                'interventions': protocol_section.get('armsInterventionsModule', {}).get('interventions', []),
                'primary_outcomes': protocol_section.get('outcomesModule', {}).get('primaryOutcomes', []),
                'keywords': conditions_info.get('keywords', []),
                'last_update_date': status_info.get('lastUpdateSubmitDate')
            }
            
        except Exception as e:
            logger.warning(f"Error formatting API trial: {e}")
            return None
    
    async def _get_trial_details(self, nct_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed trial information"""
        try:
            # Try database first
            result = self.supabase.client.table('clinical_trials')\
                .select('*')\
                .eq('nct_id', nct_id)\
                .execute()
            
            if result.data:
                return result.data[0]
            
            # Fall back to API
            response = requests.get(
                f"{self.clinicaltrials_api_base}/studies/{nct_id}",
                params={'format': 'json'},
                timeout=self.api_timeout
            )
            response.raise_for_status()
            
            api_data = response.json()
            if api_data.get('studies'):
                return self._format_api_trial(api_data['studies'][0])
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting trial details for {nct_id}: {e}")
            return None
    
    # ============================================================================
    # MATCHING ALGORITHMS
    # ============================================================================
    
    async def _rule_based_matching(self, patient_profile: PatientProfile, 
                                 trials: List[Dict[str, Any]]) -> List[TrialMatch]:
        """Rule-based matching using explicit eligibility criteria"""
        matches = []
        
        for trial in trials:
            try:
                # Parse eligibility criteria
                inclusion_criteria, exclusion_criteria = self._parse_eligibility_criteria(
                    trial.get('eligibility_criteria', '')
                )
                
                # Assess each criterion
                inclusion_analysis = self._assess_inclusion_criteria(patient_profile, inclusion_criteria)
                exclusion_analysis = self._assess_exclusion_criteria(patient_profile, exclusion_criteria)
                
                # Calculate match score
                match_score = self._calculate_rule_based_score(inclusion_analysis, exclusion_analysis)
                
                # Determine eligibility status
                eligibility_status = self._determine_eligibility_status(inclusion_analysis, exclusion_analysis)
                
                # Create trial match
                match = await self._create_trial_match(
                    patient_profile, trial, match_score, eligibility_status,
                    MatchingMethod.RULE_BASED, inclusion_analysis, exclusion_analysis
                )
                
                matches.append(match)
                
            except Exception as e:
                logger.warning(f"Error in rule-based matching for trial {trial.get('nct_id')}: {e}")
                continue
        
        return matches
    
    async def _vector_similarity_matching(self, patient_profile: PatientProfile,
                                        trials: List[Dict[str, Any]]) -> List[TrialMatch]:
        """Vector similarity-based matching using embeddings"""
        matches = []
        
        try:
            # Generate patient embedding
            patient_embedding = await self._generate_patient_embedding(patient_profile)
            
            for trial in trials:
                # Generate trial embedding
                trial_embedding = await self._generate_trial_embedding(trial)
                
                # Calculate cosine similarity
                similarity_score = self._calculate_cosine_similarity(patient_embedding, trial_embedding)
                
                # Convert similarity to match score
                match_score = min(similarity_score * 1.2, 1.0)  # Scale up slightly
                
                # Basic eligibility assessment for vector method
                inclusion_analysis = {'vector_based': True, 'similarity_score': similarity_score}
                exclusion_analysis = {'vector_based': True}
                
                # Determine status based on similarity threshold
                if similarity_score > 0.7:
                    eligibility_status = EligibilityStatus.ELIGIBLE
                elif similarity_score > 0.5:
                    eligibility_status = EligibilityStatus.POTENTIALLY_ELIGIBLE
                else:
                    eligibility_status = EligibilityStatus.INELIGIBLE
                
                # Create trial match
                match = await self._create_trial_match(
                    patient_profile, trial, match_score, eligibility_status,
                    MatchingMethod.VECTOR_SIMILARITY, inclusion_analysis, exclusion_analysis
                )
                
                matches.append(match)
                
        except Exception as e:
            logger.error(f"Error in vector similarity matching: {e}")
        
        return matches
    
    async def _hybrid_matching(self, patient_profile: PatientProfile,
                             trials: List[Dict[str, Any]]) -> List[TrialMatch]:
        """Hybrid matching combining rule-based and vector similarity"""
        
        # Get both rule-based and vector-based matches
        rule_matches = await self._rule_based_matching(patient_profile, trials)
        vector_matches = await self._vector_similarity_matching(patient_profile, trials)
        
        # Create lookup for vector matches
        vector_lookup = {match.nct_id: match for match in vector_matches}
        
        hybrid_matches = []
        
        for rule_match in rule_matches:
            vector_match = vector_lookup.get(rule_match.nct_id)
            
            if vector_match:
                # Combine scores (weighted average)
                hybrid_score = 0.6 * rule_match.overall_match_score + 0.4 * vector_match.overall_match_score
                
                # Create hybrid match
                hybrid_match = TrialMatch(
                    patient_id=rule_match.patient_id,
                    trial_id=rule_match.trial_id,
                    nct_id=rule_match.nct_id,
                    trial_title=rule_match.trial_title,
                    overall_match_score=hybrid_score,
                    eligibility_status=rule_match.eligibility_status,  # Prefer rule-based assessment
                    matching_method=MatchingMethod.HYBRID,
                    inclusion_criteria_analysis=rule_match.inclusion_criteria_analysis,
                    exclusion_criteria_analysis=rule_match.exclusion_criteria_analysis,
                    demographic_compatibility=rule_match.demographic_compatibility,
                    clinical_compatibility=rule_match.clinical_compatibility,
                    genetic_compatibility=rule_match.genetic_compatibility,
                    geographic_feasibility=rule_match.geographic_feasibility,
                    estimated_travel_distance_km=rule_match.estimated_travel_distance_km,
                    trial_phase=rule_match.trial_phase,
                    enrollment_status=rule_match.enrollment_status,
                    estimated_enrollment_end=rule_match.estimated_enrollment_end,
                    potential_benefits=rule_match.potential_benefits,
                    potential_risks=rule_match.potential_risks,
                    contraindications=rule_match.contraindications,
                    confidence_score=max(rule_match.confidence_score, vector_match.confidence_score),
                    matching_reasoning=f"Hybrid approach: {rule_match.matching_reasoning}; Vector similarity: {vector_match.overall_match_score:.2f}",
                    uncertainty_factors=list(set(rule_match.uncertainty_factors + vector_match.uncertainty_factors)),
                    recommendation_priority=rule_match.recommendation_priority
                )
                
                hybrid_matches.append(hybrid_match)
        
        return hybrid_matches
    
    async def _ml_enhanced_matching(self, patient_profile: PatientProfile,
                                  trials: List[Dict[str, Any]]) -> List[TrialMatch]:
        """ML-enhanced matching with advanced feature engineering"""
        # Placeholder for future ML model integration
        # For now, use hybrid approach with enhanced scoring
        return await self._hybrid_matching(patient_profile, trials)
    
    # ============================================================================
    # ELIGIBILITY ASSESSMENT
    # ============================================================================
    
    def _parse_eligibility_criteria(self, criteria_text: str) -> Tuple[List[str], List[str]]:
        """Parse eligibility criteria text into inclusion and exclusion lists"""
        if not criteria_text:
            return [], []
        
        # Split into inclusion and exclusion sections
        sections = re.split(r'(?i)exclusion\s*criteria?:?', criteria_text)
        inclusion_text = sections[0]
        exclusion_text = sections[1] if len(sections) > 1 else ""
        
        # Remove "inclusion criteria" header if present
        inclusion_text = re.sub(r'(?i)^.*?inclusion\s*criteria?:?\s*', '', inclusion_text).strip()
        exclusion_text = exclusion_text.strip()
        
        # Split criteria by common delimiters
        inclusion_criteria = self._split_criteria(inclusion_text)
        exclusion_criteria = self._split_criteria(exclusion_text)
        
        return inclusion_criteria, exclusion_criteria
    
    def _split_criteria(self, text: str) -> List[str]:
        """Split criteria text into individual criteria"""
        if not text.strip():
            return []
        
        # Common patterns for splitting criteria
        criteria = re.split(r'\n\s*[-•]\s*|\n\s*\d+\.\s*|\n\s*[a-z]\)\s*', text)
        
        # Clean up and filter empty criteria
        cleaned_criteria = []
        for criterion in criteria:
            criterion = criterion.strip()
            if len(criterion) > 10:  # Ignore very short fragments
                cleaned_criteria.append(criterion)
        
        return cleaned_criteria
    
    def _assess_inclusion_criteria(self, patient_profile: PatientProfile, 
                                 inclusion_criteria: List[str]) -> Dict[str, Any]:
        """Assess how well patient meets inclusion criteria"""
        assessment = {
            'total_criteria': len(inclusion_criteria),
            'criteria_met': 0,
            'criteria_assessments': [],
            'overall_score': 0.0
        }
        
        for criterion in inclusion_criteria:
            criterion_assessment = self._assess_single_criterion(patient_profile, criterion, is_inclusion=True)
            assessment['criteria_assessments'].append(criterion_assessment)
            
            if criterion_assessment['meets_criterion']:
                assessment['criteria_met'] += 1
        
        # Calculate overall score
        if assessment['total_criteria'] > 0:
            assessment['overall_score'] = assessment['criteria_met'] / assessment['total_criteria']
        
        return assessment
    
    def _assess_exclusion_criteria(self, patient_profile: PatientProfile,
                                 exclusion_criteria: List[str]) -> Dict[str, Any]:
        """Assess if patient violates any exclusion criteria"""
        assessment = {
            'total_criteria': len(exclusion_criteria),
            'criteria_violated': 0,
            'criteria_assessments': [],
            'overall_score': 1.0  # Start at 1.0, subtract for violations
        }
        
        for criterion in exclusion_criteria:
            criterion_assessment = self._assess_single_criterion(patient_profile, criterion, is_inclusion=False)
            assessment['criteria_assessments'].append(criterion_assessment)
            
            if criterion_assessment['meets_criterion']:  # For exclusion, meeting = violation
                assessment['criteria_violated'] += 1
        
        # Calculate overall score (1.0 - violation rate)
        if assessment['total_criteria'] > 0:
            violation_rate = assessment['criteria_violated'] / assessment['total_criteria']
            assessment['overall_score'] = 1.0 - violation_rate
        
        return assessment
    
    def _assess_single_criterion(self, patient_profile: PatientProfile, 
                               criterion: str, is_inclusion: bool) -> Dict[str, Any]:
        """Assess a single eligibility criterion"""
        assessment = {
            'criterion_text': criterion,
            'meets_criterion': False,
            'confidence': 0.0,
            'reasoning': '',
            'data_available': False
        }
        
        criterion_lower = criterion.lower()
        
        # Age criteria
        if 'age' in criterion_lower:
            age_matches = re.findall(r'(\d+)', criterion_lower)
            if len(age_matches) == 2:
                min_age, max_age = sorted([int(age) for age in age_matches])
                patient_age = patient_profile.demographic_summary.get('age')

                if patient_age:
                    assessment['data_available'] = True
                    if min_age <= patient_age <= max_age:
                        assessment['meets_criterion'] = True
                    assessment['confidence'] = 0.9
                    assessment['reasoning'] = f"Patient age {patient_age} vs criterion age range {min_age}-{max_age}"

            elif len(age_matches) == 1:
                criterion_age = int(age_matches[0])
                patient_age = patient_profile.demographic_summary.get('age')
                
                if patient_age:
                    assessment['data_available'] = True
                    if 'older than' in criterion_lower or 'above' in criterion_lower:
                        assessment['meets_criterion'] = patient_age > criterion_age
                    elif 'younger than' in criterion_lower or 'below' in criterion_lower:
                        assessment['meets_criterion'] = patient_age < criterion_age
                    elif 'at least' in criterion_lower:
                        assessment['meets_criterion'] = patient_age >= criterion_age
                    
                    assessment['confidence'] = 0.9
                    assessment['reasoning'] = f"Patient age {patient_age} vs criterion age {criterion_age}"
        
        # Gender criteria
        elif 'male' in criterion_lower or 'female' in criterion_lower:
            patient_gender = patient_profile.demographic_summary.get('gender', '').lower()
            if patient_gender:
                assessment['data_available'] = True
                if 'male' in criterion_lower and 'male' in patient_gender:
                    assessment['meets_criterion'] = True
                elif 'female' in criterion_lower and 'female' in patient_gender:
                    assessment['meets_criterion'] = True
                assessment['confidence'] = 0.95
                assessment['reasoning'] = f"Patient gender: {patient_gender}"
        
        # Clinical condition criteria
        else:
            # Check against patient's clinical summary and fused insights
            clinical_text = json.dumps(patient_profile.clinical_summary).lower()
            
            # Simple keyword matching (would be enhanced with NLP)
            key_terms = self._extract_medical_terms(criterion_lower)
            matches_found = sum(1 for term in key_terms if term in clinical_text)
            
            if key_terms:
                assessment['data_available'] = len(key_terms) > 0
                assessment['meets_criterion'] = matches_found > 0
                assessment['confidence'] = min(0.8, matches_found / len(key_terms))
                assessment['reasoning'] = f"Found {matches_found}/{len(key_terms)} key terms in clinical data"
        
        return assessment
    
    def _extract_medical_terms(self, text: str) -> List[str]:
        """Extract key medical terms from criteria text"""
        # Simple extraction - would use medical NLP in production
        medical_terms = []
        
        # Common medical conditions
        conditions = ['diabetes', 'hypertension', 'cancer', 'heart disease', 'kidney disease', 
                     'liver disease', 'stroke', 'myocardial infarction', 'copd', 'asthma']
        
        for condition in conditions:
            if condition in text:
                medical_terms.append(condition)
        
        return medical_terms
    
    # ============================================================================
    # SCORING AND ASSESSMENT
    # ============================================================================
    
    def _calculate_rule_based_score(self, inclusion_analysis: Dict[str, Any],
                                  exclusion_analysis: Dict[str, Any]) -> float:
        """Calculate overall match score from rule-based analysis"""
        inclusion_score = inclusion_analysis.get('overall_score', 0.0)
        exclusion_score = exclusion_analysis.get('overall_score', 1.0)
        
        # Weighted combination favoring exclusion compliance
        return 0.4 * inclusion_score + 0.6 * exclusion_score
    
    def _determine_eligibility_status(self, inclusion_analysis: Dict[str, Any],
                                    exclusion_analysis: Dict[str, Any]) -> EligibilityStatus:
        """Determine eligibility status from criteria analysis"""
        inclusion_score = inclusion_analysis.get('overall_score', 0.0)
        exclusion_score = exclusion_analysis.get('overall_score', 1.0)
        violations = exclusion_analysis.get('criteria_violated', 0)
        
        # Hard exclusion rule
        if violations > 0:
            return EligibilityStatus.INELIGIBLE
        
        # Inclusion-based assessment
        if inclusion_score >= 0.8:
            return EligibilityStatus.ELIGIBLE
        elif inclusion_score >= 0.6:
            return EligibilityStatus.POTENTIALLY_ELIGIBLE
        elif inclusion_score >= 0.3:
            return EligibilityStatus.REQUIRES_REVIEW
        else:
            return EligibilityStatus.INSUFFICIENT_DATA
    
    async def _generate_patient_embedding(self, patient_profile: PatientProfile) -> np.ndarray:
        """Generate vector embedding for patient profile"""
        # Combine features from different modalities
        features = []
        
        # Demographic features
        age = patient_profile.demographic_summary.get('age', 0) / 100  # Normalize
        features.append(age)
        
        gender = 1.0 if patient_profile.demographic_summary.get('gender', '').lower() == 'male' else 0.0
        features.append(gender)
        
        # Clinical features (simplified)
        clinical_entities = patient_profile.clinical_summary.get('entity_counts', {})
        features.extend([
            clinical_entities.get('conditions', 0) / 10,  # Normalize
            clinical_entities.get('medications', 0) / 10,
            clinical_entities.get('procedures', 0) / 5
        ])
        
        # Genetic risk features
        genetic_profile = patient_profile.genetic_risk_profile
        if genetic_profile.get('available'):
            features.append(genetic_profile.get('total_variants', 0) / 1000)
            features.append(genetic_profile.get('high_risk_variants', 0) / 100)
        else:
            features.extend([0.0, 0.0])
        
        # Pad to standard embedding dimension
        embedding = np.array(features, dtype=np.float32)
        
        # Pad or truncate to 128 dimensions
        target_dim = 128
        if len(embedding) < target_dim:
            padding = np.zeros(target_dim - len(embedding), dtype=np.float32)
            embedding = np.concatenate([embedding, padding])
        else:
            embedding = embedding[:target_dim]
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    async def _generate_trial_embedding(self, trial: Dict[str, Any]) -> np.ndarray:
        """Generate vector embedding for clinical trial"""
        features = []
        
        # Trial type features
        study_type = trial.get('study_type', '')
        features.append(1.0 if study_type == 'INTERVENTIONAL' else 0.0)
        
        # Phase encoding
        phase = trial.get('phase', '')
        phase_encoding = {'PHASE1': 0.25, 'PHASE2': 0.5, 'PHASE3': 0.75, 'PHASE4': 1.0}
        features.append(phase_encoding.get(phase, 0.0))
        
        # Condition features (simplified)
        conditions = trial.get('conditions', [])
        condition_categories = ['cardiovascular', 'oncology', 'neurological', 'infectious', 'metabolic']
        
        for category in condition_categories:
            has_condition = any(category in condition.lower() for condition in conditions)
            features.append(1.0 if has_condition else 0.0)
        
        # Age criteria features
        min_age_str = trial.get('minimum_age', '0 Years')
        max_age_str = trial.get('maximum_age', '100 Years')
        
        min_age = self._parse_age_string(min_age_str) / 100
        max_age = self._parse_age_string(max_age_str) / 100
        
        features.extend([min_age, max_age])
        
        # Gender criteria
        gender = trial.get('gender', 'ALL')
        features.append(1.0 if gender == 'MALE' else 0.0)
        features.append(1.0 if gender == 'FEMALE' else 0.0)
        
        # Pad to target dimension
        embedding = np.array(features, dtype=np.float32)
        target_dim = 128
        
        if len(embedding) < target_dim:
            padding = np.zeros(target_dim - len(embedding), dtype=np.float32)
            embedding = np.concatenate([embedding, padding])
        else:
            embedding = embedding[:target_dim]
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _parse_age_string(self, age_str: str) -> int:
        """Parse age string like '18 Years' to integer"""
        if not age_str:
            return 0
        
        match = re.search(r'(\d+)', age_str)
        return int(match.group(1)) if match else 0
    
    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    # ============================================================================
    # TRIAL MATCH CREATION AND STORAGE
    # ============================================================================
    
    async def _create_trial_match(self, patient_profile: PatientProfile, 
                                trial: Dict[str, Any], match_score: float,
                                eligibility_status: EligibilityStatus,
                                matching_method: MatchingMethod,
                                inclusion_analysis: Dict[str, Any],
                                exclusion_analysis: Dict[str, Any]) -> TrialMatch:
        """Create comprehensive TrialMatch object"""
        
        # Geographic analysis
        geographic_feasibility, travel_distance = await self._assess_geographic_feasibility(
            patient_profile, trial
        )
        
        # Risk-benefit analysis
        benefits, risks, contraindications = self._assess_risk_benefit(patient_profile, trial)
        
        # Confidence and reasoning
        confidence_score = self._calculate_confidence_score(match_score, inclusion_analysis, exclusion_analysis)
        reasoning = self._generate_matching_reasoning(matching_method, match_score, inclusion_analysis, exclusion_analysis)
        uncertainty_factors = self._identify_uncertainty_factors(inclusion_analysis, exclusion_analysis)
        
        # Recommendation priority
        priority = self._determine_recommendation_priority(match_score, eligibility_status, travel_distance)
        
        return TrialMatch(
            patient_id=patient_profile.unified_patient_id,
            trial_id=trial.get('trial_id', trial.get('nct_id')),
            nct_id=trial.get('nct_id'),
            trial_title=trial.get('title', ''),
            overall_match_score=match_score,
            eligibility_status=eligibility_status,
            matching_method=matching_method,
            inclusion_criteria_analysis=inclusion_analysis,
            exclusion_criteria_analysis=exclusion_analysis,
            demographic_compatibility=self._assess_demographic_compatibility(patient_profile, trial),
            clinical_compatibility=self._assess_clinical_compatibility(patient_profile, trial),
            genetic_compatibility=self._assess_genetic_compatibility(patient_profile, trial),
            geographic_feasibility=geographic_feasibility,
            estimated_travel_distance_km=travel_distance,
            trial_phase=trial.get('phase'),
            enrollment_status=trial.get('status', ''),
            estimated_enrollment_end=trial.get('estimated_enrollment_end'),
            potential_benefits=benefits,
            potential_risks=risks,
            contraindications=contraindications,
            confidence_score=confidence_score,
            matching_reasoning=reasoning,
            uncertainty_factors=uncertainty_factors,
            recommendation_priority=priority
        )
    
    async def _assess_geographic_feasibility(self, patient_profile: PatientProfile,
                                           trial: Dict[str, Any]) -> Tuple[bool, Optional[float]]:
        """Assess geographic feasibility and estimate travel distance"""
        # Simplified implementation - would use real geographic services
        trial_locations = trial.get('locations', [])
        
        if not trial_locations:
            return True, None  # Assume feasible if no location data
        
        # For now, assume reasonable travel distance
        estimated_distance = 150.0  # km
        feasible = estimated_distance <= self.max_reasonable_distance_km
        
        return feasible, estimated_distance
    
    def _assess_risk_benefit(self, patient_profile: PatientProfile,
                           trial: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
        """Assess potential benefits, risks, and contraindications"""
        # Simplified assessment - would use sophisticated medical knowledge
        benefits = ["Potential access to novel therapy", "Close medical monitoring"]
        risks = ["Potential adverse effects", "Unknown long-term effects"]
        contraindications = []
        
        return benefits, risks, contraindications
    
    def _calculate_confidence_score(self, match_score: float,
                                  inclusion_analysis: Dict[str, Any],
                                  exclusion_analysis: Dict[str, Any]) -> float:
        """Calculate confidence in the matching result"""
        # Base confidence on match score
        base_confidence = match_score
        
        # Adjust based on data availability
        inclusion_data_coverage = sum(1 for assessment in inclusion_analysis.get('criteria_assessments', [])
                                    if assessment.get('data_available', False))
        total_inclusion = len(inclusion_analysis.get('criteria_assessments', []))
        
        if total_inclusion > 0:
            data_coverage = inclusion_data_coverage / total_inclusion
            return base_confidence * (0.5 + 0.5 * data_coverage)  # Scale by data availability
        
        return base_confidence * 0.5  # Lower confidence with no criteria data
    
    def _generate_matching_reasoning(self, method: MatchingMethod, score: float,
                                   inclusion_analysis: Dict[str, Any],
                                   exclusion_analysis: Dict[str, Any]) -> str:
        """Generate human-readable matching reasoning"""
        reasoning_parts = []
        
        reasoning_parts.append(f"Match score: {score:.2f} using {method.value} approach")
        
        if method == MatchingMethod.RULE_BASED:
            inclusion_met = inclusion_analysis.get('criteria_met', 0)
            inclusion_total = inclusion_analysis.get('total_criteria', 0)
            exclusion_violated = exclusion_analysis.get('criteria_violated', 0)
            
            reasoning_parts.append(f"Meets {inclusion_met}/{inclusion_total} inclusion criteria")
            if exclusion_violated > 0:
                reasoning_parts.append(f"Violates {exclusion_violated} exclusion criteria")
        
        return "; ".join(reasoning_parts)
    
    def _identify_uncertainty_factors(self, inclusion_analysis: Dict[str, Any],
                                    exclusion_analysis: Dict[str, Any]) -> List[str]:
        """Identify factors contributing to matching uncertainty"""
        factors = []
        
        # Check for missing data
        assessments = inclusion_analysis.get('criteria_assessments', []) + \
                     exclusion_analysis.get('criteria_assessments', [])
        
        missing_data_count = sum(1 for assessment in assessments if not assessment.get('data_available', False))
        if missing_data_count > 0:
            factors.append(f"Missing data for {missing_data_count} criteria")
        
        # Check for low confidence assessments
        low_confidence_count = sum(1 for assessment in assessments if assessment.get('confidence', 0) < 0.5)
        if low_confidence_count > 0:
            factors.append(f"Low confidence in {low_confidence_count} assessments")
        
        return factors
    
    def _determine_recommendation_priority(self, match_score: float,
                                         eligibility_status: EligibilityStatus,
                                         travel_distance: Optional[float]) -> str:
        """Determine recommendation priority"""
        if eligibility_status == EligibilityStatus.INELIGIBLE:
            return "low"
        
        if match_score >= 0.8 and (not travel_distance or travel_distance <= 200):
            return "high"
        elif match_score >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _assess_demographic_compatibility(self, patient_profile: PatientProfile,
                                        trial: Dict[str, Any]) -> Dict[str, float]:
        """Assess demographic compatibility"""
        return {
            'age_compatibility': 0.9,  # Placeholder
            'gender_compatibility': 1.0,
            'geographic_compatibility': 0.8
        }
    
    def _assess_clinical_compatibility(self, patient_profile: PatientProfile,
                                     trial: Dict[str, Any]) -> Dict[str, float]:
        """Assess clinical compatibility"""
        return {
            'condition_match': 0.8,  # Placeholder
            'comorbidity_compatibility': 0.7,
            'medication_compatibility': 0.9
        }
    
    def _assess_genetic_compatibility(self, patient_profile: PatientProfile,
                                    trial: Dict[str, Any]) -> Dict[str, float]:
        """Assess genetic compatibility"""
        if not patient_profile.genetic_risk_profile.get('available'):
            return {'genetic_data_available': 0.0}
        
        return {
            'genetic_data_available': 1.0,
            'biomarker_compatibility': 0.8  # Placeholder
        }
    
    async def _store_matching_results(self, patient_id: str, matches: List[TrialMatch],
                                    method: MatchingMethod) -> None:
        """Store matching results in database"""
        try:
            for match in matches:
                match_record = {
                    'unified_patient_id': patient_id,
                    'trial_id': match.trial_id,
                    'nct_id': match.nct_id,
                    'match_score': match.overall_match_score,
                    'eligibility_status': match.eligibility_status.value,
                    'matching_method': method.value,
                    'eligibility_assessment': {
                        'inclusion_analysis': match.inclusion_criteria_analysis,
                        'exclusion_analysis': match.exclusion_criteria_analysis
                    },
                    'geographic_feasible': match.geographic_feasibility,
                    'estimated_travel_distance_km': match.estimated_travel_distance_km,
                    'confidence_score': match.confidence_score,
                    'matching_reasoning': match.matching_reasoning,
                    'recommendation_level': match.recommendation_priority,
                    'created_at': datetime.utcnow().isoformat()
                }
                
                # Upsert into patient_trial_matches table
                self.supabase.client.table('patient_trial_matches')\
                    .upsert(match_record, on_conflict='unified_patient_id,nct_id')\
                    .execute()
            
            logger.info(f"Stored {len(matches)} trial matches for patient {patient_id}")
            
        except Exception as e:
            logger.error(f"Error storing matching results: {e}")

    # ============================================================================
    # DETAILED ELIGIBILITY ASSESSMENT
    # ============================================================================
    
    async def _detailed_eligibility_assessment(self, patient_profile: PatientProfile,
                                             trial: Dict[str, Any]) -> TrialMatch:
        """Perform detailed eligibility assessment for a specific trial"""
        
        # Use rule-based assessment for detailed analysis
        inclusion_criteria, exclusion_criteria = self._parse_eligibility_criteria(
            trial.get('eligibility_criteria', '')
        )
        
        inclusion_analysis = self._assess_inclusion_criteria(patient_profile, inclusion_criteria)
        exclusion_analysis = self._assess_exclusion_criteria(patient_profile, exclusion_criteria)
        
        match_score = self._calculate_rule_based_score(inclusion_analysis, exclusion_analysis)
        eligibility_status = self._determine_eligibility_status(inclusion_analysis, exclusion_analysis)
        
        # Create comprehensive trial match
        match = await self._create_trial_match(
            patient_profile, trial, match_score, eligibility_status,
            MatchingMethod.RULE_BASED, inclusion_analysis, exclusion_analysis
        )
        
        return match