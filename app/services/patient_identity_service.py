"""
Unified Patient Identity Resolution Service

Advanced patient identity resolution across multiple healthcare datasets using:
- Deterministic matching (exact matches)
- Probabilistic matching (similarity-based)
- Machine learning features for identity resolution
- Privacy-preserving record linkage techniques

Integrates with existing intelligence layer architecture.
"""

import hashlib
import json
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from fuzzywuzzy import fuzz
import phonetics
import numpy as np
from app.services.supabase_service import SupabaseService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IdentityFeatures:
    """Features extracted for identity matching"""
    name_tokens: List[str]
    phonetic_codes: List[str]
    birth_date: Optional[date]
    gender: Optional[str]
    geographic_features: List[str]
    clinical_features: List[str]
    confidence_weights: Dict[str, float]

@dataclass
class MatchResult:
    """Result of identity matching"""
    unified_patient_id: str
    confidence_score: float
    matching_method: str
    matching_features: Dict[str, Any]
    conflicting_features: Dict[str, Any]

class PatientIdentityService:
    """Advanced patient identity resolution service"""
    
    def __init__(self):
        self.supabase = SupabaseService()
        
        # Matching thresholds
        self.EXACT_MATCH_THRESHOLD = 1.0
        self.HIGH_CONFIDENCE_THRESHOLD = 0.9
        self.MEDIUM_CONFIDENCE_THRESHOLD = 0.7
        self.LOW_CONFIDENCE_THRESHOLD = 0.5
        
        # Feature weights for matching
        self.FEATURE_WEIGHTS = {
            'name_exact': 0.25,
            'name_fuzzy': 0.20,
            'name_phonetic': 0.15,
            'birth_date': 0.20,
            'gender': 0.10,
            'geographic': 0.05,
            'clinical': 0.05
        }
    
    # ============================================================================
    # MAIN IDENTITY RESOLUTION METHODS
    # ============================================================================
    
    def resolve_patient_identity(self, demographics: Dict[str, Any], 
                               source_dataset: str,
                               source_patient_id: str) -> MatchResult:
        """
        Main entry point for patient identity resolution
        
        Args:
            demographics: Patient demographic information
            source_dataset: Source dataset identifier
            source_patient_id: Patient ID in source dataset
            
        Returns:
            MatchResult with unified patient ID and confidence
        """
        try:
            # Extract identity features
            features = self._extract_identity_features(demographics)
            
            # Try exact matching first
            exact_match = self._find_exact_match(features, source_dataset, source_patient_id)
            if exact_match:
                return exact_match
            
            # Try probabilistic matching
            probabilistic_match = self._find_probabilistic_match(features, source_dataset, source_patient_id)
            if probabilistic_match and probabilistic_match.confidence_score >= self.LOW_CONFIDENCE_THRESHOLD:
                return probabilistic_match
            
            # Create new unified patient if no match found
            return self._create_new_unified_patient(demographics, features, source_dataset, source_patient_id)
            
        except Exception as e:
            logger.error(f"Error resolving patient identity: {str(e)}")
            raise
    
    def validate_identity_match(self, unified_patient_id: str, 
                              new_demographics: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Validate if new demographics match existing unified patient
        
        Returns:
            (is_valid_match, confidence_score, conflict_analysis)
        """
        try:
            # Get existing patient data
            existing_patient = self.supabase.client.table('unified_patients')\
                .select('*')\
                .eq('unified_patient_id', unified_patient_id)\
                .execute()
            
            if not existing_patient.data:
                return False, 0.0, {'error': 'Patient not found'}
            
            existing_demographics = existing_patient.data[0]['demographics']
            
            # Extract features for both records
            existing_features = self._extract_identity_features(existing_demographics)
            new_features = self._extract_identity_features(new_demographics)
            
            # Calculate similarity
            similarity_score = self._calculate_feature_similarity(existing_features, new_features)
            
            # Analyze conflicts
            conflicts = self._analyze_demographic_conflicts(existing_demographics, new_demographics)
            
            is_valid = similarity_score >= self.MEDIUM_CONFIDENCE_THRESHOLD and len(conflicts) == 0
            
            return is_valid, similarity_score, conflicts
            
        except Exception as e:
            logger.error(f"Error validating identity match: {str(e)}")
            return False, 0.0, {'error': str(e)}
    
    def merge_patient_identities(self, primary_patient_id: str, 
                               secondary_patient_id: str,
                               merge_reason: str) -> bool:
        """
        Merge two unified patient identities
        
        Args:
            primary_patient_id: Patient ID to keep
            secondary_patient_id: Patient ID to merge into primary
            merge_reason: Reason for merging
            
        Returns:
            Success boolean
        """
        try:
            # Get both patient records
            primary = self.supabase.client.table('unified_patients')\
                .select('*')\
                .eq('unified_patient_id', primary_patient_id)\
                .execute()
            
            secondary = self.supabase.client.table('unified_patients')\
                .select('*')\
                .eq('unified_patient_id', secondary_patient_id)\
                .execute()
            
            if not primary.data or not secondary.data:
                logger.error("One or both patients not found for merging")
                return False
            
            # Merge demographics and data sources
            merged_demographics = self._merge_demographics(
                primary.data[0]['demographics'], 
                secondary.data[0]['demographics']
            )
            
            merged_sources = list(set(
                primary.data[0]['data_sources'] + secondary.data[0]['data_sources']
            ))
            
            # Update primary patient
            self.supabase.client.table('unified_patients')\
                .update({
                    'demographics': merged_demographics,
                    'data_sources': merged_sources,
                    'updated_at': datetime.now().isoformat()
                })\
                .eq('unified_patient_id', primary_patient_id)\
                .execute()
            
            # Update all identity mappings to point to primary patient
            self.supabase.client.table('patient_identity_mappings')\
                .update({'unified_patient_id': primary_patient_id})\
                .eq('unified_patient_id', secondary_patient_id)\
                .execute()
            
            # Delete secondary patient record
            self.supabase.client.table('unified_patients')\
                .delete()\
                .eq('unified_patient_id', secondary_patient_id)\
                .execute()
            
            logger.info(f"Successfully merged patient {secondary_patient_id} into {primary_patient_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error merging patient identities: {str(e)}")
            return False
    
    # ============================================================================
    # FEATURE EXTRACTION
    # ============================================================================
    
    def _extract_identity_features(self, demographics: Dict[str, Any]) -> IdentityFeatures:
        """Extract matching features from demographics"""
        
        # Name processing
        name_tokens = []
        phonetic_codes = []
        
        full_name = f"{demographics.get('first_name', '')} {demographics.get('last_name', '')}"
        if full_name.strip():
            name_tokens = self._tokenize_name(full_name)
            phonetic_codes = [phonetics.soundex(token) for token in name_tokens if token]
        
        # Birth date processing
        birth_date = None
        if demographics.get('birth_date'):
            if isinstance(demographics['birth_date'], str):
                try:
                    birth_date = datetime.strptime(demographics['birth_date'], '%Y-%m-%d').date()
                except ValueError:
                    pass
            elif isinstance(demographics['birth_date'], date):
                birth_date = demographics['birth_date']
        
        # Geographic features
        geographic_features = []
        for geo_field in ['city', 'state', 'country', 'zip_code']:
            if demographics.get(geo_field):
                geographic_features.append(str(demographics[geo_field]).lower())
        
        # Clinical features (if available)
        clinical_features = []
        for clinical_field in ['blood_type', 'allergies', 'medical_record_number']:
            if demographics.get(clinical_field):
                clinical_features.append(str(demographics[clinical_field]).lower())
        
        return IdentityFeatures(
            name_tokens=name_tokens,
            phonetic_codes=phonetic_codes,
            birth_date=birth_date,
            gender=demographics.get('gender', '').lower() if demographics.get('gender') else None,
            geographic_features=geographic_features,
            clinical_features=clinical_features,
            confidence_weights=self.FEATURE_WEIGHTS.copy()
        )
    
    def _tokenize_name(self, full_name: str) -> List[str]:
        """Tokenize and normalize name for matching"""
        # Remove common prefixes/suffixes
        prefixes = ['mr', 'mrs', 'ms', 'dr', 'prof']
        suffixes = ['jr', 'sr', 'ii', 'iii', 'iv']
        
        tokens = full_name.lower().replace('.', '').split()
        
        # Remove prefixes and suffixes
        filtered_tokens = []
        for token in tokens:
            if token not in prefixes and token not in suffixes and len(token) > 1:
                filtered_tokens.append(token)
        
        return filtered_tokens
    
    # ============================================================================
    # EXACT MATCHING
    # ============================================================================
    
    def _find_exact_match(self, features: IdentityFeatures, 
                         source_dataset: str, source_patient_id: str) -> Optional[MatchResult]:
        """Find exact demographic matches"""
        try:
            # Check if this source patient is already mapped
            existing_mapping = self.supabase.client.table('patient_identity_mappings')\
                .select('unified_patient_id')\
                .eq('source_dataset', source_dataset)\
                .eq('source_patient_id', source_patient_id)\
                .execute()
            
            if existing_mapping.data:
                unified_patient_id = existing_mapping.data[0]['unified_patient_id']
                return MatchResult(
                    unified_patient_id=unified_patient_id,
                    confidence_score=1.0,
                    matching_method='existing_mapping',
                    matching_features={'source_dataset': source_dataset, 'source_patient_id': source_patient_id},
                    conflicting_features={}
                )
            
            # Look for exact demographic matches
            all_patients = self.supabase.client.table('unified_patients')\
                .select('*')\
                .execute()
            
            for patient in all_patients.data:
                patient_features = self._extract_identity_features(patient['demographics'])
                
                if self._is_exact_match(features, patient_features):
                    return MatchResult(
                        unified_patient_id=patient['unified_patient_id'],
                        confidence_score=1.0,
                        matching_method='exact_demographic',
                        matching_features=self._get_matching_features(features, patient_features),
                        conflicting_features={}
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in exact matching: {str(e)}")
            return None
    
    def _is_exact_match(self, features1: IdentityFeatures, features2: IdentityFeatures) -> bool:
        """Check if two feature sets represent exact match"""
        
        # Name must match (at least one token)
        name_match = False
        for token1 in features1.name_tokens:
            for token2 in features2.name_tokens:
                if token1 == token2 and len(token1) > 2:  # Avoid matching short tokens
                    name_match = True
                    break
            if name_match:
                break
        
        if not name_match:
            return False
        
        # Birth date must match exactly (if available)
        if features1.birth_date and features2.birth_date:
            if features1.birth_date != features2.birth_date:
                return False
        
        # Gender must match (if available)
        if features1.gender and features2.gender:
            if features1.gender != features2.gender:
                return False
        
        return True
    
    # ============================================================================
    # PROBABILISTIC MATCHING
    # ============================================================================
    
    def _find_probabilistic_match(self, features: IdentityFeatures,
                                source_dataset: str, source_patient_id: str) -> Optional[MatchResult]:
        """Find probabilistic matches using similarity scoring"""
        try:
            all_patients = self.supabase.client.table('unified_patients')\
                .select('*')\
                .execute()
            
            best_match = None
            best_score = 0.0
            
            for patient in all_patients.data:
                patient_features = self._extract_identity_features(patient['demographics'])
                
                similarity_score = self._calculate_feature_similarity(features, patient_features)
                
                if similarity_score > best_score and similarity_score >= self.LOW_CONFIDENCE_THRESHOLD:
                    best_score = similarity_score
                    best_match = patient
            
            if best_match:
                patient_features = self._extract_identity_features(best_match['demographics'])
                return MatchResult(
                    unified_patient_id=best_match['unified_patient_id'],
                    confidence_score=best_score,
                    matching_method='probabilistic',
                    matching_features=self._get_matching_features(features, patient_features),
                    conflicting_features=self._get_conflicting_features(features, patient_features)
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in probabilistic matching: {str(e)}")
            return None
    
    def _calculate_feature_similarity(self, features1: IdentityFeatures, 
                                    features2: IdentityFeatures) -> float:
        """Calculate weighted similarity score between feature sets"""
        
        total_weight = 0.0
        weighted_score = 0.0
        
        # Name similarity (exact, fuzzy, phonetic)
        name_scores = self._calculate_name_similarity(features1, features2)
        for score_type, score in name_scores.items():
            weight = self.FEATURE_WEIGHTS.get(f'name_{score_type}', 0.0)
            weighted_score += score * weight
            total_weight += weight
        
        # Birth date similarity
        if features1.birth_date and features2.birth_date:
            birth_score = self._calculate_birth_date_similarity(features1.birth_date, features2.birth_date)
            weight = self.FEATURE_WEIGHTS['birth_date']
            weighted_score += birth_score * weight
            total_weight += weight
        
        # Gender similarity
        if features1.gender and features2.gender:
            gender_score = 1.0 if features1.gender == features2.gender else 0.0
            weight = self.FEATURE_WEIGHTS['gender']
            weighted_score += gender_score * weight
            total_weight += weight
        
        # Geographic similarity
        if features1.geographic_features and features2.geographic_features:
            geo_score = self._calculate_list_similarity(features1.geographic_features, features2.geographic_features)
            weight = self.FEATURE_WEIGHTS['geographic']
            weighted_score += geo_score * weight
            total_weight += weight
        
        # Clinical similarity
        if features1.clinical_features and features2.clinical_features:
            clinical_score = self._calculate_list_similarity(features1.clinical_features, features2.clinical_features)
            weight = self.FEATURE_WEIGHTS['clinical']
            weighted_score += clinical_score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_name_similarity(self, features1: IdentityFeatures, 
                                 features2: IdentityFeatures) -> Dict[str, float]:
        """Calculate various name similarity metrics"""
        
        scores = {'exact': 0.0, 'fuzzy': 0.0, 'phonetic': 0.0}
        
        if not features1.name_tokens or not features2.name_tokens:
            return scores
        
        # Exact token matching
        exact_matches = 0
        for token1 in features1.name_tokens:
            for token2 in features2.name_tokens:
                if token1 == token2:
                    exact_matches += 1
                    break
        
        scores['exact'] = exact_matches / max(len(features1.name_tokens), len(features2.name_tokens))
        
        # Fuzzy string matching
        name1 = ' '.join(features1.name_tokens)
        name2 = ' '.join(features2.name_tokens)
        scores['fuzzy'] = fuzz.token_sort_ratio(name1, name2) / 100.0
        
        # Phonetic matching
        if features1.phonetic_codes and features2.phonetic_codes:
            phonetic_matches = 0
            for code1 in features1.phonetic_codes:
                for code2 in features2.phonetic_codes:
                    if code1 == code2:
                        phonetic_matches += 1
                        break
            
            scores['phonetic'] = phonetic_matches / max(len(features1.phonetic_codes), len(features2.phonetic_codes))
        
        return scores
    
    def _calculate_birth_date_similarity(self, date1: date, date2: date) -> float:
        """Calculate birth date similarity with tolerance for data entry errors"""
        
        if date1 == date2:
            return 1.0
        
        # Calculate day difference
        day_diff = abs((date1 - date2).days)
        
        # Same year, month - might be day typo
        if date1.year == date2.year and date1.month == date2.month:
            if day_diff <= 2:  # 1-2 day difference
                return 0.9
            elif day_diff <= 7:  # Within a week
                return 0.7
        
        # Same year - might be month/day swap
        if date1.year == date2.year:
            # Check for month/day swap
            try:
                swapped_date = date(date1.year, date1.day, date1.month)
                if swapped_date == date2:
                    return 0.8
            except ValueError:
                pass
        
        # Year difference of 1 - might be year typo
        if abs(date1.year - date2.year) == 1 and date1.month == date2.month and date1.day == date2.day:
            return 0.6
        
        return 0.0
    
    def _calculate_list_similarity(self, list1: List[str], list2: List[str]) -> float:
        """Calculate similarity between two lists of strings"""
        if not list1 or not list2:
            return 0.0
        
        matches = 0
        for item1 in list1:
            for item2 in list2:
                if fuzz.ratio(item1, item2) > 80:  # 80% similarity threshold
                    matches += 1
                    break
        
        return matches / max(len(list1), len(list2))
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def _create_new_unified_patient(self, demographics: Dict[str, Any],
                                  features: IdentityFeatures, source_dataset: str,
                                  source_patient_id: str) -> MatchResult:
        """Create new unified patient identity"""
        
        # Generate master record ID
        demographics_str = json.dumps(demographics, sort_keys=True)
        master_record_id = f"UPI_{hashlib.sha256(demographics_str.encode()).hexdigest()[:16]}"
        
        # Create unified patient
        new_patient = {
            'master_record_id': master_record_id,
            'demographics': demographics,
            'data_sources': [source_dataset],
            'identity_confidence': 1.0
        }
        
        result = self.supabase.client.table('unified_patients')\
            .insert(new_patient)\
            .execute()
        
        unified_patient_id = result.data[0]['unified_patient_id']
        
        # Create identity mapping
        mapping = {
            'unified_patient_id': unified_patient_id,
            'source_dataset': source_dataset,
            'source_patient_id': source_patient_id,
            'confidence_score': 1.0,
            'matching_method': 'new_patient',
            'verified': True
        }
        
        self.supabase.client.table('patient_identity_mappings')\
            .insert(mapping)\
            .execute()
        
        return MatchResult(
            unified_patient_id=unified_patient_id,
            confidence_score=1.0,
            matching_method='new_patient',
            matching_features={'demographics': demographics},
            conflicting_features={}
        )
    
    def _get_matching_features(self, features1: IdentityFeatures, 
                             features2: IdentityFeatures) -> Dict[str, Any]:
        """Get features that match between two feature sets"""
        matching = {}
        
        # Matching name tokens
        matching_tokens = []
        for token1 in features1.name_tokens:
            for token2 in features2.name_tokens:
                if token1 == token2:
                    matching_tokens.append(token1)
                    break
        
        if matching_tokens:
            matching['name_tokens'] = matching_tokens
        
        # Matching demographic fields
        if features1.birth_date and features2.birth_date and features1.birth_date == features2.birth_date:
            matching['birth_date'] = features1.birth_date.isoformat()
        
        if features1.gender and features2.gender and features1.gender == features2.gender:
            matching['gender'] = features1.gender
        
        return matching
    
    def _get_conflicting_features(self, features1: IdentityFeatures,
                                features2: IdentityFeatures) -> Dict[str, Any]:
        """Get features that conflict between two feature sets"""
        conflicts = {}
        
        # Birth date conflicts
        if features1.birth_date and features2.birth_date and features1.birth_date != features2.birth_date:
            conflicts['birth_date'] = {
                'value1': features1.birth_date.isoformat(),
                'value2': features2.birth_date.isoformat()
            }
        
        # Gender conflicts
        if features1.gender and features2.gender and features1.gender != features2.gender:
            conflicts['gender'] = {
                'value1': features1.gender,
                'value2': features2.gender
            }
        
        return conflicts
    
    def _analyze_demographic_conflicts(self, demo1: Dict[str, Any], 
                                     demo2: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze conflicts between demographic records"""
        conflicts = {}
        
        # Check for hard conflicts in key fields
        conflict_fields = ['birth_date', 'gender', 'social_security_number']
        
        for field in conflict_fields:
            val1 = demo1.get(field)
            val2 = demo2.get(field)
            
            if val1 and val2 and val1 != val2:
                conflicts[field] = {'existing': val1, 'new': val2}
        
        return conflicts
    
    def _merge_demographics(self, primary: Dict[str, Any], 
                          secondary: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two demographic records, prioritizing primary"""
        merged = primary.copy()
        
        # Add non-conflicting fields from secondary
        for key, value in secondary.items():
            if key not in merged or not merged[key]:
                merged[key] = value
        
        return merged