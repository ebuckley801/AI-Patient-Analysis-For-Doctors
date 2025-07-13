#!/usr/bin/env python3
"""
Clinical Natural Language Processing Utilities
Advanced NLP functions for medical text analysis including negation detection,
abbreviation expansion, and temporal relationship extraction.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class NegationPattern:
    """Represents a negation pattern with context information"""
    pattern: str
    scope: int  # Number of words the negation affects
    strength: float  # Strength of negation (0.0-1.0)
    context_type: str  # Type of negation context


class ClinicalNLPProcessor:
    """Advanced NLP processor for clinical text analysis"""
    
    def __init__(self):
        self.negation_patterns = self._load_negation_patterns()
        self.medical_abbreviations = self._load_medical_abbreviations()
        self.temporal_patterns = self._load_temporal_patterns()
        self.uncertainty_patterns = self._load_uncertainty_patterns()
    
    def _load_negation_patterns(self) -> List[NegationPattern]:
        """Load comprehensive negation patterns for medical text"""
        patterns = [
            # Direct negations
            NegationPattern(r'\bno\b', 3, 1.0, 'direct'),
            NegationPattern(r'\bnot\b', 3, 1.0, 'direct'),
            NegationPattern(r'\bnone\b', 3, 1.0, 'direct'),
            NegationPattern(r'\bwithout\b', 4, 1.0, 'direct'),
            NegationPattern(r'\babsent\b', 3, 1.0, 'direct'),
            NegationPattern(r'\babsence of\b', 4, 1.0, 'direct'),
            NegationPattern(r'\bdenies\b', 5, 0.9, 'patient_reported'),
            NegationPattern(r'\bdenying\b', 5, 0.9, 'patient_reported'),
            NegationPattern(r'\bdenied\b', 5, 0.9, 'patient_reported'),
            
            # Medical negations
            NegationPattern(r'\bnegative for\b', 4, 1.0, 'test_result'),
            NegationPattern(r'\bruled out\b', 3, 0.8, 'diagnostic'),
            NegationPattern(r'\br/o\b', 3, 0.7, 'diagnostic'),  # rule out
            NegationPattern(r'\brule out\b', 3, 0.7, 'diagnostic'),
            NegationPattern(r'\bunlikely\b', 3, 0.6, 'uncertainty'),
            NegationPattern(r'\bdoubtful\b', 3, 0.6, 'uncertainty'),
            
            # Temporal negations
            NegationPattern(r'\bno longer\b', 4, 0.9, 'temporal'),
            NegationPattern(r'\bno more\b', 4, 0.9, 'temporal'),
            NegationPattern(r'\bpreviously\b', 2, 0.5, 'historical'),
            NegationPattern(r'\bformerly\b', 2, 0.5, 'historical'),
            
            # Conditional negations
            NegationPattern(r'\bif no\b', 4, 0.8, 'conditional'),
            NegationPattern(r'\bunless\b', 5, 0.7, 'conditional'),
            NegationPattern(r'\bexcept\b', 3, 0.7, 'conditional'),
            
            # Family/social history negations
            NegationPattern(r'\bfamily history negative\b', 6, 1.0, 'family_history'),
            NegationPattern(r'\bfh negative\b', 4, 1.0, 'family_history'),
            NegationPattern(r'\bno family history\b', 5, 1.0, 'family_history'),
            
            # Anatomical negations
            NegationPattern(r'\bbilateral absence\b', 4, 1.0, 'anatomical'),
            NegationPattern(r'\babsent bilaterally\b', 3, 1.0, 'anatomical'),
            NegationPattern(r'\bnot palpable\b', 3, 0.9, 'physical_exam'),
            NegationPattern(r'\bnot audible\b', 3, 0.9, 'physical_exam'),
            NegationPattern(r'\bnot visible\b', 3, 0.9, 'physical_exam'),
        ]
        return patterns
    
    def _load_medical_abbreviations(self) -> Dict[str, str]:
        """Load comprehensive medical abbreviation dictionary"""
        return {
            # Vital signs and measurements
            'bp': 'blood pressure',
            'hr': 'heart rate',
            'rr': 'respiratory rate',
            'temp': 'temperature',
            'o2 sat': 'oxygen saturation',
            'spo2': 'oxygen saturation',
            'bmi': 'body mass index',
            
            # Symptoms and conditions
            'sob': 'shortness of breath',
            'doe': 'dyspnea on exertion',
            'cp': 'chest pain',
            'abd': 'abdominal',
            'ha': 'headache',
            'n/v': 'nausea and vomiting',
            'nv': 'nausea and vomiting',
            'uti': 'urinary tract infection',
            'uri': 'upper respiratory infection',
            'mi': 'myocardial infarction',
            'cad': 'coronary artery disease',
            'chf': 'congestive heart failure',
            'copd': 'chronic obstructive pulmonary disease',
            'dm': 'diabetes mellitus',
            'htn': 'hypertension',
            'af': 'atrial fibrillation',
            'afib': 'atrial fibrillation',
            'dvt': 'deep vein thrombosis',
            'pe': 'pulmonary embolism',
            'tia': 'transient ischemic attack',
            'cva': 'cerebrovascular accident',
            'ckd': 'chronic kidney disease',
            'esrd': 'end stage renal disease',
            
            # Procedures and tests
            'ecg': 'electrocardiogram',
            'ekg': 'electrocardiogram',
            'echo': 'echocardiogram',
            'cxr': 'chest x-ray',
            'ct': 'computed tomography',
            'mri': 'magnetic resonance imaging',
            'us': 'ultrasound',
            'eeg': 'electroencephalogram',
            'emg': 'electromyogram',
            'cbc': 'complete blood count',
            'bmp': 'basic metabolic panel',
            'cmp': 'comprehensive metabolic panel',
            'pt': 'prothrombin time',
            'ptt': 'partial thromboplastin time',
            'inr': 'international normalized ratio',
            'bnp': 'brain natriuretic peptide',
            'troponin': 'troponin',
            'cpk': 'creatine phosphokinase',
            'lfts': 'liver function tests',
            'tsh': 'thyroid stimulating hormone',
            'hba1c': 'hemoglobin a1c',
            'psa': 'prostate specific antigen',
            
            # Medications
            'asa': 'aspirin',
            'nsaid': 'nonsteroidal anti-inflammatory drug',
            'ace': 'angiotensin converting enzyme',
            'arb': 'angiotensin receptor blocker',
            'bb': 'beta blocker',
            'ccb': 'calcium channel blocker',
            'ppi': 'proton pump inhibitor',
            'h2ra': 'histamine-2 receptor antagonist',
            'ssri': 'selective serotonin reuptake inhibitor',
            'snri': 'serotonin norepinephrine reuptake inhibitor',
            
            # Anatomical and directional
            'r': 'right',
            'l': 'left',
            'bil': 'bilateral',
            'ant': 'anterior',
            'post': 'posterior',
            'sup': 'superior',
            'inf': 'inferior',
            'med': 'medial',
            'lat': 'lateral',
            'prox': 'proximal',
            'dist': 'distal',
            
            # Frequencies and timing
            'bid': 'twice daily',
            'tid': 'three times daily',
            'qid': 'four times daily',
            'qd': 'once daily',
            'qod': 'every other day',
            'prn': 'as needed',
            'ac': 'before meals',
            'pc': 'after meals',
            'hs': 'at bedtime',
            'am': 'morning',
            'pm': 'evening',
            
            # Clinical assessment
            'hpi': 'history of present illness',
            'pmh': 'past medical history',
            'psh': 'past surgical history',
            'fh': 'family history',
            'sh': 'social history',
            'ros': 'review of systems',
            'pe': 'physical examination',
            'a&p': 'assessment and plan',
            'ddx': 'differential diagnosis',
            'r/o': 'rule out',
            'f/u': 'follow up',
            'rtn': 'return',
            'wbr': 'will be referred',
            
            # Units and values
            'mg': 'milligrams',
            'mcg': 'micrograms',
            'ml': 'milliliters',
            'cc': 'cubic centimeters',
            'kg': 'kilograms',
            'lbs': 'pounds',
            'cm': 'centimeters',
            'mm': 'millimeters',
            'bpm': 'beats per minute',
            'mmhg': 'millimeters of mercury',
            'wbc': 'white blood cells',
            'rbc': 'red blood cells',
            'hgb': 'hemoglobin',
            'hct': 'hematocrit',
            'plt': 'platelets',
            
            # Common phrases
            'wnl': 'within normal limits',
            'nab': 'no acute distress',
            'nad': 'no acute distress',
            'a&o': 'alert and oriented',
            'heent': 'head eyes ears nose throat',
            'cv': 'cardiovascular',
            'resp': 'respiratory',
            'gi': 'gastrointestinal',
            'gu': 'genitourinary',
            'msk': 'musculoskeletal',
            'neuro': 'neurological',
            'psych': 'psychiatric',
            'skin': 'integumentary',
        }
    
    def _load_temporal_patterns(self) -> List[Dict[str, Any]]:
        """Load temporal relationship patterns"""
        return [
            # Onset patterns
            {'pattern': r'\bstarted (\d+) (days?|weeks?|months?|years?) ago\b', 'type': 'onset', 'group': 1},
            {'pattern': r'\bbegan (\d+) (days?|weeks?|months?|years?) ago\b', 'type': 'onset', 'group': 1},
            {'pattern': r'\bfor the past (\d+) (days?|weeks?|months?|years?)\b', 'type': 'duration', 'group': 1},
            {'pattern': r'\bsince (\d+) (days?|weeks?|months?|years?) ago\b', 'type': 'onset', 'group': 1},
            {'pattern': r'\byesterday\b', 'type': 'onset', 'value': '1 day'},
            {'pattern': r'\btoday\b', 'type': 'onset', 'value': '0 days'},
            {'pattern': r'\bthis morning\b', 'type': 'onset', 'value': '0 days'},
            {'pattern': r'\blast night\b', 'type': 'onset', 'value': '1 day'},
            {'pattern': r'\bthis week\b', 'type': 'onset', 'value': '0-7 days'},
            {'pattern': r'\blast week\b', 'type': 'onset', 'value': '1 week'},
            
            # Frequency patterns
            {'pattern': r'\boccurs (\d+) times? per (day|week|month)\b', 'type': 'frequency', 'group': 1},
            {'pattern': r'\bintermittent\b', 'type': 'pattern', 'value': 'intermittent'},
            {'pattern': r'\bconstant\b', 'type': 'pattern', 'value': 'constant'},
            {'pattern': r'\bcontinuous\b', 'type': 'pattern', 'value': 'continuous'},
            {'pattern': r'\bepisodic\b', 'type': 'pattern', 'value': 'episodic'},
            
            # Progression patterns
            {'pattern': r'\bworsening\b', 'type': 'progression', 'value': 'worsening'},
            {'pattern': r'\bimproving\b', 'type': 'progression', 'value': 'improving'},
            {'pattern': r'\bstable\b', 'type': 'progression', 'value': 'stable'},
            {'pattern': r'\bgradually\b', 'type': 'progression', 'value': 'gradual'},
            {'pattern': r'\bsuddenly\b', 'type': 'progression', 'value': 'sudden'},
            {'pattern': r'\babruptly\b', 'type': 'progression', 'value': 'sudden'},
        ]
    
    def _load_uncertainty_patterns(self) -> List[Dict[str, Any]]:
        """Load uncertainty and speculation patterns"""
        return [
            {'pattern': r'\bpossible\b', 'confidence_modifier': -0.3, 'type': 'speculation'},
            {'pattern': r'\bprobable\b', 'confidence_modifier': -0.1, 'type': 'speculation'},
            {'pattern': r'\blikely\b', 'confidence_modifier': -0.1, 'type': 'speculation'},
            {'pattern': r'\bsuspected\b', 'confidence_modifier': -0.2, 'type': 'speculation'},
            {'pattern': r'\bsuggests\b', 'confidence_modifier': -0.2, 'type': 'speculation'},
            {'pattern': r'\bappears\b', 'confidence_modifier': -0.2, 'type': 'speculation'},
            {'pattern': r'\bmay be\b', 'confidence_modifier': -0.3, 'type': 'speculation'},
            {'pattern': r'\bcould be\b', 'confidence_modifier': -0.3, 'type': 'speculation'},
            {'pattern': r'\bmight be\b', 'confidence_modifier': -0.3, 'type': 'speculation'},
            {'pattern': r'\bconsistent with\b', 'confidence_modifier': -0.1, 'type': 'correlation'},
            {'pattern': r'\bquestionable\b', 'confidence_modifier': -0.4, 'type': 'uncertainty'},
            {'pattern': r'\bunclear\b', 'confidence_modifier': -0.4, 'type': 'uncertainty'},
        ]
    
    def expand_abbreviations(self, text: str) -> str:
        """Expand medical abbreviations in text while preserving context"""
        expanded_text = text.lower()
        
        # Sort by length (longest first) to avoid partial replacements
        sorted_abbrevs = sorted(self.medical_abbreviations.items(), 
                              key=lambda x: len(x[0]), reverse=True)
        
        for abbrev, expansion in sorted_abbrevs:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            expanded_text = re.sub(pattern, expansion, expanded_text, flags=re.IGNORECASE)
        
        return expanded_text
    
    def detect_negation(self, text: str, entity_position: Tuple[int, int]) -> Dict[str, Any]:
        """
        Detect if an entity is negated based on surrounding context
        
        Args:
            text: Full text containing the entity
            entity_position: (start, end) position of entity in text
            
        Returns:
            Dict containing negation information
        """
        entity_start, entity_end = entity_position
        
        # Look for negation patterns before the entity (within reasonable scope)
        pre_text = text[:entity_start].lower()
        post_text = text[entity_end:].lower()
        
        negation_info = {
            'is_negated': False,
            'negation_strength': 0.0,
            'negation_type': None,
            'negation_pattern': None,
            'confidence': 1.0
        }
        
        # Check for negation patterns before the entity
        for pattern in self.negation_patterns:
            # Look in the scope before the entity
            scope_text = pre_text[-pattern.scope * 10:]  # Approximate word scope
            
            match = re.search(pattern.pattern, scope_text, re.IGNORECASE)
            if match:
                # Calculate distance from negation to entity
                match_end = match.end()
                distance_to_entity = len(scope_text) - match_end
                
                # Negation strength decreases with distance
                distance_factor = max(0, 1 - (distance_to_entity / (pattern.scope * 10)))
                effective_strength = pattern.strength * distance_factor
                
                if effective_strength > negation_info['negation_strength']:
                    negation_info.update({
                        'is_negated': effective_strength > 0.5,
                        'negation_strength': effective_strength,
                        'negation_type': pattern.context_type,
                        'negation_pattern': pattern.pattern,
                        'confidence': effective_strength
                    })
        
        # Check for post-entity negations (less common but important)
        post_scope = post_text[:50]  # Look ahead 50 characters
        for pattern in self.negation_patterns:
            if pattern.context_type in ['test_result', 'diagnostic']:
                match = re.search(pattern.pattern, post_scope, re.IGNORECASE)
                if match:
                    negation_info.update({
                        'is_negated': True,
                        'negation_strength': pattern.strength,
                        'negation_type': pattern.context_type,
                        'negation_pattern': pattern.pattern,
                        'confidence': pattern.strength
                    })
                    break
        
        return negation_info
    
    def extract_temporal_info(self, text: str, entity_position: Tuple[int, int]) -> Dict[str, Any]:
        """Extract temporal information related to an entity"""
        entity_start, entity_end = entity_position
        
        # Look for temporal patterns in surrounding context
        context_start = max(0, entity_start - 100)
        context_end = min(len(text), entity_end + 100)
        context = text[context_start:context_end].lower()
        
        temporal_info = {
            'onset': None,
            'duration': None,
            'frequency': None,
            'pattern': None,
            'progression': None
        }
        
        for temp_pattern in self.temporal_patterns:
            match = re.search(temp_pattern['pattern'], context, re.IGNORECASE)
            if match:
                temp_type = temp_pattern['type']
                
                if 'group' in temp_pattern:
                    # Extract numeric value and unit
                    value = match.group(temp_pattern['group'])
                    unit = match.group(temp_pattern['group'] + 1)
                    temporal_info[temp_type] = f"{value} {unit}"
                elif 'value' in temp_pattern:
                    temporal_info[temp_type] = temp_pattern['value']
        
        return temporal_info
    
    def assess_uncertainty(self, text: str, entity_position: Tuple[int, int]) -> Dict[str, Any]:
        """Assess uncertainty and speculation around an entity"""
        entity_start, entity_end = entity_position
        
        # Look for uncertainty patterns in surrounding context
        context_start = max(0, entity_start - 50)
        context_end = min(len(text), entity_end + 50)
        context = text[context_start:context_end].lower()
        
        uncertainty_info = {
            'has_uncertainty': False,
            'uncertainty_type': None,
            'confidence_modifier': 0.0,
            'speculation_markers': []
        }
        
        for uncertainty_pattern in self.uncertainty_patterns:
            match = re.search(uncertainty_pattern['pattern'], context, re.IGNORECASE)
            if match:
                uncertainty_info['has_uncertainty'] = True
                uncertainty_info['uncertainty_type'] = uncertainty_pattern['type']
                uncertainty_info['confidence_modifier'] = min(
                    uncertainty_info['confidence_modifier'],
                    uncertainty_pattern['confidence_modifier']
                )
                uncertainty_info['speculation_markers'].append(match.group(0))
        
        return uncertainty_info
    
    def enhance_entity_with_nlp(self, entity: Dict[str, Any], text: str, 
                               entity_position: Tuple[int, int]) -> Dict[str, Any]:
        """
        Enhance an entity with comprehensive NLP analysis
        
        Args:
            entity: Original entity dict
            text: Full text containing the entity
            entity_position: (start, end) position of entity in text
            
        Returns:
            Enhanced entity with NLP information
        """
        enhanced_entity = entity.copy()
        
        # Add negation detection
        negation_info = self.detect_negation(text, entity_position)
        enhanced_entity['negation'] = negation_info
        
        # Add temporal information
        temporal_info = self.extract_temporal_info(text, entity_position)
        enhanced_entity['temporal'] = temporal_info
        
        # Add uncertainty assessment
        uncertainty_info = self.assess_uncertainty(text, entity_position)
        enhanced_entity['uncertainty'] = uncertainty_info
        
        # Adjust confidence based on negation and uncertainty
        original_confidence = enhanced_entity.get('confidence', 1.0)
        
        if negation_info['is_negated']:
            # If negated, flip the meaning but keep high confidence in the negation
            enhanced_entity['negated'] = True
            enhanced_entity['confidence'] = negation_info['confidence']
        else:
            # Apply uncertainty modifiers
            confidence_adjustment = uncertainty_info['confidence_modifier']
            enhanced_entity['confidence'] = max(0.1, original_confidence + confidence_adjustment)
        
        return enhanced_entity
    
    def preprocess_clinical_text(self, text: str) -> str:
        """
        Preprocess clinical text by expanding abbreviations and normalizing
        
        Args:
            text: Raw clinical text
            
        Returns:
            Preprocessed text ready for entity extraction
        """
        # Expand medical abbreviations
        expanded_text = self.expand_abbreviations(text)
        
        # Normalize common variations
        normalized_text = expanded_text
        
        # Normalize units and measurements
        normalized_text = re.sub(r'\b(\d+)mg\b', r'\1 milligrams', normalized_text, flags=re.IGNORECASE)
        normalized_text = re.sub(r'\b(\d+)ml\b', r'\1 milliliters', normalized_text, flags=re.IGNORECASE)
        normalized_text = re.sub(r'\b(\d+)bpm\b', r'\1 beats per minute', normalized_text, flags=re.IGNORECASE)
        
        # Normalize common medical phrases
        normalized_text = re.sub(r'\bc/o\b', 'complains of', normalized_text, flags=re.IGNORECASE)
        normalized_text = re.sub(r'\bs/p\b', 'status post', normalized_text, flags=re.IGNORECASE)
        normalized_text = re.sub(r'\bh/o\b', 'history of', normalized_text, flags=re.IGNORECASE)
        
        return normalized_text


# Factory function for easy import
def create_clinical_nlp_processor() -> ClinicalNLPProcessor:
    """Factory function to create ClinicalNLPProcessor instance"""
    return ClinicalNLPProcessor()