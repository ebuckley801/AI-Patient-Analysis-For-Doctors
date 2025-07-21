#!/usr/bin/env python3
"""
Enhanced Clinical Analysis Service
Integrates Faiss high-performance vector search with advanced NLP processing
for optimal clinical entity extraction and ICD-10 mapping
"""

import logging
import time
import numpy as np
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import anthropic

from app.config.config import Config
from app.utils.clinical_nlp import create_clinical_nlp_processor
from app.services.icd10_vector_matcher import ICD10VectorMatcher
from app.services.claude_icd_matcher import create_claude_icd_matcher

logger = logging.getLogger(__name__)


class EnhancedClinicalAnalysisService:
    """
    Enhanced clinical analysis service that combines:
    - Advanced NLP processing (negation, abbreviation expansion, temporal extraction)
    - High-performance Faiss vector search for ICD-10 mapping
    - Optimized clinical entity extraction workflow
    """
    
    def __init__(self, force_numpy_icd: bool = False):
        """
        Initialize enhanced clinical analysis service
        
        Args:
            force_numpy_icd: Force numpy implementation for ICD matching (for testing)
        """
        self.client = anthropic.Anthropic(api_key=Config.ANTHROPIC_KEY)
        self.nlp_processor = create_clinical_nlp_processor()
        self.icd_matcher = ICD10VectorMatcher(force_numpy=force_numpy_icd)
        
        # Initialize Claude ICD matcher for high-accuracy suggestions
        self.claude_icd_matcher = create_claude_icd_matcher()
        
        # Performance tracking
        self.analysis_stats = {
            'total_analyses': 0,
            'faiss_searches': 0,
            'numpy_searches': 0,
            'avg_analysis_time_ms': 0,
            'avg_icd_search_time_ms': 0
        }
        
        # Simple embedding cache to avoid duplicate API calls
        self.embedding_cache = {}
        
        # Medical synonyms for better ICD matching (class-level for performance)
        self.medical_synonyms = {
            'ARDS': ['Acute respiratory distress syndrome', 'Adult respiratory distress syndrome'],
            'COVID-19': ['COVID 19', 'SARS-CoV-2 infection', 'Coronavirus disease', 'novel coronavirus'],
            'DYSPNEA': ['shortness of breath', 'difficulty breathing', 'breathing difficulty'],
            'FEVER': ['pyrexia', 'elevated temperature', 'hyperthermia'],
            'COUGH': ['coughing', 'tussis'],
            'PNEUMONIA': ['lung infection', 'pulmonary infection'],
            'SEPSIS': ['septicemia', 'blood poisoning'],
            'HYPERTENSION': ['high blood pressure', 'elevated blood pressure'],
            'DIABETES': ['diabetes mellitus', 'DM'],
            'MI': ['myocardial infarction', 'heart attack'],
            'CHF': ['congestive heart failure', 'heart failure'],
            'COPD': ['chronic obstructive pulmonary disease', 'chronic airway obstruction'],
            'DVT': ['deep vein thrombosis', 'venous thrombosis'],
            'PE': ['pulmonary embolism', 'lung embolism'],
            'UTI': ['urinary tract infection', 'bladder infection']
        }
        
        logger.info(f"✅ Enhanced Clinical Analysis Service initialized")
        logger.info(f"📊 NLP Processor: {'✅ Active' if self.nlp_processor else '❌ Failed'}")
        logger.info(f"🔍 ICD Search Method: {'Faiss' if self.icd_matcher.use_faiss else 'Numpy'}")
    
    def extract_clinical_entities_enhanced(self, 
                                         patient_note: str, 
                                         patient_context: Optional[Dict] = None,
                                         include_icd_mapping: bool = True,
                                         icd_top_k: int = 5,
                                         enable_nlp_preprocessing: bool = True) -> Dict[str, Any]:
        """
        Enhanced clinical entity extraction with integrated NLP and Faiss search
        
        Args:
            patient_note: Raw patient note text
            patient_context: Patient demographics and history
            include_icd_mapping: Whether to include ICD-10 mapping
            icd_top_k: Number of top ICD matches to return
            enable_nlp_preprocessing: Whether to apply NLP preprocessing
            
        Returns:
            Comprehensive analysis result with entities, ICD mappings, and performance metrics
        """
        start_time = time.time()
        
        try:
            # Phase 1: NLP Preprocessing
            preprocessing_start = time.time()
            
            if enable_nlp_preprocessing:
                preprocessed_note = self.nlp_processor.preprocess_clinical_text(patient_note)
                logger.debug(f"📝 Preprocessed note: {len(preprocessed_note)} chars")
            else:
                preprocessed_note = patient_note
            
            preprocessing_time = (time.time() - preprocessing_start) * 1000
            
            # Phase 2: Clinical Entity Extraction
            extraction_start = time.time()
            
            prompt = self._build_enhanced_extraction_prompt(
                preprocessed_note, 
                patient_context,
                enable_nlp_preprocessing
            )
            
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2500,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result = self._parse_claude_response(response.content[0].text)
            extraction_time = (time.time() - extraction_start) * 1000
            
            # Debug: Log what Claude actually returned
            logger.info(f"Raw Claude response: {response.content[0].text}")
            logger.info(f"Parsed result before NLP enhancement: {json.dumps(result, indent=2)}")
            
            # Phase 3: NLP Enhancement
            nlp_enhancement_start = time.time()
            
            if enable_nlp_preprocessing:
                result = self._enhance_entities_with_nlp(result, patient_note, preprocessed_note)
            
            nlp_enhancement_time = (time.time() - nlp_enhancement_start) * 1000
            
            # Phase 4: High-Performance ICD-10 Mapping
            icd_mapping_start = time.time()
            icd_mapping_results = []
            
            if include_icd_mapping:
                icd_mapping_results = self._perform_enhanced_icd_mapping(
                    result, 
                    top_k=icd_top_k,
                    use_nlp_context=enable_nlp_preprocessing
                )
            
            icd_mapping_time = (time.time() - icd_mapping_start) * 1000
            
            # Phase 5: Result Assembly
            total_time = (time.time() - start_time) * 1000
            
            # Add performance metrics and metadata
            result.update({
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'model_version': "claude-3-5-sonnet-20241022",
                'nlp_enhanced': enable_nlp_preprocessing,
                'icd_search_method': 'faiss' if self.icd_matcher.use_faiss else 'numpy',
                'performance_metrics': {
                    'total_time_ms': round(total_time, 2),
                    'preprocessing_time_ms': round(preprocessing_time, 2),
                    'extraction_time_ms': round(extraction_time, 2),
                    'nlp_enhancement_time_ms': round(nlp_enhancement_time, 2),
                    'icd_mapping_time_ms': round(icd_mapping_time, 2),
                    'chars_processed': len(patient_note),
                    'chars_preprocessed': len(preprocessed_note) if enable_nlp_preprocessing else len(patient_note)
                }
            })
            
            # Only add ICD mappings if requested
            if include_icd_mapping:
                result['icd_mappings'] = icd_mapping_results
            
            # Update global stats
            self._update_performance_stats(total_time, icd_mapping_time)
            
            logger.info(f"✅ Enhanced analysis completed in {total_time:.1f}ms")
            return result
            
        except Exception as e:
            logger.error(f"❌ Enhanced clinical analysis failed: {e}")
            return self._create_error_result(str(e), time.time() - start_time)
    
    def _build_enhanced_extraction_prompt(self, 
                                        patient_note: str, 
                                        patient_context: Optional[Dict] = None,
                                        nlp_enabled: bool = True) -> str:
        """Build enhanced extraction prompt with NLP awareness"""
        
        context_info = ""
        if patient_context:
            context_info = f"""
Patient Context:
- Age: {patient_context.get('age', 'Not specified')}
- Gender: {patient_context.get('gender', 'Not specified')}
- Medical History: {patient_context.get('medical_history', 'Not specified')}
"""
        
        nlp_instructions = ""
        if nlp_enabled:
            nlp_instructions = """
🧠 NLP Processing Applied:
- Medical abbreviations have been expanded
- Text has been normalized for better analysis
- Pay special attention to negation markers and temporal indicators
- Use the expanded forms for more accurate entity extraction

Enhanced Analysis Instructions:
- Detect negated entities (pay attention to "denies", "negative for", "rule out", etc.)
- Identify temporal relationships (onset, duration, progression)
- Assess uncertainty levels (possible, likely, suspected)
- Consider medical context and relationships between entities
"""
        
        prompt = f"""You are an advanced clinical AI assistant with enhanced NLP capabilities. Extract structured medical information from the patient note below.

{context_info}{nlp_instructions}

Patient Note (Enhanced):
{patient_note}

Extract the following clinical entities with high precision:

SYMPTOMS (with negation and temporal context):
- Entity name, severity, temporal information, negation status, confidence

CONDITIONS/DIAGNOSES (with certainty assessment):
- Entity name, status (active/resolved/suspected), confidence, ICD category hint

MEDICATIONS (with dosage and timing):
- Medication name, dosage, frequency, route, confidence

VITAL SIGNS (with abnormality detection):
- Parameter name, value, unit, abnormal flag, confidence

PROCEDURES (with timing and results):
- Procedure name, date/timing, results, confidence

ABNORMAL FINDINGS:
- Finding description, severity, requires attention flag, confidence

OVERALL ASSESSMENT:
- Primary concerns list
- Risk level (low/moderate/high/critical)
- Requires immediate attention (boolean)
- Clinical summary (1-2 sentences)

Return ONLY valid JSON in this exact format:
{{
    "symptoms": [
        {{
            "entity": "symptom name",
            "severity": "mild/moderate/severe",
            "temporal": "acute/chronic/intermittent",
            "confidence": 0.0-1.0,
            "text_span": "original text",
            "negated": false,
            "onset": "timing if mentioned",
            "progression": "stable/improving/worsening"
        }}
    ],
    "conditions": [
        {{
            "entity": "condition name",
            "status": "active/resolved/suspected",
            "confidence": 0.0-1.0,
            "text_span": "original text",
            "icd_category": "system affected",
            "certainty": "confirmed/suspected/rule_out"
        }}
    ],
    "medications": [
        {{
            "entity": "medication name",
            "dosage": "amount and unit",
            "frequency": "dosing schedule",
            "route": "administration route",
            "confidence": 0.0-1.0,
            "text_span": "original text"
        }}
    ],
    "vital_signs": [
        {{
            "entity": "vital sign name",
            "value": "measurement value",
            "unit": "measurement unit",
            "abnormal": true/false,
            "confidence": 0.0-1.0,
            "text_span": "original text"
        }}
    ],
    "procedures": [
        {{
            "entity": "procedure name",
            "timing": "when performed",
            "results": "procedure results",
            "confidence": 0.0-1.0,
            "text_span": "original text"
        }}
    ],
    "abnormal_findings": [
        {{
            "entity": "finding description",
            "severity": "mild/moderate/severe",
            "requires_attention": true/false,
            "confidence": 0.0-1.0,
            "text_span": "original text"
        }}
    ],
    "overall_assessment": {{
        "primary_concerns": ["concern1", "concern2"],
        "risk_level": "low/moderate/high/critical",
        "requires_immediate_attention": true/false,
        "summary": "brief clinical summary"
    }}
}}"""
        
        return prompt
    
    def _enhance_entities_with_nlp(self, 
                                 result: Dict[str, Any], 
                                 original_text: str, 
                                 preprocessed_text: str) -> Dict[str, Any]:
        """
        Enhance extracted entities with NLP analysis
        
        Args:
            result: Initial extraction result
            original_text: Original patient note
            preprocessed_text: NLP-preprocessed text
            
        Returns:
            Enhanced result with NLP metadata
        """
        try:
            enhanced_result = result.copy()
            
            # Track NLP enhancements
            nlp_analysis = {
                'negation_detection_applied': False,
                'temporal_extraction_applied': False,
                'uncertainty_assessment_applied': False,
                'abbreviations_expanded': len(original_text) != len(preprocessed_text)
            }
            
            # Enhance each entity type
            for entity_type in ['symptoms', 'conditions', 'medications', 'vital_signs', 'procedures', 'abnormal_findings']:
                if entity_type in result:
                    enhanced_entities = []
                    
                    for entity in result[entity_type]:
                        enhanced_entity = self._enhance_single_entity(
                            entity, original_text, preprocessed_text
                        )
                        enhanced_entities.append(enhanced_entity)
                        
                        # Track what NLP features were applied
                        if 'negation' in enhanced_entity:
                            nlp_analysis['negation_detection_applied'] = True
                        if 'temporal' in enhanced_entity:
                            nlp_analysis['temporal_extraction_applied'] = True
                        if 'uncertainty' in enhanced_entity:
                            nlp_analysis['uncertainty_assessment_applied'] = True
                    
                    enhanced_result[entity_type] = enhanced_entities
            
            enhanced_result['nlp_analysis'] = nlp_analysis
            return enhanced_result
            
        except Exception as e:
            logger.error(f"❌ NLP enhancement failed: {e}")
            return result
    
    def _enhance_single_entity(self, 
                             entity: Dict[str, Any], 
                             original_text: str, 
                             preprocessed_text: str) -> Dict[str, Any]:
        """Enhance a single entity with NLP analysis"""
        
        enhanced = entity.copy()
        
        # Find entity position in text
        entity_text = entity.get('entity', '')
        text_span = entity.get('text_span', entity_text)
        
        # Find position for NLP analysis
        entity_pos = self._find_entity_position(original_text, text_span, entity_text)
        
        if entity_pos:
            # Apply negation detection
            negation_result = self.nlp_processor.detect_negation(original_text, entity_pos)
            enhanced['negation'] = negation_result
            
            # Apply temporal extraction
            temporal_result = self.nlp_processor.extract_temporal_info(original_text, entity_pos)
            enhanced['temporal'] = temporal_result
            
            # Apply uncertainty assessment
            uncertainty_result = self.nlp_processor.assess_uncertainty(original_text, entity_pos)
            enhanced['uncertainty'] = uncertainty_result
            
            # Adjust confidence based on NLP findings
            original_confidence = entity.get('confidence', 1.0)
            
            # Reduce confidence for negated entities
            if negation_result.get('is_negated'):
                enhanced['negated'] = True
                enhanced['confidence'] = min(original_confidence, 0.95)  # High confidence in negation
            
            # Adjust confidence for uncertain entities
            if uncertainty_result.get('has_uncertainty'):
                confidence_modifier = uncertainty_result.get('confidence_modifier', 0)
                enhanced['confidence'] = max(0.1, original_confidence + confidence_modifier)
        
        # Always standardize entity structure, regardless of whether we found position
        enhanced = self._standardize_entity_structure(enhanced, original_text, entity_pos)
        
        return enhanced
    
    def _find_entity_position(self, text: str, text_span: str, entity_text: str) -> Optional[Tuple[int, int]]:
        """Find entity position in text for NLP analysis"""
        try:
            # Try exact text span match first
            if text_span and text_span in text:
                start = text.lower().find(text_span.lower())
                return (start, start + len(text_span))
            
            # Fallback to entity text
            if entity_text and entity_text in text:
                start = text.lower().find(entity_text.lower())
                return (start, start + len(entity_text))
            
            # Try partial match for multi-word entities
            if entity_text and ' ' in entity_text:
                words = entity_text.split()
                for word in words:
                    if len(word) > 3 and word.lower() in text.lower():
                        start = text.lower().find(word.lower())
                        return (start, start + len(word))
            
            return None
            
        except Exception:
            return None
    
    def _perform_enhanced_icd_mapping(self, 
                                    analysis_result: Dict[str, Any], 
                                    top_k: int = 5,
                                    use_nlp_context: bool = True) -> List[Dict[str, Any]]:
        """
        Perform enhanced ICD-10 mapping using Faiss search with NLP context
        
        Args:
            analysis_result: Clinical analysis result
            top_k: Number of top ICD matches per entity
            use_nlp_context: Whether to use NLP context for better matching
            
        Returns:
            List of ICD mapping results with enhanced metadata
        """
        icd_mappings = []
        
        try:
            # Quick ICD matcher status check
            cache_info = self.icd_matcher.get_cache_info()
            logger.info(f"🔍 ICD Search: {cache_info.get('search_method', 'unknown')} ({cache_info.get('total_icd_codes', 0)} codes loaded)")
            
            # Map conditions (highest priority)
            for condition in analysis_result.get('conditions', []):
                # Defensive check: ensure condition is a dictionary
                if not isinstance(condition, dict):
                    logger.warning(f"⚠️ Skipping non-dict condition: {condition}")
                    continue
                    
                entity_text = condition.get('entity', '')
                if entity_text:
                    # Try multiple search strategies for better matching
                    search_start = time.time()
                    icd_matches = self._smart_icd_search(entity_text, condition, top_k, 'condition')
                    search_time = (time.time() - search_start) * 1000  # Convert to milliseconds
                    
                    # Safe logging with type checking
                    try:
                        match_info = []
                        for m in icd_matches[:3]:
                            if isinstance(m, dict):
                                code = m.get('code', 'No code')
                                similarity = m.get('similarity', 0)
                                match_info.append(f"{code} ({round(similarity*100)}%)")
                            else:
                                match_info.append(f"Invalid match type: {type(m)}")
                        logger.info(f"🔍 Found {len(icd_matches)} ICD matches for condition '{entity_text}': {match_info}")
                    except Exception as log_error:
                        logger.error(f"❌ Error logging condition matches: {log_error}")
                        logger.info(f"🔍 Found {len(icd_matches)} ICD matches for condition '{entity_text}' (details unavailable)")
                    
                    # Track search method
                    if self.icd_matcher.use_faiss:
                        self.analysis_stats['faiss_searches'] += 1
                    else:
                        self.analysis_stats['numpy_searches'] += 1
                    
                    if icd_matches:
                        mapping = {
                            'entity': entity_text,
                            'entity_type': 'condition',
                            'original_confidence': condition.get('confidence', 0),
                            'status': condition.get('status', 'unknown'),
                            'certainty': condition.get('certainty', 'unknown'),
                            'icd_matches': icd_matches,
                            'best_match': icd_matches[0] if icd_matches else None,
                            'search_method': 'faiss' if self.icd_matcher.use_faiss else 'numpy',
                            'search_time_ms': round(search_time, 2),
                            'enhanced_query': False  # Will be set by smart search if used
                        }
                        
                        # Add NLP context if available
                        if use_nlp_context and 'negation' in condition:
                            mapping['negated'] = condition['negation'].get('is_negated', False)
                        if use_nlp_context and 'uncertainty' in condition:
                            mapping['uncertain'] = condition['uncertainty'].get('has_uncertainty', False)
                        
                        icd_mappings.append(mapping)
            
            # Map symptoms (secondary priority)
            for symptom in analysis_result.get('symptoms', []):
                # Defensive check: ensure symptom is a dictionary
                if not isinstance(symptom, dict):
                    logger.warning(f"⚠️ Skipping non-dict symptom: {symptom}")
                    continue
                    
                entity_text = symptom.get('entity', '')
                if entity_text and not symptom.get('negated', False):  # Skip negated symptoms
                    # Try multiple search strategies for better matching
                    search_start = time.time()
                    icd_matches = self._smart_icd_search(entity_text, symptom, min(top_k, 3), 'symptom')
                    search_time = (time.time() - search_start) * 1000  # Convert to milliseconds
                    
                    # Safe logging with type checking
                    try:
                        match_info = []
                        for m in icd_matches[:3]:
                            if isinstance(m, dict):
                                code = m.get('code', 'No code')
                                similarity = m.get('similarity', 0)
                                match_info.append(f"{code} ({round(similarity*100)}%)")
                            else:
                                match_info.append(f"Invalid match type: {type(m)}")
                        logger.info(f"🔍 Found {len(icd_matches)} ICD matches for symptom '{entity_text}': {match_info}")
                    except Exception as log_error:
                        logger.error(f"❌ Error logging symptom matches: {log_error}")
                        logger.info(f"🔍 Found {len(icd_matches)} ICD matches for symptom '{entity_text}' (details unavailable)")
                    
                    if icd_matches:
                        mapping = {
                            'entity': entity_text,
                            'entity_type': 'symptom',
                            'original_confidence': symptom.get('confidence', 0),
                            'severity': symptom.get('severity', 'unknown'),
                            'icd_matches': icd_matches,
                            'best_match': icd_matches[0] if icd_matches else None,
                            'search_method': 'faiss' if self.icd_matcher.use_faiss else 'numpy',
                            'search_time_ms': round(search_time, 2),
                            'enhanced_query': False  # Will be set by smart search if used
                        }
                        icd_mappings.append(mapping)
            
            logger.info(f"🔍 ICD mapping completed: {len(icd_mappings)} mappings found")
            return icd_mappings
            
        except Exception as e:
            import traceback
            logger.error(f"❌ ICD mapping failed: {e}")
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return []
    
    def _prepare_entity_for_icd_search(self, entity: Dict[str, Any], use_nlp_context: bool) -> str:
        """
        Prepare entity text for optimal ICD search using NLP context
        
        Args:
            entity: Entity dictionary with NLP enhancements
            use_nlp_context: Whether to use NLP context
            
        Returns:
            Enhanced search query string
        """
        base_text = entity.get('entity', '')
        
        if not use_nlp_context:
            return base_text
        
        # Start with base entity text
        search_terms = [base_text]
        
        # Add severity context for conditions
        if entity.get('severity') and entity['severity'] != 'unknown':
            search_terms.append(f"{entity['severity']} {base_text}")
        
        # Add temporal context
        if 'temporal' in entity:
            temporal_info = entity['temporal']
            # Handle both string and dict temporal info
            if isinstance(temporal_info, dict):
                if temporal_info.get('onset'):
                    search_terms.append(f"{temporal_info['onset']} {base_text}")
                if temporal_info.get('progression') and temporal_info['progression'] != 'stable':
                    search_terms.append(f"{temporal_info['progression']} {base_text}")
            elif isinstance(temporal_info, str) and temporal_info not in ['unknown', 'unspecified']:
                # Use string temporal info directly
                search_terms.append(f"{temporal_info} {base_text}")
        
        # Add status context for conditions
        if entity.get('status') and entity['status'] in ['chronic', 'acute']:
            search_terms.append(f"{entity['status']} {base_text}")
        
        # Return the most descriptive term (usually the longest)
        return max(search_terms, key=len)
    
    def _generate_search_variants(self, entity_text: str, entity_data: Dict[str, Any]) -> List[str]:
        """
        Generate multiple search variants for parallel ICD search
        
        Args:
            entity_text: Original entity text
            entity_data: Full entity data with context
            
        Returns:
            List of search variant strings
        """
        variants = []
        
        # 1. Original text
        variants.append(entity_text)
        
        # 2. Medical synonyms and expansions
        medical_synonyms = {
            'chest pain': ['thoracic pain', 'thoracalgia', 'chest discomfort'],
            'shortness of breath': ['dyspnea', 'breathlessness', 'respiratory distress'],
            'heart attack': ['myocardial infarction', 'MI', 'cardiac arrest'],
            'diabetes': ['diabetes mellitus', 'DM', 'hyperglycemia'],
            'high blood pressure': ['hypertension', 'HTN', 'elevated blood pressure'],
            'fever': ['pyrexia', 'hyperthermia', 'elevated temperature'],
            'headache': ['cephalgia', 'head pain'],
            'stomach pain': ['abdominal pain', 'gastric pain', 'epigastric pain'],
            'back pain': ['dorsalgia', 'spinal pain', 'lumbar pain'],
            'pneumonia': ['lung infection', 'respiratory infection'],
            'stroke': ['cerebrovascular accident', 'CVA', 'brain attack']
        }
        
        entity_lower = entity_text.lower()
        for key, synonyms in medical_synonyms.items():
            if key in entity_lower:
                variants.extend(synonyms)
        
        # 3. Add severity context if available
        severity = entity_data.get('severity')
        if severity and severity not in ['unknown', 'unspecified']:
            variants.append(f"{severity} {entity_text}")
        
        # 4. Add temporal context if available
        temporal = entity_data.get('temporal')
        if temporal:
            if isinstance(temporal, str) and temporal not in ['unknown', 'unspecified']:
                variants.append(f"{temporal} {entity_text}")
            elif isinstance(temporal, dict):
                if temporal.get('onset'):
                    variants.append(f"{temporal['onset']} {entity_text}")
        
        # 5. Add anatomical context for pain/symptoms
        anatomical_expansions = {
            'pain': ['ache', 'discomfort', 'soreness'],
            'chest': ['thoracic', 'cardiac', 'pulmonary'],
            'stomach': ['abdominal', 'gastric', 'epigastric'],
            'head': ['cranial', 'cephalic']
        }
        
        for anatomy, expansions in anatomical_expansions.items():
            if anatomy in entity_lower:
                for expansion in expansions:
                    variants.append(entity_text.replace(anatomy, expansion))
        
        # 6. Remove duplicates and empty strings
        variants = list(set([v.strip() for v in variants if v.strip()]))
        
        # 7. Limit to reasonable number (avoid too many API calls)
        return variants[:6]
    
    def _merge_and_rank_search_results(self, variant_results: List[List[Dict[str, Any]]], variants: List[str]) -> List[Dict[str, Any]]:
        """
        Merge and rank results from multiple search variants
        
        Args:
            variant_results: List of result lists, one per variant
            variants: Original variant strings used for search
            
        Returns:
            Merged and ranked list of ICD matches
        """
        # Dictionary to collect results by ICD code
        merged_results = {}
        
        for i, (results, variant) in enumerate(zip(variant_results, variants)):
            for result in results:
                icd_code = result.get('icd_code') or result.get('code')
                if not icd_code:
                    continue
                
                if icd_code not in merged_results:
                    # First time seeing this ICD code
                    merged_results[icd_code] = {
                        'icd_code': icd_code,
                        'code': icd_code,  # Ensure both fields exist
                        'description': result.get('description', ''),
                        'similarities': [result.get('similarity', 0)],
                        'search_methods': [result.get('search_method', 'unknown')],
                        'variants_found': [variant],
                        'variant_count': 1,
                        'best_similarity': result.get('similarity', 0),
                        'avg_similarity': result.get('similarity', 0),
                        'confidence_boost': 0
                    }
                else:
                    # ICD code found via multiple variants - boost confidence
                    existing = merged_results[icd_code]
                    existing['similarities'].append(result.get('similarity', 0))
                    existing['search_methods'].append(result.get('search_method', 'unknown'))
                    existing['variants_found'].append(variant)
                    existing['variant_count'] += 1
                    existing['best_similarity'] = max(existing['best_similarity'], result.get('similarity', 0))
                    existing['avg_similarity'] = sum(existing['similarities']) / len(existing['similarities'])
                    
                    # Boost confidence for codes found via multiple variants
                    existing['confidence_boost'] = min(0.2, existing['variant_count'] * 0.05)
        
        # Convert to list and calculate final similarity scores
        final_results = []
        for icd_code, data in merged_results.items():
            # Final similarity combines best similarity + confidence boost
            final_similarity = min(1.0, data['best_similarity'] + data['confidence_boost'])
            
            final_results.append({
                'icd_code': data['icd_code'],
                'code': data['code'],
                'description': data['description'],
                'similarity': final_similarity,
                'best_similarity': data['best_similarity'],
                'avg_similarity': data['avg_similarity'],
                'variant_count': data['variant_count'],
                'variants_found': data['variants_found'],
                'search_method': 'parallel_batch',
                'confidence_boost': data['confidence_boost']
            })
        
        # Sort by final similarity (best + boost)
        final_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return final_results
    
    def _parse_claude_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Claude's JSON response with error handling"""
        try:
            # Clean up response text
            cleaned_text = response_text.strip()
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3]
            
            return json.loads(cleaned_text)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return self._empty_extraction_result(error=f"JSON parsing failed: {e}")
        except Exception as e:
            logger.error(f"Response parsing error: {e}")
            return self._empty_extraction_result(error=f"Response parsing failed: {e}")
    
    def _empty_extraction_result(self, error: Optional[str] = None) -> Dict[str, Any]:
        """Create empty extraction result structure"""
        result = {
            'symptoms': [],
            'conditions': [],
            'medications': [],
            'vital_signs': [],
            'procedures': [],
            'abnormal_findings': [],
            'overall_assessment': {
                'primary_concerns': [],
                'risk_level': 'low',
                'requires_immediate_attention': False,
                'summary': 'Analysis incomplete'
            }
        }
        
        if error:
            result['error'] = error
            
        return result
    
    def _create_error_result(self, error_message: str, elapsed_time: float) -> Dict[str, Any]:
        """Create error result with metadata"""
        result = self._empty_extraction_result(error=error_message)
        result.update({
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'nlp_enhanced': False,
            'performance_metrics': {
                'total_time_ms': round(elapsed_time * 1000, 2),
                'error': True
            }
        })
        return result
    
    def _update_performance_stats(self, total_time_ms: float, icd_time_ms: float):
        """Update global performance statistics"""
        self.analysis_stats['total_analyses'] += 1
        
        # Update running averages
        prev_avg_analysis = self.analysis_stats['avg_analysis_time_ms']
        prev_avg_icd = self.analysis_stats['avg_icd_search_time_ms']
        n = self.analysis_stats['total_analyses']
        
        self.analysis_stats['avg_analysis_time_ms'] = (
            (prev_avg_analysis * (n - 1) + total_time_ms) / n
        )
        self.analysis_stats['avg_icd_search_time_ms'] = (
            (prev_avg_icd * (n - 1) + icd_time_ms) / n
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            **self.analysis_stats,
            'nlp_processor_active': self.nlp_processor is not None,
            'icd_search_method': 'faiss' if self.icd_matcher.use_faiss else 'numpy',
            'icd_cache_info': self.icd_matcher.get_cache_info()
        }
    
    def benchmark_enhanced_analysis(self, num_tests: int = 10) -> Dict[str, Any]:
        """
        Benchmark the enhanced analysis pipeline
        
        Args:
            num_tests: Number of test analyses to run
            
        Returns:
            Comprehensive benchmark results
        """
        test_notes = [
            "Patient presents with severe chest pain and shortness of breath. No fever.",
            "45-year-old male with DM, HTN who c/o SOB x 3 days. BP 160/90, HR 110.",
            "Denies chest pain but reports possible pneumonia symptoms. CXR negative.",
            "Patient has chronic kidney disease, stage 3. Cr 2.1, BUN 45. No acute distress.",
            "Acute MI ruled out. Troponin negative x3. EKG shows old RBBB.",
        ]
        
        total_start = time.time()
        individual_times = []
        
        for i in range(num_tests):
            note = test_notes[i % len(test_notes)]
            context = {'age': 50 + (i * 5) % 50, 'gender': 'male' if i % 2 else 'female'}
            
            start = time.time()
            result = self.extract_clinical_entities_enhanced(
                note, context, include_icd_mapping=True
            )
            duration = time.time() - start
            individual_times.append(duration * 1000)
            
            if 'error' in result:
                logger.warning(f"Test {i+1} failed: {result['error']}")
        
        total_time = time.time() - total_start
        
        return {
            'num_tests': num_tests,
            'total_time_seconds': total_time,
            'avg_time_per_analysis_ms': sum(individual_times) / len(individual_times),
            'min_time_ms': min(individual_times),
            'max_time_ms': max(individual_times),
            'analyses_per_second': num_tests / total_time,
            'individual_times_ms': individual_times,
            'performance_stats': self.get_performance_stats()
        }
    
    def _standardize_entity_structure(self, entity: Dict[str, Any], original_text: str, 
                                      entity_position: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Standardize entity structure to ensure consistent field mapping
        
        Args:
            entity: Original entity from Claude response
            original_text: Full text for position lookup
            entity_position: Optional position tuple (start, end)
            
        Returns:
            Standardized entity with consistent field names
        """
        standardized = entity.copy()
        
        # Ensure we have a 'text' field with the actual entity text
        entity_text = entity.get('entity', '') or entity.get('text_span', '') or entity.get('text', '')
        
        if entity_text:
            standardized['text'] = entity_text
            
            # If we have position info, use it; otherwise try to find it
            if entity_position:
                start, end = entity_position
                standardized['start'] = start
                standardized['end'] = end
                # Verify the text matches the position
                try:
                    extracted_text = original_text[start:end]
                    if extracted_text.strip() == entity_text.strip():
                        standardized['text'] = extracted_text
                except:
                    pass
            else:
                # Try to find the entity in the text
                try:
                    start_pos = original_text.lower().find(entity_text.lower())
                    if start_pos != -1:
                        standardized['start'] = start_pos
                        standardized['end'] = start_pos + len(entity_text)
                    else:
                        standardized['start'] = 0
                        standardized['end'] = 0
                except:
                    standardized['start'] = 0
                    standardized['end'] = 0
        else:
            # If no entity text found, mark as empty
            standardized['text'] = 'Unknown entity'
            standardized['start'] = 0
            standardized['end'] = 0
            
        # Ensure other required fields exist
        if 'confidence' not in standardized:
            standardized['confidence'] = 0.5
            
        if 'type' not in standardized:
            standardized['type'] = 'unknown'
            
        return standardized
    
    def _smart_icd_search(self, entity_text: str, entity_data: Dict[str, Any], top_k: int, entity_type: str) -> List[Dict[str, Any]]:
        """
        Hybrid ICD search: Fast parallel vector search + accurate Claude AI suggestions
        
        Args:
            entity_text: Entity text to search for
            entity_data: Full entity data for context
            top_k: Number of results to return
            entity_type: Type of entity (condition, symptom, etc.)
            
        Returns:
            List of ICD code matches with similarity scores
        """
        try:
            # Strategy 1: Fast parallel vector search (existing approach)
            logger.info(f"🚀 Hybrid search for '{entity_text}' - trying vector search first...")
            
            variants = self._generate_search_variants(entity_text, entity_data)
            variant_results = self.icd_matcher.find_similar_icd_codes_batch(
                variants,
                top_k=top_k * 2,
                min_similarity=0.2  # Higher threshold for vector search
            )
            
            vector_results = self._merge_and_rank_search_results(variant_results, variants)[:top_k]
            
            # Check if vector search found good results
            has_good_vector_results = (
                vector_results and 
                len(vector_results) >= max(1, top_k // 2) and
                vector_results[0].get('similarity', 0) >= 0.3
            )
            
            if has_good_vector_results:
                logger.info(f"✅ Vector search success: {len(vector_results)} matches (best: {vector_results[0].get('similarity', 0):.3f})")
                return vector_results
                
            # Strategy 2: Claude AI suggestions (high accuracy fallback)
            logger.info(f"🤖 Vector search insufficient, trying Claude AI suggestions...")
            
            # Prepare context for Claude
            context = {
                'severity': entity_data.get('severity'),
                'temporal': entity_data.get('temporal'),
            }
            
            claude_results = self.claude_icd_matcher.suggest_icd_codes(
                entity_text=entity_text,
                entity_type=entity_type,
                top_k=top_k,
                context=context
            )
            
            if claude_results:
                # Convert Claude results to match expected format
                formatted_results = []
                for result in claude_results:
                    formatted_result = {
                        'icd_code': result.get('code'),
                        'code': result.get('code'),
                        'description': result.get('description'),
                        'similarity': result.get('confidence', 0.8),
                        'search_method': 'claude_ai',
                        'validated': result.get('validated', False),
                        'database_match': result.get('database_match', 'unknown'),
                        'reasoning': result.get('reasoning', '')
                    }
                    formatted_results.append(formatted_result)
                
                logger.info(f"✅ Claude AI success: {len(formatted_results)} suggestions (best: {formatted_results[0].get('similarity', 0):.3f})")
                return formatted_results
                
            # Strategy 3: Lower threshold vector search (last resort)
            logger.info(f"🔄 Trying lower threshold vector search as final fallback...")
            
            fallback_results = self.icd_matcher.find_similar_icd_codes(
                entity_text,
                top_k=top_k,
                min_similarity=0.05
            )
            
            if fallback_results:
                logger.info(f"✅ Fallback search found {len(fallback_results)} results")
                return fallback_results
                
            logger.info(f"❌ No ICD matches found for '{entity_text}' with any strategy")
            return []
            
        except Exception as e:
            logger.error(f"❌ Smart ICD search failed for '{entity_text}': {e}")
            return []
    
    def _clean_medical_term(self, term: str) -> str:
        """
        Clean medical terms by removing severity modifiers and extra words
        
        Args:
            term: Original medical term
            
        Returns:
            Cleaned term more likely to match ICD codes
        """
        # Remove common modifiers
        modifiers_to_remove = [
            'severe', 'moderate', 'mild', 'acute', 'chronic', 'recurrent',
            'persistent', 'intermittent', 'progressive', 'stable',
            'new onset', 'recent', 'longstanding', 'refractory'
        ]
        
        cleaned = term.lower().strip()
        
        for modifier in modifiers_to_remove:
            # Remove modifier at start of term
            if cleaned.startswith(modifier + ' '):
                cleaned = cleaned[len(modifier):].strip()
            # Remove modifier at end of term  
            if cleaned.endswith(' ' + modifier):
                cleaned = cleaned[:-len(modifier)].strip()
        
        return cleaned.title()  # Return with proper capitalization
    
    def _cached_icd_search(self, search_term: str, top_k: int, min_similarity: float) -> List[Dict[str, Any]]:
        """
        Cached ICD search to avoid duplicate API calls for embeddings
        
        Args:
            search_term: Term to search for
            top_k: Number of results
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of ICD matches
        """
        # Create cache key
        cache_key = f"{search_term.lower()}_{top_k}_{min_similarity}"
        
        # Check cache first
        if cache_key in self.embedding_cache:
            logger.info(f"🔄 Using cached result for '{search_term}'")
            return self.embedding_cache[cache_key]
        
        # Perform search and cache result
        try:
            matches = self.icd_matcher.find_similar_icd_codes(
                search_term, top_k=top_k, min_similarity=min_similarity
            )
            
            # Cache the result (limit cache size to prevent memory issues)
            if len(self.embedding_cache) < 100:
                self.embedding_cache[cache_key] = matches
            
            return matches
            
        except Exception as e:
            logger.error(f"Cached ICD search failed for '{search_term}': {e}")
            return []


def create_enhanced_clinical_analysis_service(force_numpy_icd: bool = False) -> Optional[EnhancedClinicalAnalysisService]:
    """
    Factory function to create EnhancedClinicalAnalysisService with error handling
    
    Args:
        force_numpy_icd: Force numpy implementation for ICD matching
        
    Returns:
        EnhancedClinicalAnalysisService instance or None if creation fails
    """
    try:
        return EnhancedClinicalAnalysisService(force_numpy_icd=force_numpy_icd)
    except Exception as e:
        logger.error(f"❌ Failed to create enhanced clinical analysis service: {e}")
        return None