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
        
        # Performance tracking
        self.analysis_stats = {
            'total_analyses': 0,
            'faiss_searches': 0,
            'numpy_searches': 0,
            'avg_analysis_time_ms': 0,
            'avg_icd_search_time_ms': 0
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
                'icd_mappings': icd_mapping_results,
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
            # Map conditions (highest priority)
            for condition in analysis_result.get('conditions', []):
                entity_text = condition.get('entity', '')
                if entity_text:
                    enhanced_text = self._prepare_entity_for_icd_search(condition, use_nlp_context)
                    
                    search_start = time.time()
                    icd_matches = self.icd_matcher.find_similar_icd_codes(
                        enhanced_text, top_k=top_k, min_similarity=0.1
                    )
                    search_time = (time.time() - search_start) * 1000
                    
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
                            'enhanced_query': enhanced_text != entity_text
                        }
                        
                        # Add NLP context if available
                        if use_nlp_context and 'negation' in condition:
                            mapping['negated'] = condition['negation'].get('is_negated', False)
                        if use_nlp_context and 'uncertainty' in condition:
                            mapping['uncertain'] = condition['uncertainty'].get('has_uncertainty', False)
                        
                        icd_mappings.append(mapping)
            
            # Map symptoms (secondary priority)
            for symptom in analysis_result.get('symptoms', []):
                entity_text = symptom.get('entity', '')
                if entity_text and not symptom.get('negated', False):  # Skip negated symptoms
                    enhanced_text = self._prepare_entity_for_icd_search(symptom, use_nlp_context)
                    
                    search_start = time.time()
                    icd_matches = self.icd_matcher.find_similar_icd_codes(
                        enhanced_text, top_k=min(top_k, 3), min_similarity=0.15  # Higher threshold for symptoms
                    )
                    search_time = (time.time() - search_start) * 1000
                    
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
                            'enhanced_query': enhanced_text != entity_text
                        }
                        icd_mappings.append(mapping)
            
            logger.info(f"🔍 ICD mapping completed: {len(icd_mappings)} mappings found")
            return icd_mappings
            
        except Exception as e:
            logger.error(f"❌ ICD mapping failed: {e}")
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
            if temporal_info.get('onset'):
                search_terms.append(f"{temporal_info['onset']} {base_text}")
            if temporal_info.get('progression') and temporal_info['progression'] != 'stable':
                search_terms.append(f"{temporal_info['progression']} {base_text}")
        
        # Add status context for conditions
        if entity.get('status') and entity['status'] in ['chronic', 'acute']:
            search_terms.append(f"{entity['status']} {base_text}")
        
        # Return the most descriptive term (usually the longest)
        return max(search_terms, key=len)
    
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