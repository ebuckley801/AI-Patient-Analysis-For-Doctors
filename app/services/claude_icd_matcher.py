#!/usr/bin/env python3
"""
Claude-powered ICD-10 code suggestion service
High-accuracy medical code matching using AI reasoning
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from app.services.clinical_analysis_service import ClinicalAnalysisService
from app.services.supabase_service import SupabaseService

logger = logging.getLogger(__name__)

class ClaudeICDMatcher:
    """
    Advanced ICD-10 code matching using Claude's medical knowledge
    Provides high-accuracy suggestions with reasoning and confidence scores
    """
    
    def __init__(self):
        """Initialize Claude ICD matcher with caching and validation"""
        self.clinical_service = ClinicalAnalysisService()
        self.supabase_service = SupabaseService()
        
        # Aggressive caching for performance
        self.suggestion_cache = {}
        self.validation_cache = {}  # Cache validation results for individual codes
        
        # Performance tracking
        self.stats = {
            'cache_hits': 0,
            'api_calls': 0,
            'validation_queries': 0,
            'validation_cache_hits': 0,
            'avg_response_time_ms': 0,
            'avg_validation_time_ms': 0,
            'accuracy_feedback': []
        }
    
    def _validate_code_exists(self, code: str) -> Dict[str, Any]:
        """
        Check if a single ICD code exists in database (on-demand validation)
        
        Args:
            code: Normalized ICD code (e.g., "I10", "R079")
            
        Returns:
            Dictionary with validation results
        """
        # Check cache first
        if code in self.validation_cache:
            self.stats['validation_cache_hits'] += 1
            return self.validation_cache[code]
        
        try:
            start_time = time.time()
            
            # Query database for exact code match
            response = self.supabase_service.client.table('icd_10_codes')\
                .select('icd_10_code,description')\
                .eq('icd_10_code', code)\
                .limit(1)\
                .execute()
            
            validation_time = (time.time() - start_time) * 1000
            self.stats['validation_queries'] += 1
            self._update_validation_stats(validation_time)
            
            # Process result
            if response.data and len(response.data) > 0:
                # Exact match found
                result = {
                    'exists': True,
                    'match_type': 'exact',
                    'description': response.data[0]['description'],
                    'code': response.data[0]['icd_10_code']
                }
                logger.info(f"✅ Code {code} validated in {validation_time:.0f}ms")
            else:
                # No exact match - try partial matching
                prefix_result = self._validate_code_partial(code)
                result = prefix_result
                logger.info(f"🔍 Code {code} partial validation in {validation_time:.0f}ms")
            
            # Cache the result
            self.validation_cache[code] = result
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Validation failed for {code}: {e}")
            return {
                'exists': False,
                'match_type': 'error',
                'description': '',
                'code': code,
                'error': str(e)
            }
    
    def _validate_code_partial(self, code: str) -> Dict[str, Any]:
        """
        Try partial matching for ICD codes (fallback when exact match fails)
        
        Args:
            code: ICD code to partially match
            
        Returns:
            Dictionary with partial validation results
        """
        try:
            # Try matching with first 3-4 characters
            prefixes = [code[:4], code[:3]] if len(code) > 3 else [code[:3]]
            
            for prefix in prefixes:
                if len(prefix) < 2:  # Too short to be meaningful
                    continue
                
                response = self.supabase_service.client.table('icd_10_codes')\
                    .select('icd_10_code,description')\
                    .like('icd_10_code', f'{prefix}%')\
                    .limit(5)\
                    .execute()
                
                if response.data and len(response.data) > 0:
                    # Find best partial match
                    best_match = min(response.data, 
                                   key=lambda x: abs(len(x['icd_10_code']) - len(code)))
                    
                    return {
                        'exists': True,
                        'match_type': 'partial',
                        'description': best_match['description'],
                        'code': best_match['icd_10_code'],
                        'original_code': code,
                        'prefix_used': prefix
                    }
            
            # No partial matches found
            return {
                'exists': False,
                'match_type': 'none',
                'description': '',
                'code': code
            }
            
        except Exception as e:
            logger.error(f"❌ Partial validation failed for {code}: {e}")
            return {
                'exists': False,
                'match_type': 'error',
                'description': '',
                'code': code,
                'error': str(e)
            }
    
    def suggest_icd_codes(self, entity_text: str, entity_type: str = 'condition', 
                         top_k: int = 5, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Get ICD-10 code suggestions using Claude's medical knowledge
        
        Args:
            entity_text: Medical entity to match (e.g., "chest pain", "diabetes")
            entity_type: Type of entity ('condition', 'symptom', 'procedure')
            top_k: Number of suggestions to return
            context: Additional context (severity, temporal info, etc.)
            
        Returns:
            List of ICD code suggestions with confidence and reasoning
        """
        # Create cache key
        cache_key = f"{entity_text.lower()}_{entity_type}_{top_k}_{hash(str(context))}"
        
        # Check cache first
        if cache_key in self.suggestion_cache:
            self.stats['cache_hits'] += 1
            logger.info(f"📋 Cache hit for '{entity_text}' ({entity_type})")
            return self.suggestion_cache[cache_key]
        
        try:
            # Generate Claude prompt
            prompt = self._create_icd_prompt(entity_text, entity_type, top_k, context)
            
            start_time = time.time()
            
            # Call Claude
            response = self.clinical_service.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1500,
                temperature=0.1,  # Low temperature for consistent medical responses
                messages=[{"role": "user", "content": prompt}]
            )
            
            api_time = (time.time() - start_time) * 1000
            self.stats['api_calls'] += 1
            
            # Parse Claude's response
            suggestions = self._parse_claude_icd_response(response.content[0].text)
            
            # Validate suggestions against our database
            validated_suggestions = self._validate_suggestions(suggestions, entity_text)
            
            # Cache the result
            self.suggestion_cache[cache_key] = validated_suggestions
            
            # Update stats
            self._update_performance_stats(api_time)
            
            logger.info(f"🤖 Claude ICD suggestions for '{entity_text}': {len(validated_suggestions)} codes in {api_time:.0f}ms")
            
            return validated_suggestions
            
        except Exception as e:
            logger.error(f"❌ Claude ICD suggestion failed for '{entity_text}': {e}")
            return []
    
    def _create_icd_prompt(self, entity_text: str, entity_type: str, top_k: int, context: Dict[str, Any] = None) -> str:
        """
        Create context-aware prompt for Claude based on entity type and context
        
        Args:
            entity_text: Medical entity text
            entity_type: Type of medical entity
            top_k: Number of codes requested
            context: Additional context information
            
        Returns:
            Optimized prompt string for Claude
        """
        # Base context
        severity = context.get('severity', '') if context else ''
        temporal = context.get('temporal', '') if context else ''
        patient_age = context.get('age', '') if context else ''
        
        context_info = ""
        if severity:
            context_info += f"\n- Severity: {severity}"
        if temporal:
            context_info += f"\n- Temporal aspect: {temporal}"
        if patient_age:
            context_info += f"\n- Patient age: {patient_age}"
        
        # Entity-type specific prompting
        if entity_type == 'condition':
            prompt = f"""You are a medical coding expert. Given the medical condition "{entity_text}", provide the most relevant ICD-10 codes.

Consider:
- Primary vs secondary diagnoses
- Chronic vs acute presentations  
- Common complications and variants
- Specificity levels (use most specific appropriate code){context_info}

Return the top {top_k} most relevant ICD-10 codes in this exact JSON format:
[
  {{
    "code": "ICD_CODE",
    "description": "Full ICD description", 
    "confidence": 0.95,
    "reasoning": "Why this code is appropriate",
    "specificity": "high/medium/low"
  }}
]

IMPORTANT: You can provide codes with or without periods (e.g., "R07.9" or "R079") - both formats will be accepted.

Focus on:
1. Most common presentations first
2. Clinically relevant codes
3. Appropriate specificity level
4. Clear reasoning for each suggestion"""

        elif entity_type == 'symptom':
            prompt = f"""You are a medical coding expert. Given the symptom "{entity_text}", provide the most relevant ICD-10 codes.

Consider:
- Direct symptom codes (R codes)
- Most likely underlying conditions
- Differential diagnoses
- Age-appropriate conditions{context_info}

Return the top {top_k} most relevant ICD-10 codes in this exact JSON format:
[
  {{
    "code": "ICD_CODE", 
    "description": "Full ICD description",
    "confidence": 0.90,
    "reasoning": "Why this code matches the symptom",
    "category": "symptom/condition"
  }}
]

IMPORTANT: You can provide codes with or without periods (e.g., "R07.9" or "R079") - both formats will be accepted.

Priority:
1. Direct symptom codes (R category)
2. Common underlying conditions  
3. Emergency/serious conditions to consider
4. Age-specific considerations"""

        else:  # procedure or other
            prompt = f"""You are a medical coding expert. Given the medical term "{entity_text}", provide the most relevant ICD-10 codes.

Consider:
- Procedure codes if applicable
- Associated conditions
- Common clinical scenarios{context_info}

Return the top {top_k} most relevant ICD-10 codes in this exact JSON format:
[
  {{
    "code": "ICD_CODE",
    "description": "Full ICD description", 
    "confidence": 0.85,
    "reasoning": "Why this code is relevant"
  }}
]

IMPORTANT: You can provide codes with or without periods (e.g., "R07.9" or "R079") - both formats will be accepted."""
        
        return prompt
    
    def _parse_claude_icd_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse Claude's JSON response into structured suggestions
        
        Args:
            response_text: Raw response from Claude
            
        Returns:
            List of parsed ICD suggestions
        """
        try:
            # Clean the response - remove any text before/after JSON
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                logger.error(f"❌ No JSON found in Claude response")
                return []
            
            json_text = response_text[json_start:json_end]
            suggestions = json.loads(json_text)
            
            # Validate structure
            if not isinstance(suggestions, list):
                logger.error(f"❌ Claude response is not a list")
                return []
            
            parsed_suggestions = []
            for suggestion in suggestions:
                if isinstance(suggestion, dict) and 'code' in suggestion and 'description' in suggestion:
                    # Standardize fields
                    parsed_suggestion = {
                        'code': suggestion['code'].strip(),
                        'description': suggestion['description'].strip(),
                        'confidence': float(suggestion.get('confidence', 0.8)),
                        'reasoning': suggestion.get('reasoning', ''),
                        'source': 'claude_ai',
                        'specificity': suggestion.get('specificity', 'medium'),
                        'category': suggestion.get('category', 'condition')
                    }
                    parsed_suggestions.append(parsed_suggestion)
                else:
                    logger.warning(f"⚠️ Invalid suggestion format: {suggestion}")
            
            logger.info(f"📝 Parsed {len(parsed_suggestions)} suggestions from Claude")
            return parsed_suggestions
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing error: {e}")
            logger.error(f"Response text: {response_text[:500]}...")
            return []
        except Exception as e:
            logger.error(f"❌ Error parsing Claude response: {e}")
            return []
    
    def _normalize_icd_code(self, code: str) -> str:
        """
        Normalize ICD code to match database format (remove periods)
        
        Args:
            code: ICD code potentially with periods (e.g., "R07.9")
            
        Returns:
            Normalized code without periods (e.g., "R079")
        """
        return code.replace('.', '').strip().upper()
    
    def _validate_suggestions(self, suggestions: List[Dict[str, Any]], entity_text: str) -> List[Dict[str, Any]]:
        """
        Validate Claude's suggestions using on-demand database queries
        
        Args:
            suggestions: List of ICD suggestions from Claude
            entity_text: Original entity text for context
            
        Returns:
            List of validated suggestions with database matches
        """
        if not suggestions:
            return suggestions
        
        validated = []
        
        for suggestion in suggestions:
            original_code = suggestion['code']
            normalized_code = self._normalize_icd_code(original_code)
            
            # Store both original and normalized versions
            suggestion['code_original'] = original_code
            suggestion['code'] = normalized_code
            
            # On-demand validation
            validation_result = self._validate_code_exists(normalized_code)
            
            if validation_result['exists']:
                if validation_result['match_type'] == 'exact':
                    # Exact match - highest confidence
                    suggestion['database_match'] = 'exact'
                    suggestion['database_description'] = validation_result['description']
                    suggestion['validated'] = True
                    validated.append(suggestion)
                    
                elif validation_result['match_type'] == 'partial':
                    # Partial match - good confidence
                    suggestion['database_match'] = 'partial'
                    suggestion['database_description'] = validation_result['description']
                    suggestion['suggested_alternative'] = validation_result['code']
                    suggestion['validated'] = True
                    suggestion['confidence'] *= 0.85  # Slight reduction for partial matches
                    validated.append(suggestion)
                    
            else:
                # No match found
                suggestion['database_match'] = validation_result['match_type']  # 'none' or 'error'
                suggestion['validated'] = False
                suggestion['confidence'] *= 0.6  # Reduce confidence for unvalidated codes
                validated.append(suggestion)
                
                if validation_result['match_type'] != 'error':
                    logger.warning(f"⚠️ Code {original_code} → {normalized_code} not found in database for '{entity_text}'")
        
        # Sort by confidence (validated suggestions first)
        validated.sort(key=lambda x: (x['validated'], x['confidence']), reverse=True)
        
        validated_count = len([s for s in validated if s['validated']])
        logger.info(f"✅ Validated {validated_count} of {len(suggestions)} suggestions using on-demand queries")
        
        return validated
    
    def _update_performance_stats(self, api_time_ms: float):
        """Update performance statistics"""
        prev_avg = self.stats['avg_response_time_ms']
        call_count = self.stats['api_calls']
        
        self.stats['avg_response_time_ms'] = (
            (prev_avg * (call_count - 1) + api_time_ms) / call_count
        )
    
    def _update_validation_stats(self, validation_time_ms: float):
        """Update validation performance statistics"""
        prev_avg = self.stats['avg_validation_time_ms']
        query_count = self.stats['validation_queries']
        
        self.stats['avg_validation_time_ms'] = (
            (prev_avg * (query_count - 1) + validation_time_ms) / query_count
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance and accuracy statistics"""
        total_requests = self.stats['cache_hits'] + self.stats['api_calls']
        cache_hit_rate = self.stats['cache_hits'] / total_requests if total_requests > 0 else 0
        
        total_validations = self.stats['validation_cache_hits'] + self.stats['validation_queries']
        validation_cache_hit_rate = self.stats['validation_cache_hits'] / total_validations if total_validations > 0 else 0
        
        return {
            'total_requests': total_requests,
            'cache_hits': self.stats['cache_hits'],
            'api_calls': self.stats['api_calls'],
            'cache_hit_rate': cache_hit_rate,
            'avg_response_time_ms': self.stats['avg_response_time_ms'],
            'cached_entities': len(self.suggestion_cache),
            
            # Validation stats
            'validation_queries': self.stats['validation_queries'],
            'validation_cache_hits': self.stats['validation_cache_hits'],
            'validation_cache_hit_rate': validation_cache_hit_rate,
            'avg_validation_time_ms': self.stats['avg_validation_time_ms'],
            'cached_validations': len(self.validation_cache)
        }
    
    def clear_cache(self):
        """Clear all caches (for testing or memory management)"""
        self.suggestion_cache.clear()
        self.validation_cache.clear()
        logger.info("🗑️ Claude ICD matcher caches cleared")


def create_claude_icd_matcher() -> ClaudeICDMatcher:
    """Factory function to create ClaudeICDMatcher instance"""
    return ClaudeICDMatcher()