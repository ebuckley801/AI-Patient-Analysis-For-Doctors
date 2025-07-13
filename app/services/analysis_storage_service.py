#!/usr/bin/env python3
"""
Analysis Storage Service for Intelligence Layer (Phase 2)
Handles persistent storage of clinical analysis results, entity mappings, and caching.
"""

import os
import json
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from decimal import Decimal

from app.services.supabase_service import get_supabase_client


class AnalysisStorageService:
    """Service for storing and retrieving clinical analysis results"""
    
    def __init__(self):
        self.supabase = get_supabase_client()
    
    def generate_session_id(self) -> str:
        """Generate a unique session ID for analysis tracking"""
        return f"session_{uuid.uuid4().hex[:16]}_{int(datetime.now().timestamp())}"
    
    def generate_cache_key(self, note_text: str, patient_context: Optional[Dict] = None, 
                          analysis_type: str = "extract") -> str:
        """Generate a cache key from note text and context"""
        # Create a consistent hash from the inputs
        content = f"{note_text.strip()}"
        if patient_context:
            # Sort the context dict to ensure consistent hashing
            sorted_context = json.dumps(patient_context, sort_keys=True)
            content += f"|{sorted_context}"
        content += f"|{analysis_type}"
        
        return hashlib.sha256(content.encode()).hexdigest()
    
    def generate_note_text_hash(self, note_text: str) -> str:
        """Generate SHA256 hash of note text for cache lookup"""
        return hashlib.sha256(note_text.strip().encode()).hexdigest()
    
    def create_analysis_session(self, note_id: Optional[str] = None, 
                              patient_id: Optional[str] = None,
                              analysis_type: str = "extract",
                              request_data: Dict[str, Any] = None) -> str:
        """
        Create a new analysis session and return the session ID
        
        Args:
            note_id: Optional note identifier
            patient_id: Optional patient identifier  
            analysis_type: Type of analysis ('extract', 'diagnose', 'batch', 'priority')
            request_data: The original request data
            
        Returns:
            session_id: Unique session identifier
        """
        try:
            session_id = self.generate_session_id()
            
            session_data = {
                'session_id': session_id,
                'note_id': note_id,
                'patient_id': patient_id,
                'analysis_type': analysis_type,
                'status': 'pending',
                'request_data': request_data or {},
                'created_at': datetime.now().isoformat()
            }
            
            result = self.supabase.table('analysis_sessions').insert(session_data).execute()
            
            if result.data:
                return session_id
            else:
                raise Exception("Failed to create analysis session")
                
        except Exception as e:
            print(f"Error creating analysis session: {str(e)}")
            raise
    
    def update_analysis_session(self, session_id: str, **updates) -> bool:
        """Update an existing analysis session with new data"""
        try:
            # Add updated timestamp
            updates['updated_at'] = datetime.now().isoformat()
            
            result = self.supabase.table('analysis_sessions')\
                                 .update(updates)\
                                 .eq('session_id', session_id)\
                                 .execute()
            
            return len(result.data) > 0
            
        except Exception as e:
            print(f"Error updating analysis session {session_id}: {str(e)}")
            return False
    
    def store_clinical_entities(self, session_id: str, entities: List[Dict[str, Any]]) -> List[str]:
        """
        Store clinical entities extracted from analysis
        
        Args:
            session_id: Session identifier
            entities: List of clinical entities with metadata
            
        Returns:
            List of entity IDs that were created
        """
        try:
            entity_records = []
            
            for entity in entities:
                entity_record = {
                    'session_id': session_id,
                    'entity_type': entity.get('type', 'unknown'),
                    'entity_text': entity.get('entity', ''),
                    'confidence': float(entity.get('confidence', 0.0)),
                    'severity': entity.get('severity'),
                    'status': entity.get('status'),
                    'temporal_info': entity.get('temporal_info'),
                    'negation': entity.get('negation', False),
                    'text_span': entity.get('text_span'),
                    'normalized_form': entity.get('normalized_form'),
                    'additional_context': entity.get('additional_context', {}),
                    'extraction_method': 'claude_ai',
                    'created_at': datetime.now().isoformat()
                }
                entity_records.append(entity_record)
            
            if entity_records:
                result = self.supabase.table('clinical_entities').insert(entity_records).execute()
                
                if result.data:
                    return [record['id'] for record in result.data]
                else:
                    raise Exception("Failed to store clinical entities")
            
            return []
            
        except Exception as e:
            print(f"Error storing clinical entities for session {session_id}: {str(e)}")
            raise
    
    def store_icd_mappings(self, session_id: str, entity_id: str, 
                          icd_mappings: List[Dict[str, Any]]) -> List[str]:
        """
        Store ICD-10 mappings for a clinical entity
        
        Args:
            session_id: Session identifier
            entity_id: Clinical entity identifier
            icd_mappings: List of ICD-10 mappings with similarity scores
            
        Returns:
            List of mapping IDs that were created
        """
        try:
            mapping_records = []
            
            for rank, mapping in enumerate(icd_mappings, 1):
                mapping_record = {
                    'entity_id': entity_id,
                    'session_id': session_id,
                    'icd_10_code': mapping.get('icd_10_code', ''),
                    'icd_description': mapping.get('description', ''),
                    'similarity_score': float(mapping.get('similarity_score', 0.0)),
                    'mapping_confidence': float(mapping.get('confidence', 0.0)),
                    'mapping_method': mapping.get('method', 'vector_similarity'),
                    'is_primary_mapping': rank == 1,  # First mapping is primary
                    'rank_order': rank,
                    'icd_category': mapping.get('category'),
                    'mapping_notes': mapping.get('notes'),
                    'verified_by_clinician': False,
                    'created_at': datetime.now().isoformat()
                }
                mapping_records.append(mapping_record)
            
            if mapping_records:
                result = self.supabase.table('entity_icd_mappings').insert(mapping_records).execute()
                
                if result.data:
                    return [record['id'] for record in result.data]
                else:
                    raise Exception("Failed to store ICD mappings")
            
            return []
            
        except Exception as e:
            print(f"Error storing ICD mappings for entity {entity_id}: {str(e)}")
            raise
    
    def cache_analysis_result(self, note_text: str, patient_context: Optional[Dict],
                            analysis_type: str, result: Dict[str, Any],
                            confidence_threshold: float = 0.7,
                            cache_ttl_days: int = 7) -> bool:
        """
        Cache analysis result for future retrieval
        
        Args:
            note_text: Original note text
            patient_context: Patient context used in analysis
            analysis_type: Type of analysis performed
            result: Complete analysis result to cache
            confidence_threshold: Confidence threshold used
            cache_ttl_days: Number of days to keep in cache
            
        Returns:
            True if caching was successful
        """
        try:
            cache_key = self.generate_cache_key(note_text, patient_context, analysis_type)
            note_text_hash = self.generate_note_text_hash(note_text)
            
            cache_record = {
                'cache_key': cache_key,
                'note_text_hash': note_text_hash,
                'patient_context': patient_context or {},
                'cached_result': result,
                'analysis_type': analysis_type,
                'confidence_threshold': confidence_threshold,
                'hit_count': 0,
                'expires_at': (datetime.now() + timedelta(days=cache_ttl_days)).isoformat(),
                'created_at': datetime.now().isoformat()
            }
            
            # Use upsert to handle potential duplicates
            result = self.supabase.table('analysis_cache')\
                                 .upsert(cache_record, on_conflict='cache_key')\
                                 .execute()
            
            return len(result.data) > 0
            
        except Exception as e:
            print(f"Error caching analysis result: {str(e)}")
            return False
    
    def get_cached_analysis(self, note_text: str, patient_context: Optional[Dict],
                          analysis_type: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached analysis result if available and not expired
        
        Args:
            note_text: Original note text
            patient_context: Patient context
            analysis_type: Type of analysis
            
        Returns:
            Cached result or None if not found/expired
        """
        try:
            cache_key = self.generate_cache_key(note_text, patient_context, analysis_type)
            
            result = self.supabase.table('analysis_cache')\
                                 .select('*')\
                                 .eq('cache_key', cache_key)\
                                 .gt('expires_at', datetime.now().isoformat())\
                                 .execute()
            
            if result.data and len(result.data) > 0:
                cache_record = result.data[0]
                
                # Update hit count and last accessed
                self.supabase.table('analysis_cache')\
                            .update({
                                'hit_count': cache_record['hit_count'] + 1,
                                'last_accessed': datetime.now().isoformat()
                            })\
                            .eq('cache_key', cache_key)\
                            .execute()
                
                return cache_record['cached_result']
            
            return None
            
        except Exception as e:
            print(f"Error retrieving cached analysis: {str(e)}")
            return None
    
    def get_analysis_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve analysis session by ID"""
        try:
            result = self.supabase.table('analysis_sessions')\
                                 .select('*')\
                                 .eq('session_id', session_id)\
                                 .execute()
            
            if result.data and len(result.data) > 0:
                return result.data[0]
            
            return None
            
        except Exception as e:
            print(f"Error retrieving analysis session {session_id}: {str(e)}")
            return None
    
    def get_session_entities(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all clinical entities for a session"""
        try:
            result = self.supabase.table('clinical_entities')\
                                 .select('*')\
                                 .eq('session_id', session_id)\
                                 .order('confidence', desc=True)\
                                 .execute()
            
            return result.data or []
            
        except Exception as e:
            print(f"Error retrieving entities for session {session_id}: {str(e)}")
            return []
    
    def get_entity_icd_mappings(self, entity_id: str) -> List[Dict[str, Any]]:
        """Retrieve ICD-10 mappings for a clinical entity"""
        try:
            result = self.supabase.table('entity_icd_mappings')\
                                 .select('*')\
                                 .eq('entity_id', entity_id)\
                                 .order('rank_order')\
                                 .execute()
            
            return result.data or []
            
        except Exception as e:
            print(f"Error retrieving ICD mappings for entity {entity_id}: {str(e)}")
            return []
    
    def get_priority_findings(self, note_id: str = None, patient_id: str = None,
                            risk_threshold: str = "high") -> List[Dict[str, Any]]:
        """
        Retrieve high-priority findings for a note or patient
        
        Args:
            note_id: Optional note identifier
            patient_id: Optional patient identifier
            risk_threshold: Minimum risk level ('moderate', 'high', 'critical')
            
        Returns:
            List of analysis sessions with priority findings
        """
        try:
            query = self.supabase.table('analysis_sessions')\
                                .select('*')\
                                .eq('status', 'completed')
            
            # Filter by risk level
            risk_levels = {
                'moderate': ['moderate', 'high', 'critical'],
                'high': ['high', 'critical'], 
                'critical': ['critical']
            }
            
            if risk_threshold in risk_levels:
                query = query.in_('risk_level', risk_levels[risk_threshold])
            
            # Filter by note_id or patient_id
            if note_id:
                query = query.eq('note_id', note_id)
            elif patient_id:
                query = query.eq('patient_id', patient_id)
            
            # Order by risk level and creation time
            result = query.order('requires_immediate_attention', desc=True)\
                         .order('created_at', desc=True)\
                         .execute()
            
            return result.data or []
            
        except Exception as e:
            print(f"Error retrieving priority findings: {str(e)}")
            return []
    
    def cleanup_expired_cache(self) -> int:
        """Clean up expired cache entries and return count of deleted entries"""
        try:
            # Call the database function we created
            result = self.supabase.rpc('cleanup_expired_cache').execute()
            
            if result.data is not None:
                return result.data
            else:
                return 0
                
        except Exception as e:
            print(f"Error cleaning up expired cache: {str(e)}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        try:
            # Get cache counts
            total_result = self.supabase.table('analysis_cache')\
                                      .select('id', count='exact')\
                                      .execute()
            
            expired_result = self.supabase.table('analysis_cache')\
                                        .select('id', count='exact')\
                                        .lt('expires_at', datetime.now().isoformat())\
                                        .execute()
            
            # Get hit statistics
            hits_result = self.supabase.table('analysis_cache')\
                                     .select('hit_count, analysis_type')\
                                     .execute()
            
            total_entries = total_result.count or 0
            expired_entries = expired_result.count or 0
            active_entries = total_entries - expired_entries
            
            # Calculate hit statistics
            total_hits = sum(record['hit_count'] for record in (hits_result.data or []))
            hit_rate = (total_hits / total_entries) if total_entries > 0 else 0
            
            return {
                'total_cache_entries': total_entries,
                'active_cache_entries': active_entries,
                'expired_cache_entries': expired_entries,
                'total_cache_hits': total_hits,
                'cache_hit_rate': round(hit_rate, 3),
                'cleanup_needed': expired_entries > 0
            }
            
        except Exception as e:
            print(f"Error getting cache stats: {str(e)}")
            return {
                'total_cache_entries': 0,
                'active_cache_entries': 0,
                'expired_cache_entries': 0,
                'total_cache_hits': 0,
                'cache_hit_rate': 0.0,
                'cleanup_needed': False,
                'error': str(e)
            }