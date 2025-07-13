#!/usr/bin/env python3
"""
Async Clinical Analysis Service
High-performance async processing for large-scale clinical text analysis
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import time

from app.services.clinical_analysis_service import ClinicalAnalysisService
from app.services.analysis_storage_service import AnalysisStorageService
from app.services.icd10_vector_matcher import ICD10VectorMatcher

logger = logging.getLogger(__name__)


@dataclass
class BatchAnalysisResult:
    """Result of batch analysis operation"""
    note_id: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time_ms: Optional[int] = None
    session_id: Optional[str] = None


@dataclass
class BatchAnalysisConfig:
    """Configuration for batch analysis operations"""
    max_concurrent: int = 10
    timeout_seconds: int = 30
    include_icd_mapping: bool = True
    include_storage: bool = True
    chunk_size: int = 50
    retry_attempts: int = 2
    retry_delay: float = 1.0


class AsyncClinicalAnalysis:
    """Async clinical analysis service for high-performance batch processing"""
    
    def __init__(self):
        self.clinical_service = ClinicalAnalysisService()
        self.storage_service = AnalysisStorageService()
        self.icd_matcher = ICD10VectorMatcher()
        self.executor = ThreadPoolExecutor(max_workers=20)
    
    async def analyze_note_async(self, note_data: Dict[str, Any], config: BatchAnalysisConfig) -> BatchAnalysisResult:
        """
        Analyze a single note asynchronously
        
        Args:
            note_data: Dict with note_id, note_text, patient_context
            config: Batch analysis configuration
            
        Returns:
            BatchAnalysisResult with analysis results
        """
        start_time = time.time()
        note_id = note_data.get('note_id', f'note_{int(time.time())}')
        
        try:
            # Extract required data
            note_text = note_data.get('note_text', '')
            patient_context = note_data.get('patient_context', {})
            patient_id = note_data.get('patient_id')
            
            if not note_text or len(note_text.strip()) < 10:
                return BatchAnalysisResult(
                    note_id=note_id,
                    success=False,
                    error="Note text is too short or empty"
                )
            
            session_id = None
            
            # Create analysis session if storage enabled
            if config.include_storage:
                try:
                    session_id = self.storage_service.create_analysis_session(
                        note_id=note_id,
                        patient_id=patient_id,
                        analysis_type='batch',
                        request_data={
                            'note_text': note_text,
                            'patient_context': patient_context
                        }
                    )
                    
                    # Update session status
                    self.storage_service.update_analysis_session(session_id, status='processing')
                    
                except Exception as storage_error:
                    logger.warning(f"Storage session creation failed for {note_id}: {storage_error}")
            
            # Check cache first if storage enabled
            if config.include_storage:
                try:
                    cached_result = self.storage_service.get_cached_analysis(
                        note_text, patient_context, 'extract'
                    )
                    if cached_result:
                        # Update session with cached result
                        if session_id:
                            self.storage_service.update_analysis_session(
                                session_id,
                                status='completed',
                                response_data=cached_result
                            )
                        
                        processing_time = int((time.time() - start_time) * 1000)
                        cached_result['from_cache'] = True
                        cached_result['processing_time_ms'] = processing_time
                        
                        return BatchAnalysisResult(
                            note_id=note_id,
                            success=True,
                            data=cached_result,
                            processing_time_ms=processing_time,
                            session_id=session_id
                        )
                except Exception as cache_error:
                    logger.warning(f"Cache lookup failed for {note_id}: {cache_error}")
            
            # Run clinical analysis in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            analysis_result = await loop.run_in_executor(
                self.executor,
                self.clinical_service.extract_clinical_entities,
                note_text,
                patient_context
            )
            
            # Check for analysis errors
            if 'error' in analysis_result:
                if session_id:
                    self.storage_service.update_analysis_session(
                        session_id,
                        status='failed',
                        error_message=analysis_result['error']
                    )
                
                return BatchAnalysisResult(
                    note_id=note_id,
                    success=False,
                    error=analysis_result['error'],
                    session_id=session_id
                )
            
            # Add ICD mapping if requested
            if config.include_icd_mapping:
                try:
                    analysis_result = await loop.run_in_executor(
                        self.executor,
                        self.icd_matcher.map_clinical_entities_to_icd,
                        analysis_result
                    )
                except Exception as icd_error:
                    logger.warning(f"ICD mapping failed for {note_id}: {icd_error}")
                    # Continue without ICD mapping
            
            # Store results if storage enabled
            if config.include_storage and session_id:
                try:
                    # Store entities
                    all_entities = []
                    for entity_type in ['symptoms', 'conditions', 'medications', 'vital_signs', 'procedures', 'abnormal_findings']:
                        for entity in analysis_result.get(entity_type, []):
                            entity_with_type = entity.copy()
                            entity_with_type['type'] = entity_type[:-1] if entity_type.endswith('s') else entity_type
                            all_entities.append(entity_with_type)
                    
                    if all_entities:
                        await loop.run_in_executor(
                            self.executor,
                            self.storage_service.store_clinical_entities,
                            session_id,
                            all_entities
                        )
                    
                    # Update session with success
                    assessment = analysis_result.get('overall_assessment', {})
                    await loop.run_in_executor(
                        self.executor,
                        self.storage_service.update_analysis_session,
                        session_id,
                        {
                            'status': 'completed',
                            'response_data': analysis_result,
                            'risk_level': assessment.get('risk_level', 'low'),
                            'requires_immediate_attention': assessment.get('requires_immediate_attention', False)
                        }
                    )
                    
                    # Cache the result
                    await loop.run_in_executor(
                        self.executor,
                        self.storage_service.cache_analysis_result,
                        note_text,
                        patient_context,
                        'extract',
                        analysis_result
                    )
                    
                except Exception as storage_error:
                    logger.warning(f"Storage operations failed for {note_id}: {storage_error}")
            
            processing_time = int((time.time() - start_time) * 1000)
            analysis_result['processing_time_ms'] = processing_time
            analysis_result['from_cache'] = False
            
            return BatchAnalysisResult(
                note_id=note_id,
                success=True,
                data=analysis_result,
                processing_time_ms=processing_time,
                session_id=session_id
            )
            
        except asyncio.TimeoutError:
            return BatchAnalysisResult(
                note_id=note_id,
                success=False,
                error=f"Analysis timeout after {config.timeout_seconds} seconds"
            )
        except Exception as e:
            logger.error(f"Error in async analysis for {note_id}: {str(e)}")
            return BatchAnalysisResult(
                note_id=note_id,
                success=False,
                error=str(e)
            )
    
    async def batch_analyze_notes(self, notes: List[Dict[str, Any]], 
                                config: Optional[BatchAnalysisConfig] = None) -> Dict[str, Any]:
        """
        Analyze multiple notes concurrently with optimized performance
        
        Args:
            notes: List of note dictionaries with note_text, patient_context, etc.
            config: Configuration for batch processing
            
        Returns:
            Dict with results, summary, and performance metrics
        """
        if config is None:
            config = BatchAnalysisConfig()
        
        start_time = time.time()
        total_notes = len(notes)
        
        logger.info(f"Starting batch analysis of {total_notes} notes with {config.max_concurrent} concurrent workers")
        
        # Initialize results
        results = []
        summary = {
            'total_notes': total_notes,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'cached_results': 0,
            'total_entities': 0,
            'high_priority_cases': 0,
            'total_processing_time_ms': 0,
            'average_processing_time_ms': 0,
            'concurrent_workers': config.max_concurrent,
            'chunks_processed': 0
        }
        
        # Process notes in chunks to manage memory and concurrency
        for chunk_start in range(0, total_notes, config.chunk_size):
            chunk_end = min(chunk_start + config.chunk_size, total_notes)
            chunk = notes[chunk_start:chunk_end]
            chunk_number = (chunk_start // config.chunk_size) + 1
            
            logger.info(f"Processing chunk {chunk_number}/{((total_notes - 1) // config.chunk_size) + 1} "
                       f"({len(chunk)} notes)")
            
            # Create semaphore to limit concurrent operations
            semaphore = asyncio.Semaphore(config.max_concurrent)
            
            async def analyze_with_semaphore(note_data):
                async with semaphore:
                    for attempt in range(config.retry_attempts):
                        try:
                            result = await asyncio.wait_for(
                                self.analyze_note_async(note_data, config),
                                timeout=config.timeout_seconds
                            )
                            return result
                        except asyncio.TimeoutError:
                            if attempt < config.retry_attempts - 1:
                                logger.warning(f"Timeout on attempt {attempt + 1} for note {note_data.get('note_id')}, retrying...")
                                await asyncio.sleep(config.retry_delay * (attempt + 1))
                            else:
                                return BatchAnalysisResult(
                                    note_id=note_data.get('note_id', 'unknown'),
                                    success=False,
                                    error=f"Timeout after {config.retry_attempts} attempts"
                                )
                        except Exception as e:
                            if attempt < config.retry_attempts - 1:
                                logger.warning(f"Error on attempt {attempt + 1} for note {note_data.get('note_id')}: {e}, retrying...")
                                await asyncio.sleep(config.retry_delay * (attempt + 1))
                            else:
                                return BatchAnalysisResult(
                                    note_id=note_data.get('note_id', 'unknown'),
                                    success=False,
                                    error=str(e)
                                )
            
            # Process chunk concurrently
            chunk_tasks = [analyze_with_semaphore(note_data) for note_data in chunk]
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
            
            # Process chunk results
            for result in chunk_results:
                if isinstance(result, Exception):
                    logger.error(f"Unexpected error in batch processing: {result}")
                    results.append(BatchAnalysisResult(
                        note_id='unknown',
                        success=False,
                        error=str(result)
                    ))
                    summary['failed_analyses'] += 1
                else:
                    results.append(result)
                    
                    if result.success:
                        summary['successful_analyses'] += 1
                        
                        if result.data:
                            # Check if from cache
                            if result.data.get('from_cache'):
                                summary['cached_results'] += 1
                            
                            # Count entities
                            entity_count = sum(
                                len(result.data.get(k, [])) 
                                for k in ['symptoms', 'conditions', 'medications', 'vital_signs', 'procedures', 'abnormal_findings']
                            )
                            summary['total_entities'] += entity_count
                            
                            # Check for high priority
                            if result.data.get('overall_assessment', {}).get('requires_immediate_attention'):
                                summary['high_priority_cases'] += 1
                            
                            # Add processing time
                            if result.processing_time_ms:
                                summary['total_processing_time_ms'] += result.processing_time_ms
                    else:
                        summary['failed_analyses'] += 1
            
            summary['chunks_processed'] += 1
            
            # Small delay between chunks to prevent overwhelming the system
            if chunk_end < total_notes:
                await asyncio.sleep(0.1)
        
        # Calculate final metrics
        total_time_ms = int((time.time() - start_time) * 1000)
        if summary['successful_analyses'] > 0:
            summary['average_processing_time_ms'] = summary['total_processing_time_ms'] // summary['successful_analyses']
        
        summary['batch_total_time_ms'] = total_time_ms
        summary['cache_hit_rate'] = summary['cached_results'] / total_notes if total_notes > 0 else 0
        summary['success_rate'] = summary['successful_analyses'] / total_notes if total_notes > 0 else 0
        
        logger.info(f"Batch analysis completed: {summary['successful_analyses']}/{total_notes} successful "
                   f"({summary['success_rate']:.1%} success rate, {summary['cache_hit_rate']:.1%} cache hit rate)")
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            result_dict = {
                'note_id': result.note_id,
                'success': result.success,
                'processing_time_ms': result.processing_time_ms,
                'session_id': result.session_id
            }
            
            if result.success and result.data:
                result_dict['data'] = result.data
            elif not result.success and result.error:
                result_dict['error'] = result.error
            
            serializable_results.append(result_dict)
        
        return {
            'success': True,
            'results': serializable_results,
            'summary': summary,
            'config_used': {
                'max_concurrent': config.max_concurrent,
                'chunk_size': config.chunk_size,
                'timeout_seconds': config.timeout_seconds,
                'include_icd_mapping': config.include_icd_mapping,
                'include_storage': config.include_storage
            },
            'processed_at': datetime.utcnow().isoformat()
        }
    
    async def priority_scan_async(self, notes: List[Dict[str, Any]], 
                                risk_threshold: str = 'high') -> Dict[str, Any]:
        """
        Async priority scan for identifying high-risk cases quickly
        
        Args:
            notes: List of notes to scan
            risk_threshold: Minimum risk level to flag
            
        Returns:
            Dict with priority findings and risk assessment
        """
        config = BatchAnalysisConfig(
            max_concurrent=15,  # Higher concurrency for priority scanning
            timeout_seconds=20,  # Shorter timeout for faster scanning
            include_icd_mapping=False,  # Skip ICD mapping for speed
            include_storage=False,  # Skip storage for speed
            chunk_size=30
        )
        
        start_time = time.time()
        
        logger.info(f"Starting priority scan of {len(notes)} notes for {risk_threshold}+ risk cases")
        
        # Process with async batch analysis
        batch_result = await self.batch_analyze_notes(notes, config)
        
        # Filter for priority cases
        priority_cases = []
        risk_levels = {
            'moderate': ['moderate', 'high', 'critical'],
            'high': ['high', 'critical'],
            'critical': ['critical']
        }
        
        target_levels = risk_levels.get(risk_threshold, ['critical'])
        
        for result in batch_result['results']:
            if result['success'] and result.get('data'):
                assessment = result['data'].get('overall_assessment', {})
                risk_level = assessment.get('risk_level', 'low')
                
                if risk_level in target_levels or assessment.get('requires_immediate_attention'):
                    priority_cases.append({
                        'note_id': result['note_id'],
                        'risk_level': risk_level,
                        'requires_immediate_attention': assessment.get('requires_immediate_attention', False),
                        'primary_concerns': assessment.get('primary_concerns', []),
                        'processing_time_ms': result.get('processing_time_ms', 0)
                    })
        
        scan_time_ms = int((time.time() - start_time) * 1000)
        
        return {
            'success': True,
            'priority_cases': priority_cases,
            'scan_summary': {
                'total_notes_scanned': len(notes),
                'priority_cases_found': len(priority_cases),
                'risk_threshold': risk_threshold,
                'scan_time_ms': scan_time_ms,
                'average_time_per_note_ms': scan_time_ms // len(notes) if notes else 0
            },
            'batch_processing_stats': batch_result['summary'],
            'scanned_at': datetime.utcnow().isoformat()
        }
    
    def __del__(self):
        """Cleanup executor when service is destroyed"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)