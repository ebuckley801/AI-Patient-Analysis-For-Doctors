"""
Multi-Modal Vector Similarity Service

Extends existing Faiss infrastructure to support cross-dataset similarity search across:
- Clinical text data (existing)
- ICD-10 codes (existing) 
- MIMIC-IV vital signs and trajectories
- UK Biobank genetic risk profiles
- FAERS adverse event patterns
- Clinical trial eligibility profiles

Built on top of existing FaissICD10VectorMatcher architecture.
"""

import asyncio
import logging
import numpy as np
import pickle
import json
import hashlib
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

from app.services.faiss_icd10_matcher import FaissICD10VectorMatcher
from app.services.supabase_service import SupabaseService
from app.services.patient_identity_service import PatientIdentityService

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

class ModalityType(Enum):
    """Types of medical data modalities"""
    CLINICAL_TEXT = "clinical_text"
    ICD_CODES = "icd_codes"
    VITAL_SIGNS = "vital_signs"
    GENETIC_PROFILE = "genetic_profile"
    ADVERSE_EVENTS = "adverse_events"
    TRIAL_ELIGIBILITY = "trial_eligibility"
    DEMOGRAPHICS = "demographics"

@dataclass
class MultiModalEmbedding:
    """Represents a multi-modal embedding with metadata"""
    vector: np.ndarray
    modality: ModalityType
    patient_id: str
    data_source: str
    content_hash: str
    metadata: Dict[str, Any]
    timestamp: datetime

@dataclass
class SimilarityResult:
    """Result of cross-modal similarity search"""
    patient_id: str
    modality: ModalityType
    data_source: str
    similarity_score: float
    content_summary: str
    metadata: Dict[str, Any]
    ranking: int

class MultiModalVectorService:
    """Service for multi-modal medical data vector similarity search"""
    
    def __init__(self, base_index_path: str = "data/indexes/multimodal"):
        """
        Initialize multi-modal vector service
        
        Args:
            base_index_path: Base directory for storing modality-specific indexes
        """
        if not FAISS_AVAILABLE:
            raise ImportError("Faiss is required for multi-modal vector service")
        
        self.base_index_path = Path(base_index_path)
        self.base_index_path.mkdir(parents=True, exist_ok=True)
        
        self.supabase = SupabaseService()
        self.identity_service = PatientIdentityService()
        
        # Existing ICD-10 matcher for backwards compatibility
        self.icd_matcher = FaissICD10VectorMatcher()
        
        # Modality-specific indexes
        self.modality_indexes = {}
        self.modality_metadata = {}
        
        # Common embedding dimension (standardize across modalities)
        self.embedding_dimension = 1536
        
        # Initialize all modality indexes
        self._initialize_modality_indexes()
    
    # ============================================================================
    # INITIALIZATION AND INDEX MANAGEMENT
    # ============================================================================
    
    def _initialize_modality_indexes(self):
        """Initialize Faiss indexes for each modality"""
        for modality in ModalityType:
            index_path = self.base_index_path / f"{modality.value}_index.bin"
            metadata_path = self.base_index_path / f"{modality.value}_metadata.pkl"
            
            try:
                if index_path.exists() and metadata_path.exists():
                    # Load existing index
                    self.modality_indexes[modality] = faiss.read_index(str(index_path))
                    with open(metadata_path, 'rb') as f:
                        self.modality_metadata[modality] = pickle.load(f)
                    logger.info(f"✅ Loaded existing {modality.value} index")
                else:
                    # Create new index
                    self.modality_indexes[modality] = faiss.IndexFlatL2(self.embedding_dimension)
                    self.modality_metadata[modality] = []
                    logger.info(f"🆕 Created new {modality.value} index")
                    
            except Exception as e:
                logger.error(f"❌ Error initializing {modality.value} index: {e}")
                self.modality_indexes[modality] = faiss.IndexFlatL2(self.embedding_dimension)
                self.modality_metadata[modality] = []
    
    def save_all_indexes(self):
        """Save all modality indexes to disk"""
        for modality in ModalityType:
            try:
                index_path = self.base_index_path / f"{modality.value}_index.bin"
                metadata_path = self.base_index_path / f"{modality.value}_metadata.pkl"
                
                # Save index
                if modality in self.modality_indexes:
                    faiss.write_index(self.modality_indexes[modality], str(index_path))
                
                # Save metadata
                if modality in self.modality_metadata:
                    with open(metadata_path, 'wb') as f:
                        pickle.dump(self.modality_metadata[modality], f)
                
                logger.info(f"💾 Saved {modality.value} index and metadata")
                
            except Exception as e:
                logger.error(f"❌ Error saving {modality.value} index: {e}")
    
    # ============================================================================
    # EMBEDDING GENERATION FOR DIFFERENT MODALITIES
    # ============================================================================
    
    async def generate_clinical_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for clinical text using existing infrastructure"""
        # This would integrate with existing Claude/OpenAI embedding generation
        # For now, return a placeholder that demonstrates the interface
        return np.random.randn(self.embedding_dimension).astype(np.float32)
    
    async def generate_vital_signs_embedding(self, vital_data: Dict[str, Any]) -> np.ndarray:
        """
        Generate embedding for vital signs trajectory
        
        Args:
            vital_data: Dictionary containing vital signs data over time
            
        Returns:
            Numpy array representing vital signs pattern
        """
        try:
            # Extract key vital signs features
            features = []
            
            # Statistical features for each vital type
            for vital_type in ['heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic', 
                             'temperature', 'respiratory_rate', 'oxygen_saturation']:
                values = vital_data.get(vital_type, [])
                if values:
                    # Basic statistics
                    features.extend([
                        np.mean(values),
                        np.std(values),
                        np.min(values),
                        np.max(values),
                        np.percentile(values, 25),
                        np.percentile(values, 75)
                    ])
                else:
                    features.extend([0.0] * 6)  # Fill with zeros if no data
            
            # Temporal features (trends, variability)
            if vital_data.get('timestamps'):
                # Calculate trends and patterns over time
                for vital_type in ['heart_rate', 'blood_pressure_systolic']:
                    values = vital_data.get(vital_type, [])
                    if len(values) > 1:
                        # Linear trend
                        x = np.arange(len(values))
                        trend = np.polyfit(x, values, 1)[0]
                        features.append(trend)
                        
                        # Coefficient of variation
                        cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
                        features.append(cv)
                    else:
                        features.extend([0.0, 0.0])
            
            # Pad or truncate to embedding dimension
            embedding = np.array(features[:self.embedding_dimension], dtype=np.float32)
            if len(embedding) < self.embedding_dimension:
                padding = np.zeros(self.embedding_dimension - len(embedding), dtype=np.float32)
                embedding = np.concatenate([embedding, padding])
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating vital signs embedding: {e}")
            return np.random.randn(self.embedding_dimension).astype(np.float32)
    
    async def generate_genetic_profile_embedding(self, genetic_data: Dict[str, Any]) -> np.ndarray:
        """
        Generate embedding for genetic risk profile
        
        Args:
            genetic_data: Dictionary containing genetic variants and risk scores
            
        Returns:
            Numpy array representing genetic risk profile
        """
        try:
            features = []
            
            # Polygenic risk scores for common diseases
            common_conditions = [
                'coronary_artery_disease', 'diabetes_type2', 'hypertension',
                'breast_cancer', 'prostate_cancer', 'alzheimer_disease',
                'depression', 'schizophrenia', 'inflammatory_bowel_disease'
            ]
            
            for condition in common_conditions:
                risk_score = genetic_data.get(f'prs_{condition}', 0.0)
                features.append(risk_score)
            
            # Major variant effects
            high_impact_variants = genetic_data.get('high_impact_variants', [])
            features.append(len(high_impact_variants))  # Count of high-impact variants
            
            # Pharmacogenomic markers
            pharmaco_markers = genetic_data.get('pharmacogenomic_variants', {})
            for drug_class in ['warfarin', 'clopidogrel', 'statins', 'antidepressants']:
                features.append(pharmaco_markers.get(drug_class, 0.0))
            
            # Ancestry components
            ancestry = genetic_data.get('ancestry_proportions', {})
            for population in ['european', 'african', 'asian', 'american', 'oceanic']:
                features.append(ancestry.get(population, 0.0))
            
            # Pad or truncate to embedding dimension
            embedding = np.array(features[:self.embedding_dimension], dtype=np.float32)
            if len(embedding) < self.embedding_dimension:
                padding = np.zeros(self.embedding_dimension - len(embedding), dtype=np.float32)
                embedding = np.concatenate([embedding, padding])
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating genetic profile embedding: {e}")
            return np.random.randn(self.embedding_dimension).astype(np.float32)
    
    async def generate_adverse_events_embedding(self, ae_data: Dict[str, Any]) -> np.ndarray:
        """
        Generate embedding for adverse event profile
        
        Args:
            ae_data: Dictionary containing adverse event history
            
        Returns:
            Numpy array representing adverse event pattern
        """
        try:
            features = []
            
            # Event frequency by system organ class
            soc_counts = ae_data.get('system_organ_class_counts', {})
            common_socs = [
                'cardiac_disorders', 'nervous_system_disorders', 'gastrointestinal_disorders',
                'skin_disorders', 'respiratory_disorders', 'psychiatric_disorders',
                'renal_disorders', 'hepatic_disorders', 'blood_disorders'
            ]
            
            for soc in common_socs:
                features.append(soc_counts.get(soc, 0))
            
            # Severity distribution
            severity_counts = ae_data.get('severity_distribution', {})
            for severity in ['mild', 'moderate', 'severe', 'life_threatening']:
                features.append(severity_counts.get(severity, 0))
            
            # Drug class associations
            drug_class_ae = ae_data.get('drug_class_associations', {})
            common_drug_classes = [
                'antibiotics', 'nsaids', 'antidepressants', 'antihypertensives',
                'anticoagulants', 'chemotherapy', 'immunosuppressants'
            ]
            
            for drug_class in common_drug_classes:
                features.append(drug_class_ae.get(drug_class, 0))
            
            # Temporal patterns
            features.append(ae_data.get('total_events', 0))
            features.append(ae_data.get('unique_drugs', 0))
            features.append(ae_data.get('avg_time_to_onset_days', 0))
            
            # Pad or truncate to embedding dimension
            embedding = np.array(features[:self.embedding_dimension], dtype=np.float32)
            if len(embedding) < self.embedding_dimension:
                padding = np.zeros(self.embedding_dimension - len(embedding), dtype=np.float32)
                embedding = np.concatenate([embedding, padding])
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating adverse events embedding: {e}")
            return np.random.randn(self.embedding_dimension).astype(np.float32)
    
    # ============================================================================
    # CROSS-MODAL SIMILARITY SEARCH
    # ============================================================================
    
    async def add_patient_embedding(self, unified_patient_id: str, modality: ModalityType,
                                  data_source: str, content: Dict[str, Any]) -> bool:
        """
        Add patient embedding to appropriate modality index
        
        Args:
            unified_patient_id: Patient's unified identifier
            modality: Type of medical data
            data_source: Source dataset
            content: Raw data content
            
        Returns:
            Success boolean
        """
        try:
            # Generate embedding based on modality
            if modality == ModalityType.CLINICAL_TEXT:
                embedding = await self.generate_clinical_text_embedding(content.get('text', ''))
            elif modality == ModalityType.VITAL_SIGNS:
                embedding = await self.generate_vital_signs_embedding(content)
            elif modality == ModalityType.GENETIC_PROFILE:
                embedding = await self.generate_genetic_profile_embedding(content)
            elif modality == ModalityType.ADVERSE_EVENTS:
                embedding = await self.generate_adverse_events_embedding(content)
            else:
                logger.warning(f"Unsupported modality: {modality}")
                return False
            
            # Create content hash for deduplication
            content_hash = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
            
            # Check if embedding already exists
            existing_hashes = [meta.get('content_hash') for meta in self.modality_metadata[modality]]
            if content_hash in existing_hashes:
                logger.info(f"Embedding already exists for {modality.value}")
                return True
            
            # Add to index
            embedding_norm = embedding.reshape(1, -1)
            faiss.normalize_L2(embedding_norm)
            self.modality_indexes[modality].add(embedding_norm)
            
            # Add metadata
            metadata = {
                'patient_id': unified_patient_id,
                'data_source': data_source,
                'content_hash': content_hash,
                'content_summary': self._generate_content_summary(content, modality),
                'timestamp': datetime.now().isoformat(),
                'modality': modality.value
            }
            
            self.modality_metadata[modality].append(metadata)
            
            # Store in database for persistence
            await self._store_embedding_in_database(unified_patient_id, modality, 
                                                  data_source, content_hash, 
                                                  embedding, metadata)
            
            logger.info(f"✅ Added {modality.value} embedding for patient {unified_patient_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding patient embedding: {e}")
            return False
    
    async def search_similar_patients(self, query_patient_id: str, 
                                    target_modality: ModalityType,
                                    source_modalities: List[ModalityType] = None,
                                    top_k: int = 10,
                                    min_similarity: float = 0.1) -> List[SimilarityResult]:
        """
        Find patients similar to query patient across modalities
        
        Args:
            query_patient_id: Patient to find similarities for
            target_modality: Modality to search in
            source_modalities: Modalities to use as query (if None, use all)
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of similar patients with similarity scores
        """
        try:
            if source_modalities is None:
                source_modalities = list(ModalityType)
            
            all_results = []
            
            # Search each source modality
            for source_modality in source_modalities:
                if source_modality not in self.modality_indexes:
                    continue
                
                # Find query patient's embeddings in source modality
                query_embeddings = await self._get_patient_embeddings(query_patient_id, source_modality)
                
                for query_embedding in query_embeddings:
                    # Search target modality
                    target_index = self.modality_indexes[target_modality]
                    target_metadata = self.modality_metadata[target_modality]
                    
                    if target_index.ntotal == 0:
                        continue
                    
                    # Perform similarity search
                    query_vector = query_embedding.reshape(1, -1)
                    faiss.normalize_L2(query_vector)
                    
                    search_k = min(top_k * 2, target_index.ntotal)
                    distances, indices = target_index.search(query_vector, search_k)
                    
                    # Convert to similarity results
                    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                        if idx >= 0 and idx < len(target_metadata):
                            similarity = 1.0 - (distance / 2.0)  # Convert L2 to cosine similarity
                            
                            if similarity >= min_similarity:
                                metadata = target_metadata[idx]
                                
                                # Skip same patient
                                if metadata['patient_id'] == query_patient_id:
                                    continue
                                
                                result = SimilarityResult(
                                    patient_id=metadata['patient_id'],
                                    modality=target_modality,
                                    data_source=metadata['data_source'],
                                    similarity_score=similarity,
                                    content_summary=metadata['content_summary'],
                                    metadata=metadata,
                                    ranking=len(all_results) + 1
                                )
                                
                                all_results.append(result)
            
            # Sort by similarity and deduplicate by patient
            all_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Deduplicate by patient ID, keeping highest similarity
            seen_patients = set()
            unique_results = []
            
            for result in all_results:
                if result.patient_id not in seen_patients:
                    seen_patients.add(result.patient_id)
                    result.ranking = len(unique_results) + 1
                    unique_results.append(result)
                    
                    if len(unique_results) >= top_k:
                        break
            
            return unique_results
            
        except Exception as e:
            logger.error(f"Error searching similar patients: {e}")
            return []
    
    async def cross_modal_patient_analysis(self, unified_patient_id: str) -> Dict[str, Any]:
        """
        Comprehensive cross-modal analysis for a patient
        
        Args:
            unified_patient_id: Patient to analyze
            
        Returns:
            Dictionary with cross-modal insights and similar patient cohorts
        """
        try:
            analysis = {
                'patient_id': unified_patient_id,
                'modalities_available': [],
                'cross_modal_similarities': {},
                'risk_cohorts': {},
                'clinical_insights': []
            }
            
            # Check which modalities have data for this patient
            for modality in ModalityType:
                embeddings = await self._get_patient_embeddings(unified_patient_id, modality)
                if embeddings:
                    analysis['modalities_available'].append(modality.value)
            
            # Find similar patients in each modality
            for target_modality in ModalityType:
                if target_modality.value in analysis['modalities_available']:
                    similar_patients = await self.search_similar_patients(
                        unified_patient_id, target_modality, top_k=5
                    )
                    
                    analysis['cross_modal_similarities'][target_modality.value] = [
                        {
                            'patient_id': result.patient_id,
                            'similarity': result.similarity_score,
                            'data_source': result.data_source
                        }
                        for result in similar_patients
                    ]
            
            # Identify risk cohorts (patients with similar genetic + clinical profiles)
            if (ModalityType.GENETIC_PROFILE.value in analysis['modalities_available'] and
                ModalityType.CLINICAL_TEXT.value in analysis['modalities_available']):
                
                genetic_similar = await self.search_similar_patients(
                    unified_patient_id, ModalityType.GENETIC_PROFILE, top_k=20
                )
                
                clinical_similar = await self.search_similar_patients(
                    unified_patient_id, ModalityType.CLINICAL_TEXT, top_k=20
                )
                
                # Find intersection - patients similar in both modalities
                genetic_ids = set(result.patient_id for result in genetic_similar)
                clinical_ids = set(result.patient_id for result in clinical_similar)
                risk_cohort_ids = genetic_ids & clinical_ids
                
                analysis['risk_cohorts']['genetic_clinical'] = list(risk_cohort_ids)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in cross-modal analysis: {e}")
            return {'error': str(e)}
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    async def _get_patient_embeddings(self, patient_id: str, 
                                    modality: ModalityType) -> List[np.ndarray]:
        """Get all embeddings for a patient in a specific modality"""
        try:
            # Query database for stored embeddings
            embeddings = self.supabase.client.table('multimodal_embeddings')\
                .select('*')\
                .eq('unified_patient_id', patient_id)\
                .eq('data_type', modality.value)\
                .execute()
            
            embedding_vectors = []
            for record in embeddings.data:
                vector_data = record.get('embedding_vector', [])
                if vector_data:
                    vector = np.array(vector_data, dtype=np.float32)
                    embedding_vectors.append(vector)
            
            return embedding_vectors
            
        except Exception as e:
            logger.error(f"Error getting patient embeddings: {e}")
            return []
    
    async def _store_embedding_in_database(self, patient_id: str, modality: ModalityType,
                                         data_source: str, content_hash: str,
                                         embedding: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Store embedding in database for persistence"""
        try:
            record = {
                'unified_patient_id': patient_id,
                'data_source': data_source,
                'data_type': modality.value,
                'content_hash': content_hash,
                'content_summary': metadata.get('content_summary', ''),
                'embedding_vector': embedding.tolist(),
                'embedding_model': 'multimodal_v1',
                'vector_dimension': len(embedding)
            }
            
            self.supabase.client.table('multimodal_embeddings')\
                .upsert(record, on_conflict='content_hash,embedding_model')\
                .execute()
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing embedding in database: {e}")
            return False
    
    def _generate_content_summary(self, content: Dict[str, Any], 
                                modality: ModalityType) -> str:
        """Generate human-readable summary of content"""
        try:
            if modality == ModalityType.CLINICAL_TEXT:
                return content.get('text', '')[:200] + "..." if len(content.get('text', '')) > 200 else content.get('text', '')
            
            elif modality == ModalityType.VITAL_SIGNS:
                vital_types = list(content.keys())
                return f"Vital signs: {', '.join(vital_types[:5])}"
            
            elif modality == ModalityType.GENETIC_PROFILE:
                variant_count = len(content.get('variants', []))
                risk_conditions = list(content.get('risk_scores', {}).keys())
                return f"{variant_count} variants, risk for: {', '.join(risk_conditions[:3])}"
            
            elif modality == ModalityType.ADVERSE_EVENTS:
                event_count = content.get('total_events', 0)
                drug_count = content.get('unique_drugs', 0)
                return f"{event_count} adverse events from {drug_count} drugs"
            
            else:
                return json.dumps(content)[:200]
                
        except Exception as e:
            logger.error(f"Error generating content summary: {e}")
            return "Content summary unavailable"
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the multi-modal service"""
        stats = {
            'modalities': {},
            'total_patients': set(),
            'cross_modal_coverage': {},
            'service_status': 'operational'
        }
        
        try:
            for modality in ModalityType:
                if modality in self.modality_indexes:
                    index = self.modality_indexes[modality]
                    metadata = self.modality_metadata.get(modality, [])
                    
                    patient_ids = set(meta['patient_id'] for meta in metadata)
                    stats['total_patients'].update(patient_ids)
                    
                    stats['modalities'][modality.value] = {
                        'total_embeddings': index.ntotal if hasattr(index, 'ntotal') else 0,
                        'unique_patients': len(patient_ids),
                        'data_sources': list(set(meta['data_source'] for meta in metadata))
                    }
            
            stats['total_unique_patients'] = len(stats['total_patients'])
            stats['total_patients'] = list(stats['total_patients'])
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting service stats: {e}")
            stats['service_status'] = f'error: {str(e)}'
            return stats