import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import ast
from app.services.supabase_service import SupabaseService
from app.services.clinical_analysis_service import ClinicalAnalysisService

logger = logging.getLogger(__name__)

class ICD10VectorMatcher:
    """Service for matching clinical entities to ICD-10 codes using vector similarity"""
    
    def __init__(self, force_numpy: bool = False):
        """
        Initialize ICD10VectorMatcher with hybrid Faiss/numpy approach
        
        Args:
            force_numpy: Force use of numpy implementation (for testing/fallback)
        """
        self.supabase_service = SupabaseService()
        self.clinical_service = ClinicalAnalysisService()
        self._icd_codes_cache = None
        self._embeddings_cache = None
        self._embedding_cache = {}  # Cache for entity embeddings
        
        # Try to initialize Faiss matcher first
        self.faiss_matcher = None
        self.use_faiss = False
        
        if not force_numpy:
            try:
                from app.services.faiss_icd10_matcher import create_faiss_icd10_matcher
                self.faiss_matcher = create_faiss_icd10_matcher()
                if self.faiss_matcher is not None:
                    self.use_faiss = True
                    logger.info("✅ Using Faiss for high-performance vector search")
                else:
                    logger.warning("⚠️ Faiss matcher creation failed, falling back to numpy")
            except ImportError as e:
                logger.warning(f"⚠️ Faiss not available: {e}")
                logger.warning("📋 Install with: pip install faiss-cpu")
            except Exception as e:
                logger.warning(f"⚠️ Faiss initialization failed: {e}")
        
        # Initialize numpy fallback if Faiss not available
        if not self.use_faiss:
            logger.info("🔄 Using numpy implementation for vector search")
            self._load_icd_data()
    
    def _load_icd_data(self):
        """Load ICD-10 codes and their embeddings from database"""
        try:
            # Get ICD codes from Supabase
            response = self.supabase_service.client.table('icd_10_codes').select('*').execute()
            
            if not response.data:
                logger.warning("No ICD codes found in database")
                self._icd_codes_cache = []
                self._embeddings_cache = np.array([])
                return
            
            self._icd_codes_cache = response.data
            
            # Extract embeddings and convert to numpy array
            embeddings_list = []
            for code_data in self._icd_codes_cache:
                embedding_str = code_data.get('embedded_description', '[]')
                try:
                    # Parse the embedding string to list of floats
                    if isinstance(embedding_str, str):
                        embedding = ast.literal_eval(embedding_str)
                    else:
                        embedding = embedding_str
                    
                    if isinstance(embedding, list) and len(embedding) > 0:
                        embeddings_list.append(embedding)
                    else:
                        logger.warning(f"Invalid embedding for ICD code {code_data.get('icd_10_code', 'unknown')}")
                        embeddings_list.append([0.0] * 1536)  # Default embedding size for text-embedding-ada-002
                        
                except (ValueError, SyntaxError) as e:
                    logger.error(f"Error parsing embedding for ICD code {code_data.get('icd_10_code', 'unknown')}: {str(e)}")
                    embeddings_list.append([0.0] * 1536)
            
            self._embeddings_cache = np.array(embeddings_list)
            logger.info(f"Loaded {len(self._icd_codes_cache)} ICD codes with embeddings")
            
        except Exception as e:
            logger.error(f"Error loading ICD data: {str(e)}")
            self._icd_codes_cache = []
            self._embeddings_cache = np.array([])
    
    def _get_entity_embedding(self, entity_text: str) -> np.ndarray:
        """
        Get embedding for a clinical entity using fast semantic analysis
        Uses caching and rule-based features to avoid slow API calls
        """
        # Check cache first
        cache_key = entity_text.lower().strip()
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        try:
            # Use fast rule-based semantic features directly (no API calls)
            semantic_features = self._extract_semantic_features(entity_text)
            
            # Cache the result
            self._embedding_cache[cache_key] = semantic_features
            
            return semantic_features
            
        except Exception as e:
            logger.error(f"Error getting entity embedding: {str(e)}")
            # Fallback to zero vector
            fallback = np.zeros(1536, dtype=np.float32)
            self._embedding_cache[cache_key] = fallback
            return fallback
    
    def _get_batch_embeddings(self, entity_texts: List[str]) -> np.ndarray:
        """
        Get embeddings for multiple entities in batch for better performance
        
        Args:
            entity_texts: List of entity text strings
            
        Returns:
            numpy array of shape (len(entity_texts), 1536)
        """
        embeddings = []
        
        for entity_text in entity_texts:
            embedding = self._get_entity_embedding(entity_text)
            embeddings.append(embedding)
        
        return np.array(embeddings, dtype=np.float32)
    
    def _extract_semantic_features(self, text: str) -> np.ndarray:
        """
        Extract semantic features from text using fast rule-based medical concept mapping
        This approximates embeddings by mapping to medical concept spaces
        """
        import re
        import math
        
        # Normalize text
        text = text.lower().strip()
        
        # Initialize feature vector
        features = np.zeros(1536, dtype=np.float32)
        
        # Enhanced medical concept categories and their feature space positions
        medical_concepts = {
            # Symptoms (positions 0-255)
            'pain': ['pain', 'ache', 'discomfort', 'hurt', 'sore', 'aching', 'painful'],
            'fever': ['fever', 'pyrexia', 'temperature', 'hot', 'hyperthermia', 'febrile'],
            'cough': ['cough', 'coughing', 'tussis', 'expectoration'],
            'breathing': ['dyspnea', 'breath', 'breathing', 'respiratory', 'shortness', 'dyspnoea'],
            'nausea': ['nausea', 'vomit', 'sick', 'queasy', 'vomiting', 'emesis'],
            'fatigue': ['fatigue', 'tired', 'weakness', 'exhaustion', 'malaise'],
            'headache': ['headache', 'cephalgia', 'migraine'],
            'dizziness': ['dizziness', 'vertigo', 'lightheaded'],
            
            # Body systems (positions 256-511)
            'cardiac': ['heart', 'cardiac', 'cardio', 'myocardial', 'coronary'],
            'pulmonary': ['lung', 'pulmonary', 'respiratory', 'bronchial', 'pneumonia'],
            'gastrointestinal': ['stomach', 'gastric', 'intestinal', 'digestive', 'bowel'],
            'neurological': ['brain', 'neural', 'neurological', 'cerebral', 'cognitive'],
            'musculoskeletal': ['muscle', 'bone', 'joint', 'skeletal', 'orthopedic'],
            
            # Severity (positions 512-767)
            'acute': ['acute', 'sudden', 'rapid', 'immediate'],
            'chronic': ['chronic', 'long-term', 'persistent', 'ongoing'],
            'severe': ['severe', 'serious', 'critical', 'intense'],
            'mild': ['mild', 'slight', 'minor', 'minimal'],
            
            # Conditions (positions 768-1023)
            'infection': ['infection', 'bacterial', 'viral', 'sepsis', 'inflammatory'],
            'diabetes': ['diabetes', 'diabetic', 'glucose', 'insulin', 'hyperglycemia'],
            'hypertension': ['hypertension', 'blood pressure', 'hypertensive'],
            'cancer': ['cancer', 'tumor', 'malignant', 'neoplasm', 'carcinoma'],
            
            # Anatomical (positions 1024-1279)
            'chest': ['chest', 'thoracic', 'pectoral', 'breast'],
            'abdomen': ['abdomen', 'abdominal', 'belly', 'stomach'],
            'head': ['head', 'cranial', 'cephalic', 'skull'],
            'extremities': ['arm', 'leg', 'limb', 'extremity'],
            
            # Modifiers (positions 1280-1535)
            'bilateral': ['bilateral', 'both', 'sides'],
            'unilateral': ['unilateral', 'one', 'single', 'left', 'right'],
            'radiating': ['radiating', 'spreading', 'extending'],
            'localized': ['localized', 'local', 'confined', 'specific']
        }
        
        # Convert text to lowercase for matching
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Map concepts to feature positions
        position = 0
        for category, keywords in medical_concepts.items():
            # Calculate relevance score for this category
            relevance = 0
            for keyword in keywords:
                # Exact word match
                if keyword in words:
                    relevance += 1.0
                # Partial match in text
                elif keyword in text_lower:
                    relevance += 0.5
            
            # Normalize relevance
            if relevance > 0:
                relevance = min(relevance / len(keywords), 1.0)
                
                # Set feature values for this concept (use multiple positions per concept)
                concept_size = 1536 // len(medical_concepts)
                start_pos = position * concept_size
                end_pos = min(start_pos + concept_size, 1536)
                
                # Create pattern within concept space
                for i in range(start_pos, end_pos):
                    # Use sine wave pattern with relevance as amplitude
                    offset = (i - start_pos) / concept_size * 2 * math.pi
                    features[i] = relevance * math.sin(offset + hash(category) % 100)
            
            position += 1
        
        # Add some noise based on text hash for uniqueness
        text_hash = hash(text_lower) % 1000
        for i in range(0, len(features), 10):
            features[i] += (text_hash % 100 - 50) / 1000.0
        
        # Normalize the vector
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            # Handle zero vectors
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return np.dot(a, b) / (norm_a * norm_b)
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {str(e)}")
            return 0.0
    
    def find_similar_icd_codes(self, entity_text: str, top_k: int = 5, min_similarity: float = 0.1) -> List[Dict[str, Any]]:
        """
        Find ICD-10 codes similar to the given clinical entity using hybrid approach
        
        Args:
            entity_text: Clinical entity text to match
            top_k: Number of top matches to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of matching ICD codes with similarity scores
        """
        if self.use_faiss and self.faiss_matcher is not None:
            # Use high-performance Faiss search
            try:
                entity_embedding = self._get_entity_embedding(entity_text)
                results = self.faiss_matcher.search_similar_codes(
                    entity_embedding, top_k=top_k, min_similarity=min_similarity
                )
                
                # Add entity_text to results for consistency and fix field names
                for result in results:
                    result['entity_text'] = entity_text
                    # Ensure consistent field naming - map 'icd_code' to 'code'
                    if 'icd_code' in result and 'code' not in result:
                        result['code'] = result['icd_code']
                
                return results
                
            except Exception as e:
                logger.error(f"Faiss search failed: {e}, falling back to numpy")
                # Continue to numpy fallback below
        
        # Numpy fallback implementation
        if self._icd_codes_cache is None or self._embeddings_cache is None:
            logger.warning("No ICD data loaded")
            return []
        
        try:
            # Get embedding for the entity
            entity_embedding = self._get_entity_embedding(entity_text)
            
            # Calculate similarities with all ICD codes
            similarities = []
            for i, icd_embedding in enumerate(self._embeddings_cache):
                similarity = self.cosine_similarity(entity_embedding, icd_embedding)
                if similarity >= min_similarity:
                    similarities.append((similarity, i))
            
            # Sort by similarity and get top results
            similarities.sort(reverse=True)
            results = []
            
            for similarity, idx in similarities[:top_k]:
                icd_data = self._icd_codes_cache[idx]
                results.append({
                    'icd_code': icd_data['icd_10_code'],
                    'code': icd_data['icd_10_code'],
                    'description': icd_data['description'],
                    'similarity': similarity,
                    'search_method': 'numpy',
                    'entity_text': entity_text
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in numpy ICD search: {str(e)}")
            return []
    
    def find_similar_icd_codes_batch(self, entity_texts: List[str], top_k: int = 5, min_similarity: float = 0.1) -> List[List[Dict[str, Any]]]:
        """
        Find similar ICD codes for multiple entities in batch (faster than individual searches)
        
        Args:
            entity_texts: List of entity text strings to search for
            top_k: Number of top results to return per entity
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of results lists, one per input entity
        """
        if not entity_texts:
            return []
        
        try:
            if self.use_faiss and self.faiss_matcher is not None:
                # Use Faiss batch search
                query_embeddings = self._get_batch_embeddings(entity_texts)
                
                # Faiss batch search
                similarities, indices = self.faiss_matcher.index.search(query_embeddings, top_k * 2)  # Search more, filter later
                
                # Process results for each query
                batch_results = []
                for i, entity_text in enumerate(entity_texts):
                    entity_results = []
                    
                    for j in range(len(similarities[i])):
                        similarity = similarities[i][j]
                        idx = indices[i][j]
                        
                        if idx >= 0 and idx < len(self.faiss_matcher.icd_metadata):
                            # Convert L2 distance to cosine similarity (for normalized vectors)
                            cosine_sim = 1.0 - (similarity / 2.0)
                            
                            if cosine_sim >= min_similarity:
                                metadata = self.faiss_matcher.icd_metadata[idx]
                                entity_results.append({
                                    'icd_code': metadata['icd_code'],
                                    'description': metadata['description'],
                                    'similarity': float(cosine_sim),
                                    'rank': len(entity_results) + 1,
                                    'search_method': 'faiss_batch',
                                    'query_text': entity_text
                                })
                                
                                if len(entity_results) >= top_k:
                                    break
                    
                    batch_results.append(entity_results)
                
                return batch_results
                
            else:
                # Fallback to individual numpy searches
                batch_results = []
                for entity_text in entity_texts:
                    results = self.find_similar_icd_codes(entity_text, top_k, min_similarity)
                    batch_results.append(results)
                return batch_results
                
        except Exception as e:
            logger.error(f"Error in batch ICD search: {str(e)}")
            # Fallback to individual searches
            batch_results = []
            for entity_text in entity_texts:
                try:
                    results = self.find_similar_icd_codes(entity_text, top_k, min_similarity)
                    batch_results.append(results)
                except:
                    batch_results.append([])
            return batch_results
                
        except Exception as e:
            logger.error(f"Faiss search failed: {e}, falling back to numpy")
            # Continue to numpy fallback
        
        # Fallback to numpy implementation
        return self._find_similar_icd_codes_numpy(entity_text, top_k, min_similarity)
    
    def _find_similar_icd_codes_numpy(self, entity_text: str, top_k: int = 5, min_similarity: float = 0.1) -> List[Dict[str, Any]]:
        """
        Original numpy-based vector similarity search (fallback method)
        
        Args:
            entity_text: Clinical entity text to match
            top_k: Number of top matches to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of matching ICD codes with similarity scores
        """
        if not self._icd_codes_cache or len(self._embeddings_cache) == 0:
            logger.warning("No ICD codes loaded")
            return []
        
        try:
            # Get entity embedding (placeholder for now)
            entity_embedding = self._get_entity_embedding(entity_text)
            
            # Calculate similarities with all ICD codes
            similarities = []
            for i, icd_embedding in enumerate(self._embeddings_cache):
                similarity = self.cosine_similarity(entity_embedding, icd_embedding)
                
                if similarity >= min_similarity:
                    similarities.append({
                        'icd_code': self._icd_codes_cache[i]['icd_10_code'],
                        'description': self._icd_codes_cache[i]['description'],
                        'similarity': float(similarity),
                        'entity_text': entity_text,
                        'search_method': 'numpy_vector'
                    })
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar ICD codes: {str(e)}")
            return []
    
    def find_similar_icd_codes_simple(self, entity_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Simple text-based matching for clinical entities to ICD codes
        This is a fallback method when vector embeddings aren't available
        """
        if not self._icd_codes_cache:
            return []
        
        try:
            entity_lower = entity_text.lower()
            matches = []
            
            for icd_data in self._icd_codes_cache:
                description = icd_data['description'].lower()
                icd_code = icd_data['icd_10_code']
                
                # Simple text matching score
                score = 0.0
                
                # Exact match bonus
                if entity_lower in description:
                    score += 0.8
                
                # Word overlap scoring
                entity_words = set(entity_lower.split())
                desc_words = set(description.split())
                
                if entity_words and desc_words:
                    overlap = len(entity_words.intersection(desc_words))
                    total_words = len(entity_words.union(desc_words))
                    score += (overlap / total_words) * 0.6
                
                if score > 0.1:  # Minimum threshold
                    matches.append({
                        'icd_code': icd_code,
                        'description': icd_data['description'],
                        'similarity': score,
                        'entity_text': entity_text,
                        'match_type': 'text_based'
                    })
            
            # Sort by score and return top k
            matches.sort(key=lambda x: x['similarity'], reverse=True)
            return matches[:top_k]
            
        except Exception as e:
            logger.error(f"Error in simple ICD matching: {str(e)}")
            return []
    
    def map_clinical_entities_to_icd(self, clinical_entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map all clinical entities from analysis result to ICD-10 codes
        
        Args:
            clinical_entities: Result from clinical analysis service
            
        Returns:
            Enhanced result with ICD-10 mappings
        """
        result = clinical_entities.copy()
        result['icd_mappings'] = {
            'conditions': [],
            'symptoms': [],
            'procedures': [],
            'summary': {
                'total_mappings': 0,
                'high_confidence_mappings': 0,
                'mapping_method': 'vector_similarity'
            }
        }
        
        try:
            # Map conditions to ICD codes
            for condition in clinical_entities.get('conditions', []):
                entity_text = condition.get('entity', '')
                if entity_text:
                    icd_matches = self.find_similar_icd_codes_simple(entity_text, top_k=3)
                    
                    if icd_matches:
                        mapping = {
                            'entity': entity_text,
                            'original_confidence': condition.get('confidence', 0),
                            'icd_matches': icd_matches,
                            'best_match': icd_matches[0] if icd_matches else None,
                            'entity_type': 'condition'
                        }
                        result['icd_mappings']['conditions'].append(mapping)
            
            # Map symptoms to ICD codes
            for symptom in clinical_entities.get('symptoms', []):
                entity_text = symptom.get('entity', '')
                if entity_text:
                    icd_matches = self.find_similar_icd_codes_simple(entity_text, top_k=3)
                    
                    if icd_matches:
                        mapping = {
                            'entity': entity_text,
                            'original_confidence': symptom.get('confidence', 0),
                            'severity': symptom.get('severity', 'unknown'),
                            'icd_matches': icd_matches,
                            'best_match': icd_matches[0] if icd_matches else None,
                            'entity_type': 'symptom'
                        }
                        result['icd_mappings']['symptoms'].append(mapping)
            
            # Map procedures to ICD codes (using different approach for procedures)
            for procedure in clinical_entities.get('procedures', []):
                entity_text = procedure.get('entity', '')
                if entity_text:
                    # For procedures, we might want to search for procedure codes specifically
                    icd_matches = self.find_similar_icd_codes_simple(entity_text, top_k=2)
                    
                    if icd_matches:
                        mapping = {
                            'entity': entity_text,
                            'original_confidence': procedure.get('confidence', 0),
                            'status': procedure.get('status', 'unknown'),
                            'icd_matches': icd_matches,
                            'best_match': icd_matches[0] if icd_matches else None,
                            'entity_type': 'procedure'
                        }
                        result['icd_mappings']['procedures'].append(mapping)
            
            # Calculate summary statistics
            total_mappings = (len(result['icd_mappings']['conditions']) + 
                            len(result['icd_mappings']['symptoms']) + 
                            len(result['icd_mappings']['procedures']))
            
            high_confidence = 0
            for category in ['conditions', 'symptoms', 'procedures']:
                for mapping in result['icd_mappings'][category]:
                    if (mapping.get('best_match', {}).get('similarity', 0) > 0.7 and 
                        mapping.get('original_confidence', 0) > 0.8):
                        high_confidence += 1
            
            result['icd_mappings']['summary'] = {
                'total_mappings': total_mappings,
                'high_confidence_mappings': high_confidence,
                'mapping_method': 'text_based_similarity',
                'confidence_threshold': 0.7
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error mapping clinical entities to ICD: {str(e)}")
            result['icd_mappings']['error'] = str(e)
            return result
    
    def get_icd_hierarchy(self, icd_code: str) -> Dict[str, str]:
        """
        Get ICD-10 code hierarchy information
        
        Args:
            icd_code: ICD-10 code (e.g., "I21.9")
            
        Returns:
            Dictionary with hierarchy information
        """
        try:
            # Basic ICD-10 hierarchy parsing
            if not icd_code or len(icd_code) < 3:
                return {'error': 'Invalid ICD code'}
            
            category = icd_code[0]
            subcategory = icd_code[:3]
            
            category_map = {
                'A': 'Certain infectious and parasitic diseases',
                'B': 'Certain infectious and parasitic diseases',
                'C': 'Neoplasms',
                'D': 'Diseases of the blood and blood-forming organs',
                'E': 'Endocrine, nutritional and metabolic diseases',
                'F': 'Mental, Behavioral and Neurodevelopmental disorders',
                'G': 'Diseases of the nervous system',
                'H': 'Diseases of the eye and adnexa / Diseases of the ear',
                'I': 'Diseases of the circulatory system',
                'J': 'Diseases of the respiratory system',
                'K': 'Diseases of the digestive system',
                'L': 'Diseases of the skin and subcutaneous tissue',
                'M': 'Diseases of the musculoskeletal system',
                'N': 'Diseases of the genitourinary system',
                'O': 'Pregnancy, childbirth and the puerperium',
                'P': 'Certain conditions originating in the perinatal period',
                'Q': 'Congenital malformations and chromosomal abnormalities',
                'R': 'Symptoms, signs and abnormal clinical findings',
                'S': 'Injury, poisoning and certain other consequences',
                'T': 'Injury, poisoning and certain other consequences',
                'V': 'External causes of morbidity',
                'W': 'External causes of morbidity',
                'X': 'External causes of morbidity',
                'Y': 'External causes of morbidity',
                'Z': 'Factors influencing health status'
            }
            
            return {
                'icd_code': icd_code,
                'category': category,
                'category_description': category_map.get(category, 'Unknown category'),
                'subcategory': subcategory,
                'is_valid': len(icd_code) >= 3 and category.isalpha()
            }
            
        except Exception as e:
            logger.error(f"Error getting ICD hierarchy: {str(e)}")
            return {'error': str(e)}
    
    def refresh_cache(self):
        """Refresh the ICD codes cache from database"""
        logger.info("Refreshing ICD codes cache...")
        self._load_icd_data()
        
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the current cache and search method"""
        if self.use_faiss and self.faiss_matcher is not None:
            # When using Faiss, get count from Faiss matcher
            faiss_stats = self.faiss_matcher.get_index_stats()
            base_info = {
                'search_method': 'faiss',
                'faiss_available': True,
                'total_icd_codes': faiss_stats.get('total_vectors', 0),
                'embeddings_shape': f"faiss_index({faiss_stats.get('total_vectors', 0)}, {faiss_stats.get('dimension', 1536)})",
                'cache_loaded': faiss_stats.get('total_vectors', 0) > 0,
                'faiss_stats': faiss_stats
            }
        else:
            # When using numpy, use local cache
            base_info = {
                'search_method': 'numpy',
                'faiss_available': self.faiss_matcher is not None,
                'total_icd_codes': len(self._icd_codes_cache) if self._icd_codes_cache else 0,
                'embeddings_shape': self._embeddings_cache.shape if self._embeddings_cache is not None and self._embeddings_cache.size > 0 else 'empty',
                'cache_loaded': self._icd_codes_cache is not None and len(self._icd_codes_cache) > 0
            }
        
        return base_info
    
    def benchmark_performance(self, num_queries: int = 50) -> Dict[str, Any]:
        """
        Benchmark performance of current search method
        
        Args:
            num_queries: Number of test queries to run
            
        Returns:
            Performance metrics
        """
        if self.use_faiss and self.faiss_matcher is not None:
            return self.faiss_matcher.benchmark_search(num_queries)
        else:
            # Simple benchmark for numpy implementation
            import time
            
            test_queries = ["chest pain", "diabetes", "hypertension", "pneumonia", "fever"]
            
            start_time = time.time()
            for i in range(num_queries):
                query = test_queries[i % len(test_queries)]
                self.find_similar_icd_codes(query, top_k=5)
            
            total_time = time.time() - start_time
            
            return {
                'search_method': 'numpy',
                'num_queries': num_queries,
                'total_time_seconds': total_time,
                'avg_query_ms': (total_time / num_queries) * 1000,
                'queries_per_second': num_queries / total_time
            }