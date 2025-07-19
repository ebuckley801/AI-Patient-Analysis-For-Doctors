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
            response = self.supabase_service.client.table('icd_codes').select('*').execute()
            
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
        Get embedding for a clinical entity using Claude's text analysis
        For now, we'll use a simple approach - in production you'd want to use 
        the same embedding model used for ICD codes
        """
        # This is a placeholder - in practice you'd use the same embedding model
        # that was used to create the ICD code embeddings
        # For now, we'll create a simple text-based similarity
        
        # Use Claude to expand the entity text for better matching
        try:
            expanded_text = self._expand_entity_for_matching(entity_text)
            # For now, return a placeholder embedding - you'd replace this with actual embeddings
            # from the same model used for ICD codes (likely OpenAI's text-embedding-ada-002)
            return np.random.rand(1536)  # Placeholder embedding
        except Exception as e:
            logger.error(f"Error getting entity embedding: {str(e)}")
            return np.random.rand(1536)
    
    def _expand_entity_for_matching(self, entity_text: str) -> str:
        """Expand entity text with medical synonyms and context for better matching"""
        try:
            prompt = f"""Given the clinical entity "{entity_text}", provide expanded medical terminology including:
1. Medical synonyms
2. Related conditions
3. Standard medical terminology
4. ICD-10 relevant description

Return only the expanded text without explanation."""
            
            response = self.clinical_service.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=200,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Error expanding entity text: {str(e)}")
            return entity_text
    
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
                
                # Add entity_text to results for consistency
                for result in results:
                    result['entity_text'] = entity_text
                
                return results
                
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
        base_info = {
            'search_method': 'faiss' if self.use_faiss else 'numpy',
            'faiss_available': self.faiss_matcher is not None,
            'total_icd_codes': len(self._icd_codes_cache) if self._icd_codes_cache else 0,
            'embeddings_shape': self._embeddings_cache.shape if self._embeddings_cache is not None and self._embeddings_cache.size > 0 else 'empty',
            'cache_loaded': self._icd_codes_cache is not None and len(self._icd_codes_cache) > 0
        }
        
        # Add Faiss-specific information if available
        if self.use_faiss and self.faiss_matcher is not None:
            faiss_stats = self.faiss_matcher.get_index_stats()
            base_info.update({
                'faiss_stats': faiss_stats
            })
        
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