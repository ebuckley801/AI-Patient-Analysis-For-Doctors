import pytest
from unittest.mock import Mock, patch
from app.services.icd10_vector_matcher import ICD10VectorMatcher

class TestICD10VectorMatcher:
    
    def setup_method(self):
        """Set up test fixtures"""
        # Mock the Supabase service to avoid database calls in tests
        with patch('app.services.icd10_vector_matcher.SupabaseService') as mock_supabase:
            # Mock ICD codes data
            mock_response = Mock()
            mock_response.data = [
                {
                    'icd_10_code': 'I21.9',
                    'description': 'Acute myocardial infarction, unspecified',
                    'embedded_description': '[0.1, 0.2, 0.3]'  # Simplified embedding
                },
                {
                    'icd_10_code': 'R06.00',
                    'description': 'Dyspnea, unspecified',
                    'embedded_description': '[0.2, 0.1, 0.4]'
                },
                {
                    'icd_10_code': 'R50.9',
                    'description': 'Fever, unspecified',
                    'embedded_description': '[0.3, 0.4, 0.1]'
                }
            ]
            
            mock_supabase_instance = Mock()
            mock_supabase_instance.client.table.return_value.select.return_value.execute.return_value = mock_response
            mock_supabase.return_value = mock_supabase_instance
            
            self.matcher = ICD10VectorMatcher()
    
    def test_initialization(self):
        """Test that the matcher initializes correctly"""
        assert self.matcher is not None
        assert hasattr(self.matcher, '_icd_codes_cache')
        assert hasattr(self.matcher, '_embeddings_cache')
    
    def test_get_cache_info(self):
        """Test cache information retrieval"""
        cache_info = self.matcher.get_cache_info()
        
        assert 'total_icd_codes' in cache_info
        assert 'embeddings_shape' in cache_info
        assert 'cache_loaded' in cache_info
        assert isinstance(cache_info['total_icd_codes'], int)
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        import numpy as np
        
        # Test identical vectors
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([1, 0, 0])
        similarity = self.matcher.cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 1e-10
        
        # Test orthogonal vectors
        vec3 = np.array([1, 0, 0])
        vec4 = np.array([0, 1, 0])
        similarity = self.matcher.cosine_similarity(vec3, vec4)
        assert abs(similarity - 0.0) < 1e-10
        
        # Test zero vector
        vec5 = np.array([0, 0, 0])
        vec6 = np.array([1, 1, 1])
        similarity = self.matcher.cosine_similarity(vec5, vec6)
        assert similarity == 0.0
    
    def test_find_similar_icd_codes_simple(self):
        """Test simple text-based ICD code matching"""
        # Test exact match
        matches = self.matcher.find_similar_icd_codes_simple("myocardial infarction", top_k=3)
        
        assert len(matches) > 0
        assert matches[0]['icd_code'] == 'I21.9'
        assert 'similarity' in matches[0]
        assert matches[0]['similarity'] > 0
        assert matches[0]['entity_text'] == "myocardial infarction"
        assert matches[0]['match_type'] == 'text_based'
    
    def test_find_similar_icd_codes_simple_partial_match(self):
        """Test partial text matching"""
        matches = self.matcher.find_similar_icd_codes_simple("fever", top_k=2)
        
        assert len(matches) > 0
        fever_match = next((m for m in matches if 'fever' in m['description'].lower()), None)
        assert fever_match is not None
        assert fever_match['icd_code'] == 'R50.9'
    
    def test_find_similar_icd_codes_simple_no_match(self):
        """Test when no matches are found"""
        matches = self.matcher.find_similar_icd_codes_simple("completely unrelated term xyz123", top_k=3)
        
        # Should return empty list or very low similarity matches
        assert isinstance(matches, list)
        # If matches exist, they should have very low similarity
        for match in matches:
            assert match['similarity'] <= 0.5
    
    def test_map_clinical_entities_to_icd(self):
        """Test mapping clinical entities to ICD codes"""
        # Sample clinical analysis result
        clinical_entities = {
            'symptoms': [
                {
                    'entity': 'chest pain',
                    'severity': 'severe',
                    'confidence': 0.9
                },
                {
                    'entity': 'shortness of breath',
                    'severity': 'moderate',
                    'confidence': 0.8
                }
            ],
            'conditions': [
                {
                    'entity': 'myocardial infarction',
                    'status': 'active',
                    'confidence': 0.95
                }
            ],
            'procedures': [
                {
                    'entity': 'cardiac catheterization',
                    'status': 'planned',
                    'confidence': 0.9
                }
            ]
        }
        
        result = self.matcher.map_clinical_entities_to_icd(clinical_entities)
        
        # Check structure
        assert 'icd_mappings' in result
        assert 'conditions' in result['icd_mappings']
        assert 'symptoms' in result['icd_mappings']
        assert 'procedures' in result['icd_mappings']
        assert 'summary' in result['icd_mappings']
        
        # Check that conditions were mapped
        condition_mappings = result['icd_mappings']['conditions']
        assert len(condition_mappings) > 0
        
        mi_mapping = condition_mappings[0]
        assert mi_mapping['entity'] == 'myocardial infarction'
        assert mi_mapping['original_confidence'] == 0.95
        assert 'icd_matches' in mi_mapping
        assert 'best_match' in mi_mapping
        
        # Check symptoms mapping
        symptom_mappings = result['icd_mappings']['symptoms']
        assert len(symptom_mappings) > 0
        
        # Check summary statistics
        summary = result['icd_mappings']['summary']
        assert 'total_mappings' in summary
        assert 'high_confidence_mappings' in summary
        assert 'mapping_method' in summary
        assert summary['total_mappings'] > 0
    
    def test_get_icd_hierarchy(self):
        """Test ICD code hierarchy parsing"""
        # Test valid cardiovascular code
        hierarchy = self.matcher.get_icd_hierarchy('I21.9')
        
        assert hierarchy['icd_code'] == 'I21.9'
        assert hierarchy['category'] == 'I'
        assert 'circulatory system' in hierarchy['category_description'].lower()
        assert hierarchy['subcategory'] == 'I21'
        assert hierarchy['is_valid'] is True
        
        # Test respiratory code
        hierarchy = self.matcher.get_icd_hierarchy('J44.1')
        assert hierarchy['category'] == 'J'
        assert 'respiratory' in hierarchy['category_description'].lower()
        
        # Test invalid code
        hierarchy = self.matcher.get_icd_hierarchy('XYZ')
        assert hierarchy['is_valid'] is False
        
        # Test empty code
        hierarchy = self.matcher.get_icd_hierarchy('')
        assert 'error' in hierarchy
    
    def test_empty_icd_cache(self):
        """Test behavior when ICD cache is empty"""
        # Create matcher with empty cache
        with patch('app.services.icd10_vector_matcher.SupabaseService') as mock_supabase:
            mock_response = Mock()
            mock_response.data = []  # Empty data
            
            mock_supabase_instance = Mock()
            mock_supabase_instance.client.table.return_value.select.return_value.execute.return_value = mock_response
            mock_supabase.return_value = mock_supabase_instance
            
            empty_matcher = ICD10VectorMatcher()
            
            # Test that methods handle empty cache gracefully
            matches = empty_matcher.find_similar_icd_codes_simple("test", top_k=3)
            assert matches == []
            
            cache_info = empty_matcher.get_cache_info()
            assert cache_info['total_icd_codes'] == 0
            assert cache_info['cache_loaded'] is False
    
    @patch('app.services.icd10_vector_matcher.ClinicalAnalysisService')
    def test_expand_entity_for_matching(self, mock_clinical_service):
        """Test entity text expansion for better matching"""
        # Mock Claude response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "acute coronary syndrome, heart attack, myocardial infarction"
        
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_clinical_service.return_value.client = mock_client
        
        expanded = self.matcher._expand_entity_for_matching("heart attack")
        
        assert isinstance(expanded, str)
        assert len(expanded) > 0
        assert "myocardial infarction" in expanded.lower()
    
    def test_refresh_cache(self):
        """Test cache refresh functionality"""
        initial_cache_size = len(self.matcher._icd_codes_cache)
        
        # Refresh cache (should not error)
        self.matcher.refresh_cache()
        
        # Cache should still exist
        assert self.matcher._icd_codes_cache is not None
        # Size might be the same or different depending on mock data
        assert len(self.matcher._icd_codes_cache) >= 0