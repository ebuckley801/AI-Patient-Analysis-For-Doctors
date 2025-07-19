"""
Comprehensive Test Suite for Multi-Modal Medical Data Integration

Tests all components of the multi-modal medical data integration system:
- Data ingestion pipelines
- Patient identity resolution
- Vector similarity search
- Data fusion services
- Clinical trials matching
- API endpoints

Uses pytest with async support and comprehensive mocking.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, Mock, AsyncMock
import numpy as np
from typing import Dict, List, Any

# Import services to test
from app.services.multimodal_data_service import (
    MultiModalDataService, DataIngestionResult, PatientIdentity
)
from app.services.multimodal_vector_service import (
    MultiModalVectorService, ModalityType, SimilarityResult
)
from app.services.patient_identity_service import (
    PatientIdentityService, IdentityFeatures, MatchResult
)
from app.services.data_fusion_service import (
    DataFusionService, PatientProfile, FusedInsight, Evidence, EvidenceLevel, RiskLevel
)
from app.services.clinical_trials_matching_service import (
    ClinicalTrialsMatchingService, TrialMatch, EligibilityStatus, MatchingMethod
)

# Test fixtures
@pytest.fixture
def mock_supabase_service():
    """Mock Supabase service for testing"""
    mock_service = Mock()
    mock_client = Mock()
    mock_table = Mock()
    
    mock_service.client = mock_client
    mock_client.table.return_value = mock_table
    mock_table.select.return_value = mock_table
    mock_table.insert.return_value = mock_table
    mock_table.upsert.return_value = mock_table
    mock_table.eq.return_value = mock_table
    mock_table.execute.return_value = Mock(data=[])
    
    return mock_service

@pytest.fixture
def sample_patient_demographics():
    """Sample patient demographics for testing"""
    return {
        'first_name': 'John',
        'last_name': 'Doe',
        'birth_date': '1970-01-15',
        'gender': 'M',
        'age': 53,
        'ethnicity': 'White',
        'country': 'US'
    }

@pytest.fixture
def sample_mimic_admission():
    """Sample MIMIC-IV admission data"""
    return {
        'subject_id': 12345,
        'hadm_id': 67890,
        'age': 65,
        'gender': 'M',
        'ethnicity': 'WHITE',
        'admission_type': 'EMERGENCY',
        'admission_location': 'EMERGENCY ROOM ADMIT',
        'discharge_location': 'HOME',
        'insurance': 'Medicare',
        'admittime': '2023-01-15 10:30:00',
        'dischtime': '2023-01-20 14:15:00',
        'diagnosis': 'Acute myocardial infarction'
    }

@pytest.fixture
def sample_biobank_participant():
    """Sample UK Biobank participant data"""
    return {
        'eid': 123456,
        'birth_year': 1960,
        'sex': 'Female',
        'ethnic_background': 'British',
        'assessment_centre': 'Newcastle',
        'genotyping_array': 'UK BiLEVE Axiom Array'
    }

@pytest.fixture
def sample_genetic_data():
    """Sample genetic variant data"""
    return {
        'eid': 123456,
        'variant_type': 'snp',
        'variant_id': 'rs123456',
        'chromosome': '1',
        'position': 123456789,
        'allele_1': 'A',
        'allele_2': 'G',
        'genotype': 'AG',
        'risk_score': 0.75,
        'confidence': 0.95,
        'associated_conditions': ['cardiovascular disease', 'diabetes']
    }

@pytest.fixture
def sample_clinical_trial():
    """Sample clinical trial data"""
    return {
        'nct_id': 'NCT12345678',
        'title': 'Study of Novel Cardiovascular Treatment',
        'brief_summary': 'Testing new treatment for heart disease',
        'study_type': 'INTERVENTIONAL',
        'phase': 'PHASE3',
        'status': 'RECRUITING',
        'conditions': ['Cardiovascular Disease', 'Myocardial Infarction'],
        'eligibility_criteria': 'Inclusion Criteria:\n- Age 18-75 years\n- Diagnosed with cardiovascular disease\nExclusion Criteria:\n- Pregnancy\n- Severe kidney disease',
        'minimum_age': '18 Years',
        'maximum_age': '75 Years',
        'gender': 'ALL',
        'locations': [{'facility': 'Test Hospital', 'city': 'Boston', 'state': 'MA'}]
    }

# ============================================================================
# MULTIMODAL DATA SERVICE TESTS
# ============================================================================

class TestMultiModalDataService:
    """Test suite for MultiModalDataService"""
    
    @pytest.fixture
    def service(self, mock_supabase_service):
        """Create service instance with mocked dependencies"""
        with patch('app.services.multimodal_data_service.SupabaseService', return_value=mock_supabase_service):
            return MultiModalDataService()
    
    def test_create_unified_patient(self, service, sample_patient_demographics):
        """Test unified patient creation"""
        # Mock existing patient check
        service.supabase.client.table().select().eq().execute.return_value = Mock(data=[])
        
        # Mock patient creation
        mock_result = Mock(data=[{'unified_patient_id': 'test-uuid'}])
        service.supabase.client.table().insert().execute.return_value = mock_result
        
        # Test patient creation
        unified_id = service.create_unified_patient(
            sample_patient_demographics, 'local', 'patient_123'
        )
        
        assert unified_id == 'test-uuid'
    
    @pytest.mark.asyncio
    async def test_ingest_mimic_admissions(self, service, sample_mimic_admission):
        """Test MIMIC-IV admissions ingestion"""
        # Mock unified patient creation
        service.create_unified_patient = Mock(return_value='test-uuid')
        
        # Mock database insertion
        service.supabase.client.table().upsert().execute.return_value = Mock()
        
        # Test ingestion
        result = await service.ingest_mimic_admissions([sample_mimic_admission])
        
        assert isinstance(result, DataIngestionResult)
        assert result.success is True
        assert result.records_processed == 1
        assert len(result.errors) == 0
    
    @pytest.mark.asyncio
    async def test_ingest_biobank_participants(self, service, sample_biobank_participant):
        """Test UK Biobank participants ingestion"""
        # Mock unified patient creation
        service.create_unified_patient = Mock(return_value='test-uuid')
        
        # Mock database insertion
        service.supabase.client.table().upsert().execute.return_value = Mock()
        
        # Test ingestion
        result = await service.ingest_biobank_participants([sample_biobank_participant])
        
        assert isinstance(result, DataIngestionResult)
        assert result.success is True
        assert result.records_processed == 1
    
    @pytest.mark.asyncio
    async def test_fetch_clinical_trials(self, service):
        """Test clinical trials API fetching"""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'studies': [
                {
                    'protocolSection': {
                        'identificationModule': {'nctId': 'NCT12345678', 'briefTitle': 'Test Study'},
                        'statusModule': {'overallStatus': 'RECRUITING'},
                        'conditionsModule': {'conditions': ['Heart Disease']}
                    }
                }
            ]
        }
        mock_response.raise_for_status = Mock()

        with patch('requests.get', return_value=mock_response):
            trials = await service.fetch_clinical_trials(['heart disease'])

        assert len(trials) == 1
        assert trials[0]['protocolSection']['identificationModule']['nctId'] == 'NCT12345678'

# ============================================================================
# PATIENT IDENTITY SERVICE TESTS
# ============================================================================

class TestPatientIdentityService:
    """Test suite for PatientIdentityService"""
    
    @pytest.fixture
    def service(self, mock_supabase_service):
        """Create service instance with mocked dependencies"""
        with patch('app.services.patient_identity_service.SupabaseService', return_value=mock_supabase_service):
            return PatientIdentityService()
    
    def test_extract_identity_features(self, service, sample_patient_demographics):
        """Test identity feature extraction"""
        features = service._extract_identity_features(sample_patient_demographics)
        
        assert isinstance(features, IdentityFeatures)
        assert len(features.name_tokens) > 0
        assert features.birth_date is not None
        assert features.gender == 'm'
    
    def test_exact_matching(self, service, sample_patient_demographics):
        """Test exact patient matching"""
        # Mock database queries
        service.supabase.client.table().select().eq().execute.return_value = Mock(data=[])
        
        # Mock unified patients
        mock_patients = Mock(data=[
            {
                'unified_patient_id': 'existing-uuid',
                'demographics': sample_patient_demographics
            }
        ])
        service.supabase.client.table().select().execute.return_value = mock_patients
        
        features = service._extract_identity_features(sample_patient_demographics)
        match = service._find_exact_match(features, 'local', 'patient_123')
        
        assert match is not None
        assert match.confidence_score == 1.0
    
    def test_probabilistic_matching(self, service, sample_patient_demographics):
        """Test probabilistic patient matching"""
        # Create slightly different demographics
        different_demographics = sample_patient_demographics.copy()
        different_demographics['first_name'] = 'Jonathan'  # Variation of John
        
        # Mock unified patients
        mock_patients = Mock(data=[
            {
                'unified_patient_id': 'existing-uuid',
                'demographics': different_demographics
            }
        ])
        service.supabase.client.table().select().execute.return_value = mock_patients
        
        features = service._extract_identity_features(sample_patient_demographics)
        match = service._find_probabilistic_match(features, 'local', 'patient_123')
        
        assert match is not None
        assert 0.5 <= match.confidence_score < 1.0  # Should be high but not perfect match

# ============================================================================
# MULTIMODAL VECTOR SERVICE TESTS
# ============================================================================

class TestMultiModalVectorService:
    """Test suite for MultiModalVectorService"""
    
    @pytest.fixture
    def service(self, mock_supabase_service):
        """Create service instance with mocked dependencies"""
        with patch('app.services.multimodal_vector_service.SupabaseService', return_value=mock_supabase_service):
            with patch('app.services.multimodal_vector_service.PatientIdentityService'):
                with patch('faiss.IndexFlatL2'):
                    return MultiModalVectorService()
    
    @pytest.mark.asyncio
    async def test_generate_clinical_text_embedding(self, service):
        """Test clinical text embedding generation"""
        embedding = await service.generate_clinical_text_embedding("Patient has chest pain and shortness of breath")
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (service.embedding_dimension,)
    
    @pytest.mark.asyncio
    async def test_generate_vital_signs_embedding(self, service):
        """Test vital signs embedding generation"""
        vital_data = {
            'heart_rate': [80, 85, 90, 88],
            'blood_pressure_systolic': [120, 125, 130, 128],
            'temperature': [98.6, 99.0, 98.8, 98.5],
            'timestamps': ['2023-01-01T10:00:00', '2023-01-01T14:00:00', '2023-01-01T18:00:00', '2023-01-01T22:00:00']
        }
        
        embedding = await service.generate_vital_signs_embedding(vital_data)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (service.embedding_dimension,)
        assert not np.all(embedding == 0)  # Should have non-zero values
    
    @pytest.mark.asyncio
    async def test_generate_genetic_profile_embedding(self, service):
        """Test genetic profile embedding generation"""
        genetic_data = {
            'prs_coronary_artery_disease': 0.75,
            'prs_diabetes_type2': 0.45,
            'prs_hypertension': 0.60,
            'high_impact_variants': ['BRCA1_mutation', 'LDLR_variant'],
            'pharmacogenomic_variants': {'warfarin': 0.8, 'statins': 0.6},
            'ancestry_proportions': {'european': 0.8, 'african': 0.1, 'asian': 0.1}
        }
        
        embedding = await service.generate_genetic_profile_embedding(genetic_data)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (service.embedding_dimension,)
    
    @pytest.mark.asyncio
    async def test_add_patient_embedding(self, service):
        """Test adding patient embedding to index"""
        # Mock embedding storage
        service._store_embedding_in_database = AsyncMock(return_value=True)
        
        content = {'text': 'Patient has diabetes and hypertension'}
        result = await service.add_patient_embedding(
            'test-patient-id', ModalityType.CLINICAL_TEXT, 'local', content
        )
        
        assert result is True
    
    def test_get_service_stats(self, service):
        """Test service statistics retrieval"""
        stats = service.get_service_stats()
        
        assert isinstance(stats, dict)
        assert 'modalities' in stats
        assert 'total_unique_patients' in stats

# ============================================================================
# DATA FUSION SERVICE TESTS
# ============================================================================

class TestDataFusionService:
    """Test suite for DataFusionService"""
    
    @pytest.fixture
    def service(self, mock_supabase_service):
        """Create service instance with mocked dependencies"""
        with patch('app.services.data_fusion_service.SupabaseService', return_value=mock_supabase_service):
            with patch('app.services.data_fusion_service.MultiModalVectorService'):
                with patch('app.services.data_fusion_service.PatientIdentityService'):
                    with patch('app.services.data_fusion_service.ClinicalAnalysisService'):
                        return DataFusionService()
    
    @pytest.mark.asyncio
    async def test_gather_patient_data(self, service):
        """Test patient data gathering from multiple sources"""
        # Mock database responses
        unified_patient_mock = Mock(data=[{
            'demographics': {'age': 45, 'gender': 'M'},
            'data_sources': ['local', 'mimic']
        }])
        service.supabase.client.table().select().eq().execute.return_value = unified_patient_mock
        
        raw_data = await service._gather_patient_data('test-patient-id')
        
        assert isinstance(raw_data, dict)
        assert 'demographics' in raw_data
        assert 'clinical_entities' in raw_data
        assert 'mimic_admissions' in raw_data
    
    @pytest.mark.asyncio
    async def test_extract_evidence_from_data(self, service):
        """Test evidence extraction from raw data"""
        raw_data = {
            'clinical_entities': [{
                'entity_type': 'condition',
                'entity_text': 'hypertension',
                'confidence': 0.9,
                'severity': 'moderate',
                'created_at': datetime.now(timezone.utc).isoformat()
            }],
            'biobank_genetics': [{
                'variant_id': 'rs123456',
                'variant_type': 'SNP',
                'risk_score': 0.8,
                'confidence': 0.95,
                'associated_conditions': ['cardiovascular disease'],
                'created_at': datetime.now(timezone.utc).isoformat()
            }]
        }
        
        evidence = await service._extract_evidence_from_data(raw_data)
        
        assert isinstance(evidence, list)
        assert len(evidence) >= 2  # Should have clinical and genetic evidence
        assert all(isinstance(e, Evidence) for e in evidence)
    
    def test_group_evidence_by_domain(self, service):
        """Test evidence grouping by clinical domain"""
        evidence_list = [
            Evidence(
                source_modality=ModalityType.CLINICAL_TEXT,
                data_source='local',
                evidence_type='diagnostic',
                finding='cardiac condition: chest pain',
                confidence=0.8,
                supporting_data={},
                timestamp=datetime.now(timezone.utc)
            ),
            Evidence(
                source_modality=ModalityType.VITAL_SIGNS,
                data_source='mimic',
                evidence_type='prognostic',
                finding='heart rate elevated',
                confidence=0.9,
                supporting_data={},
                timestamp=datetime.now(timezone.utc)
            )
        ]
        
        grouped = service._group_evidence_by_domain(evidence_list)
        
        assert isinstance(grouped, dict)
        assert 'cardiovascular' in grouped
        assert len(grouped['cardiovascular']) == 2

# ============================================================================
# CLINICAL TRIALS MATCHING SERVICE TESTS
# ============================================================================

class TestClinicalTrialsMatchingService:
    """Test suite for ClinicalTrialsMatchingService"""
    
    @pytest.fixture
    def service(self, mock_supabase_service):
        """Create service instance with mocked dependencies"""
        with patch('app.services.clinical_trials_matching_service.SupabaseService', return_value=mock_supabase_service):
            with patch('app.services.clinical_trials_matching_service.MultiModalVectorService'):
                with patch('app.services.clinical_trials_matching_service.PatientIdentityService'):
                    with patch('app.services.clinical_trials_matching_service.DataFusionService'):
                        return ClinicalTrialsMatchingService()
    
    def test_parse_eligibility_criteria(self, service):
        """Test eligibility criteria parsing"""
        criteria_text = """
        Inclusion Criteria:
        - Age 18-75 years
        - Diagnosed with cardiovascular disease
        
        Exclusion Criteria:
        - Pregnancy
        - Severe kidney disease
        """
        
        inclusion, exclusion = service._parse_eligibility_criteria(criteria_text)
        
        assert len(inclusion) >= 2
        assert len(exclusion) >= 2
        assert any('age' in criterion.lower() for criterion in inclusion)
        assert any('pregnancy' in criterion.lower() for criterion in exclusion)
    
    def test_assess_single_criterion_age(self, service):
        """Test single criterion assessment for age"""
        # Mock patient profile
        mock_profile = Mock()
        mock_profile.demographic_summary = {'age': 45}
        
        criterion = "Age 18-75 years"
        assessment = service._assess_single_criterion(mock_profile, criterion, is_inclusion=True)
        
        assert assessment['meets_criterion'] is True
        assert assessment['confidence'] > 0.8
        assert assessment['data_available'] is True
    
    def test_assess_single_criterion_gender(self, service):
        """Test single criterion assessment for gender"""
        # Mock patient profile
        mock_profile = Mock()
        mock_profile.demographic_summary = {'gender': 'male'}
        
        criterion = "Male participants only"
        assessment = service._assess_single_criterion(mock_profile, criterion, is_inclusion=True)
        
        assert assessment['meets_criterion'] is True
        assert assessment['confidence'] > 0.9
    
    @pytest.mark.asyncio
    async def test_generate_patient_embedding(self, service):
        """Test patient embedding generation for trial matching"""
        # Mock patient profile
        mock_profile = Mock()
        mock_profile.demographic_summary = {'age': 45, 'gender': 'male'}
        mock_profile.clinical_summary = {'entity_counts': {'conditions': 3, 'medications': 5}}
        mock_profile.genetic_risk_profile = {'available': False}
        
        embedding = await service._generate_patient_embedding(mock_profile)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (128,)  # Standard embedding dimension
    
    @pytest.mark.asyncio
    async def test_generate_trial_embedding(self, service, sample_clinical_trial):
        """Test trial embedding generation"""
        embedding = await service._generate_trial_embedding(sample_clinical_trial)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (128,)
    
    def test_calculate_cosine_similarity(self, service):
        """Test cosine similarity calculation"""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        vec3 = np.array([1.0, 0.0, 0.0])
        
        # Orthogonal vectors should have 0 similarity
        similarity_orthogonal = service._calculate_cosine_similarity(vec1, vec2)
        assert abs(similarity_orthogonal - 0.0) < 1e-6
        
        # Identical vectors should have 1.0 similarity
        similarity_identical = service._calculate_cosine_similarity(vec1, vec3)
        assert abs(similarity_identical - 1.0) < 1e-6

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestMultiModalIntegration:
    """Integration tests for multi-modal system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_patient_analysis(self, mock_supabase_service):
        """Test complete end-to-end patient analysis workflow"""
        # This would test the complete workflow from data ingestion to analysis
        # For brevity, we'll test the key integration points
        
        with patch('app.services.multimodal_data_service.SupabaseService', return_value=mock_supabase_service):
            with patch('app.services.data_fusion_service.SupabaseService', return_value=mock_supabase_service):
                
                # Initialize services
                data_service = MultiModalDataService()
                fusion_service = DataFusionService()
                
                # Mock patient creation
                data_service.create_unified_patient = Mock(return_value='test-patient-uuid')
                
                # Mock data gathering
                fusion_service._gather_patient_data = AsyncMock(return_value={
                    'demographics': {'age': 45, 'gender': 'M'},
                    'clinical_entities': [],
                    'mimic_admissions': [],
                    'biobank_genetics': []
                })
                
                # Mock profile storage
                fusion_service._store_patient_profile = AsyncMock(return_value=True)
                
                # Test workflow
                patient_id = 'test-patient-uuid'
                profile = await fusion_service.create_comprehensive_patient_profile(patient_id)
                
                assert isinstance(profile, PatientProfile)
                assert profile.unified_patient_id == patient_id

# ============================================================================
# API ENDPOINT TESTS
# ============================================================================

class TestMultiModalAPIEndpoints:
    """Test suite for multi-modal API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client for API endpoints"""
        from app import create_app
        app = create_app()
        app.config['TESTING'] = True
        
        with app.test_client() as client:
            yield client
    
    def test_multimodal_health_endpoint(self, client):
        """Test multi-modal health check endpoint"""
        response = client.get('/api/multimodal/health')
        
        assert response.status_code in [200, 503]  # May be unavailable in test
        
        data = response.get_json()
        assert 'status' in data
        assert 'timestamp' in data
        assert 'services' in data
    
    def test_similarity_search_validation(self, client):
        """Test similarity search input validation"""
        # Test missing required fields
        response = client.post('/api/multimodal/similarity/patients', json={})
        
        assert response.status_code == 400
        
        data = response.get_json()
        assert data['success'] is False
        assert 'error' in data
    
    def test_identity_resolution_validation(self, client):
        """Test identity resolution input validation"""
        # Test missing required fields
        response = client.post('/api/multimodal/identity/resolve', json={})
        
        assert response.status_code == 400
        
        data = response.get_json()
        assert data['success'] is False

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance tests for multi-modal system"""
    
    @pytest.mark.asyncio
    async def test_batch_ingestion_performance(self, mock_supabase_service):
        """Test performance of batch data ingestion"""
        with patch('app.services.multimodal_data_service.SupabaseService', return_value=mock_supabase_service):
            service = MultiModalDataService()
            service.create_unified_patient = Mock(return_value='test-uuid')
            
            # Create large batch of test data
            batch_data = []
            for i in range(100):  # Test with 100 records
                batch_data.append({
                    'subject_id': i,
                    'hadm_id': i + 10000,
                    'age': 50 + (i % 30),
                    'gender': 'M' if i % 2 else 'F',
                    'diagnosis': f'Test diagnosis {i}'
                })
            
            start_time = datetime.now(timezone.utc)
            result = await service.ingest_mimic_admissions(batch_data)
            end_time = datetime.now(timezone.utc)
            
            processing_time = (end_time - start_time).total_seconds()
            
            assert result.success
            assert result.records_processed == 100
            assert processing_time < 30  # Should complete within 30 seconds
            assert processing_time / 100 < 0.5  # Less than 0.5 seconds per record

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test error handling across multi-modal services"""
    
    def test_database_connection_error(self, mock_supabase_service):
        """Test handling of database connection errors"""
        # Mock database error
        mock_supabase_service.client.table().select().execute.side_effect = Exception("Database connection failed")
        
        with patch('app.services.multimodal_data_service.SupabaseService', return_value=mock_supabase_service):
            service = MultiModalDataService()
            
            # Should handle database errors gracefully
            with pytest.raises(Exception):
                service.create_unified_patient({}, 'local', 'test')
    
    @pytest.mark.asyncio
    async def test_api_timeout_error(self, mock_supabase_service):
        """Test handling of API timeout errors"""
        with patch('app.services.multimodal_data_service.SupabaseService', return_value=mock_supabase_service):
            service = MultiModalDataService()
            
            # Mock requests timeout
            with patch('requests.get', side_effect=Exception("Request timeout")):
                trials = await service.fetch_clinical_trials(['test'])
                
                # Should return empty list on error, not crash
                assert trials == []

# ============================================================================
# TEST UTILITIES
# ============================================================================

def create_mock_patient_profile(patient_id: str = 'test-patient') -> PatientProfile:
    """Create a mock patient profile for testing"""
    return PatientProfile(
        unified_patient_id=patient_id,
        demographic_summary={'age': 45, 'gender': 'M'},
        clinical_summary={'entity_counts': {'conditions': 3}},
        genetic_risk_profile={'available': False},
        vital_signs_patterns={'available': False},
        adverse_event_history={'available': False},
        trial_eligibility={'available': False},
        risk_stratification={'overall': RiskLevel.MODERATE},
        fused_insights=[],
        data_completeness={'overall': 0.6},
        last_updated=datetime.now(timezone.utc)
    )

def create_mock_evidence(modality: ModalityType, finding: str) -> Evidence:
    """Create mock evidence for testing"""
    return Evidence(
        source_modality=modality,
        data_source='test',
        evidence_type='diagnostic',
        finding=finding,
        confidence=0.8,
        supporting_data={},
        timestamp=datetime.now(timezone.utc)
    )

# Run tests if executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v'])