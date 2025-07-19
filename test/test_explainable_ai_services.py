"""Comprehensive tests for explainable AI services"""
import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock
import json
from datetime import datetime

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import services to test
from app.services.pubmed_service import PubMedService, RateLimiter
from app.services.pubmed_cache_service import PubMedCacheService
from app.services.uncertainty_service import UncertaintyCalculator
from app.services.pathway_explorer import TreatmentPathwayExplorer
from app.services.explainable_clinical_service import ExplainableClinicalService

class TestRateLimiter(unittest.TestCase):
    """Test rate limiting functionality"""
    
    def setUp(self):
        self.rate_limiter = RateLimiter(requests_per_second=2)
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter initializes correctly"""
        self.assertEqual(self.rate_limiter.requests_per_second, 2)
        self.assertEqual(self.rate_limiter.min_interval, 0.5)
        self.assertEqual(self.rate_limiter.last_request_time, 0)
    
    def test_wait_if_needed_first_request(self):
        """Test first request doesn't need to wait"""
        import time
        start_time = time.time()
        self.rate_limiter.wait_if_needed()
        end_time = time.time()
        
        # First request should be immediate
        self.assertLess(end_time - start_time, 0.1)

class TestPubMedService(unittest.TestCase):
    """Test PubMed API integration service"""
    
    def setUp(self):
        self.pubmed_service = PubMedService()
    
    def test_build_base_params(self):
        """Test building base API parameters"""
        params = self.pubmed_service._build_base_params()
        
        self.assertIn('email', params)
        self.assertIn('tool', params)
        self.assertEqual(params['tool'], 'PatientAnalysis')
    
    def test_build_clinical_query(self):
        """Test building clinical queries"""
        query = self.pubmed_service.build_clinical_query(
            condition="diabetes",
            treatment="metformin",
            study_type="clinical_trial"
        )
        
        self.assertIn("diabetes", query)
        self.assertIn("metformin", query)
        self.assertIn("clinical trial[Publication Type]", query)
    
    @patch('requests.get')
    def test_make_request_success(self, mock_get):
        """Test successful API request"""
        mock_response = Mock()
        mock_response.text = "<xml>test response</xml>"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.pubmed_service._make_request('search', {'term': 'diabetes'})
        
        self.assertEqual(result, "<xml>test response</xml>")
        mock_get.assert_called_once()
    
    @patch('requests.get')
    def test_make_request_failure(self, mock_get):
        """Test failed API request"""
        mock_get.side_effect = Exception("Network error")
        
        result = self.pubmed_service._make_request('search', {'term': 'diabetes'})
        
        self.assertIsNone(result)
    
    def test_parse_search_response(self):
        """Test parsing search response XML"""
        xml_response = """
        <eSearchResult>
            <IdList>
                <Id>12345</Id>
                <Id>67890</Id>
            </IdList>
        </eSearchResult>
        """
        
        pmids = self.pubmed_service._parse_search_response(xml_response)
        
        self.assertEqual(pmids, ['12345', '67890'])
    
    def test_parse_search_response_invalid_xml(self):
        """Test parsing invalid XML"""
        invalid_xml = "<invalid>xml"
        
        pmids = self.pubmed_service._parse_search_response(invalid_xml)
        
        self.assertEqual(pmids, [])

class TestPubMedCacheService(unittest.TestCase):
    """Test PubMed cache service"""
    
    def setUp(self):
        self.cache_service = PubMedCacheService()
        # Mock the supabase client
        self.cache_service.supabase = Mock()
    
    def test_generate_query_hash(self):
        """Test query hash generation"""
        hash1 = PubMedCacheService.generate_query_hash("diabetes", 10)
        hash2 = PubMedCacheService.generate_query_hash("diabetes", 10)
        hash3 = PubMedCacheService.generate_query_hash("diabetes", 20)
        
        # Same query should produce same hash
        self.assertEqual(hash1, hash2)
        # Different query should produce different hash
        self.assertNotEqual(hash1, hash3)
    
    def test_cache_search_results_success(self):
        """Test successful cache storage"""
        mock_result = Mock()
        mock_result.data = [{'id': 1}]
        self.cache_service.supabase.client.table.return_value.upsert.return_value.execute.return_value = mock_result
        
        success = self.cache_service.cache_search_results(
            query_hash="test_hash",
            query="test query",
            results=[{'pmid': '12345'}]
        )
        
        self.assertTrue(success)
    
    def test_cache_search_results_failure(self):
        """Test failed cache storage"""
        self.cache_service.supabase.client.table.return_value.upsert.return_value.execute.side_effect = Exception("DB error")
        
        success = self.cache_service.cache_search_results(
            query_hash="test_hash",
            query="test query",
            results=[{'pmid': '12345'}]
        )
        
        self.assertFalse(success)

class TestUncertaintyCalculator(unittest.TestCase):
    """Test uncertainty quantification service"""
    
    def setUp(self):
        self.uncertainty_calculator = UncertaintyCalculator()
    
    def test_calculate_confidence_intervals(self):
        """Test confidence interval calculation"""
        entity = {
            'entity': 'diabetes',
            'confidence': 0.8,
            'text_span': 'patient has diabetes'
        }
        
        result = self.uncertainty_calculator.calculate_confidence_intervals(entity)
        
        self.assertIn('entity', result)
        self.assertIn('confidence_interval', result)
        self.assertIn('lower', result['confidence_interval'])
        self.assertIn('upper', result['confidence_interval'])
        self.assertEqual(result['entity'], 'diabetes')
    
    def test_categorize_confidence(self):
        """Test confidence categorization"""
        high_conf = self.uncertainty_calculator._categorize_confidence(0.9)
        medium_conf = self.uncertainty_calculator._categorize_confidence(0.7)
        low_conf = self.uncertainty_calculator._categorize_confidence(0.3)
        
        self.assertEqual(high_conf, 'high')
        self.assertEqual(medium_conf, 'medium')
        self.assertEqual(low_conf, 'low')
    
    def test_assess_diagnostic_uncertainty_empty_entities(self):
        """Test uncertainty assessment with no entities"""
        result = self.uncertainty_calculator.assess_diagnostic_uncertainty([])
        
        self.assertEqual(result['overall_confidence'], 0.0)
        self.assertIn('no_entities_detected', result['uncertainty_sources'])
        self.assertEqual(result['recommendation'], 'insufficient_data')
    
    def test_assess_diagnostic_uncertainty_with_entities(self):
        """Test uncertainty assessment with entities"""
        entities = {
            'symptoms': [
                {'entity': 'fever', 'confidence': 0.9},
                {'entity': 'cough', 'confidence': 0.7}
            ],
            'conditions': [
                {'entity': 'pneumonia', 'confidence': 0.8}
            ]
        }
        
        result = self.uncertainty_calculator.assess_diagnostic_uncertainty(entities)
        
        self.assertGreater(result['overall_confidence'], 0)
        self.assertIn('confidence_range', result)
        self.assertIsInstance(result['uncertainty_sources'], list)
    
    def test_create_uncertainty_visualization(self):
        """Test uncertainty visualization data creation"""
        analysis = {
            'overall_confidence': 0.8,
            'confidence_range': {'lower': 0.7, 'upper': 0.9},
            'entity_count': 5,
            'high_confidence_entities': 3,
            'low_confidence_entities': 1,
            'uncertainty_sources': ['some_uncertainty']
        }
        
        viz_data = self.uncertainty_calculator.create_uncertainty_visualization(analysis)
        
        self.assertIn('confidence_distribution', viz_data)
        self.assertIn('uncertainty_heatmap', viz_data)
        self.assertIn('evidence_strength', viz_data)
        self.assertIn('chart_data', viz_data)

class TestTreatmentPathwayExplorer(unittest.TestCase):
    """Test treatment pathway exploration service"""
    
    def setUp(self):
        self.pathway_explorer = TreatmentPathwayExplorer()
    
    def test_generate_alternative_pathways(self):
        """Test pathway generation"""
        primary_diagnosis = {
            'entity': 'hypertension',
            'confidence': 0.8
        }
        patient_context = {
            'age': 45,
            'gender': 'M',
            'medical_history': []
        }
        
        pathways = self.pathway_explorer.generate_alternative_pathways(
            primary_diagnosis=primary_diagnosis,
            patient_context=patient_context,
            max_pathways=3
        )
        
        self.assertIsInstance(pathways, list)
        self.assertLessEqual(len(pathways), 3)
        
        if pathways:
            pathway = pathways[0]
            self.assertIn('pathway_id', pathway)
            self.assertIn('treatment_sequence', pathway)
            self.assertIn('evidence_strength', pathway)
    
    def test_rank_pathways_by_evidence(self):
        """Test pathway ranking"""
        pathways = [
            {
                'evidence_strength': 0.7,
                'supporting_studies': ['study1'],
                'estimated_outcomes': {'success_rate': 0.8, 'side_effect_rate': 0.1}
            },
            {
                'evidence_strength': 0.9,
                'supporting_studies': ['study1', 'study2'],
                'estimated_outcomes': {'success_rate': 0.9, 'side_effect_rate': 0.05}
            }
        ]
        
        ranked = self.pathway_explorer.rank_pathways_by_evidence(pathways)
        
        self.assertEqual(len(ranked), 2)
        # Higher evidence should be ranked first
        self.assertGreater(ranked[0]['evidence_score'], ranked[1]['evidence_score'])
        self.assertEqual(ranked[0]['rank'], 1)
        self.assertEqual(ranked[1]['rank'], 2)
    
    def test_check_contraindications(self):
        """Test contraindication checking"""
        pathway = {
            'treatment_sequence': [
                {'intervention': 'ace_inhibitor'},
                {'intervention': 'aspirin'}
            ]
        }
        patient_context = {
            'age': 15,  # Pediatric
            'allergies': ['aspirin'],
            'medical_history': ['kidney_disease'],
            'current_medications': []
        }
        
        contraindications = self.pathway_explorer.check_contraindications(
            pathway=pathway,
            patient_context=patient_context
        )
        
        self.assertIsInstance(contraindications, list)
        # Should detect aspirin contraindication in pediatric patient
        aspirin_contraindications = [c for c in contraindications if 'aspirin' in c]
        self.assertTrue(len(aspirin_contraindications) > 0)
    
    def test_get_patient_specific_notes(self):
        """Test patient-specific note generation"""
        pathways = [
            {'evidence_score': 0.9},
            {'evidence_score': 0.5}  # Low evidence pathway
        ]
        patient_context = {
            'age': 70,  # Geriatric
            'gender': 'F',
            'medical_history': ['diabetes']
        }
        
        notes = self.pathway_explorer.get_patient_specific_notes(
            pathways=pathways,
            patient_context=patient_context
        )
        
        self.assertIsInstance(notes, list)
        # Should include geriatric considerations
        geriatric_notes = [n for n in notes if 'geriatric' in n.lower()]
        self.assertTrue(len(geriatric_notes) > 0)

class TestExplainableClinicalService(unittest.TestCase):
    """Test explainable clinical analysis service"""
    
    def setUp(self):
        # Mock dependencies
        with patch('app.services.explainable_clinical_service.PubMedService'), \
             patch('app.services.explainable_clinical_service.PubMedCacheService'), \
             patch('app.services.explainable_clinical_service.UncertaintyCalculator'), \
             patch('app.services.explainable_clinical_service.AnalysisStorageService'):
            self.explainable_service = ExplainableClinicalService()
    
    @patch.object(ExplainableClinicalService, 'extract_clinical_entities')
    def test_analyze_with_explanation_basic(self, mock_extract):
        """Test basic explainable analysis"""
        # Mock the base extraction
        mock_extract.return_value = {
            'symptoms': [{'entity': 'fever', 'confidence': 0.8}],
            'conditions': [{'entity': 'flu', 'confidence': 0.7}],
            'overall_assessment': {'risk_level': 'low'}
        }
        
        # Mock other methods
        self.explainable_service.gather_literature_evidence = Mock(return_value=[])
        self.explainable_service.uncertainty_calculator.assess_diagnostic_uncertainty = Mock(
            return_value={'overall_confidence': 0.8}
        )
        self.explainable_service.storage_service.create_analysis_session = Mock(return_value='session_123')
        self.explainable_service.storage_service.store_reasoning_chain = Mock(return_value=True)
        self.explainable_service.storage_service.store_uncertainty_analysis = Mock(return_value=True)
        
        result = self.explainable_service.analyze_with_explanation(
            patient_note="Patient has fever and feels unwell",
            patient_context={'age': 30, 'gender': 'M'}
        )
        
        self.assertIn('analysis', result)
        self.assertIn('explanation', result)
        self.assertIn('session_id', result)
        self.assertIn('processing_time', result)
        
        explanation = result['explanation']
        self.assertIn('reasoning_chain', explanation)
        self.assertIn('evidence_sources', explanation)
        self.assertIn('uncertainty_analysis', explanation)
    
    def test_generate_reasoning_chain(self):
        """Test reasoning chain generation"""
        entities = {
            'symptoms': [
                {'entity': 'chest pain', 'confidence': 0.9},
                {'entity': 'shortness of breath', 'confidence': 0.8}
            ],
            'vital_signs': [
                {'entity': 'heart rate', 'value': '110', 'abnormal': True, 'confidence': 0.95}
            ],
            'conditions': [
                {'entity': 'myocardial infarction', 'confidence': 0.7}
            ],
            'overall_assessment': {
                'risk_level': 'high',
                'requires_immediate_attention': True,
                'primary_concerns': ['cardiac event']
            }
        }
        
        patient_context = {'age': 55, 'gender': 'M'}
        
        reasoning_chain = self.explainable_service.generate_reasoning_chain(
            entities=entities,
            patient_context=patient_context,
            depth='detailed'
        )
        
        self.assertIsInstance(reasoning_chain, list)
        self.assertGreater(len(reasoning_chain), 0)
        
        # Check first step
        if reasoning_chain:
            step = reasoning_chain[0]
            self.assertIn('step', step)
            self.assertIn('reasoning', step)
            self.assertIn('evidence_type', step)
            self.assertIn('confidence', step)
    
    def test_health_check(self):
        """Test service health check"""
        # Mock the extract_clinical_entities method
        self.explainable_service.extract_clinical_entities = Mock(
            return_value={'symptoms': []}
        )
        
        is_healthy = self.explainable_service.health_check()
        
        self.assertTrue(is_healthy)

class TestExplainableAIIntegration(unittest.TestCase):
    """Integration tests for explainable AI components"""
    
    def setUp(self):
        self.sample_patient_note = """
        Patient presents with chest pain, shortness of breath, and elevated heart rate.
        Blood pressure is 140/90. Patient appears anxious and reports pain started 2 hours ago.
        EKG shows ST elevation. Troponin levels are elevated.
        """
        
        self.sample_patient_context = {
            'age': 58,
            'gender': 'M',
            'medical_history': ['hypertension', 'diabetes'],
            'current_medications': ['metformin', 'lisinopril']
        }
    
    def test_end_to_end_workflow_simulation(self):
        """Test simulated end-to-end workflow"""
        # This test simulates the workflow without actual API calls
        
        # Step 1: Mock clinical analysis
        mock_analysis = {
            'symptoms': [
                {'entity': 'chest pain', 'confidence': 0.9, 'severity': 'severe'},
                {'entity': 'shortness of breath', 'confidence': 0.8, 'severity': 'moderate'}
            ],
            'vital_signs': [
                {'entity': 'blood pressure', 'value': '140/90', 'abnormal': True, 'confidence': 0.95},
                {'entity': 'heart rate', 'value': 'elevated', 'abnormal': True, 'confidence': 0.9}
            ],
            'conditions': [
                {'entity': 'myocardial infarction', 'confidence': 0.85, 'status': 'suspected'}
            ],
            'overall_assessment': {
                'risk_level': 'critical',
                'requires_immediate_attention': True,
                'primary_concerns': ['acute coronary syndrome']
            }
        }
        
        # Step 2: Test uncertainty calculation
        uncertainty_calc = UncertaintyCalculator()
        uncertainty_result = uncertainty_calc.assess_diagnostic_uncertainty(mock_analysis)
        
        self.assertGreater(uncertainty_result['overall_confidence'], 0.7)
        self.assertEqual(uncertainty_result['high_confidence_entities'], 4)  # All entities have >0.8 confidence
        
        # Step 3: Test pathway generation
        pathway_explorer = TreatmentPathwayExplorer()
        pathways = pathway_explorer.generate_alternative_pathways(
            primary_diagnosis=mock_analysis['conditions'][0],
            patient_context=self.sample_patient_context
        )
        
        self.assertGreater(len(pathways), 0)
        
        # Step 4: Test contraindication checking
        if pathways:
            contraindications = pathway_explorer.check_contraindications(
                pathway=pathways[0],
                patient_context=self.sample_patient_context
            )
            self.assertIsInstance(contraindications, list)
    
    def test_error_handling_workflow(self):
        """Test error handling in workflow"""
        # Test with empty/invalid data
        uncertainty_calc = UncertaintyCalculator()
        
        # Test with empty entities
        result = uncertainty_calc.assess_diagnostic_uncertainty([])
        self.assertEqual(result['overall_confidence'], 0.0)
        
        # Test with malformed entities
        malformed_entities = {'invalid': 'data'}
        result = uncertainty_calc.assess_diagnostic_uncertainty(malformed_entities)
        self.assertIn('overall_confidence', result)

if __name__ == '__main__':
    # Run specific test suites
    test_suites = [
        unittest.TestLoader().loadTestsFromTestCase(TestRateLimiter),
        unittest.TestLoader().loadTestsFromTestCase(TestPubMedService),
        unittest.TestLoader().loadTestsFromTestCase(TestPubMedCacheService),
        unittest.TestLoader().loadTestsFromTestCase(TestUncertaintyCalculator),
        unittest.TestLoader().loadTestsFromTestCase(TestTreatmentPathwayExplorer),
        unittest.TestLoader().loadTestsFromTestCase(TestExplainableClinicalService),
        unittest.TestLoader().loadTestsFromTestCase(TestExplainableAIIntegration)
    ]
    
    # Combine all test suites
    combined_suite = unittest.TestSuite(test_suites)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(combined_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"EXPLAINABLE AI SERVICES TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Error:')[-1].strip()}")