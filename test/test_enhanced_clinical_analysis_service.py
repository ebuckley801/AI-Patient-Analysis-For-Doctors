#!/usr/bin/env python3
"""
Test Enhanced Clinical Analysis Service
Tests for the integrated Faiss + NLP clinical analysis pipeline
"""

import unittest
import sys
import os
import time
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from app.services.enhanced_clinical_analysis import EnhancedClinicalAnalysisService, create_enhanced_clinical_analysis_service


class TestEnhancedClinicalAnalysisService(unittest.TestCase):
    """Test cases for Enhanced Clinical Analysis Service"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Sample clinical notes for testing
        self.sample_notes = {
            'chest_pain': "Patient is a 65-year-old male with h/o DM, HTN who presents with c/o severe CP and SOB. Denies fever. BP 160/90, HR 110 bpm.",
            'diabetes': "45 y/o F with DM type 2, well controlled on metformin. HbA1c 6.8%. No diabetic complications noted.",
            'negated_symptoms': "Patient denies chest pain, shortness of breath, or palpitations. No fever present. Vitals stable.",
            'complex_case': "Pt c/o SOB x 3 days, worsening. Possible pneumonia vs. CHF exacerbation. CXR shows bilateral infiltrates. BNP elevated.",
            'temporal_case': "Chest pain started 2 days ago, intermittent episodes, worsening since this morning. No radiation."
        }
        
        # Mock ICD data for testing
        self.mock_icd_data = [
            {
                'icd_10_code': 'I21.9',
                'description': 'Acute myocardial infarction, unspecified',
                'embedded_description': [0.1] * 1536
            },
            {
                'icd_10_code': 'E11.9',
                'description': 'Type 2 diabetes mellitus without complications',
                'embedded_description': [0.2] * 1536
            },
            {
                'icd_10_code': 'J44.1',
                'description': 'Chronic obstructive pulmonary disease with acute exacerbation',
                'embedded_description': [0.3] * 1536
            }
        ]
    
    @patch('app.services.enhanced_clinical_analysis.anthropic.Anthropic')
    @patch('app.services.enhanced_clinical_analysis.create_clinical_nlp_processor')
    @patch('app.services.enhanced_clinical_analysis.ICD10VectorMatcher')
    def test_service_initialization(self, mock_icd_matcher, mock_nlp_processor, mock_anthropic):
        """Test enhanced service initialization"""
        
        # Mock dependencies
        mock_nlp_processor.return_value = Mock()
        mock_icd_matcher.return_value = Mock()
        mock_icd_matcher.return_value.use_faiss = True
        mock_anthropic.return_value = Mock()
        
        # Create service
        service = EnhancedClinicalAnalysisService()
        
        # Verify initialization
        self.assertIsNotNone(service.client)
        self.assertIsNotNone(service.nlp_processor)
        self.assertIsNotNone(service.icd_matcher)
        self.assertIn('total_analyses', service.analysis_stats)
        self.assertEqual(service.analysis_stats['total_analyses'], 0)
    
    @patch('app.services.enhanced_clinical_analysis.anthropic.Anthropic')
    @patch('app.services.enhanced_clinical_analysis.create_clinical_nlp_processor')
    @patch('app.services.enhanced_clinical_analysis.ICD10VectorMatcher')
    def test_enhanced_extraction_with_faiss(self, mock_icd_matcher, mock_nlp_processor, mock_anthropic):
        """Test enhanced clinical extraction with Faiss integration"""
        
        # Setup mocks
        self._setup_comprehensive_mocks(mock_anthropic, mock_nlp_processor, mock_icd_matcher, use_faiss=True)
        
        service = EnhancedClinicalAnalysisService()
        
        # Test enhanced extraction
        result = service.extract_clinical_entities_enhanced(
            self.sample_notes['chest_pain'],
            patient_context={'age': 65, 'gender': 'male'},
            include_icd_mapping=True,
            enable_nlp_preprocessing=True
        )
        
        # Verify enhanced result structure
        self.assertIn('performance_metrics', result)
        self.assertIn('icd_mappings', result)
        self.assertIn('nlp_analysis', result)
        self.assertTrue(result['nlp_enhanced'])
        self.assertEqual(result['icd_search_method'], 'faiss')
        
        # Verify performance metrics
        metrics = result['performance_metrics']
        self.assertIn('total_time_ms', metrics)
        self.assertIn('preprocessing_time_ms', metrics)
        self.assertIn('icd_mapping_time_ms', metrics)
        self.assertGreater(metrics['total_time_ms'], 0)
        
        # Verify ICD mappings
        self.assertIsInstance(result['icd_mappings'], list)
        if result['icd_mappings']:
            mapping = result['icd_mappings'][0]
            self.assertIn('search_method', mapping)
            self.assertIn('search_time_ms', mapping)
            self.assertEqual(mapping['search_method'], 'faiss')
    
    @patch('app.services.enhanced_clinical_analysis.anthropic.Anthropic')
    @patch('app.services.enhanced_clinical_analysis.create_clinical_nlp_processor')
    @patch('app.services.enhanced_clinical_analysis.ICD10VectorMatcher')
    def test_enhanced_extraction_with_numpy_fallback(self, mock_icd_matcher, mock_nlp_processor, mock_anthropic):
        """Test enhanced clinical extraction with numpy fallback"""
        
        # Setup mocks with numpy fallback
        self._setup_comprehensive_mocks(mock_anthropic, mock_nlp_processor, mock_icd_matcher, use_faiss=False)
        
        service = EnhancedClinicalAnalysisService(force_numpy_icd=True)
        
        # Test extraction
        result = service.extract_clinical_entities_enhanced(
            self.sample_notes['diabetes'],
            include_icd_mapping=True
        )
        
        # Verify numpy search method
        self.assertEqual(result['icd_search_method'], 'numpy')
        
        # Verify ICD mappings use numpy
        if result['icd_mappings']:
            mapping = result['icd_mappings'][0]
            self.assertEqual(mapping['search_method'], 'numpy')
    
    @patch('app.services.enhanced_clinical_analysis.anthropic.Anthropic')
    @patch('app.services.enhanced_clinical_analysis.create_clinical_nlp_processor')
    @patch('app.services.enhanced_clinical_analysis.ICD10VectorMatcher')
    def test_nlp_enhancement_integration(self, mock_icd_matcher, mock_nlp_processor, mock_anthropic):
        """Test NLP enhancement integration"""
        
        # Setup mocks with detailed NLP responses
        mock_nlp = Mock()
        mock_nlp.preprocess_clinical_text.return_value = "Patient complains of severe chest pain and shortness of breath. Denies fever. Blood pressure 160/90, heart rate 110 beats per minute."
        mock_nlp.detect_negation.return_value = {
            'is_negated': True,
            'negation_strength': 0.9,
            'negation_type': 'patient_reported',
            'confidence': 0.9
        }
        mock_nlp.extract_temporal_info.return_value = {
            'onset': '2 days',
            'duration': None,
            'progression': 'worsening'
        }
        mock_nlp.assess_uncertainty.return_value = {
            'has_uncertainty': False,
            'confidence_modifier': 0.0
        }
        mock_nlp_processor.return_value = mock_nlp
        
        # Setup other mocks
        self._setup_basic_mocks(mock_anthropic, mock_icd_matcher)
        
        service = EnhancedClinicalAnalysisService()
        
        # Test with negated symptoms
        result = service.extract_clinical_entities_enhanced(
            self.sample_notes['negated_symptoms'],
            enable_nlp_preprocessing=True
        )
        
        # Verify NLP analysis was applied
        self.assertIn('nlp_analysis', result)
        nlp_analysis = result['nlp_analysis']
        self.assertTrue(nlp_analysis['abbreviations_expanded'])
        self.assertTrue(nlp_analysis['negation_detection_applied'])
        
        # Verify entities were enhanced
        if result.get('symptoms'):
            symptom = result['symptoms'][0]
            self.assertIn('negation', symptom)
            self.assertIn('temporal', symptom)
            self.assertIn('uncertainty', symptom)
    
    @patch('app.services.enhanced_clinical_analysis.anthropic.Anthropic')
    @patch('app.services.enhanced_clinical_analysis.create_clinical_nlp_processor')
    @patch('app.services.enhanced_clinical_analysis.ICD10VectorMatcher')
    def test_icd_mapping_with_nlp_context(self, mock_icd_matcher, mock_nlp_processor, mock_anthropic):
        """Test ICD mapping enhanced with NLP context"""
        
        # Setup mocks
        self._setup_comprehensive_mocks(mock_anthropic, mock_nlp_processor, mock_icd_matcher)
        
        service = EnhancedClinicalAnalysisService()
        
        # Test complex case with temporal information
        result = service.extract_clinical_entities_enhanced(
            self.sample_notes['temporal_case'],
            include_icd_mapping=True,
            enable_nlp_preprocessing=True
        )
        
        # Verify ICD mappings include enhanced queries
        if result['icd_mappings']:
            mapping = result['icd_mappings'][0]
            self.assertIn('enhanced_query', mapping)
            self.assertIn('search_time_ms', mapping)
            
            # Should include NLP context
            if 'negated' in mapping:
                self.assertIsInstance(mapping['negated'], bool)
    
    @patch('app.services.enhanced_clinical_analysis.anthropic.Anthropic')
    @patch('app.services.enhanced_clinical_analysis.create_clinical_nlp_processor')
    @patch('app.services.enhanced_clinical_analysis.ICD10VectorMatcher')
    def test_performance_tracking(self, mock_icd_matcher, mock_nlp_processor, mock_anthropic):
        """Test performance tracking and statistics"""
        
        # Setup mocks
        self._setup_comprehensive_mocks(mock_anthropic, mock_nlp_processor, mock_icd_matcher)
        
        service = EnhancedClinicalAnalysisService()
        
        # Run multiple analyses
        for i, note in enumerate(list(self.sample_notes.values())[:3]):
            result = service.extract_clinical_entities_enhanced(note, include_icd_mapping=True)
            
            # Verify performance metrics are tracked
            self.assertIn('performance_metrics', result)
            self.assertGreater(result['performance_metrics']['total_time_ms'], 0)
        
        # Verify global stats are updated
        stats = service.get_performance_stats()
        self.assertEqual(stats['total_analyses'], 3)
        self.assertGreater(stats['avg_analysis_time_ms'], 0)
        self.assertIn('icd_search_method', stats)
    
    @patch('app.services.enhanced_clinical_analysis.anthropic.Anthropic')
    @patch('app.services.enhanced_clinical_analysis.create_clinical_nlp_processor')
    @patch('app.services.enhanced_clinical_analysis.ICD10VectorMatcher')
    def test_error_handling(self, mock_icd_matcher, mock_nlp_processor, mock_anthropic):
        """Test error handling in enhanced analysis"""
        
        # Setup mocks with errors
        mock_anthropic.return_value.messages.create.side_effect = Exception("API Error")
        mock_nlp_processor.return_value = Mock()
        mock_icd_matcher.return_value = Mock()
        
        service = EnhancedClinicalAnalysisService()
        
        # Test error handling
        result = service.extract_clinical_entities_enhanced("Test note")
        
        # Verify error result structure
        self.assertIn('error', result)
        self.assertIn('performance_metrics', result)
        self.assertTrue(result['performance_metrics']['error'])
        self.assertFalse(result['nlp_enhanced'])
    
    @patch('app.services.enhanced_clinical_analysis.anthropic.Anthropic')
    @patch('app.services.enhanced_clinical_analysis.create_clinical_nlp_processor')
    @patch('app.services.enhanced_clinical_analysis.ICD10VectorMatcher')
    def test_benchmark_functionality(self, mock_icd_matcher, mock_nlp_processor, mock_anthropic):
        """Test benchmark functionality"""
        
        # Setup mocks
        self._setup_comprehensive_mocks(mock_anthropic, mock_nlp_processor, mock_icd_matcher)
        
        service = EnhancedClinicalAnalysisService()
        
        # Run benchmark
        benchmark_result = service.benchmark_enhanced_analysis(num_tests=3)
        
        # Verify benchmark results
        self.assertEqual(benchmark_result['num_tests'], 3)
        self.assertIn('avg_time_per_analysis_ms', benchmark_result)
        self.assertIn('analyses_per_second', benchmark_result)
        self.assertIn('performance_stats', benchmark_result)
        self.assertEqual(len(benchmark_result['individual_times_ms']), 3)
    
    def test_factory_function(self):
        """Test factory function for creating enhanced service"""
        
        # Test successful creation (with mocks)
        with patch('app.services.enhanced_clinical_analysis.EnhancedClinicalAnalysisService') as mock_service:
            mock_instance = Mock()
            mock_service.return_value = mock_instance
            
            service = create_enhanced_clinical_analysis_service()
            self.assertEqual(service, mock_instance)
        
        # Test error handling
        with patch('app.services.enhanced_clinical_analysis.EnhancedClinicalAnalysisService', side_effect=Exception("Test error")):
            service = create_enhanced_clinical_analysis_service()
            self.assertIsNone(service)
    
    def test_entity_position_finding(self):
        """Test entity position finding for NLP analysis"""
        
        with patch('app.services.enhanced_clinical_analysis.anthropic.Anthropic'):
            with patch('app.services.enhanced_clinical_analysis.create_clinical_nlp_processor'):
                with patch('app.services.enhanced_clinical_analysis.ICD10VectorMatcher'):
                    service = EnhancedClinicalAnalysisService()
                    
                    # Test exact match
                    text = "Patient has severe chest pain and fever"
                    pos = service._find_entity_position(text, "chest pain", "chest pain")
                    self.assertEqual(pos, (19, 29))
                    
                    # Test partial match
                    pos = service._find_entity_position(text, "", "fever")
                    self.assertEqual(pos, (34, 39))
                    
                    # Test not found
                    pos = service._find_entity_position(text, "nonexistent", "nonexistent")
                    self.assertIsNone(pos)
    
    def _setup_comprehensive_mocks(self, mock_anthropic, mock_nlp_processor, mock_icd_matcher, use_faiss=True):
        """Setup comprehensive mocks for testing"""
        
        # Mock Anthropic response
        mock_response = Mock()
        mock_response.content = [Mock(text=self._get_sample_claude_response())]
        mock_anthropic.return_value.messages.create.return_value = mock_response
        
        # Mock NLP processor with all required methods
        mock_nlp = Mock()
        mock_nlp.preprocess_clinical_text.return_value = "Processed clinical text"
        mock_nlp.detect_negation.return_value = {
            'is_negated': False,
            'negation_strength': 0.0,
            'confidence': 0.9
        }
        mock_nlp.extract_temporal_info.return_value = {
            'onset': None,
            'duration': None,
            'progression': None
        }
        mock_nlp.assess_uncertainty.return_value = {
            'has_uncertainty': False,
            'confidence_modifier': 0.0
        }
        # Ensure all NLP methods exist on the mock
        mock_nlp.enhance_entity_with_nlp = Mock(return_value={})
        mock_nlp_processor.return_value = mock_nlp
        
        # Mock ICD matcher
        mock_icd = Mock()
        mock_icd.use_faiss = use_faiss
        mock_icd.find_similar_icd_codes.return_value = [
            {
                'icd_code': 'I21.9',
                'description': 'Acute myocardial infarction',
                'similarity': 0.95,
                'search_method': 'faiss' if use_faiss else 'numpy'
            }
        ]
        mock_icd.get_cache_info.return_value = {
            'search_method': 'faiss' if use_faiss else 'numpy',
            'total_icd_codes': 1000
        }
        mock_icd_matcher.return_value = mock_icd
    
    def _setup_basic_mocks(self, mock_anthropic, mock_icd_matcher):
        """Setup basic mocks"""
        # Setup Anthropic response
        mock_response = Mock()
        mock_response.content = [Mock(text=self._get_sample_claude_response())]
        mock_anthropic.return_value.messages.create.return_value = mock_response
        
        # Setup ICD matcher
        mock_icd_instance = Mock()
        mock_icd_instance.use_faiss = True
        mock_icd_instance.find_similar_icd_codes.return_value = []
        mock_icd_instance.get_cache_info.return_value = {'search_method': 'faiss'}
        mock_icd_matcher.return_value = mock_icd_instance
    
    def _get_sample_claude_response(self):
        """Get sample Claude response for testing"""
        return """{
            "symptoms": [
                {
                    "entity": "chest pain",
                    "severity": "severe",
                    "temporal": "acute",
                    "confidence": 0.95,
                    "text_span": "chest pain",
                    "negated": false,
                    "onset": "2 days ago",
                    "progression": "worsening"
                }
            ],
            "conditions": [
                {
                    "entity": "acute myocardial infarction",
                    "status": "suspected",
                    "confidence": 0.85,
                    "text_span": "MI",
                    "icd_category": "cardiovascular",
                    "certainty": "suspected"
                }
            ],
            "medications": [],
            "vital_signs": [
                {
                    "entity": "blood pressure",
                    "value": "160/90",
                    "unit": "mmHg",
                    "abnormal": true,
                    "confidence": 0.98,
                    "text_span": "BP 160/90"
                }
            ],
            "procedures": [],
            "abnormal_findings": [],
            "overall_assessment": {
                "primary_concerns": ["chest pain", "hypertension"],
                "risk_level": "high",
                "requires_immediate_attention": true,
                "summary": "65-year-old male with acute chest pain and hypertension"
            }
        }"""


def run_enhanced_analysis_integration_demo():
    """
    Demo function showing enhanced clinical analysis with Faiss integration
    """
    print("🚀 Enhanced Clinical Analysis + Faiss Integration Demo")
    print("=" * 70)
    
    try:
        # Demo with comprehensive mocks
        with patch('app.services.enhanced_clinical_analysis.anthropic.Anthropic') as mock_anthropic:
            with patch('app.services.enhanced_clinical_analysis.create_clinical_nlp_processor') as mock_nlp:
                with patch('app.services.enhanced_clinical_analysis.ICD10VectorMatcher') as mock_icd:
                    
                    # Setup demo mocks
                    mock_response = Mock()
                    mock_response.content = [Mock(text="""{
                        "symptoms": [
                            {
                                "entity": "chest pain",
                                "severity": "severe",
                                "confidence": 0.95,
                                "text_span": "chest pain",
                                "negated": false
                            }
                        ],
                        "conditions": [
                            {
                                "entity": "myocardial infarction",
                                "status": "suspected",
                                "confidence": 0.85,
                                "certainty": "suspected"
                            }
                        ],
                        "overall_assessment": {
                            "primary_concerns": ["chest pain"],
                            "risk_level": "high",
                            "requires_immediate_attention": true,
                            "summary": "Acute cardiac event suspected"
                        }
                    }""")]
                    mock_anthropic.return_value.messages.create.return_value = mock_response
                    
                    # Mock NLP processor
                    mock_nlp_instance = Mock()
                    mock_nlp_instance.preprocess_clinical_text.return_value = "Patient complains of severe chest pain and shortness of breath"
                    mock_nlp_instance.detect_negation.return_value = {'is_negated': False, 'confidence': 0.9}
                    mock_nlp_instance.extract_temporal_info.return_value = {'onset': '2 hours', 'progression': 'worsening'}
                    mock_nlp_instance.assess_uncertainty.return_value = {'has_uncertainty': False, 'confidence_modifier': 0.0}
                    mock_nlp.return_value = mock_nlp_instance
                    
                    # Mock ICD matcher with Faiss
                    mock_icd_instance = Mock()
                    mock_icd_instance.use_faiss = True
                    mock_icd_instance.find_similar_icd_codes.return_value = [
                        {
                            'icd_code': 'I21.9',
                            'description': 'Acute myocardial infarction, unspecified',
                            'similarity': 0.95,
                            'search_method': 'faiss'
                        }
                    ]
                    mock_icd_instance.get_cache_info.return_value = {
                        'search_method': 'faiss',
                        'total_icd_codes': 70000,
                        'faiss_available': True
                    }
                    mock_icd.return_value = mock_icd_instance
                    
                    # Create enhanced service
                    print("🔧 Initializing Enhanced Clinical Analysis Service...")
                    service = EnhancedClinicalAnalysisService()
                    
                    # Demo note
                    demo_note = "Patient is a 65 y/o male with h/o DM, HTN who presents with c/o severe CP and SOB x 2 hours. Denies fever. BP 160/90, HR 110 bpm."
                    patient_context = {'age': 65, 'gender': 'male', 'medical_history': 'diabetes, hypertension'}
                    
                    print(f"\n📝 Demo Clinical Note:")
                    print(f"   '{demo_note}'")
                    
                    # Run enhanced analysis
                    print(f"\n⚡ Running Enhanced Analysis...")
                    start_time = time.time()
                    
                    result = service.extract_clinical_entities_enhanced(
                        demo_note,
                        patient_context=patient_context,
                        include_icd_mapping=True,
                        enable_nlp_preprocessing=True
                    )
                    
                    analysis_time = time.time() - start_time
                    
                    print(f"✅ Analysis completed in {analysis_time:.3f}s")
                    
                    # Display results
                    print(f"\n📊 Enhanced Analysis Results:")
                    print(f"   • NLP Enhanced: {result.get('nlp_enhanced', False)}")
                    print(f"   • Search Method: {result.get('icd_search_method', 'unknown')}")
                    print(f"   • Entities Found: {len(result.get('symptoms', []))} symptoms, {len(result.get('conditions', []))} conditions")
                    print(f"   • ICD Mappings: {len(result.get('icd_mappings', []))}")
                    
                    # Performance metrics
                    if 'performance_metrics' in result:
                        metrics = result['performance_metrics']
                        print(f"\n⏱️ Performance Breakdown:")
                        print(f"   • Total Time: {metrics.get('total_time_ms', 0):.1f}ms")
                        print(f"   • Preprocessing: {metrics.get('preprocessing_time_ms', 0):.1f}ms")
                        print(f"   • ICD Mapping: {metrics.get('icd_mapping_time_ms', 0):.1f}ms")
                    
                    # Service stats
                    stats = service.get_performance_stats()
                    print(f"\n📈 Service Statistics:")
                    print(f"   • Total Analyses: {stats['total_analyses']}")
                    print(f"   • Avg Analysis Time: {stats['avg_analysis_time_ms']:.1f}ms")
                    print(f"   • ICD Search Method: {stats['icd_search_method']}")
                    
                    # Benchmark
                    print(f"\n🏁 Running Performance Benchmark...")
                    benchmark = service.benchmark_enhanced_analysis(num_tests=5)
                    
                    print(f"📊 Benchmark Results:")
                    print(f"   • Avg Time per Analysis: {benchmark['avg_time_per_analysis_ms']:.1f}ms")
                    print(f"   • Analyses per Second: {benchmark['analyses_per_second']:.1f}")
                    print(f"   • Min Time: {benchmark['min_time_ms']:.1f}ms")
                    print(f"   • Max Time: {benchmark['max_time_ms']:.1f}ms")
                    
                    print(f"\n✨ Enhanced Clinical Analysis Demo Complete!")
                    print("Key Integration Benefits:")
                    print("   • Advanced NLP preprocessing with medical abbreviation expansion")
                    print("   • Sophisticated negation and temporal detection")
                    print("   • High-performance Faiss vector search for ICD-10 mapping")
                    print("   • Comprehensive performance tracking and optimization")
                    print("   • Production-ready scalability for healthcare systems")
                    
                    return True
    
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Running Enhanced Clinical Analysis Service Tests")
    print("=" * 80)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run integration demo
    print("\n" + "=" * 80)
    run_enhanced_analysis_integration_demo()