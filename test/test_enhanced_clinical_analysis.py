#!/usr/bin/env python3
"""
Test Enhanced Clinical Analysis Service
Tests for the clinical analysis service with NLP enhancements
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.services.clinical_analysis_service import ClinicalAnalysisService


class TestEnhancedClinicalAnalysis(unittest.TestCase):
    """Test cases for Enhanced Clinical Analysis Service"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock the Anthropic client to avoid actual API calls
        with patch('app.services.clinical_analysis_service.anthropic.Anthropic'):
            self.service = ClinicalAnalysisService()
        
        # Mock the Claude response
        self.mock_client = Mock()
        self.service.client = self.mock_client
        
        # Sample test data
        self.sample_note_with_abbreviations = """
        Pt is 65 y/o male with h/o DM, HTN, CAD who presents with c/o SOB and CP.
        Patient denies fever but reports N/V. No chest pain at rest.
        Vitals: BP 160/90, HR 110 bpm, RR 22, O2 sat 94%.
        Possible pneumonia vs. CHF exacerbation.
        Started 3 days ago, worsening since yesterday.
        ECG negative for STEMI. BNP elevated.
        """
        
        self.sample_claude_response = """{
            "symptoms": [
                {
                    "entity": "shortness of breath",
                    "severity": "moderate",
                    "temporal": "acute",
                    "confidence": 0.95,
                    "text_span": "SOB",
                    "negated": false
                },
                {
                    "entity": "chest pain",
                    "severity": "moderate", 
                    "temporal": "episodic",
                    "confidence": 0.90,
                    "text_span": "CP",
                    "negated": false
                },
                {
                    "entity": "nausea and vomiting",
                    "severity": "mild",
                    "temporal": "acute",
                    "confidence": 0.85,
                    "text_span": "N/V",
                    "negated": false
                },
                {
                    "entity": "fever",
                    "severity": "unknown",
                    "temporal": "acute", 
                    "confidence": 0.90,
                    "text_span": "fever",
                    "negated": true
                }
            ],
            "conditions": [
                {
                    "entity": "diabetes mellitus",
                    "status": "active",
                    "confidence": 0.95,
                    "text_span": "DM",
                    "icd_category": "endocrine"
                },
                {
                    "entity": "hypertension",
                    "status": "active", 
                    "confidence": 0.95,
                    "text_span": "HTN",
                    "icd_category": "cardiovascular"
                },
                {
                    "entity": "pneumonia",
                    "status": "suspected",
                    "confidence": 0.60,
                    "text_span": "pneumonia",
                    "icd_category": "respiratory"
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
                },
                {
                    "entity": "heart rate",
                    "value": "110",
                    "unit": "bpm",
                    "abnormal": true,
                    "confidence": 0.98,
                    "text_span": "HR 110 bpm"
                }
            ],
            "procedures": [],
            "abnormal_findings": [
                {
                    "entity": "elevated BNP",
                    "severity": "moderate",
                    "requires_attention": true,
                    "confidence": 0.90,
                    "text_span": "BNP elevated"
                }
            ],
            "overall_assessment": {
                "primary_concerns": ["shortness of breath", "chest pain", "elevated BNP"],
                "risk_level": "high",
                "requires_immediate_attention": true,
                "summary": "65-year-old male with acute cardiopulmonary symptoms"
            }
        }"""
        
        self.patient_context = {
            'age': 65,
            'gender': 'male',
            'medical_history': 'diabetes, hypertension, coronary artery disease'
        }
    
    @patch('app.services.clinical_analysis_service.create_clinical_nlp_processor')
    def test_service_initialization_with_nlp(self, mock_nlp_factory):
        """Test that service initializes with NLP processor"""
        mock_nlp_processor = Mock()
        mock_nlp_factory.return_value = mock_nlp_processor
        
        with patch('app.services.clinical_analysis_service.anthropic.Anthropic'):
            service = ClinicalAnalysisService()
        
        mock_nlp_factory.assert_called_once()
        self.assertEqual(service.nlp_processor, mock_nlp_processor)
    
    def test_abbreviation_expansion_in_preprocessing(self):
        """Test that abbreviations are expanded during preprocessing"""
        # Mock Claude response
        mock_response = Mock()
        mock_response.content = [Mock(text=self.sample_claude_response)]
        self.mock_client.messages.create.return_value = mock_response
        
        # Mock NLP processor methods properly
        mock_expanded_text = self.sample_note_with_abbreviations.replace('SOB', 'shortness of breath').replace('CP', 'chest pain')
        
        # Create a proper mock for the nlp_processor
        mock_nlp_processor = Mock()
        mock_nlp_processor.preprocess_clinical_text.return_value = mock_expanded_text
        mock_nlp_processor.enhance_entity_with_nlp.return_value = {'entity': 'test', 'confidence': 0.9}
        self.service.nlp_processor = mock_nlp_processor
        
        result = self.service.extract_clinical_entities(self.sample_note_with_abbreviations, self.patient_context)
        
        # Verify preprocessing was called
        mock_nlp_processor.preprocess_clinical_text.assert_called_once_with(self.sample_note_with_abbreviations)
        
        # Verify that the processed text was used in the prompt
        call_args = self.mock_client.messages.create.call_args
        prompt_text = call_args[1]['messages'][0]['content']
        self.assertIn('shortness of breath', prompt_text)  # Should contain expanded abbreviation
    
    def test_enhanced_prompt_generation(self):
        """Test that enhanced prompts include NLP instructions"""
        # Mock Claude response
        mock_response = Mock()
        mock_response.content = [Mock(text=self.sample_claude_response)]
        self.mock_client.messages.create.return_value = mock_response
        
        # Mock NLP processor
        mock_nlp_processor = Mock()
        mock_nlp_processor.preprocess_clinical_text.return_value = self.sample_note_with_abbreviations
        mock_nlp_processor.enhance_entity_with_nlp.return_value = {'entity': 'test', 'confidence': 0.9}
        self.service.nlp_processor = mock_nlp_processor
        
        result = self.service.extract_clinical_entities(self.sample_note_with_abbreviations, self.patient_context)
        
        # Verify enhanced prompt was generated
        call_args = self.mock_client.messages.create.call_args
        prompt_text = call_args[1]['messages'][0]['content']
        
        self.assertIn('Enhanced Negation Detection', prompt_text)
        self.assertIn('Temporal Context', prompt_text)
        self.assertIn('Uncertainty Markers', prompt_text)
        self.assertIn('Abbreviation Awareness', prompt_text)
    
    def test_nlp_enhancement_applied_to_entities(self):
        """Test that NLP enhancement is applied to extracted entities"""
        # Mock Claude response
        mock_response = Mock()
        mock_response.content = [Mock(text=self.sample_claude_response)]
        self.mock_client.messages.create.return_value = mock_response
        
        # Mock NLP processor
        mock_nlp_processor = Mock()
        mock_nlp_processor.preprocess_clinical_text.return_value = self.sample_note_with_abbreviations
        self.service.nlp_processor = mock_nlp_processor
        
        # Mock enhanced entity
        def mock_enhance(entity, text, pos):
            enhanced = entity.copy()
            enhanced['negation'] = {
                'is_negated': entity.get('negated', False),
                'confidence': 0.9
            }
            enhanced['temporal'] = {'onset': '3 days', 'progression': 'worsening'}
            enhanced['uncertainty'] = {'has_uncertainty': False}
            return enhanced
        
        mock_nlp_processor.enhance_entity_with_nlp.side_effect = mock_enhance
        
        result = self.service.extract_clinical_entities(self.sample_note_with_abbreviations, self.patient_context)
        
        # Verify NLP enhancement was applied
        self.assertTrue(result['nlp_enhanced'])
        self.assertIn('nlp_analysis', result)
        
        # Check that entities were enhanced
        symptoms = result.get('symptoms', [])
        if symptoms:
            first_symptom = symptoms[0]
            self.assertIn('negation', first_symptom)
            self.assertIn('temporal', first_symptom)
            self.assertIn('uncertainty', first_symptom)
    
    def test_negation_detection_integration(self):
        """Test negation detection integration with clinical analysis"""
        # Mock Claude response with negated entity
        claude_response_with_negation = """{
            "symptoms": [
                {
                    "entity": "fever",
                    "severity": "unknown",
                    "confidence": 0.90,
                    "text_span": "fever",
                    "negated": true
                }
            ],
            "conditions": [],
            "medications": [],
            "vital_signs": [],
            "procedures": [],
            "abnormal_findings": [],
            "overall_assessment": {
                "primary_concerns": [],
                "risk_level": "low",
                "requires_immediate_attention": false,
                "summary": "Patient denies fever"
            }
        }"""
        
        mock_response = Mock()
        mock_response.content = [Mock(text=claude_response_with_negation)]
        self.mock_client.messages.create.return_value = mock_response
        
        # Mock NLP processor with negation detection
        mock_nlp_processor = Mock()
        mock_nlp_processor.preprocess_clinical_text.return_value = "Patient denies fever"
        self.service.nlp_processor = mock_nlp_processor
        
        def mock_enhance_with_negation(entity, text, pos):
            enhanced = entity.copy()
            if entity.get('negated'):
                enhanced['negation'] = {
                    'is_negated': True,
                    'negation_strength': 0.9,
                    'negation_type': 'patient_reported',
                    'confidence': 0.9
                }
                enhanced['negated'] = True
            return enhanced
        
        mock_nlp_processor.enhance_entity_with_nlp.side_effect = mock_enhance_with_negation
        
        result = self.service.extract_clinical_entities("Patient denies fever", self.patient_context)
        
        # Verify negation was properly detected and enhanced
        symptoms = result.get('symptoms', [])
        if symptoms:
            fever_symptom = symptoms[0]
            self.assertTrue(fever_symptom.get('negated'))
            self.assertTrue(fever_symptom['negation']['is_negated'])
            self.assertEqual(fever_symptom['negation']['negation_type'], 'patient_reported')
    
    def test_temporal_information_extraction(self):
        """Test temporal information extraction"""
        # Mock NLP processor with temporal information
        mock_nlp_processor = Mock()
        mock_nlp_processor.preprocess_clinical_text.return_value = self.sample_note_with_abbreviations
        self.service.nlp_processor = mock_nlp_processor
        
        def mock_enhance_with_temporal(entity, text, pos):
            enhanced = entity.copy()
            enhanced['temporal'] = {
                'onset': '3 days',
                'duration': None,
                'frequency': None,
                'pattern': None,
                'progression': 'worsening'
            }
            return enhanced
        
        mock_nlp_processor.enhance_entity_with_nlp.side_effect = mock_enhance_with_temporal
        
        # Mock Claude response
        mock_response = Mock()
        mock_response.content = [Mock(text=self.sample_claude_response)]
        self.mock_client.messages.create.return_value = mock_response
        
        result = self.service.extract_clinical_entities(self.sample_note_with_abbreviations, self.patient_context)
        
        # Verify temporal information was extracted
        symptoms = result.get('symptoms', [])
        if symptoms:
            first_symptom = symptoms[0]
            self.assertIn('temporal', first_symptom)
            self.assertEqual(first_symptom['temporal']['onset'], '3 days')
            self.assertEqual(first_symptom['temporal']['progression'], 'worsening')
    
    def test_uncertainty_assessment(self):
        """Test uncertainty assessment for clinical entities"""
        # Mock NLP processor with uncertainty assessment
        mock_nlp_processor = Mock()
        mock_nlp_processor.preprocess_clinical_text.return_value = "Possible pneumonia"
        self.service.nlp_processor = mock_nlp_processor
        
        def mock_enhance_with_uncertainty(entity, text, pos):
            enhanced = entity.copy()
            if 'pneumonia' in entity.get('entity', ''):
                enhanced['uncertainty'] = {
                    'has_uncertainty': True,
                    'uncertainty_type': 'speculation',
                    'confidence_modifier': -0.3,
                    'speculation_markers': ['possible']
                }
                # Adjust confidence based on uncertainty
                enhanced['confidence'] = max(0.1, enhanced.get('confidence', 1.0) - 0.3)
            return enhanced
        
        mock_nlp_processor.enhance_entity_with_nlp.side_effect = mock_enhance_with_uncertainty
        
        # Mock Claude response with uncertain entity
        uncertain_response = """{
            "symptoms": [],
            "conditions": [
                {
                    "entity": "pneumonia",
                    "status": "suspected",
                    "confidence": 0.60,
                    "text_span": "pneumonia"
                }
            ],
            "medications": [],
            "vital_signs": [],
            "procedures": [],
            "abnormal_findings": [],
            "overall_assessment": {
                "primary_concerns": ["possible pneumonia"],
                "risk_level": "moderate",
                "requires_immediate_attention": false,
                "summary": "Possible pneumonia requires further evaluation"
            }
        }"""
        
        mock_response = Mock()
        mock_response.content = [Mock(text=uncertain_response)]
        self.mock_client.messages.create.return_value = mock_response
        
        result = self.service.extract_clinical_entities("Possible pneumonia", self.patient_context)
        
        # Verify uncertainty assessment
        conditions = result.get('conditions', [])
        if conditions:
            pneumonia_condition = conditions[0]
            self.assertIn('uncertainty', pneumonia_condition)
            self.assertTrue(pneumonia_condition['uncertainty']['has_uncertainty'])
            self.assertEqual(pneumonia_condition['uncertainty']['uncertainty_type'], 'speculation')
            # Confidence should be reduced due to uncertainty
            self.assertLess(pneumonia_condition['confidence'], 0.6)
    
    def test_entity_position_finding(self):
        """Test entity position finding for NLP enhancement"""
        text = "Patient has chest pain and fever"
        
        # Test exact text span match
        result = self.service._find_entity_position(text, "chest pain", "chest pain")
        self.assertEqual(result, (12, 22))
        
        # Test entity text fallback
        result = self.service._find_entity_position(text, "", "fever")
        self.assertEqual(result, (27, 32))
        
        # Test multi-word entity partial match
        result = self.service._find_entity_position(text, "", "chest pain syndrome")
        self.assertEqual(result, (12, 17))  # Should find "chest"
        
        # Test not found
        result = self.service._find_entity_position(text, "nonexistent", "nonexistent")
        self.assertIsNone(result)
    
    def test_comprehensive_nlp_integration(self):
        """Test comprehensive integration of all NLP features"""
        complex_note = """
        Patient is a 45-year-old female who presents with c/o SOB and CP that started 2 days ago.
        She denies fever but reports possible N/V. No chest pain at rest currently.
        Possible pneumonia vs. anxiety attack. Started suddenly, worsening since this morning.
        Vitals: BP 140/85, HR 95 bpm. ECG negative for acute changes.
        """
        
        # Mock Claude response
        mock_response = Mock()
        mock_response.content = [Mock(text=self.sample_claude_response)]
        self.mock_client.messages.create.return_value = mock_response
        
        # Mock comprehensive NLP processing
        mock_nlp_processor = Mock()
        mock_nlp_processor.preprocess_clinical_text.return_value = complex_note.replace('SOB', 'shortness of breath').replace('CP', 'chest pain')
        self.service.nlp_processor = mock_nlp_processor
        
        def comprehensive_enhance(entity, text, pos):
            enhanced = entity.copy()
            entity_name = entity.get('entity', '').lower()
            
            # Add preprocessing info to ALL entities
            enhanced['preprocessed'] = {
                'abbreviations_expanded': True,
                'original_span': entity.get('text_span', '')
            }
            
            # Add negation detection
            if 'fever' in entity_name:
                enhanced['negation'] = {
                    'is_negated': True,
                    'negation_type': 'patient_reported',
                    'confidence': 0.9
                }
                enhanced['negated'] = True
            
            # Add temporal information
            if 'shortness of breath' in entity_name or 'chest pain' in entity_name:
                enhanced['temporal'] = {
                    'onset': '2 days',
                    'progression': 'worsening'
                }
            
            # Add uncertainty assessment
            if 'pneumonia' in entity_name:
                enhanced['uncertainty'] = {
                    'has_uncertainty': True,
                    'uncertainty_type': 'speculation',
                    'confidence_modifier': -0.3
                }
                enhanced['confidence'] = max(0.1, enhanced.get('confidence', 1.0) - 0.3)
            
            return enhanced
        
        mock_nlp_processor.enhance_entity_with_nlp.side_effect = comprehensive_enhance
        
        result = self.service.extract_clinical_entities(complex_note, self.patient_context)
        
        # Verify comprehensive NLP enhancement
        self.assertTrue(result['nlp_enhanced'])
        self.assertIn('nlp_analysis', result)
        
        nlp_analysis = result['nlp_analysis']
        self.assertTrue(nlp_analysis['negation_detection_applied'])
        self.assertTrue(nlp_analysis['temporal_extraction_applied'])
        self.assertTrue(nlp_analysis['uncertainty_assessment_applied'])
        self.assertTrue(nlp_analysis['abbreviations_expanded'])
        
        # Verify that preprocessing was applied
        mock_nlp_processor.preprocess_clinical_text.assert_called_once()
        
        # Verify that entities were enhanced with NLP
        # Note: Only entities that can be found in the text will be enhanced
        enhanced_entities_found = False
        for entity_type in ['symptoms', 'conditions', 'vital_signs']:
            entities = result.get(entity_type, [])
            for entity in entities:
                if 'preprocessed' in entity:
                    enhanced_entities_found = True
                    break
            if enhanced_entities_found:
                break
        
        # At least some entities should have been enhanced
        self.assertTrue(enhanced_entities_found, "At least some entities should have NLP enhancements")
    
    def test_error_handling_in_nlp_enhancement(self):
        """Test error handling when NLP enhancement fails"""
        # Mock Claude response
        mock_response = Mock()
        mock_response.content = [Mock(text=self.sample_claude_response)]
        self.mock_client.messages.create.return_value = mock_response
        
        # Mock NLP processor that raises an exception
        mock_nlp_processor = Mock()
        mock_nlp_processor.preprocess_clinical_text.return_value = self.sample_note_with_abbreviations
        mock_nlp_processor.enhance_entity_with_nlp.side_effect = Exception("NLP enhancement failed")
        self.service.nlp_processor = mock_nlp_processor
        
        # Should not crash, should continue with original entities
        result = self.service.extract_clinical_entities(self.sample_note_with_abbreviations, self.patient_context)
        
        # Should still have results
        self.assertIn('symptoms', result)
        self.assertIn('conditions', result)
        
        # Should have NLP analysis metadata
        self.assertTrue(result['nlp_enhanced'])


def run_enhanced_analysis_demo():
    """
    Demo function showing enhanced clinical analysis capabilities
    """
    print("🧠 Enhanced Clinical Analysis Demo")
    print("=" * 50)
    
    try:
        # This would demonstrate the enhanced analysis in practice
        # but requires actual API keys, so we'll show the concept
        
        demo_notes = [
            # Note with abbreviations
            "Pt is 72 y/o M with h/o DM, HTN, CAD who presents with SOB and CP. Denies fever.",
            
            # Note with negations
            "Patient denies chest pain, shortness of breath, or palpitations. No fever present.",
            
            # Note with temporal information
            "Chest pain started 3 days ago, worsening since yesterday morning. Intermittent episodes.",
            
            # Note with uncertainty
            "Possible pneumonia vs. heart failure. Symptoms suggest viral infection. May be dehydration.",
            
            # Complex note with multiple features
            "45 y/o F with c/o SOB x 2 days. No CP at rest. Possible anxiety vs. PE. Started suddenly."
        ]
        
        print("\n📝 Demo Clinical Notes:")
        for i, note in enumerate(demo_notes, 1):
            print(f"   {i}. {note}")
        
        print("\n🔧 Enhanced Processing Features:")
        print("   • Medical abbreviation expansion (100+ abbreviations)")
        print("   • Sophisticated negation detection (20+ patterns)")
        print("   • Temporal relationship extraction (onset, duration, progression)")
        print("   • Uncertainty and speculation assessment")
        print("   • Clinical context awareness")
        print("   • Entity position tracking for NLP analysis")
        
        print("\n📊 Expected Enhancements:")
        print("   • 'SOB' → 'shortness of breath'")
        print("   • 'CP' → 'chest pain'")
        print("   • 'denies chest pain' → negated: true")
        print("   • 'started 3 days ago' → onset: '3 days'")
        print("   • 'possible pneumonia' → uncertainty: speculation")
        print("   • 'worsening' → progression: 'worsening'")
        
        print("\n✨ Enhanced Clinical Analysis Demo Complete!")
        print("Integration Points:")
        print("   • Seamless integration with existing Claude analysis")
        print("   • Preserves original API interface")
        print("   • Adds comprehensive NLP metadata")
        print("   • Maintains high confidence in enhanced entities")
        print("   • Graceful error handling if NLP fails")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Running Enhanced Clinical Analysis Tests")
    print("=" * 60)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run demo
    print("\n" + "=" * 60)
    run_enhanced_analysis_demo()