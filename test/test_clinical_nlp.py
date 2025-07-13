#!/usr/bin/env python3
"""
Test Clinical NLP Utilities
Tests for negation detection, abbreviation expansion, and temporal extraction
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.utils.clinical_nlp import ClinicalNLPProcessor, create_clinical_nlp_processor


class TestClinicalNLPProcessor(unittest.TestCase):
    """Test cases for Clinical NLP Processor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.nlp_processor = create_clinical_nlp_processor()
    
    def test_abbreviation_expansion(self):
        """Test medical abbreviation expansion"""
        test_cases = [
            # Basic abbreviations
            ("Patient has sob and cp", "patient has shortness of breath and chest pain"),
            ("BP 140/90, HR 110", "blood pressure 140/90, heart rate 110"),
            ("UTI diagnosed, prescribed antibiotics", "urinary tract infection diagnosed, prescribed antibiotics"),
            
            # Multiple abbreviations
            ("Pt with DM, HTN, and CHF", "pt with diabetes mellitus, hypertension, and congestive heart failure"),
            ("ECG shows AF, BNP elevated", "electrocardiogram shows atrial fibrillation, brain natriuretic peptide elevated"),
            
            # Case insensitive
            ("SOB and CP present", "shortness of breath and chest pain present"),
            ("htn and dm controlled", "hypertension and diabetes mellitus controlled"),
            
            # With boundaries (shouldn't expand partial matches)
            ("subpoena document", "subpoena document"),  # Should not expand 'bp' in 'subpoena'
            ("absorption rate", "absorption rate"),  # Should not expand 'sob' in 'absorption'
        ]
        
        for original, expected in test_cases:
            with self.subTest(original=original):
                result = self.nlp_processor.expand_abbreviations(original)
                self.assertEqual(result.lower(), expected.lower())
    
    def test_negation_detection_direct(self):
        """Test direct negation detection"""
        test_cases = [
            # Direct negations
            {
                'text': "Patient has no fever today",
                'entity_pos': (12, 17),  # "fever"
                'expected_negated': True,
                'expected_strength': 1.0
            },
            {
                'text': "No chest pain reported",
                'entity_pos': (3, 13),  # "chest pain"
                'expected_negated': True,
                'expected_strength': 1.0
            },
            {
                'text': "Patient denies shortness of breath",
                'entity_pos': (15, 34),  # "shortness of breath"
                'expected_negated': True,
                'expected_strength': 0.9
            },
            {
                'text': "Absence of nausea and vomiting",
                'entity_pos': (11, 17),  # "nausea"
                'expected_negated': True,
                'expected_strength': 1.0
            },
            
            # Not negated
            {
                'text': "Patient has fever",
                'entity_pos': (12, 17),  # "fever"
                'expected_negated': False,
                'expected_strength': 0.0
            }
        ]
        
        for case in test_cases:
            with self.subTest(text=case['text']):
                result = self.nlp_processor.detect_negation(case['text'], case['entity_pos'])
                self.assertEqual(result['is_negated'], case['expected_negated'])
                if case['expected_negated']:
                    self.assertGreater(result['negation_strength'], 0.5)
    
    def test_negation_detection_medical(self):
        """Test medical-specific negation patterns"""
        test_cases = [
            {
                'text': "Chest X-ray negative for pneumonia",
                'entity_pos': (25, 34),  # "pneumonia"
                'expected_negated': True,
                'negation_type': 'test_result'
            },
            {
                'text': "We will rule out myocardial infarction",
                'entity_pos': (17, 39),  # "myocardial infarction"
                'expected_negated': True,
                'negation_type': 'diagnostic'
            },
            {
                'text': "Stroke is unlikely given the presentation",
                'entity_pos': (0, 6),  # "Stroke"
                'expected_negated': True,
                'negation_type': 'uncertainty'
            }
        ]
        
        for case in test_cases:
            with self.subTest(text=case['text']):
                result = self.nlp_processor.detect_negation(case['text'], case['entity_pos'])
                self.assertTrue(result['is_negated'])
                self.assertEqual(result['negation_type'], case['negation_type'])
    
    def test_temporal_extraction(self):
        """Test temporal information extraction"""
        test_cases = [
            {
                'text': "Chest pain started 3 days ago",
                'entity_pos': (0, 10),  # "Chest pain"
                'expected_onset': "3 days"
            },
            {
                'text': "Headache for the past 2 weeks",
                'entity_pos': (0, 8),  # "Headache"
                'expected_duration': "2 weeks"
            },
            {
                'text': "Fever began yesterday morning",
                'entity_pos': (0, 5),  # "Fever"
                'expected_onset': "1 day"
            },
            {
                'text': "Intermittent abdominal pain worsening over time",
                'entity_pos': (12, 27),  # "abdominal pain"
                'expected_pattern': "intermittent",
                'expected_progression': "worsening"
            }
        ]
        
        for case in test_cases:
            with self.subTest(text=case['text']):
                result = self.nlp_processor.extract_temporal_info(case['text'], case['entity_pos'])
                
                if 'expected_onset' in case:
                    self.assertEqual(result['onset'], case['expected_onset'])
                if 'expected_duration' in case:
                    self.assertEqual(result['duration'], case['expected_duration'])
                if 'expected_pattern' in case:
                    self.assertEqual(result['pattern'], case['expected_pattern'])
                if 'expected_progression' in case:
                    self.assertEqual(result['progression'], case['expected_progression'])
    
    def test_uncertainty_assessment(self):
        """Test uncertainty and speculation detection"""
        test_cases = [
            {
                'text': "Possible pneumonia on chest X-ray",
                'entity_pos': (9, 18),  # "pneumonia"
                'expected_uncertainty': True,
                'expected_type': 'speculation'
            },
            {
                'text': "Symptoms suggest viral infection",
                'entity_pos': (17, 32),  # "viral infection"
                'expected_uncertainty': True,
                'expected_type': 'speculation'
            },
            {
                'text': "Findings consistent with heart failure",
                'entity_pos': (25, 38),  # "heart failure"
                'expected_uncertainty': True,
                'expected_type': 'correlation'
            },
            {
                'text': "Questionable diagnosis of diabetes",
                'entity_pos': (26, 34),  # "diabetes"
                'expected_uncertainty': True,
                'expected_type': 'uncertainty'
            }
        ]
        
        for case in test_cases:
            with self.subTest(text=case['text']):
                result = self.nlp_processor.assess_uncertainty(case['text'], case['entity_pos'])
                self.assertTrue(result['has_uncertainty'])
                self.assertEqual(result['uncertainty_type'], case['expected_type'])
                self.assertLess(result['confidence_modifier'], 0)  # Should reduce confidence
    
    def test_entity_enhancement_integration(self):
        """Test complete entity enhancement with all NLP features"""
        entity = {
            'entity': 'chest pain',
            'confidence': 0.9,
            'severity': 'moderate'
        }
        
        # Test with negated entity
        text = "Patient denies any chest pain or discomfort"
        entity_pos = (19, 29)  # "chest pain"
        
        enhanced_entity = self.nlp_processor.enhance_entity_with_nlp(entity, text, entity_pos)
        
        # Check negation
        self.assertTrue(enhanced_entity['negated'])
        self.assertTrue(enhanced_entity['negation']['is_negated'])
        self.assertEqual(enhanced_entity['negation']['negation_type'], 'patient_reported')
        
        # Check confidence adjustment
        self.assertGreater(enhanced_entity['confidence'], 0.8)  # High confidence in negation
    
    def test_preprocess_clinical_text(self):
        """Test complete clinical text preprocessing"""
        original_text = "Pt c/o SOB and CP. BP 160/90, HR 110 bpm. H/O DM and HTN."
        
        processed = self.nlp_processor.preprocess_clinical_text(original_text)
        
        # Check abbreviation expansion
        self.assertIn("shortness of breath", processed.lower())
        self.assertIn("chest pain", processed.lower())
        self.assertIn("blood pressure", processed.lower())
        self.assertIn("heart rate", processed.lower())
        self.assertIn("beats per minute", processed.lower())
        self.assertIn("diabetes mellitus", processed.lower())
        self.assertIn("hypertension", processed.lower())
        
        # Check phrase normalization
        self.assertIn("complains of", processed.lower())
        self.assertIn("history of", processed.lower())
    
    def test_complex_clinical_scenarios(self):
        """Test complex real-world clinical scenarios"""
        
        # Scenario 1: Multiple negations with temporal info
        text1 = "Patient denies chest pain since last week but reports shortness of breath started yesterday"
        
        # Test chest pain (negated)
        cp_result = self.nlp_processor.detect_negation(text1, (15, 25))  # "chest pain"
        self.assertTrue(cp_result['is_negated'])
        
        # Test shortness of breath (not negated)
        sob_pos = text1.lower().find("shortness of breath")
        sob_result = self.nlp_processor.detect_negation(text1, (sob_pos, sob_pos + 19))
        self.assertFalse(sob_result['is_negated'])
        
        # Scenario 2: Uncertainty with temporal progression
        text2 = "Possible pneumonia that may be worsening over the past 3 days"
        pneumonia_pos = text2.lower().find("pneumonia")
        
        uncertainty = self.nlp_processor.assess_uncertainty(text2, (pneumonia_pos, pneumonia_pos + 9))
        self.assertTrue(uncertainty['has_uncertainty'])
        
        temporal = self.nlp_processor.extract_temporal_info(text2, (pneumonia_pos, pneumonia_pos + 9))
        self.assertEqual(temporal['progression'], 'worsening')
        self.assertEqual(temporal['duration'], '3 days')
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        
        # Empty text
        result = self.nlp_processor.detect_negation("", (0, 0))
        self.assertFalse(result['is_negated'])
        
        # Invalid entity position
        result = self.nlp_processor.detect_negation("Patient has fever", (100, 105))
        self.assertFalse(result['is_negated'])
        
        # Entity at text boundary
        text = "fever"
        result = self.nlp_processor.detect_negation(text, (0, 5))
        self.assertFalse(result['is_negated'])
        
        # Very long text
        long_text = "Patient has fever. " * 100
        result = self.nlp_processor.detect_negation(long_text, (12, 17))
        self.assertFalse(result['is_negated'])


def run_clinical_nlp_demo():
    """
    Demo function showing Clinical NLP capabilities
    """
    print("🧠 Clinical NLP Processor Demo")
    print("=" * 50)
    
    processor = create_clinical_nlp_processor()
    
    # Demo text with various clinical elements
    demo_text = """
    Patient is a 65 y/o male with h/o DM, HTN who presents with c/o SOB and CP.
    Pt denies fever but reports N/V. No chest pain at rest.
    Possible pneumonia vs. CHF exacerbation.
    Started 3 days ago, worsening since yesterday.
    ECG negative for STEMI. BNP elevated.
    """
    
    print(f"\n📝 Original Text:")
    print(demo_text.strip())
    
    # Demo abbreviation expansion
    print(f"\n📖 After Abbreviation Expansion:")
    expanded = processor.expand_abbreviations(demo_text)
    print(expanded.strip())
    
    # Demo full preprocessing
    print(f"\n🔧 After Full Preprocessing:")
    preprocessed = processor.preprocess_clinical_text(demo_text)
    print(preprocessed.strip())
    
    # Demo negation detection on specific entities
    print(f"\n🚫 Negation Detection Examples:")
    
    test_entities = [
        ("fever", "Should be negated (denies fever)"),
        ("chest pain", "Context dependent"),
        ("pneumonia", "Uncertain but not negated"),
        ("STEMI", "Should be negated (negative for)")
    ]
    
    for entity, description in test_entities:
        entity_pos = demo_text.lower().find(entity.lower())
        if entity_pos != -1:
            negation = processor.detect_negation(demo_text, (entity_pos, entity_pos + len(entity)))
            print(f"   • {entity}: {'NEGATED' if negation['is_negated'] else 'NOT NEGATED'} "
                  f"(confidence: {negation['confidence']:.2f}) - {description}")
    
    # Demo temporal extraction
    print(f"\n⏰ Temporal Information Extraction:")
    sob_pos = demo_text.lower().find("sob")
    if sob_pos != -1:
        temporal = processor.extract_temporal_info(demo_text, (sob_pos, sob_pos + 3))
        print(f"   • SOB onset: {temporal.get('onset', 'Not found')}")
        print(f"   • SOB progression: {temporal.get('progression', 'Not found')}")
    
    print(f"\n✨ Clinical NLP Demo Complete!")
    print("Key Features Demonstrated:")
    print("   • Medical abbreviation expansion (100+ abbreviations)")
    print("   • Sophisticated negation detection (20+ patterns)")
    print("   • Temporal relationship extraction")
    print("   • Uncertainty and speculation assessment")
    print("   • Clinical context awareness")


if __name__ == "__main__":
    print("🧪 Running Clinical NLP Tests")
    print("=" * 60)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run demo
    print("\n" + "=" * 60)
    run_clinical_nlp_demo()