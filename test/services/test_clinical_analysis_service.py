import pytest
import json
from unittest.mock import Mock, patch
from app.services.clinical_analysis_service import ClinicalAnalysisService

class TestClinicalAnalysisService:
    
    def setup_method(self):
        """Set up test fixtures"""
        self.service = ClinicalAnalysisService()
        
        # Sample patient note for testing
        self.sample_note = """
        Patient presents with chest pain that started 2 hours ago. 
        Pain is described as sharp, 8/10 severity, radiating to left arm. 
        Blood pressure 150/95, heart rate 110 bpm, temperature 98.6°F.
        Patient has history of hypertension, currently taking lisinopril 10mg daily.
        ECG shows ST elevation in leads II, III, aVF. 
        Troponin levels elevated at 2.5 ng/mL.
        Impression: Acute ST-elevation myocardial infarction.
        Plan: Emergency cardiac catheterization.
        """
        
        self.sample_patient_context = {
            "age": 65,
            "gender": "male",
            "medical_history": "hypertension, diabetes"
        }
        
    @patch('anthropic.Anthropic')
    def test_extract_clinical_entities_success(self, mock_anthropic):
        """Test successful clinical entity extraction"""
        # Mock Claude response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = json.dumps({
            "symptoms": [
                {
                    "entity": "chest pain",
                    "severity": "severe", 
                    "temporal": "acute",
                    "confidence": 0.95,
                    "text_span": "chest pain",
                    "negated": False
                }
            ],
            "conditions": [
                {
                    "entity": "acute ST-elevation myocardial infarction",
                    "status": "active",
                    "confidence": 0.9,
                    "text_span": "Acute ST-elevation myocardial infarction",
                    "icd_category": "cardiovascular"
                }
            ],
            "medications": [
                {
                    "entity": "lisinopril",
                    "dosage": "10mg",
                    "frequency": "daily",
                    "status": "current",
                    "confidence": 0.95,
                    "text_span": "lisinopril 10mg daily"
                }
            ],
            "vital_signs": [
                {
                    "entity": "blood pressure",
                    "value": "150/95",
                    "unit": "mmHg",
                    "abnormal": True,
                    "confidence": 0.95,
                    "text_span": "Blood pressure 150/95"
                }
            ],
            "procedures": [
                {
                    "entity": "emergency cardiac catheterization",
                    "status": "planned",
                    "date": "",
                    "confidence": 0.9,
                    "text_span": "Emergency cardiac catheterization"
                }
            ],
            "abnormal_findings": [
                {
                    "entity": "ST elevation",
                    "severity": "critical",
                    "requires_attention": True,
                    "confidence": 0.95,
                    "text_span": "ST elevation in leads II, III, aVF"
                }
            ],
            "overall_assessment": {
                "primary_concerns": ["acute myocardial infarction"],
                "risk_level": "critical",
                "requires_immediate_attention": True,
                "summary": "Patient presenting with acute STEMI requiring immediate intervention"
            }
        })
        
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        # Test extraction
        result = self.service.extract_clinical_entities(
            self.sample_note, 
            self.sample_patient_context
        )
        
        # Assertions
        assert result is not None
        assert 'symptoms' in result
        assert 'conditions' in result
        assert 'medications' in result
        assert 'vital_signs' in result
        assert 'procedures' in result
        assert 'abnormal_findings' in result
        assert 'overall_assessment' in result
        assert 'analysis_timestamp' in result
        assert 'model_version' in result
        
        # Check specific extractions
        assert len(result['symptoms']) > 0
        assert result['symptoms'][0]['entity'] == 'chest pain'
        assert result['overall_assessment']['requires_immediate_attention'] == True
        
    @patch('anthropic.Anthropic')
    def test_extract_clinical_entities_api_error(self, mock_anthropic):
        """Test handling of API errors"""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic.return_value = mock_client
        
        result = self.service.extract_clinical_entities(self.sample_note)
        
        # Should return empty result with error
        assert result is not None
        assert 'error' in result
        assert result['error'] == "API Error"
        assert result['symptoms'] == []
        assert result['overall_assessment']['summary'] == "Analysis failed"
        
    def test_get_high_priority_findings(self):
        """Test filtering of high-priority findings"""
        sample_result = {
            "symptoms": [
                {
                    "entity": "chest pain",
                    "severity": "severe",
                    "confidence": 0.9
                },
                {
                    "entity": "mild headache", 
                    "severity": "mild",
                    "confidence": 0.7
                }
            ],
            "abnormal_findings": [
                {
                    "entity": "ST elevation",
                    "severity": "critical",
                    "requires_attention": True,
                    "confidence": 0.95
                }
            ],
            "vital_signs": [
                {
                    "entity": "blood pressure",
                    "value": "150/95",
                    "abnormal": True,
                    "confidence": 0.9
                }
            ],
            "overall_assessment": {
                "requires_immediate_attention": True,
                "summary": "Critical patient condition",
                "risk_level": "critical"
            }
        }
        
        high_priority = self.service.get_high_priority_findings(sample_result)
        
        # Should find multiple high-priority items
        assert len(high_priority) > 0
        
        # Check for urgent assessment
        urgent_items = [item for item in high_priority if item['type'] == 'urgent_assessment']
        assert len(urgent_items) == 1
        
        # Check for abnormal findings
        abnormal_items = [item for item in high_priority if item['type'] == 'abnormal_finding']
        assert len(abnormal_items) == 1
        
        # Check for severe symptoms
        severe_symptoms = [item for item in high_priority if item['type'] == 'severe_symptom']
        assert len(severe_symptoms) == 1
        
    def test_batch_extract_entities(self):
        """Test batch processing of multiple notes"""
        notes_data = [
            {
                "note_id": "note_1",
                "note_text": "Patient has fever and cough",
                "patient_context": {"age": 45, "gender": "female"}
            },
            {
                "note_id": "note_2", 
                "note_text": "Follow-up visit, patient feeling better",
                "patient_context": {"age": 45, "gender": "female"}
            }
        ]
        
        with patch.object(self.service, 'extract_clinical_entities') as mock_extract:
            mock_extract.return_value = {
                "symptoms": [],
                "conditions": [],
                "medications": [],
                "vital_signs": [],
                "procedures": [],
                "abnormal_findings": [],
                "overall_assessment": {
                    "primary_concerns": [],
                    "risk_level": "low",
                    "requires_immediate_attention": False,
                    "summary": "Mock result"
                }
            }
            
            results = self.service.batch_extract_entities(notes_data)
            
            assert len(results) == 2
            assert results[0]['note_id'] == "note_1"
            assert results[1]['note_id'] == "note_2"
            assert mock_extract.call_count == 2
            
    def test_empty_extraction_result(self):
        """Test empty result structure"""
        result = self.service._empty_extraction_result("Test error")
        
        required_keys = ['symptoms', 'conditions', 'medications', 'vital_signs', 
                        'procedures', 'abnormal_findings', 'overall_assessment']
        
        for key in required_keys:
            assert key in result
            
        assert result['error'] == "Test error"
        assert 'analysis_timestamp' in result
        assert 'model_version' in result