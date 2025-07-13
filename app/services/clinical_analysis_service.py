import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import anthropic
from app.config.config import Config

logger = logging.getLogger(__name__)

class ClinicalAnalysisService:
    """Service for extracting clinical insights from patient notes using Claude AI"""
    
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=Config.ANTHROPIC_KEY)
        
    def extract_clinical_entities(self, patient_note: str, patient_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Extract clinical entities from patient note text
        
        Args:
            patient_note: Raw text of the patient note
            patient_context: Optional patient demographics (age, gender, etc.)
            
        Returns:
            Dict containing extracted entities with confidence scores
        """
        try:
            prompt = self._build_extraction_prompt(patient_note, patient_context)
            
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0.1,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            result = self._parse_claude_response(response.content[0].text)
            result['analysis_timestamp'] = datetime.utcnow().isoformat()
            result['model_version'] = "claude-3-5-sonnet-20241022"
            
            return result
            
        except Exception as e:
            logger.error(f"Error in clinical entity extraction: {str(e)}")
            return self._empty_extraction_result(error=str(e))
    
    def _build_extraction_prompt(self, patient_note: str, patient_context: Optional[Dict] = None) -> str:
        """Build the prompt for Claude to extract clinical entities"""
        
        context_info = ""
        if patient_context:
            context_info = f"""
Patient Context:
- Age: {patient_context.get('age', 'Not specified')}
- Gender: {patient_context.get('gender', 'Not specified')}
- Medical History: {patient_context.get('medical_history', 'Not specified')}
"""
        
        prompt = f"""You are a clinical AI assistant analyzing patient notes to extract structured medical information. Extract the following clinical entities from the patient note below:

{context_info}

Patient Note:
{patient_note}

Please extract and categorize the following information in JSON format:

{{
  "symptoms": [
    {{
      "entity": "symptom name",
      "severity": "mild/moderate/severe/critical",
      "temporal": "acute/chronic/onset_date",
      "confidence": 0.0-1.0,
      "text_span": "exact text from note",
      "negated": true/false
    }}
  ],
  "conditions": [
    {{
      "entity": "condition/diagnosis name",
      "status": "active/resolved/suspected/ruled_out",
      "confidence": 0.0-1.0,
      "text_span": "exact text from note",
      "icd_category": "general category if obvious"
    }}
  ],
  "medications": [
    {{
      "entity": "medication name",
      "dosage": "dosage if mentioned",
      "frequency": "frequency if mentioned",
      "status": "current/discontinued/prescribed",
      "confidence": 0.0-1.0,
      "text_span": "exact text from note"
    }}
  ],
  "vital_signs": [
    {{
      "entity": "vital sign type",
      "value": "measured value",
      "unit": "unit of measurement",
      "abnormal": true/false,
      "confidence": 0.0-1.0,
      "text_span": "exact text from note"
    }}
  ],
  "procedures": [
    {{
      "entity": "procedure name",
      "status": "planned/completed/cancelled",
      "date": "date if mentioned",
      "confidence": 0.0-1.0,
      "text_span": "exact text from note"
    }}
  ],
  "abnormal_findings": [
    {{
      "entity": "abnormal finding",
      "severity": "mild/moderate/severe/critical",
      "requires_attention": true/false,
      "confidence": 0.0-1.0,
      "text_span": "exact text from note"
    }}
  ],
  "overall_assessment": {{
    "primary_concerns": ["list of main medical concerns"],
    "risk_level": "low/moderate/high/critical",
    "requires_immediate_attention": true/false,
    "summary": "brief clinical summary"
  }}
}}

Important guidelines:
1. Only extract entities that are explicitly mentioned in the text
2. Pay attention to negation (e.g., "no fever" should have negated: true)
3. Confidence scores should reflect certainty based on clinical context
4. Use medical terminology consistently
5. Flag any concerning findings that need immediate attention
6. If information is unclear or ambiguous, lower the confidence score
7. Include exact text spans for traceability

Return only the JSON object, no additional text."""

        return prompt
    
    def _parse_claude_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Claude's JSON response and validate structure"""
        try:
            # Extract JSON from response (handle cases where Claude adds explanation)
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            result = json.loads(response_text)
            
            # Validate required structure
            required_keys = ['symptoms', 'conditions', 'medications', 'vital_signs', 'procedures', 'abnormal_findings', 'overall_assessment']
            for key in required_keys:
                if key not in result:
                    result[key] = []
            
            # Ensure overall_assessment has required structure
            if 'overall_assessment' not in result or not isinstance(result['overall_assessment'], dict):
                result['overall_assessment'] = {
                    "primary_concerns": [],
                    "risk_level": "low",
                    "requires_immediate_attention": False,
                    "summary": "Unable to generate assessment"
                }
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response as JSON: {str(e)}")
            return self._empty_extraction_result(error="JSON parsing error")
        except Exception as e:
            logger.error(f"Error parsing Claude response: {str(e)}")
            return self._empty_extraction_result(error=str(e))
    
    def _empty_extraction_result(self, error: str = None) -> Dict[str, Any]:
        """Return empty extraction result structure"""
        return {
            "symptoms": [],
            "conditions": [],
            "medications": [],
            "vital_signs": [],
            "procedures": [],
            "abnormal_findings": [],
            "overall_assessment": {
                "primary_concerns": [],
                "risk_level": "unknown",
                "requires_immediate_attention": False,
                "summary": "Analysis failed"
            },
            "error": error,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "model_version": "claude-3-5-sonnet-20241022"
        }
    
    def batch_extract_entities(self, notes_with_context: List[Dict]) -> List[Dict[str, Any]]:
        """
        Extract entities from multiple patient notes
        
        Args:
            notes_with_context: List of dicts with 'note_text' and optional 'patient_context'
            
        Returns:
            List of extraction results
        """
        results = []
        for note_data in notes_with_context:
            note_text = note_data.get('note_text', '')
            patient_context = note_data.get('patient_context')
            
            result = self.extract_clinical_entities(note_text, patient_context)
            result['note_id'] = note_data.get('note_id')
            results.append(result)
            
        return results
    
    def get_high_priority_findings(self, extraction_result: Dict[str, Any]) -> List[Dict]:
        """
        Filter extraction results to return only high-priority findings
        
        Args:
            extraction_result: Result from extract_clinical_entities
            
        Returns:
            List of high-priority findings requiring attention
        """
        high_priority = []
        
        # Check overall assessment
        if extraction_result.get('overall_assessment', {}).get('requires_immediate_attention'):
            high_priority.append({
                'type': 'urgent_assessment',
                'summary': extraction_result['overall_assessment'].get('summary', ''),
                'risk_level': extraction_result['overall_assessment'].get('risk_level', 'unknown')
            })
        
        # Check abnormal findings
        for finding in extraction_result.get('abnormal_findings', []):
            if finding.get('requires_attention') or finding.get('severity') in ['severe', 'critical']:
                high_priority.append({
                    'type': 'abnormal_finding',
                    'entity': finding.get('entity'),
                    'severity': finding.get('severity'),
                    'confidence': finding.get('confidence')
                })
        
        # Check critical vital signs
        for vital in extraction_result.get('vital_signs', []):
            if vital.get('abnormal') and vital.get('confidence', 0) > 0.7:
                high_priority.append({
                    'type': 'abnormal_vital',
                    'entity': vital.get('entity'),
                    'value': vital.get('value'),
                    'confidence': vital.get('confidence')
                })
        
        # Check severe symptoms
        for symptom in extraction_result.get('symptoms', []):
            if symptom.get('severity') in ['severe', 'critical'] and symptom.get('confidence', 0) > 0.7:
                high_priority.append({
                    'type': 'severe_symptom',
                    'entity': symptom.get('entity'),
                    'severity': symptom.get('severity'),
                    'confidence': symptom.get('confidence')
                })
        
        return high_priority