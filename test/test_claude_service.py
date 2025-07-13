#!/usr/bin/env python3
"""
Simple demonstration of the Clinical Analysis Service
This will test the Claude integration with a sample patient note
"""

import json
import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.services.clinical_analysis_service import ClinicalAnalysisService

def test_clinical_analysis():
    """Test the clinical analysis service with a sample patient note"""
    
    # Sample patient note
    sample_note = """
    Patient is a 58-year-old male presenting to the emergency department with 
    acute onset chest pain that started approximately 3 hours ago while at rest. 
    
    The pain is described as crushing, substernal, 9/10 severity, with radiation 
    to the left arm and jaw. Associated symptoms include shortness of breath, 
    nausea, and diaphoresis.
    
    Past medical history significant for hypertension and hyperlipidemia. 
    Current medications include metoprolol 50mg twice daily and atorvastatin 40mg nightly.
    
    Physical exam reveals an anxious-appearing male in moderate distress. 
    Vital signs: BP 165/98, HR 105, RR 22, O2 sat 96% on room air, temp 98.2°F.
    
    ECG shows ST-segment elevation in leads II, III, and aVF consistent with 
    inferior STEMI. Troponin I elevated at 5.2 ng/mL.
    
    Assessment: Acute ST-elevation myocardial infarction (inferior)
    Plan: Immediate cardiac catheterization, aspirin 325mg, metoprolol, 
    atorvastatin, and heparin per protocol.
    """
    
    patient_context = {
        "age": 58,
        "gender": "male",
        "medical_history": "hypertension, hyperlipidemia"
    }
    
    try:
        print("🔬 Testing Clinical Analysis Service...")
        print("-" * 60)
        
        # Initialize service
        service = ClinicalAnalysisService()
        
        # Extract clinical entities
        print("📝 Analyzing patient note...")
        result = service.extract_clinical_entities(sample_note, patient_context)
        
        # Display results
        print("\n✅ Analysis Complete!")
        print(f"📊 Model: {result.get('model_version', 'Unknown')}")
        print(f"⏰ Timestamp: {result.get('analysis_timestamp', 'Unknown')}")
        
        # Show extracted entities
        print(f"\n🔍 Extracted Entities:")
        print(f"  • Symptoms: {len(result.get('symptoms', []))}")
        print(f"  • Conditions: {len(result.get('conditions', []))}")
        print(f"  • Medications: {len(result.get('medications', []))}")
        print(f"  • Vital Signs: {len(result.get('vital_signs', []))}")
        print(f"  • Procedures: {len(result.get('procedures', []))}")
        print(f"  • Abnormal Findings: {len(result.get('abnormal_findings', []))}")
        
        # Show overall assessment
        assessment = result.get('overall_assessment', {})
        print(f"\n🏥 Clinical Assessment:")
        print(f"  • Risk Level: {assessment.get('risk_level', 'Unknown')}")
        print(f"  • Immediate Attention: {assessment.get('requires_immediate_attention', False)}")
        print(f"  • Summary: {assessment.get('summary', 'No summary available')}")
        
        # Show some specific extractions
        if result.get('conditions'):
            print(f"\n🩺 Key Conditions:")
            for condition in result['conditions'][:3]:  # Show first 3
                print(f"  • {condition.get('entity', 'Unknown')} (confidence: {condition.get('confidence', 0):.2f})")
        
        if result.get('symptoms'):
            print(f"\n😷 Key Symptoms:")
            for symptom in result['symptoms'][:3]:  # Show first 3
                print(f"  • {symptom.get('entity', 'Unknown')} - {symptom.get('severity', 'Unknown')} (confidence: {symptom.get('confidence', 0):.2f})")
        
        # Test high priority findings
        print(f"\n🚨 High Priority Findings:")
        high_priority = service.get_high_priority_findings(result)
        for finding in high_priority[:5]:  # Show first 5
            print(f"  • {finding.get('type', 'Unknown')}: {finding.get('entity', 'N/A')}")
        
        print(f"\n✨ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_clinical_analysis()
    sys.exit(0 if success else 1)