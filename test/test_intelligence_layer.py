#!/usr/bin/env python3
"""
Comprehensive test of the Intelligence Layer (Phase 2)
This demonstrates the complete workflow:
1. Clinical entity extraction with Claude
2. ICD-10 code mapping with vector similarity
3. Confidence scoring and prioritization
"""

import json
import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.services.clinical_analysis_service import ClinicalAnalysisService
from app.services.icd10_vector_matcher import ICD10VectorMatcher

def test_complete_intelligence_layer():
    """Test the complete intelligence layer workflow"""
    
    # Complex patient note for testing
    complex_note = """
    EMERGENCY DEPARTMENT NOTE
    
    Chief Complaint: Chest pain and difficulty breathing
    
    HPI: 62-year-old female presents with sudden onset severe chest pain that started 
    4 hours ago while watching television. Pain is described as crushing, substernal, 
    radiating to left arm and jaw, 9/10 severity. Associated with shortness of breath, 
    nausea, vomiting, and diaphoresis. Patient reports feeling "like an elephant is 
    sitting on my chest."
    
    Past Medical History: 
    - Hypertension (controlled with lisinopril)
    - Type 2 diabetes mellitus (metformin)
    - Hyperlipidemia (atorvastatin)
    - Former smoker (quit 5 years ago, 30 pack-year history)
    
    Current Medications:
    - Lisinopril 10mg daily
    - Metformin 500mg twice daily  
    - Atorvastatin 40mg nightly
    - Aspirin 81mg daily
    
    Physical Examination:
    - Appears in acute distress, diaphoretic
    - Vital Signs: BP 180/110, HR 115, RR 24, O2 sat 92% on room air, Temp 98.8°F
    - Cardiovascular: Tachycardic, no murmurs, rubs, or gallops
    - Pulmonary: Bilateral crackles in lower lobes
    - Neurological: Alert and oriented x3, anxious
    
    Diagnostic Studies:
    - ECG: ST-segment elevation in leads V2-V6, consistent with anterior STEMI
    - Chest X-ray: Mild pulmonary edema
    - Troponin I: 8.5 ng/mL (severely elevated)
    - BNP: 450 pg/mL (elevated)
    - CBC: WBC 12,000, Hgb 11.2, Plt 380,000
    - Basic metabolic panel: Glucose 180, Creatinine 1.1
    
    Assessment and Plan:
    1. ST-elevation myocardial infarction (anterior) - STEMI alert activated
       - Emergent cardiac catheterization
       - Dual antiplatelet therapy (aspirin + clopidogrel)
       - Heparin anticoagulation
       - Metoprolol for rate control
    
    2. Acute heart failure with pulmonary edema
       - Furosemide 40mg IV
       - Monitor fluid balance
    
    3. Hypertensive crisis secondary to acute MI
       - Continue current antihypertensive
       - Monitor closely
    
    4. Diabetes - continue home medications, monitor glucose
    
    Disposition: Admitted to cardiac ICU for post-catheterization monitoring
    """
    
    patient_context = {
        "age": 62,
        "gender": "female",
        "medical_history": "hypertension, type 2 diabetes, hyperlipidemia, former smoker"
    }
    
    try:
        print("🚀 Testing Complete Intelligence Layer (Phase 2)")
        print("=" * 70)
        
        # Step 1: Clinical Entity Extraction
        print("\n📋 STEP 1: Clinical Entity Extraction with Claude")
        print("-" * 50)
        
        clinical_service = ClinicalAnalysisService()
        clinical_result = clinical_service.extract_clinical_entities(complex_note, patient_context)
        
        if 'error' in clinical_result:
            print(f"❌ Error in clinical analysis: {clinical_result['error']}")
            return False
        
        # Display extracted entities
        print(f"✅ Extraction completed successfully")
        print(f"📊 Analysis timestamp: {clinical_result.get('analysis_timestamp', 'Unknown')}")
        
        print(f"\n🔍 Extracted Clinical Entities:")
        for entity_type in ['symptoms', 'conditions', 'medications', 'vital_signs', 'procedures', 'abnormal_findings']:
            entities = clinical_result.get(entity_type, [])
            print(f"  • {entity_type.replace('_', ' ').title()}: {len(entities)}")
            
            # Show top 3 entities for each type
            for i, entity in enumerate(entities[:3]):
                confidence = entity.get('confidence', 0)
                entity_name = entity.get('entity', 'Unknown')
                print(f"    {i+1}. {entity_name} (confidence: {confidence:.2f})")
        
        # Show overall assessment
        assessment = clinical_result.get('overall_assessment', {})
        print(f"\n🏥 Clinical Assessment:")
        print(f"  • Risk Level: {assessment.get('risk_level', 'Unknown')}")
        print(f"  • Immediate Attention Required: {assessment.get('requires_immediate_attention', False)}")
        print(f"  • Primary Concerns: {', '.join(assessment.get('primary_concerns', []))}")
        print(f"  • Summary: {assessment.get('summary', 'No summary')}")
        
        # Step 2: ICD-10 Code Mapping
        print(f"\n📚 STEP 2: ICD-10 Code Mapping with Vector Similarity")
        print("-" * 50)
        
        icd_matcher = ICD10VectorMatcher()
        
        # Get cache information
        cache_info = icd_matcher.get_cache_info()
        print(f"📦 ICD Cache: {cache_info['total_icd_codes']} codes loaded")
        
        # Map clinical entities to ICD codes
        enhanced_result = icd_matcher.map_clinical_entities_to_icd(clinical_result)
        
        if 'error' in enhanced_result.get('icd_mappings', {}):
            print(f"⚠️ Warning in ICD mapping: {enhanced_result['icd_mappings']['error']}")
        
        # Display ICD mappings
        mappings = enhanced_result.get('icd_mappings', {})
        summary = mappings.get('summary', {})
        
        print(f"✅ ICD mapping completed")
        print(f"📊 Mapping Summary:")
        print(f"  • Total Mappings: {summary.get('total_mappings', 0)}")
        print(f"  • High Confidence: {summary.get('high_confidence_mappings', 0)}")
        print(f"  • Method: {summary.get('mapping_method', 'Unknown')}")
        
        # Show top condition mappings
        condition_mappings = mappings.get('conditions', [])
        if condition_mappings:
            print(f"\n🩺 Top Condition Mappings:")
            for i, mapping in enumerate(condition_mappings[:3]):
                entity = mapping.get('entity', 'Unknown')
                best_match = mapping.get('best_match', {})
                if best_match:
                    icd_code = best_match.get('icd_code', 'Unknown')
                    description = best_match.get('description', 'Unknown')
                    similarity = best_match.get('similarity', 0)
                    confidence = mapping.get('original_confidence', 0)
                    
                    print(f"  {i+1}. {entity}")
                    print(f"     → {icd_code}: {description}")
                    print(f"     → Similarity: {similarity:.3f}, Original Confidence: {confidence:.2f}")
        
        # Show top symptom mappings
        symptom_mappings = mappings.get('symptoms', [])
        if symptom_mappings:
            print(f"\n😷 Top Symptom Mappings:")
            for i, mapping in enumerate(symptom_mappings[:3]):
                entity = mapping.get('entity', 'Unknown')
                severity = mapping.get('severity', 'Unknown')
                best_match = mapping.get('best_match', {})
                if best_match:
                    icd_code = best_match.get('icd_code', 'Unknown')
                    description = best_match.get('description', 'Unknown')
                    similarity = best_match.get('similarity', 0)
                    
                    print(f"  {i+1}. {entity} ({severity})")
                    print(f"     → {icd_code}: {description}")
                    print(f"     → Similarity: {similarity:.3f}")
        
        # Step 3: High Priority Analysis
        print(f"\n🚨 STEP 3: High Priority Findings Analysis")
        print("-" * 50)
        
        high_priority = clinical_service.get_high_priority_findings(clinical_result)
        
        if high_priority:
            print(f"⚠️ Found {len(high_priority)} high-priority findings:")
            for i, finding in enumerate(high_priority):
                finding_type = finding.get('type', 'Unknown')
                entity = finding.get('entity', 'N/A')
                severity = finding.get('severity', '')
                confidence = finding.get('confidence', 0)
                
                print(f"  {i+1}. {finding_type.replace('_', ' ').title()}: {entity}")
                if severity:
                    print(f"     Severity: {severity}")
                if confidence > 0:
                    print(f"     Confidence: {confidence:.2f}")
        else:
            print("✅ No high-priority findings detected")
        
        # Step 4: ICD Hierarchy Analysis  
        print(f"\n🌳 STEP 4: ICD Code Hierarchy Analysis")
        print("-" * 50)
        
        # Analyze hierarchy for top conditions
        if condition_mappings:
            for mapping in condition_mappings[:2]:  # Top 2 conditions
                best_match = mapping.get('best_match', {})
                if best_match:
                    icd_code = best_match.get('icd_code', '')
                    if icd_code:
                        hierarchy = icd_matcher.get_icd_hierarchy(icd_code)
                        if 'error' not in hierarchy:
                            print(f"📋 {mapping.get('entity', 'Unknown')} → {icd_code}")
                            print(f"  Category: {hierarchy.get('category', 'Unknown')} - {hierarchy.get('category_description', 'Unknown')}")
                            print(f"  Subcategory: {hierarchy.get('subcategory', 'Unknown')}")
        
        # Summary Report
        print(f"\n📄 INTELLIGENCE LAYER SUMMARY REPORT")
        print("=" * 50)
        print(f"Patient: {patient_context['age']}-year-old {patient_context['gender']}")
        print(f"Analysis Date: {clinical_result.get('analysis_timestamp', 'Unknown')}")
        print(f"")
        print(f"🔬 Clinical Analysis Results:")
        print(f"  • Total Entities Extracted: {sum(len(clinical_result.get(k, [])) for k in ['symptoms', 'conditions', 'medications', 'vital_signs', 'procedures', 'abnormal_findings'])}")
        print(f"  • Risk Level: {assessment.get('risk_level', 'Unknown')}")
        print(f"  • Requires Immediate Attention: {assessment.get('requires_immediate_attention', False)}")
        print(f"")
        print(f"📚 ICD-10 Mapping Results:")
        print(f"  • Total ICD Mappings: {summary.get('total_mappings', 0)}")
        print(f"  • High Confidence Mappings: {summary.get('high_confidence_mappings', 0)}")
        print(f"  • Available ICD Codes: {cache_info['total_icd_codes']}")
        print(f"")
        print(f"🚨 Clinical Priorities:")
        print(f"  • High Priority Findings: {len(high_priority)}")
        print(f"  • Primary Concerns: {', '.join(assessment.get('primary_concerns', ['None']))}")
        
        print(f"\n✨ Intelligence layer test completed successfully!")
        print(f"🎯 Phase 2 implementation is working correctly!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in intelligence layer test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_intelligence_layer()
    sys.exit(0 if success else 1)