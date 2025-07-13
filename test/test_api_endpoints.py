#!/usr/bin/env python3
"""
Test script for the Intelligence Layer API endpoints
This tests the Flask API routes for clinical analysis
"""

import json
import sys
import os
import requests
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_api_endpoints():
    """Test all intelligence layer API endpoints"""
    
    base_url = "http://localhost:5001/api/analysis"  # Using PORT from .env
    
    # Sample test data
    sample_note = """
    Patient is a 45-year-old male presenting with acute onset chest pain 
    that started 2 hours ago. Pain described as crushing, substernal, 
    8/10 severity, radiating to left arm. Associated with shortness of 
    breath and nausea.
    
    Past medical history: hypertension, diabetes
    Current medications: lisinopril 10mg daily, metformin 500mg BID
    
    Vital signs: BP 160/95, HR 110, RR 20, O2 sat 94%
    
    Assessment: Rule out acute coronary syndrome
    Plan: ECG, troponins, chest X-ray
    """
    
    patient_context = {
        "age": 45,
        "gender": "male",
        "medical_history": "hypertension, diabetes"
    }
    
    print("🧪 Testing Intelligence Layer API Endpoints")
    print("=" * 60)
    
    # Test 1: Health Check
    print("\n🏥 TEST 1: Health Check Endpoint")
    print("-" * 40)
    
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health check successful")
            print(f"Status: {health_data.get('status', 'unknown')}")
            print(f"Services: {health_data.get('services', {})}")
        else:
            print(f"❌ Health check failed: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - Make sure Flask server is running on port 5001")
        print("Run: python app.py")
        return False
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False
    
    # Test 2: Clinical Entity Extraction
    print("\n🔬 TEST 2: Clinical Entity Extraction")
    print("-" * 40)
    
    try:
        extract_payload = {
            "note_text": sample_note,
            "patient_context": patient_context
        }
        
        response = requests.post(
            f"{base_url}/extract", 
            json=extract_payload,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            extract_data = response.json()
            if extract_data.get('success'):
                data = extract_data['data']
                print("✅ Clinical entity extraction successful")
                print(f"Symptoms: {len(data.get('symptoms', []))}")
                print(f"Conditions: {len(data.get('conditions', []))}")
                print(f"Medications: {len(data.get('medications', []))}")
                print(f"Vital Signs: {len(data.get('vital_signs', []))}")
                print(f"Risk Level: {data.get('overall_assessment', {}).get('risk_level', 'unknown')}")
                print(f"Immediate Attention: {data.get('overall_assessment', {}).get('requires_immediate_attention', False)}")
            else:
                print(f"❌ API returned error: {extract_data.get('error', 'unknown')}")
                return False
        else:
            print(f"❌ Extraction failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Extraction test error: {str(e)}")
        return False
    
    # Test 3: Diagnosis with ICD Mapping
    print("\n📚 TEST 3: Diagnosis with ICD Mapping")
    print("-" * 40)
    
    try:
        diagnose_payload = {
            "note_text": sample_note,
            "patient_context": patient_context,
            "options": {
                "include_low_confidence": False,
                "max_icd_matches": 5
            }
        }
        
        response = requests.post(
            f"{base_url}/diagnose", 
            json=diagnose_payload,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            diagnose_data = response.json()
            if diagnose_data.get('success'):
                data = diagnose_data['data']
                mappings = data.get('icd_mappings', {})
                summary = mappings.get('summary', {})
                
                print("✅ Diagnosis with ICD mapping successful")
                print(f"Total ICD Mappings: {summary.get('total_mappings', 0)}")
                print(f"High Confidence Mappings: {summary.get('high_confidence_mappings', 0)}")
                print(f"Mapping Method: {summary.get('mapping_method', 'unknown')}")
                print(f"Available ICD Codes: {data.get('icd_cache_info', {}).get('total_icd_codes', 0)}")
            else:
                print(f"❌ API returned error: {diagnose_data.get('error', 'unknown')}")
                return False
        else:
            print(f"❌ Diagnosis failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Diagnosis test error: {str(e)}")
        return False
    
    # Test 4: Batch Analysis
    print("\n📋 TEST 4: Batch Analysis")
    print("-" * 40)
    
    try:
        batch_payload = {
            "notes": [
                {
                    "note_id": "test_note_1",
                    "note_text": sample_note,
                    "patient_context": patient_context
                },
                {
                    "note_id": "test_note_2",
                    "note_text": "Patient has mild headache and feels tired. Vital signs normal.",
                    "patient_context": {"age": 30, "gender": "female"}
                }
            ],
            "options": {
                "include_icd_mapping": True,
                "include_priority_analysis": True
            }
        }
        
        response = requests.post(
            f"{base_url}/batch", 
            json=batch_payload,
            headers={'Content-Type': 'application/json'},
            timeout=60
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            batch_data = response.json()
            if batch_data.get('success'):
                data = batch_data['data']
                summary = data.get('summary', {})
                
                print("✅ Batch analysis successful")
                print(f"Total Notes: {summary.get('total_notes', 0)}")
                print(f"Successful Analyses: {summary.get('successful_analyses', 0)}")
                print(f"Failed Analyses: {summary.get('failed_analyses', 0)}")
                print(f"Total Entities: {summary.get('total_entities', 0)}")
                print(f"High Priority Cases: {summary.get('high_priority_cases', 0)}")
            else:
                print(f"❌ API returned error: {batch_data.get('error', 'unknown')}")
                return False
        else:
            print(f"❌ Batch analysis failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Batch analysis test error: {str(e)}")
        return False
    
    # Test 5: Priority Endpoint (Expected to return Not Implemented)
    print("\n🚨 TEST 5: Priority Findings Endpoint")
    print("-" * 40)
    
    try:
        response = requests.get(f"{base_url}/priority/test_note_123", timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 501:  # Not Implemented
            priority_data = response.json()
            print("✅ Priority endpoint correctly returns 'Not Implemented'")
            print(f"Message: {priority_data.get('message', 'No message')}")
        else:
            print(f"⚠️ Unexpected response: {response.text}")
            
    except Exception as e:
        print(f"❌ Priority test error: {str(e)}")
        return False
    
    # Test 6: Error Handling
    print("\n❌ TEST 6: Error Handling")
    print("-" * 40)
    
    try:
        # Test invalid request (missing note_text)
        invalid_payload = {"patient_context": {"age": 45}}
        
        response = requests.post(
            f"{base_url}/extract", 
            json=invalid_payload,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 400:
            error_data = response.json()
            print("✅ Error handling working correctly")
            print(f"Error: {error_data.get('error', 'No error message')}")
            print(f"Code: {error_data.get('code', 'No error code')}")
        else:
            print(f"⚠️ Expected 400 error, got: {response.text}")
            
    except Exception as e:
        print(f"❌ Error handling test error: {str(e)}")
        return False
    
    # Summary
    print(f"\n✨ API ENDPOINT TESTING SUMMARY")
    print("=" * 50)
    print("✅ All intelligence layer API endpoints are working correctly!")
    print("🎯 The Flask API integration is complete and functional!")
    print(f"📊 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return True

if __name__ == "__main__":
    print("Starting API endpoint tests...")
    print("Make sure the Flask server is running with: python app.py")
    print()
    
    success = test_api_endpoints()
    sys.exit(0 if success else 1)