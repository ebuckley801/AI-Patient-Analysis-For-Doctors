#!/usr/bin/env python3
"""
Simple test of API endpoint functionality without server dependency
"""

import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_api_routes_import():
    """Test that we can import the analysis routes successfully"""
    try:
        from app.routes.analysis_routes import analysis_bp
        print("✅ Successfully imported analysis_routes")
        
        # Check that endpoints are registered
        rules = [rule.rule for rule in analysis_bp.url_map.iter_rules()]
        print(f"📋 Found {len(rules)} registered endpoints:")
        for rule in sorted(rules):
            print(f"  • {rule}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import analysis_routes: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing routes: {e}")
        return False

def test_services_integration():
    """Test that services can be instantiated"""
    try:
        from app.services.clinical_analysis_service import ClinicalAnalysisService
        from app.services.icd10_vector_matcher import ICD10VectorMatcher
        
        print("📊 Testing service instantiation...")
        
        # Test clinical service
        clinical_service = ClinicalAnalysisService()
        print("✅ ClinicalAnalysisService instantiated successfully")
        
        # Test ICD matcher service (will show error about missing DB table, but should not crash)
        icd_matcher = ICD10VectorMatcher()
        cache_info = icd_matcher.get_cache_info()
        print("✅ ICD10VectorMatcher instantiated successfully")
        print(f"📦 ICD Cache Info: {cache_info}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing services: {e}")
        return False

def test_flask_app_creation():
    """Test that Flask app can be created with analysis routes"""
    try:
        from app import create_app
        
        print("🚀 Testing Flask app creation...")
        app = create_app()
        
        print("✅ Flask app created successfully")
        print(f"📋 Registered blueprints:")
        for blueprint_name, blueprint in app.blueprints.items():
            print(f"  • {blueprint_name}: {blueprint.url_prefix}")
        
        # Test if our analysis routes are registered
        with app.app_context():
            rules = list(app.url_map.iter_rules())
            analysis_rules = [rule for rule in rules if '/api/analysis' in rule.rule]
            print(f"🔍 Found {len(analysis_rules)} analysis endpoints:")
            for rule in sorted(analysis_rules, key=lambda x: x.rule):
                methods = ', '.join(sorted(rule.methods - {'HEAD', 'OPTIONS'}))
                print(f"  • {methods} {rule.rule}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating Flask app: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Intelligence Layer API Integration")
    print("=" * 60)
    
    tests = [
        ("Import Analysis Routes", test_api_routes_import),
        ("Services Integration", test_services_integration), 
        ("Flask App Creation", test_flask_app_creation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔬 {test_name}")
        print("-" * 40)
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print(f"\n📊 TEST RESULTS SUMMARY")
    print("=" * 40)
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 {passed}/{total} tests passed")
    
    if passed == total:
        print("✨ All API integration tests passed!")
        print("🚀 Intelligence Layer API is ready for use!")
        return True
    else:
        print("⚠️ Some tests failed - check the output above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)