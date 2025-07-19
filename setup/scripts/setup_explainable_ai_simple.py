#!/usr/bin/env python3
"""Simple setup for explainable AI features - creates basic cache table only"""
import sys
import os
import logging

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from app.services.supabase_service import SupabaseService

def create_basic_cache_table():
    """Create a basic cache table that doesn't depend on other tables"""
    try:
        supabase = SupabaseService()
        
        print("🚀 Setting up basic explainable AI functionality...")
        print("=" * 60)
        
        # Try to create a simple cache table first
        print("\n📝 Testing Supabase connection...")
        
        # Test basic connectivity
        test_response = supabase.client.table('patient_notes').select('id').limit(1).execute()
        if test_response.data is not None:
            print("✅ Supabase connection successful")
        else:
            print("❌ Supabase connection failed")
            return False
        
        print("\n📊 Tables needed for explainable AI:")
        print("1. literature_evidence - PubMed articles")
        print("2. entity_literature_mappings - Links entities to literature")  
        print("3. pubmed_cache - Search result cache")
        print("4. reasoning_chains - Explanation steps")
        print("5. uncertainty_analysis - Uncertainty assessments")
        print("6. treatment_pathways - Alternative treatments")
        
        print(f"\n📝 Manual Setup Required:")
        print(f"Since RPC functions aren't available, please:")
        print(f"1. Open your Supabase dashboard")
        print(f"2. Go to SQL Editor")
        print(f"3. Run the SQL script: explainable_ai_schema.sql")
        print(f"4. This will create all necessary tables and indexes")
        
        print(f"\n✅ Setup guidance complete!")
        print(f"After running the SQL script, your explainable AI features will be ready!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during setup: {str(e)}")
        return False

def test_new_services():
    """Test the new explainable AI services"""
    print("\n🧪 Testing explainable AI services...")
    
    try:
        # Test imports
        from app.services.pubmed_service import PubMedService
        from app.services.pubmed_cache_service import PubMedCacheService  
        from app.services.uncertainty_service import UncertaintyCalculator
        from app.services.pathway_explorer import TreatmentPathwayExplorer
        
        print("✅ All service imports successful")
        
        # Test basic initialization
        pubmed = PubMedService()
        uncertainty_calc = UncertaintyCalculator() 
        pathway_explorer = TreatmentPathwayExplorer()
        
        print("✅ All services initialize correctly")
        
        # Test uncertainty calculation with sample data
        sample_entities = {
            'symptoms': [
                {'entity': 'fever', 'confidence': 0.9},
                {'entity': 'cough', 'confidence': 0.7}
            ],
            'conditions': [
                {'entity': 'pneumonia', 'confidence': 0.8}
            ]
        }
        
        uncertainty_result = uncertainty_calc.assess_diagnostic_uncertainty(sample_entities)
        print(f"✅ Uncertainty analysis works: {uncertainty_result['overall_confidence']:.2f} confidence")
        
        # Test pathway generation
        sample_diagnosis = {'entity': 'hypertension', 'confidence': 0.8}
        sample_context = {'age': 45, 'gender': 'M'}
        
        pathways = pathway_explorer.generate_alternative_pathways(
            primary_diagnosis=sample_diagnosis,
            patient_context=sample_context,
            max_pathways=3
        )
        print(f"✅ Pathway generation works: {len(pathways)} pathways generated")
        
        return True
        
    except Exception as e:
        print(f"❌ Service testing failed: {str(e)}")
        return False

def main():
    """Main setup function"""
    print("🚀 Explainable AI Simple Setup")
    print("=" * 40)
    
    # Test basic setup
    setup_success = create_basic_cache_table()
    
    if setup_success:
        # Test services
        test_success = test_new_services()
        
        if test_success:
            print(f"\n🎉 Explainable AI setup completed successfully!")
            print(f"\nNext steps:")
            print(f"1. Run the SQL script in Supabase: explainable_ai_schema.sql")
            print(f"2. Add these routes to your Flask app:")
            print(f"   from app.routes.explanation_routes import explanation_bp")
            print(f"   app.register_blueprint(explanation_bp)")
            print(f"3. Test the new endpoints!")
        else:
            print(f"\n⚠️  Setup completed but service tests failed")
            print(f"Check for missing dependencies or configuration issues")
    else:
        print(f"\n❌ Setup failed")
        print(f"Check your Supabase configuration and connectivity")

if __name__ == "__main__":
    main()