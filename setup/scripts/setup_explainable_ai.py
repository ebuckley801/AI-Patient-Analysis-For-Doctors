#!/usr/bin/env python3
"""Setup script for explainable AI features"""
import os
import sys
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'requests',
        'xml.etree.ElementTree', 
        'statistics',
        'anthropic',
        'supabase'
    ]
    
    # Add current directory to Python path for app imports
    import sys
    sys.path.insert(0, os.getcwd())
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if '.' in package:
                # Handle submodules like xml.etree.ElementTree
                main_package = package.split('.')[0]
                __import__(main_package)
            else:
                __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages before proceeding.")
        return False
    
    print("✅ All dependencies are available!")
    return True

def update_config():
    """Update configuration for explainable AI features"""
    print("\n🔧 Configuration setup...")
    
    config_updates = """
# Add these environment variables to your .env file:

# PubMed API Configuration (Optional - for literature integration)
PUBMED_API_KEY=your_ncbi_api_key_here
PUBMED_EMAIL=your_email@domain.com
PUBMED_TOOL_NAME=PatientAnalysis

# Explainability Settings
EXPLANATION_CACHE_TTL=604800  # 7 days
MAX_LITERATURE_RESULTS=20
DEFAULT_UNCERTAINTY_THRESHOLD=0.7

# Redis Configuration (Optional - for enhanced caching)
REDIS_URL=redis://localhost:6379/0
"""
    
    env_file = Path('.env')
    
    print("📝 Configuration requirements:")
    print(config_updates)
    
    if env_file.exists():
        print(f"✅ Found existing .env file")
        print("💡 Please add the above configuration to your .env file")
    else:
        print("⚠️  No .env file found")
        print("💡 Create a .env file and add the above configuration")
    
    return True

def setup_database():
    """Provide instructions for database setup"""
    print("\n🗄️  Database setup instructions:")
    
    print("""
To set up the explainable AI database tables, run:

    python app/utils/create_explainable_ai_db.py

This will create the following tables:
  • literature_evidence - Stores PubMed articles
  • entity_literature_mappings - Links entities to literature
  • pubmed_cache - Caches PubMed search results
  • reasoning_chains - Stores explanation reasoning steps
  • uncertainty_analysis - Stores uncertainty assessments
  • treatment_pathways - Stores alternative treatment options

Note: Make sure your Supabase database is accessible and the existing
intelligence layer tables are already created.
""")

def setup_flask_routes():
    """Provide instructions for Flask route integration"""
    print("\n🌐 Flask route integration:")
    
    print("""
To integrate the explainable AI routes into your Flask app, add this to your main app.py:

    from app.routes.explanation_routes import explanation_bp
    app.register_blueprint(explanation_bp)

New API endpoints will be available at:
  • POST /api/explanation/analyze - Explainable clinical analysis
  • GET /api/explanation/literature/<entity_id> - Literature evidence
  • POST /api/explanation/pathways - Treatment pathway exploration
  • GET /api/explanation/uncertainty/<analysis_id> - Uncertainty analysis
  • GET /api/explanation/health - Health check
  • GET /api/explanation/cache/stats - Cache statistics
""")

def run_tests():
    """Provide instructions for running tests"""
    print("\n🧪 Testing instructions:")
    
    print("""
To test the explainable AI services, run:

    python test/test_explainable_ai_services.py

This comprehensive test suite covers:
  • PubMed API integration and rate limiting
  • Cache service functionality
  • Uncertainty quantification algorithms
  • Treatment pathway exploration
  • Explainable clinical analysis service
  • End-to-end workflow simulation

For individual service testing:
    python -m pytest test/test_explainable_ai_services.py::TestPubMedService -v
    python -m pytest test/test_explainable_ai_services.py::TestUncertaintyCalculator -v
""")

def performance_considerations():
    """Provide performance optimization guidance"""
    print("\n⚡ Performance considerations:")
    
    print("""
For optimal performance:

1. PubMed API Rate Limiting:
   • Respects 9 requests/second limit (with API key: 10/sec)
   • Automatic rate limiting prevents API blocks
   • Consider getting NCBI API key for higher limits

2. Caching Strategy:
   • Literature searches cached for 7 days
   • Query results stored in database
   • Cache hit ratio monitoring available

3. Database Optimization:
   • Indexes created for optimal query performance
   • Automatic cleanup of expired cache entries
   • Consider Redis for enhanced caching

4. Memory Management:
   • Large literature results are paginated
   • Batch processing with configurable limits
   • Cleanup functions for old data
""")

def security_notes():
    """Provide security guidance"""
    print("\n🔒 Security considerations:")
    
    print("""
Security best practices:

1. API Keys:
   • Store PubMed API key securely in environment variables
   • Never commit API keys to version control
   • Use service accounts for production

2. Input Validation:
   • All inputs are sanitized through existing validation layer
   • SQL injection protection via parameterized queries
   • Rate limiting prevents abuse

3. Data Privacy:
   • Literature data is public domain
   • Patient context is processed securely
   • Audit trail for all analysis sessions

4. Access Control:
   • Same authentication as existing API endpoints
   • Consider role-based access for admin functions
   • Monitor usage through logs
""")

def main():
    """Main setup function"""
    print("🚀 Setting up Explainable AI Features for Patient Analysis")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Configuration
    update_config()
    
    # Database setup
    setup_database()
    
    # Flask integration
    setup_flask_routes()
    
    # Testing
    run_tests()
    
    # Performance
    performance_considerations()
    
    # Security
    security_notes()
    
    print(f"\n{'=' * 60}")
    print("🎉 Explainable AI setup completed!")
    print("=" * 60)
    
    print("""
Next steps:
1. Update your .env file with the required configuration
2. Run the database setup script
3. Integrate the routes into your Flask app
4. Run the tests to verify everything works
5. Start using the new explainable AI features!

For detailed implementation guidance, see:
  • .claude/explainable_ai_implementation_plan.md
  • test/test_explainable_ai_services.py for usage examples
""")

if __name__ == "__main__":
    main()