"""Database schema creation for explainable AI features"""
import sys
import os
import logging

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from app.services.supabase_service import SupabaseService

logger = logging.getLogger(__name__)

class ExplainableAIDBCreator:
    """Creates database tables for explainable AI features"""
    
    def __init__(self):
        self.supabase = SupabaseService()
    
    def create_all_tables(self):
        """Create all explainable AI related tables"""
        print("Creating explainable AI database tables...")
        
        # Create tables in order (dependencies first)
        self.create_literature_evidence_table()
        self.create_entity_literature_mappings_table()
        self.create_pubmed_cache_table()
        self.create_reasoning_chains_table()
        self.create_uncertainty_analysis_table()
        self.create_treatment_pathways_table()
        
        # Create indexes for performance
        self.create_indexes()
        
        print("✅ All explainable AI tables created successfully!")
    
    def create_literature_evidence_table(self):
        """Create literature_evidence table for storing PubMed articles"""
        sql = """
        CREATE TABLE IF NOT EXISTS literature_evidence (
            evidence_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            pmid VARCHAR(20) NOT NULL UNIQUE,
            title TEXT NOT NULL,
            abstract TEXT,
            authors TEXT[],
            journal VARCHAR(500),
            publication_date DATE,
            study_type VARCHAR(100),
            evidence_quality_score DECIMAL(3,2),
            keywords TEXT[],
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
        """
        
        try:
            self.supabase.client.rpc('exec_sql', {'sql': sql}).execute()
            print("✅ Created literature_evidence table")
        except Exception as e:
            print(f"❌ Error creating literature_evidence table: {str(e)}")
    
    def create_entity_literature_mappings_table(self):
        """Create entity_literature_mappings table for linking entities to literature"""
        sql = """
        CREATE TABLE IF NOT EXISTS entity_literature_mappings (
            mapping_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            entity_id UUID REFERENCES clinical_entities(entity_id),
            evidence_id UUID REFERENCES literature_evidence(evidence_id),
            relevance_score DECIMAL(3,2),
            context_type VARCHAR(100),
            created_at TIMESTAMP DEFAULT NOW()
        );
        """
        
        try:
            self.supabase.client.rpc('exec_sql', {'sql': sql}).execute()
            print("✅ Created entity_literature_mappings table")
        except Exception as e:
            print(f"❌ Error creating entity_literature_mappings table: {str(e)}")
    
    def create_pubmed_cache_table(self):
        """Create pubmed_cache table for caching search results"""
        sql = """
        CREATE TABLE IF NOT EXISTS pubmed_cache (
            cache_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            query_hash VARCHAR(32) NOT NULL UNIQUE,
            query TEXT NOT NULL,
            results JSONB NOT NULL,
            result_count INTEGER DEFAULT 0,
            cached_at TIMESTAMP DEFAULT NOW(),
            last_accessed TIMESTAMP DEFAULT NOW(),
            access_count INTEGER DEFAULT 1
        );
        """
        
        try:
            self.supabase.client.rpc('exec_sql', {'sql': sql}).execute()
            print("✅ Created pubmed_cache table")
        except Exception as e:
            print(f"❌ Error creating pubmed_cache table: {str(e)}")
    
    def create_reasoning_chains_table(self):
        """Create reasoning_chains table for storing explanation chains"""
        sql = """
        CREATE TABLE IF NOT EXISTS reasoning_chains (
            chain_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id UUID REFERENCES analysis_sessions(session_id),
            step_number INTEGER NOT NULL,
            reasoning TEXT NOT NULL,
            evidence_type VARCHAR(100),
            confidence DECIMAL(3,2),
            supporting_literature TEXT[],
            created_at TIMESTAMP DEFAULT NOW()
        );
        """
        
        try:
            self.supabase.client.rpc('exec_sql', {'sql': sql}).execute()
            print("✅ Created reasoning_chains table")
        except Exception as e:
            print(f"❌ Error creating reasoning_chains table: {str(e)}")
    
    def create_uncertainty_analysis_table(self):
        """Create uncertainty_analysis table for storing uncertainty assessments"""
        sql = """
        CREATE TABLE IF NOT EXISTS uncertainty_analysis (
            analysis_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id UUID REFERENCES analysis_sessions(session_id),
            overall_confidence DECIMAL(3,2),
            uncertainty_sources TEXT[],
            recommendation VARCHAR(200),
            confidence_range JSONB,
            uncertainty_visualization JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """
        
        try:
            self.supabase.client.rpc('exec_sql', {'sql': sql}).execute()
            print("✅ Created uncertainty_analysis table")
        except Exception as e:
            print(f"❌ Error creating uncertainty_analysis table: {str(e)}")
    
    def create_treatment_pathways_table(self):
        """Create treatment_pathways table for storing alternative treatment options"""
        sql = """
        CREATE TABLE IF NOT EXISTS treatment_pathways (
            pathway_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id UUID REFERENCES analysis_sessions(session_id),
            pathway_name VARCHAR(200) NOT NULL,
            treatment_sequence JSONB NOT NULL,
            evidence_strength DECIMAL(3,2),
            contraindications TEXT[],
            estimated_outcomes JSONB,
            supporting_studies TEXT[],
            rank_score DECIMAL(3,2),
            created_at TIMESTAMP DEFAULT NOW()
        );
        """
        
        try:
            self.supabase.client.rpc('exec_sql', {'sql': sql}).execute()
            print("✅ Created treatment_pathways table")
        except Exception as e:
            print(f"❌ Error creating treatment_pathways table: {str(e)}")
    
    def create_indexes(self):
        """Create indexes for optimal performance"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_literature_pmid ON literature_evidence(pmid);",
            "CREATE INDEX IF NOT EXISTS idx_entity_literature ON entity_literature_mappings(entity_id, relevance_score DESC);",
            "CREATE INDEX IF NOT EXISTS idx_pubmed_cache_hash ON pubmed_cache(query_hash);",
            "CREATE INDEX IF NOT EXISTS idx_pubmed_cache_accessed ON pubmed_cache(last_accessed DESC);",
            "CREATE INDEX IF NOT EXISTS idx_reasoning_session ON reasoning_chains(session_id, step_number);",
            "CREATE INDEX IF NOT EXISTS idx_uncertainty_session ON uncertainty_analysis(session_id);",
            "CREATE INDEX IF NOT EXISTS idx_pathways_session ON treatment_pathways(session_id, rank_score DESC);"
        ]
        
        for index_sql in indexes:
            try:
                self.supabase.client.rpc('exec_sql', {'sql': index_sql}).execute()
            except Exception as e:
                print(f"❌ Error creating index: {str(e)}")
        
        print("✅ Created database indexes")
    
    def create_cache_cleanup_function(self):
        """Create PostgreSQL function for automatic cache cleanup"""
        sql = """
        CREATE OR REPLACE FUNCTION cleanup_expired_cache()
        RETURNS INTEGER AS $$
        DECLARE
            deleted_count INTEGER;
        BEGIN
            DELETE FROM pubmed_cache 
            WHERE cached_at < NOW() - INTERVAL '7 days';
            
            GET DIAGNOSTICS deleted_count = ROW_COUNT;
            RETURN deleted_count;
        END;
        $$ LANGUAGE plpgsql;
        """
        
        try:
            self.supabase.client.rpc('exec_sql', {'sql': sql}).execute()
            print("✅ Created cache cleanup function")
        except Exception as e:
            print(f"❌ Error creating cleanup function: {str(e)}")

if __name__ == "__main__":
    try:
        creator = ExplainableAIDBCreator()
        creator.create_all_tables()
        creator.create_cache_cleanup_function()
        print("\n🎉 Explainable AI database setup completed successfully!")
        
    except Exception as e:
        print(f"\n💥 Database setup failed: {str(e)}")
        logger.error(f"Database setup error: {str(e)}")