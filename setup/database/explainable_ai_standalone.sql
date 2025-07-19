-- Standalone Explainable AI Database Schema
-- Use this if intelligence layer tables don't exist yet
-- Can be connected later via ALTER TABLE commands

-- Table 1: Literature Evidence (PubMed articles) - Standalone
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

-- Table 2: Entity Literature Mappings - Standalone (no foreign keys initially)
CREATE TABLE IF NOT EXISTS entity_literature_mappings (
    mapping_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id UUID, -- Will reference clinical_entities(entity_id) when available
    evidence_id UUID REFERENCES literature_evidence(evidence_id),
    relevance_score DECIMAL(3,2),
    context_type VARCHAR(100), -- 'diagnosis', 'treatment', 'prognosis'
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table 3: PubMed Cache - Standalone
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

-- Table 4: Reasoning Chains - Standalone (no foreign keys initially)
CREATE TABLE IF NOT EXISTS reasoning_chains (
    chain_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255), -- Will reference analysis_sessions(session_id) when available
    step_number INTEGER NOT NULL,
    reasoning TEXT NOT NULL,
    evidence_type VARCHAR(100),
    confidence DECIMAL(3,2),
    supporting_literature TEXT[],
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table 5: Uncertainty Analysis - Standalone (no foreign keys initially)
CREATE TABLE IF NOT EXISTS uncertainty_analysis (
    analysis_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255), -- Will reference analysis_sessions(session_id) when available
    overall_confidence DECIMAL(3,2),
    uncertainty_sources TEXT[],
    recommendation VARCHAR(200),
    confidence_range JSONB,
    uncertainty_visualization JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table 6: Treatment Pathways - Standalone (no foreign keys initially)
CREATE TABLE IF NOT EXISTS treatment_pathways (
    pathway_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255), -- Will reference analysis_sessions(session_id) when available
    pathway_name VARCHAR(200) NOT NULL,
    treatment_sequence JSONB NOT NULL,
    evidence_strength DECIMAL(3,2),
    contraindications TEXT[],
    estimated_outcomes JSONB,
    supporting_studies TEXT[],
    rank_score DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create Indexes for Performance
CREATE INDEX IF NOT EXISTS idx_literature_pmid ON literature_evidence(pmid);
CREATE INDEX IF NOT EXISTS idx_entity_literature_entity ON entity_literature_mappings(entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_literature_relevance ON entity_literature_mappings(relevance_score DESC);
CREATE INDEX IF NOT EXISTS idx_pubmed_cache_hash ON pubmed_cache(query_hash);
CREATE INDEX IF NOT EXISTS idx_pubmed_cache_accessed ON pubmed_cache(last_accessed DESC);
CREATE INDEX IF NOT EXISTS idx_reasoning_session ON reasoning_chains(session_id, step_number);
CREATE INDEX IF NOT EXISTS idx_uncertainty_session ON uncertainty_analysis(session_id);
CREATE INDEX IF NOT EXISTS idx_pathways_session ON treatment_pathways(session_id, rank_score DESC);

-- Cache Cleanup Function
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

-- When intelligence layer is available, run these commands to add foreign keys:
/*
-- Add foreign key to entity_literature_mappings
ALTER TABLE entity_literature_mappings 
ADD CONSTRAINT fk_entity_literature_entity 
FOREIGN KEY (entity_id) REFERENCES clinical_entities(entity_id) ON DELETE CASCADE;

-- Add foreign key to reasoning_chains
ALTER TABLE reasoning_chains 
ADD CONSTRAINT fk_reasoning_session 
FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id) ON DELETE CASCADE;

-- Add foreign key to uncertainty_analysis
ALTER TABLE uncertainty_analysis 
ADD CONSTRAINT fk_uncertainty_session 
FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id) ON DELETE CASCADE;

-- Add foreign key to treatment_pathways
ALTER TABLE treatment_pathways 
ADD CONSTRAINT fk_pathways_session 
FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id) ON DELETE CASCADE;
*/

-- Comment: Run cleanup_expired_cache() to clean up expired cache entries
-- SELECT cleanup_expired_cache();