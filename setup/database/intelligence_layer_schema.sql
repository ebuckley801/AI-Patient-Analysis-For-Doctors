-- Intelligence Layer Database Schema
-- Run this script in your Supabase SQL Editor FIRST before explainable AI tables

-- Table 1: Analysis Sessions (Track analysis requests and metadata)
CREATE TABLE IF NOT EXISTS analysis_sessions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    note_id VARCHAR(255),
    patient_id VARCHAR(255),
    analysis_type VARCHAR(50) NOT NULL, -- 'extract', 'diagnose', 'batch', 'priority'
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed'
    request_data JSONB NOT NULL,
    response_data JSONB,
    error_message TEXT,
    confidence_scores JSONB, -- Overall confidence metrics
    processing_time_ms INTEGER,
    tokens_used INTEGER,
    risk_level VARCHAR(20), -- 'low', 'moderate', 'high', 'critical'
    requires_immediate_attention BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table 2: Clinical Entities (Store extracted clinical entities with confidence scores)
CREATE TABLE IF NOT EXISTS clinical_entities (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    entity_id UUID DEFAULT gen_random_uuid() UNIQUE, -- For external references
    session_id VARCHAR(255) NOT NULL,
    entity_type VARCHAR(50) NOT NULL, -- 'symptom', 'condition', 'medication', 'vital_sign', 'procedure', 'abnormal_finding'
    entity_text TEXT NOT NULL,
    confidence DECIMAL(3,2) NOT NULL, -- 0.00-1.00
    severity VARCHAR(20), -- 'mild', 'moderate', 'severe', 'critical'
    status VARCHAR(20), -- 'active', 'resolved', 'chronic', 'suspected'
    temporal_info VARCHAR(50), -- 'current', 'past', 'family_history', etc.
    negation BOOLEAN DEFAULT FALSE, -- True if entity is negated/absent
    text_span TEXT, -- Original text from note
    normalized_form TEXT, -- Standardized/normalized entity name
    additional_context JSONB, -- Flexible field for entity-specific data
    extraction_method VARCHAR(50) DEFAULT 'claude_ai', -- 'claude_ai', 'nlp', 'manual'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id) ON DELETE CASCADE
);

-- Table 3: Entity ICD Mappings (Store ICD-10 mappings for entities)
CREATE TABLE IF NOT EXISTS entity_icd_mappings (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    entity_id UUID NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    icd_10_code VARCHAR(10) NOT NULL,
    icd_description TEXT NOT NULL,
    similarity_score DECIMAL(5,4), -- 0.0000-1.0000 for vector similarity
    mapping_confidence DECIMAL(3,2) NOT NULL, -- Combined confidence (entity + similarity)
    mapping_method VARCHAR(50) NOT NULL, -- 'vector_similarity', 'text_match', 'manual'
    is_primary_mapping BOOLEAN DEFAULT FALSE, -- True for the best/primary mapping
    rank_order INTEGER DEFAULT 1, -- Ranking of this mapping among alternatives
    icd_category VARCHAR(100), -- ICD-10 category/chapter
    mapping_notes TEXT, -- Additional notes about the mapping
    verified_by_clinician BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    FOREIGN KEY (entity_id) REFERENCES clinical_entities(entity_id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id) ON DELETE CASCADE
);

-- Table 4: Analysis Cache (Cache analysis results for performance)
CREATE TABLE IF NOT EXISTS analysis_cache (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    cache_key VARCHAR(255) UNIQUE NOT NULL, -- Hash of note_text + context
    note_text_hash VARCHAR(64) NOT NULL, -- SHA256 hash of note text
    patient_context JSONB,
    cached_result JSONB NOT NULL, -- Complete analysis result
    analysis_type VARCHAR(50) NOT NULL, -- 'extract', 'diagnose'
    confidence_threshold DECIMAL(3,2), -- Confidence threshold used
    hit_count INTEGER DEFAULT 0, -- Number of times this cache was used
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() + INTERVAL '7 days'), -- Cache TTL
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create Indexes for Performance
-- Analysis sessions indexes
CREATE INDEX IF NOT EXISTS idx_analysis_sessions_note_id ON analysis_sessions(note_id);
CREATE INDEX IF NOT EXISTS idx_analysis_sessions_patient_id ON analysis_sessions(patient_id);
CREATE INDEX IF NOT EXISTS idx_analysis_sessions_status ON analysis_sessions(status);
CREATE INDEX IF NOT EXISTS idx_analysis_sessions_risk_level ON analysis_sessions(risk_level);
CREATE INDEX IF NOT EXISTS idx_analysis_sessions_created_at ON analysis_sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_analysis_sessions_immediate_attention ON analysis_sessions(requires_immediate_attention);

-- Clinical entities indexes
CREATE INDEX IF NOT EXISTS idx_clinical_entities_session_id ON clinical_entities(session_id);
CREATE INDEX IF NOT EXISTS idx_clinical_entities_entity_id ON clinical_entities(entity_id);
CREATE INDEX IF NOT EXISTS idx_clinical_entities_type ON clinical_entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_clinical_entities_confidence ON clinical_entities(confidence);
CREATE INDEX IF NOT EXISTS idx_clinical_entities_severity ON clinical_entities(severity);
CREATE INDEX IF NOT EXISTS idx_clinical_entities_text ON clinical_entities USING gin(to_tsvector('english', entity_text));

-- Entity ICD mappings indexes
CREATE INDEX IF NOT EXISTS idx_entity_icd_mappings_entity_id ON entity_icd_mappings(entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_icd_mappings_session_id ON entity_icd_mappings(session_id);
CREATE INDEX IF NOT EXISTS idx_entity_icd_mappings_icd_code ON entity_icd_mappings(icd_10_code);
CREATE INDEX IF NOT EXISTS idx_entity_icd_mappings_confidence ON entity_icd_mappings(mapping_confidence);
CREATE INDEX IF NOT EXISTS idx_entity_icd_mappings_primary ON entity_icd_mappings(is_primary_mapping);

-- Analysis cache indexes
CREATE INDEX IF NOT EXISTS idx_analysis_cache_key ON analysis_cache(cache_key);
CREATE INDEX IF NOT EXISTS idx_analysis_cache_hash ON analysis_cache(note_text_hash);
CREATE INDEX IF NOT EXISTS idx_analysis_cache_expires ON analysis_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_analysis_cache_type ON analysis_cache(analysis_type);

-- Cache Cleanup Function
CREATE OR REPLACE FUNCTION cleanup_expired_cache()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM analysis_cache WHERE expires_at < NOW();
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Trigger for updating updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_analysis_sessions_updated_at
    BEFORE UPDATE ON analysis_sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Comment: Run this query to clean up expired cache entries
-- SELECT cleanup_expired_cache();