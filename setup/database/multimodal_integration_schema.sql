-- Multi-Modal Medical Data Integration Schema
-- Extension to existing Intelligence Layer and Explainable AI schemas
-- Supports MIMIC-IV, UK Biobank, FAERS, and Clinical Trials integration

-- ============================================================================
-- UNIFIED PATIENT IDENTITY MANAGEMENT
-- ============================================================================

-- Master Patient Index for cross-dataset patient resolution
CREATE TABLE IF NOT EXISTS unified_patients (
    unified_patient_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    master_record_id VARCHAR(255) UNIQUE NOT NULL, -- Primary identifier
    demographics JSONB NOT NULL, -- Age, gender, basic demographics
    identity_confidence DECIMAL(3,2) DEFAULT 1.00, -- Confidence in identity resolution
    data_sources TEXT[] DEFAULT '{}', -- List of contributing datasets
    privacy_level VARCHAR(20) DEFAULT 'standard', -- 'public', 'standard', 'restricted'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Cross-dataset patient identity mappings
CREATE TABLE IF NOT EXISTS patient_identity_mappings (
    mapping_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    unified_patient_id UUID REFERENCES unified_patients(unified_patient_id),
    source_dataset VARCHAR(50) NOT NULL, -- 'mimic', 'biobank', 'faers', 'trials', 'local'
    source_patient_id VARCHAR(255) NOT NULL,
    confidence_score DECIMAL(3,2) NOT NULL, -- Identity matching confidence
    matching_method VARCHAR(50) NOT NULL, -- 'exact', 'probabilistic', 'manual'
    matching_features JSONB, -- Features used for matching
    verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(source_dataset, source_patient_id)
);

-- ============================================================================
-- MIMIC-IV CRITICAL CARE DATA
-- ============================================================================

-- MIMIC-IV patient admissions and ICU stays
CREATE TABLE IF NOT EXISTS mimic_admissions (
    admission_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    unified_patient_id UUID REFERENCES unified_patients(unified_patient_id),
    mimic_subject_id INTEGER NOT NULL,
    mimic_hadm_id INTEGER NOT NULL UNIQUE,
    admission_type VARCHAR(50),
    admission_location VARCHAR(100),
    discharge_location VARCHAR(100),
    insurance VARCHAR(50),
    language VARCHAR(20),
    marital_status VARCHAR(20),
    ethnicity VARCHAR(100),
    hospital_expire_flag BOOLEAN,
    admit_time TIMESTAMP WITH TIME ZONE,
    discharge_time TIMESTAMP WITH TIME ZONE,
    deathtime TIMESTAMP WITH TIME ZONE,
    edregtime TIMESTAMP WITH TIME ZONE,
    edouttime TIMESTAMP WITH TIME ZONE,
    diagnosis TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- MIMIC-IV ICU stays
CREATE TABLE IF NOT EXISTS mimic_icu_stays (
    icu_stay_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    admission_id UUID REFERENCES mimic_admissions(admission_id),
    mimic_stay_id INTEGER NOT NULL UNIQUE,
    first_careunit VARCHAR(50),
    last_careunit VARCHAR(50),
    intime TIMESTAMP WITH TIME ZONE,
    outtime TIMESTAMP WITH TIME ZONE,
    los_hours DECIMAL(8,2), -- Length of stay in hours
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- MIMIC-IV vital signs and monitoring data
CREATE TABLE IF NOT EXISTS mimic_vitals (
    vital_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    icu_stay_id UUID REFERENCES mimic_icu_stays(icu_stay_id),
    charttime TIMESTAMP WITH TIME ZONE NOT NULL,
    vital_type VARCHAR(100) NOT NULL, -- 'heart_rate', 'blood_pressure', 'temperature', etc.
    value DECIMAL(10,4),
    unit VARCHAR(20),
    value_normalized DECIMAL(10,4), -- Standardized value
    abnormal_flag BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- MIMIC-IV procedures and interventions
CREATE TABLE IF NOT EXISTS mimic_procedures (
    procedure_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    admission_id UUID REFERENCES mimic_admissions(admission_id),
    seq_num INTEGER,
    chartdate DATE,
    icd_code VARCHAR(10),
    icd_version INTEGER,
    procedure_description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- UK BIOBANK GENETIC AND LIFESTYLE DATA
-- ============================================================================

-- UK Biobank participant data
CREATE TABLE IF NOT EXISTS biobank_participants (
    participant_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    unified_patient_id UUID REFERENCES unified_patients(unified_patient_id),
    biobank_eid INTEGER NOT NULL UNIQUE,
    assessment_centre VARCHAR(100),
    genotyping_array VARCHAR(50),
    genetic_ethnic_grouping VARCHAR(100),
    birth_country VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Genetic variants and risk scores
CREATE TABLE IF NOT EXISTS biobank_genetics (
    genetic_record_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    participant_id UUID REFERENCES biobank_participants(participant_id),
    variant_type VARCHAR(20) NOT NULL, -- 'snp', 'cnv', 'prs'
    variant_id VARCHAR(100) NOT NULL, -- rs number or variant identifier
    chromosome VARCHAR(5),
    position BIGINT,
    allele_1 VARCHAR(10),
    allele_2 VARCHAR(10),
    genotype VARCHAR(20),
    risk_score DECIMAL(8,4), -- Polygenic risk score
    confidence DECIMAL(3,2),
    associated_conditions TEXT[], -- Known disease associations
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Lifestyle and environmental factors
CREATE TABLE IF NOT EXISTS biobank_lifestyle (
    lifestyle_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    participant_id UUID REFERENCES biobank_participants(participant_id),
    assessment_date DATE,
    category VARCHAR(50) NOT NULL, -- 'diet', 'exercise', 'smoking', 'alcohol', 'sleep'
    metric_name VARCHAR(100) NOT NULL,
    value DECIMAL(10,2),
    unit VARCHAR(20),
    categorical_value VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Disease outcomes and diagnoses
CREATE TABLE IF NOT EXISTS biobank_diagnoses (
    diagnosis_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    participant_id UUID REFERENCES biobank_participants(participant_id),
    source VARCHAR(50), -- 'self_report', 'hospital', 'death_registry'
    icd_code VARCHAR(10),
    icd_version INTEGER,
    diagnosis_date DATE,
    age_at_diagnosis DECIMAL(4,1),
    diagnosis_description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- FDA ADVERSE EVENT REPORTING SYSTEM (FAERS)
-- ============================================================================

-- FAERS case reports
CREATE TABLE IF NOT EXISTS faers_cases (
    case_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    unified_patient_id UUID REFERENCES unified_patients(unified_patient_id),
    faers_case_number VARCHAR(50) NOT NULL UNIQUE,
    case_version INTEGER,
    report_type VARCHAR(20), -- 'initial', 'followup'
    serious_adverse_event BOOLEAN,
    country VARCHAR(3), -- ISO country code
    report_date DATE,
    receive_date DATE,
    reporter_type VARCHAR(50), -- 'physician', 'pharmacist', 'consumer', etc.
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Drug information from FAERS
CREATE TABLE IF NOT EXISTS faers_drugs (
    drug_record_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id UUID REFERENCES faers_cases(case_id),
    drug_sequence INTEGER,
    generic_name VARCHAR(500),
    brand_name VARCHAR(500),
    active_ingredient VARCHAR(500),
    dosage_form VARCHAR(100),
    route VARCHAR(100),
    dose_amount VARCHAR(200),
    dose_unit VARCHAR(50),
    dose_frequency VARCHAR(100),
    indication TEXT,
    drug_start_date DATE,
    drug_end_date DATE,
    drug_characterization VARCHAR(20), -- 'suspect', 'concomitant', 'interacting'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Adverse events from FAERS
CREATE TABLE IF NOT EXISTS faers_adverse_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id UUID REFERENCES faers_cases(case_id),
    meddra_pt_code INTEGER, -- Preferred Term code
    meddra_pt_name VARCHAR(500), -- Preferred Term name
    meddra_soc_code INTEGER, -- System Organ Class code
    meddra_soc_name VARCHAR(200), -- System Organ Class name
    severity VARCHAR(20),
    outcome VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- CLINICAL TRIALS DATA
-- ============================================================================

-- Clinical trials from ClinicalTrials.gov
CREATE TABLE IF NOT EXISTS clinical_trials (
    trial_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    nct_id VARCHAR(20) NOT NULL UNIQUE,
    title TEXT NOT NULL,
    brief_summary TEXT,
    detailed_description TEXT,
    study_type VARCHAR(50), -- 'interventional', 'observational'
    phase VARCHAR(20), -- 'phase_1', 'phase_2', etc.
    status VARCHAR(50), -- 'recruiting', 'active', 'completed', etc.
    enrollment_count INTEGER,
    primary_completion_date DATE,
    study_completion_date DATE,
    sponsor_name VARCHAR(500),
    responsible_party VARCHAR(500),
    study_design JSONB, -- Detailed study design information
    eligibility_criteria TEXT,
    contact_information JSONB,
    locations JSONB, -- Study locations
    conditions TEXT[], -- Medical conditions studied
    interventions JSONB, -- Interventions/treatments
    primary_outcomes JSONB,
    secondary_outcomes JSONB,
    keywords TEXT[],
    mesh_terms TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Patient-trial matching results
CREATE TABLE IF NOT EXISTS patient_trial_matches (
    match_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    unified_patient_id UUID REFERENCES unified_patients(unified_patient_id),
    trial_id UUID REFERENCES clinical_trials(trial_id),
    session_id VARCHAR(255) REFERENCES analysis_sessions(session_id),
    match_score DECIMAL(3,2) NOT NULL, -- Overall matching confidence
    eligibility_assessment JSONB, -- Detailed eligibility evaluation
    inclusion_criteria_met JSONB, -- Which inclusion criteria are satisfied
    exclusion_criteria_violated JSONB, -- Which exclusion criteria are violated
    matching_method VARCHAR(50) NOT NULL, -- 'rule_based', 'ml_similarity', 'hybrid'
    geographic_feasible BOOLEAN DEFAULT TRUE,
    estimated_travel_distance_km DECIMAL(8,2),
    matching_reasoning TEXT,
    recommendation_level VARCHAR(20), -- 'high', 'medium', 'low', 'exclude'
    clinician_reviewed BOOLEAN DEFAULT FALSE,
    patient_notified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- CROSS-DATASET VECTOR EMBEDDINGS
-- ============================================================================

-- Vector embeddings for cross-dataset similarity search
CREATE TABLE IF NOT EXISTS multimodal_embeddings (
    embedding_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    unified_patient_id UUID REFERENCES unified_patients(unified_patient_id),
    data_source VARCHAR(50) NOT NULL, -- Source dataset
    data_type VARCHAR(50) NOT NULL, -- 'clinical_text', 'genetics', 'vitals', 'adverse_events'
    content_hash VARCHAR(64) NOT NULL, -- SHA256 of content
    content_summary TEXT,
    embedding_vector DECIMAL(10,8)[], -- High-dimensional vector (adjust as needed)
    embedding_model VARCHAR(100) NOT NULL, -- Model used for embedding
    vector_dimension INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(content_hash, embedding_model)
);

-- ============================================================================
-- PERFORMANCE INDEXES
-- ============================================================================

-- Unified patients indexes
CREATE INDEX IF NOT EXISTS idx_unified_patients_record_id ON unified_patients(master_record_id);
CREATE INDEX IF NOT EXISTS idx_patient_identity_mappings_unified ON patient_identity_mappings(unified_patient_id);
CREATE INDEX IF NOT EXISTS idx_patient_identity_mappings_source ON patient_identity_mappings(source_dataset, source_patient_id);

-- MIMIC-IV indexes
CREATE INDEX IF NOT EXISTS idx_mimic_admissions_patient ON mimic_admissions(unified_patient_id);
CREATE INDEX IF NOT EXISTS idx_mimic_admissions_hadm ON mimic_admissions(mimic_hadm_id);
CREATE INDEX IF NOT EXISTS idx_mimic_icu_stays_admission ON mimic_icu_stays(admission_id);
CREATE INDEX IF NOT EXISTS idx_mimic_vitals_icu_stay ON mimic_vitals(icu_stay_id);
CREATE INDEX IF NOT EXISTS idx_mimic_vitals_time ON mimic_vitals(charttime);
CREATE INDEX IF NOT EXISTS idx_mimic_procedures_admission ON mimic_procedures(admission_id);

-- UK Biobank indexes
CREATE INDEX IF NOT EXISTS idx_biobank_participants_patient ON biobank_participants(unified_patient_id);
CREATE INDEX IF NOT EXISTS idx_biobank_genetics_participant ON biobank_genetics(participant_id);
CREATE INDEX IF NOT EXISTS idx_biobank_genetics_variant ON biobank_genetics(variant_id);
CREATE INDEX IF NOT EXISTS idx_biobank_lifestyle_participant ON biobank_lifestyle(participant_id);
CREATE INDEX IF NOT EXISTS idx_biobank_diagnoses_participant ON biobank_diagnoses(participant_id);

-- FAERS indexes
CREATE INDEX IF NOT EXISTS idx_faers_cases_patient ON faers_cases(unified_patient_id);
CREATE INDEX IF NOT EXISTS idx_faers_drugs_case ON faers_drugs(case_id);
CREATE INDEX IF NOT EXISTS idx_faers_events_case ON faers_adverse_events(case_id);

-- Clinical trials indexes
CREATE INDEX IF NOT EXISTS idx_clinical_trials_nct ON clinical_trials(nct_id);
CREATE INDEX IF NOT EXISTS idx_clinical_trials_status ON clinical_trials(status);
CREATE INDEX IF NOT EXISTS idx_clinical_trials_conditions ON clinical_trials USING gin(conditions);
CREATE INDEX IF NOT EXISTS idx_patient_trial_matches_patient ON patient_trial_matches(unified_patient_id);
CREATE INDEX IF NOT EXISTS idx_patient_trial_matches_trial ON patient_trial_matches(trial_id);
CREATE INDEX IF NOT EXISTS idx_patient_trial_matches_score ON patient_trial_matches(match_score DESC);

-- Vector embeddings indexes
CREATE INDEX IF NOT EXISTS idx_multimodal_embeddings_patient ON multimodal_embeddings(unified_patient_id);
CREATE INDEX IF NOT EXISTS idx_multimodal_embeddings_source ON multimodal_embeddings(data_source, data_type);
CREATE INDEX IF NOT EXISTS idx_multimodal_embeddings_hash ON multimodal_embeddings(content_hash);

-- ============================================================================
-- UTILITY FUNCTIONS
-- ============================================================================

-- Function to generate unified patient ID from demographics
CREATE OR REPLACE FUNCTION generate_unified_patient_id(
    demographics_data JSONB
) RETURNS VARCHAR(255) AS $$
BEGIN
    RETURN 'UPI_' || encode(sha256(demographics_data::text::bytea), 'hex')::varchar(32);
END;
$$ LANGUAGE plpgsql;

-- Function to calculate patient similarity across datasets
CREATE OR REPLACE FUNCTION calculate_patient_similarity(
    patient_id_1 UUID,
    patient_id_2 UUID
) RETURNS DECIMAL(3,2) AS $$
DECLARE
    similarity_score DECIMAL(3,2) := 0.00;
BEGIN
    -- Placeholder for sophisticated similarity calculation
    -- Would implement demographic, genetic, clinical similarity
    RETURN similarity_score;
END;
$$ LANGUAGE plpgsql;

-- Trigger for updating unified patient timestamps
CREATE TRIGGER update_unified_patients_updated_at
    BEFORE UPDATE ON unified_patients
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_clinical_trials_updated_at
    BEFORE UPDATE ON clinical_trials
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Comments for implementation notes
-- 1. Vector embeddings table uses DECIMAL arrays - consider switching to specialized vector extension
-- 2. Genetic data structure simplified - expand based on specific genetic analysis needs
-- 3. FAERS data follows FDA structure but may need customization for specific use cases
-- 4. Clinical trials matching algorithm placeholders - implement sophisticated matching logic
-- 5. Patient identity resolution requires sophisticated probabilistic matching algorithms