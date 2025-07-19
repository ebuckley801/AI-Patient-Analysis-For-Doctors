# Patient Analysis - Clinical Decision Support System

A secure healthcare application for analyzing patient data and ICD-10 codes using Supabase. Features comprehensive data validation, sanitization, REST API with security middleware, and **Phase 2 Intelligence Layer** with Claude AI integration for clinical entity extraction and ICD-10 mapping.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables in `.env`:
```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_service_role_key
ROOT_DIR=path_to_your_csv_files
PORT=5000
ANTHROPIC_KEY=your_anthropic_api_key
```

## Flask API

### Launch the API Server
```bash
python app.py
```

The API will be available at: `http://localhost:5000` (or configured PORT)

### Security Features
- Input validation and sanitization
- SQL injection prevention
- XSS protection
- Rate limiting (100 requests/hour)
- Security headers
- Request logging

### API Endpoints

#### Patient Routes (`/api/patients/`)
- `GET /api/patients/` - Get all patients (paginated)
- `GET /api/patients/<patient_id>` - Get notes for specific patient
- `POST /api/patients/` - Create new patient note
- `PUT /api/patients/<patient_id>` - Update patient note
- `DELETE /api/patients/<patient_id>` - Delete all notes for patient
- `GET /api/patients/search?q=<query>` - Search patient notes

#### Note Routes (`/api/notes/`)
- `GET /api/notes/` - Get all notes (paginated)
- `GET /api/notes/<note_id>` - Get specific note
- `POST /api/notes/` - Create new note
- `PUT /api/notes/<note_id>` - Update specific note
- `DELETE /api/notes/<note_id>` - Delete specific note
- `GET /api/notes/search?q=<query>&field=<field>` - Search notes
- `GET /api/notes/patient/<patient_id>` - Get notes by patient

#### Intelligence Layer Routes (Phase 2) 🧠
- `POST /api/analysis/extract` - Extract clinical entities from patient notes (with enhanced NLP & persistence)
- `POST /api/analysis/extract-enhanced` - **Enhanced extraction with Faiss + advanced NLP** 🚀 🆕
- `POST /api/analysis/diagnose` - Get ICD-10 mappings with confidence scores  
- `POST /api/analysis/batch` - Process multiple notes for clinical insights (up to 50 notes)
- `POST /api/analysis/batch-async` - **High-performance async batch processing (up to 1000 notes)** 🚀
- `POST /api/analysis/priority-scan` - **Rapid priority triage scanning (up to 2000 notes)** ⚡
- `GET /api/analysis/priority/<note_id>` - Get high-priority findings for a note (fully implemented)
- `GET /api/analysis/health` - Health check for intelligence layer services (includes storage stats)
- `GET /api/analysis/performance-stats` - **Comprehensive performance statistics** 📊 🆕
- `POST /api/analysis/benchmark` - **Performance benchmarking tool** 🏁 🆕

#### Example API Usage
```bash
# Get all patients
curl http://localhost:5000/api/patients/

# Get specific patient
curl http://localhost:5000/api/patients/123

# Create new patient note
curl -X POST http://localhost:5000/api/patients/ \
  -H "Content-Type: application/json" \
  -d '{"patient_id": 123, "patient_uid": "uid-123", "patient_note": "New note", "age": 30, "gender": "F"}'

# Search notes
curl "http://localhost:5000/api/notes/search?q=symptoms"

# Extract clinical entities (Phase 2)
curl -X POST http://localhost:5000/api/analysis/extract \
  -H "Content-Type: application/json" \
  -d '{"note_text": "Patient has chest pain and fever", "patient_context": {"age": 45, "gender": "M"}}'

# Get ICD-10 mappings (Phase 2)
curl -X POST http://localhost:5000/api/analysis/diagnose \
  -H "Content-Type: application/json" \
  -d '{"note_text": "Patient diagnosed with acute myocardial infarction"}'

# Ingest MIMIC-IV data (Multimodal)
curl -X POST http://localhost:5000/api/multimodal/ingest/mimic \
  -H "Content-Type: application/json" \
  -d '{"data": {"patient_id": "mimic_123", "admission_details": "..."}}'

# Find similar patients (Multimodal)
curl -X POST http://localhost:5000/api/multimodal/similarity/patients \
  -H "Content-Type: application/json" \
  -d '{"query_patient_id": "patient_456", "target_modality": "CLINICAL_TEXT"}'
```

## Intelligence Layer API (Phase 2) 🧠

### Overview
The Intelligence Layer provides AI-powered clinical analysis capabilities using Claude AI for extracting clinical insights from patient notes and mapping them to ICD-10 codes.

### Core Features
- **Clinical Entity Extraction**: Identifies symptoms, conditions, medications, vital signs, procedures
- **Enhanced NLP Processing**: Advanced negation detection, abbreviation expansion, temporal extraction 🆕
- **High-Performance Vector Search**: Faiss integration for 50-100x faster ICD-10 mapping 🚀 🆕
- **Risk Assessment**: Automatic classification (low/moderate/high/critical) with confidence scoring
- **ICD-10 Mapping**: Semantic similarity matching to relevant medical codes with vector optimization
- **Batch Processing**: Analyze multiple patient notes simultaneously
- **Async High-Performance Processing**: Up to 1000 notes concurrently with configurable performance 🚀
- **Priority Triage Scanning**: Rapid identification of high-risk cases from large note volumes ⚡
- **Priority Detection**: Flags high-priority findings requiring immediate attention
- **Persistent Storage**: Database storage for analysis results, entities, and ICD mappings
- **Smart Caching**: Automatic result caching with 7-day TTL for performance optimization
- **Session Tracking**: Complete audit trail of analysis requests and results
- **Performance Monitoring**: Comprehensive benchmarking and statistics tracking 📊 🆕

### API Endpoints

#### Extract Clinical Entities
```bash
POST /api/analysis/extract
Content-Type: application/json

{
  "note_text": "Patient note content",
  "patient_context": {
    "age": 45,
    "gender": "M",
    "medical_history": "hypertension, diabetes"
  }
}
```

**Response**: Clinical entities with confidence scores, risk assessment, and priority flags

#### Diagnose with ICD-10 Mapping
```bash
POST /api/analysis/diagnose
Content-Type: application/json

{
  "note_text": "Patient note content",
  "patient_context": {"age": 45, "gender": "M"},
  "options": {
    "include_low_confidence": false,
    "max_icd_matches": 5
  }
}
```

**Response**: Complete analysis + ICD-10 code mappings with similarity scores

#### Batch Analysis
```bash
POST /api/analysis/batch
Content-Type: application/json

{
  "notes": [
    {
      "note_id": "note_1",
      "note_text": "Patient note 1",
      "patient_context": {"age": 30, "gender": "F"}
    }
  ],
  "options": {
    "include_icd_mapping": true,
    "include_priority_analysis": true
  }
}
```

**Response**: Array of analysis results + summary statistics

#### Async Batch Processing (High Performance) 🚀
```bash
POST /api/analysis/batch-async
Content-Type: application/json

{
  "notes": [
    {
      "note_id": "async_1",
      "note_text": "Patient note content",
      "patient_context": {"age": 45, "gender": "M"},
      "patient_id": "patient_123"
    }
  ],
  "config": {
    "max_concurrent": 10,
    "timeout_seconds": 30,
    "include_icd_mapping": true,
    "chunk_size": 50
  }
}
```

**Response**: Comprehensive batch results with performance metrics, cache hit rates, and timing statistics

#### Priority Triage Scanning ⚡
```bash
POST /api/analysis/priority-scan
Content-Type: application/json

{
  "notes": [...],
  "risk_threshold": "high"
}
```

**Response**: Rapid identification of high-risk cases optimized for large-scale triage

#### Priority Findings
```bash
GET /api/analysis/priority/<note_id>?risk_threshold=high&include_details=true
```

**Response**: High-priority findings for a specific note with optional entity details

#### Health Check
```bash
GET /api/analysis/health
```

**Response**: Service status, ICD cache info, storage statistics, cache performance metrics

## Multi-Modal Integration API 🧬

### Overview
The Multi-Modal Integration API provides advanced capabilities for fusing and analyzing diverse healthcare datasets, including MIMIC-IV, UK Biobank, FAERS, and ClinicalTrials.gov. It enables unified patient identity management, cross-dataset similarity search, and sophisticated data fusion for comprehensive patient insights.

### Core Features
- **Unified Patient Identity**: Advanced identity resolution and cross-dataset patient matching.
- **Multi-Modal Data Ingestion**: Pipelines for various healthcare data sources.
- **Cross-Modal Vector Search**: High-performance similarity search across different medical modalities using Faiss.
- **Sophisticated Data Fusion**: Aggregation of evidence with weighted confidence scoring and uncertainty quantification.
- **Clinical Trials Matching**: Advanced algorithms for matching patients to relevant clinical trials.

### API Endpoints

#### Data Ingestion
- `POST /api/multimodal/ingest/mimic` - Ingest MIMIC-IV data
- `POST /api/multimodal/ingest/biobank` - Ingest UK Biobank data
- `POST /api/multimodal/ingest/faers` - Ingest FAERS data

#### Patient Identity Management
- `POST /api/multimodal/identity/resolve` - Resolve patient identity across datasets
- `POST /api/multimodal/identity/validate` - Validate identity match

#### Cross-Modal Analysis
- `POST /api/multimodal/similarity/patients` - Find similar patients across different modalities
- `GET /api/multimodal/analysis/cross-modal/<patient_id>` - Comprehensive cross-modal analysis for a patient

#### Clinical Trials
- `POST /api/multimodal/trials/fetch` - Fetch trials from ClinicalTrials.gov API
- `POST /api/multimodal/trials/match/<patient_id>` - Match patient to clinical trials

#### System Management
- `GET /api/multimodal/health` - Service health check for multimodal services
- `GET /api/multimodal/stats` - System statistics for multimodal integration

### Testing the API

For detailed testing instructions, see:
- `test/test_multimodal_integration.py` - Automated tests for multimodal endpoints.

## Database Setup

### Intelligence Layer Tables (Phase 2) 🆕
1. **Set up intelligence layer persistence**: `python app/utils/create_intelligence_db.py`
   - Creates `analysis_sessions`, `clinical_entities`, `entity_icd_mappings`, `analysis_cache` tables
   - Adds database indexes for optimal performance
   - Creates cache cleanup functions and triggers

### Multi-Modal Database Extensions 🧬
1. **Set up multimodal integration schema**: `python setup/database/multimodal_integration_schema.sql`
   - Extends the existing Supabase schema with tables for unified patient management, MIMIC-IV, UK Biobank, FAERS, Clinical Trials data, and multi-modal embeddings.

### ICD-10 Codes
1. Place your ICD-10 CSV file in the root directory as `icd_10_codes.csv`
2. Ensure CSV has columns: `embedded_description`, `icd_10_code`, `description`
3. Run: `python app/utils/create_icd10_db.py`

### Patient Notes
1. Place your patient notes CSV file in the root directory as `patient_notes.csv`
2. Ensure CSV has columns: `patient_id`, `patient_uid`, `patient_note`, `age`, `gender`
3. Run: `python app/utils/create_patient_note_db.py`

## Running Tests

### Run All Tests
```bash
source venv/bin/activate
python -m pytest test/ -v
```

### Test Intelligence Layer (Phase 2 + Option B Enhancements)
```bash
source venv/bin/activate

# Core Intelligence Layer Tests
python test/test_intelligence_layer.py               # Complete workflow demonstration
python test/test_claude_service.py                   # Claude integration test
python test/test_simple_api.py                       # API integration test (no server needed)
python test/test_api_endpoints.py                    # Full API endpoint test (requires server)

# Database Persistence Tests
python test/test_analysis_storage_service.py         # Database persistence tests
python test/test_create_intelligence_db.py           # Database schema tests
python test/test_analysis_routes_persistence.py      # API persistence integration tests

# Enhanced NLP & Async Processing Tests (New!)
python test/test_clinical_nlp.py                     # Enhanced NLP features
python test/test_async_clinical_analysis.py          # Async batch processing
python test/test_enhanced_clinical_analysis.py       # NLP-enhanced clinical analysis

# Unit Tests
python -m pytest test/test_clinical_analysis_service.py -v
python -m pytest test/test_icd10_vector_matcher.py -v
```

### Test Multi-Modal Integration 🧬
```bash
source venv/bin/activate
python test/test_multimodal_integration.py           # Comprehensive tests for multimodal features
```

### Run Tests for Specific Module
```bash
python -m pytest test/utils/          # Database utility tests
python -m pytest test/routes/         # API endpoint tests
python -m pytest test/services/       # Intelligence layer tests
python -m pytest test/utils/test_create_icd10_db.py
```

### Run Individual Test File
```bash
python -m unittest test.utils.test_create_icd10_db
python -m unittest test.utils.test_create_patient_note_db
python -m unittest test.routes.test_patient_routes
python -m unittest test.routes.test_note_routes
```

### Run Tests with Coverage Report
```bash
pip install pytest-cov
python -m pytest test/ --cov=app --cov-report=html
```

## Test Structure

```
test/
├── config/          # Tests for configuration files
├── models/          # Tests for data models
├── routes/          # Tests for API routes
│   ├── test_patient_routes.py
│   └── test_note_routes.py
├── services/        # Tests for business logic
│   ├── test_supabase_service.py
│   ├── test_clinical_analysis_service.py    # Phase 2: Claude integration
│   ├── test_icd10_vector_matcher.py         # Phase 2: ICD-10 mapping
│   └── test_multimodal_integration.py       # Multi-Modal Integration tests
├── utils/           # Tests for utility functions
│   ├── test_create_icd10_db.py
│   ├── test_create_patient_note_db.py
│   ├── test_validation.py
│   └── test_sanitization.py
├── test_claude_service.py                 # Phase 2: Claude integration demo
├── test_intelligence_layer.py            # Phase 2: Complete workflow test
├── test_api_endpoints.py                 # Phase 2: Full API endpoint testing
├── test_simple_api.py                    # Phase 2: API integration test (no server)
├── test_analysis_storage_service.py      # Phase 2: Database persistence tests
├── test_create_intelligence_db.py        # Phase 2: Database schema tests
├── test_analysis_routes_persistence.py   # Phase 2: API persistence integration
├── test_clinical_nlp.py                  # Option B: Enhanced NLP features
├── test_async_clinical_analysis.py       # Option B: Async batch processing
├── test_enhanced_clinical_analysis.py    # Option B: NLP-enhanced analysis
└── test_api_manual.md                    # Phase 2: Manual API testing guide
```

## Development Guidelines

- Every new file in `app/` must have a corresponding test file in `test/`
- Tests must cover happy path, edge cases, and error handling
- All tests must pass before code changes are considered complete
- See `.claude/test_rules.md` for detailed testing requirements

## Project Structure

```
Patient-Analysis/
├── .claude/         # Claude Code configuration
├── .git/            # Git repository
├── app/
│   ├── __init__.py
│   ├── config/      # Configuration files
│   ├── middleware/  # Security middleware
│   │   └── security.py
│   ├── models/      # Data models
│   │   └── patient.py
│   ├── routes/      # API routes
│   │   ├── patient_routes.py
│   │   ├── note_routes.py
│   │   └── multimodal_routes.py            # Multi-Modal Integration routes
│   ├── services/    # Business logic
│   │   ├── supabase_service.py
│   │   ├── clinical_analysis_service.py    # Phase 2: Claude AI integration (enhanced)
│   │   ├── icd10_vector_matcher.py         # Phase 2: ICD-10 mapping
│   │   ├── analysis_storage_service.py     # Phase 2: Database persistence
│   │   ├── async_clinical_analysis.py      # Option B: High-performance async processing
│   │   ├── multimodal_data_service.py      # Multi-Modal Integration: Data ingestion
│   │   ├── patient_identity_service.py     # Multi-Modal Integration: Identity resolution
│   │   ├── multimodal_vector_service.py    # Multi-Modal Integration: Vector search
│   │   ├── data_fusion_service.py          # Multi-Modal Integration: Data fusion
│   │   └── clinical_trials_matching_service.py # Multi-Modal Integration: Clinical trials matching
│   └── utils/       # Utility functions
│       ├── create_icd10_db.py
│       ├── create_patient_note_db.py
│       ├── create_intelligence_db.py       # Phase 2: Intelligence layer tables
│       ├── clinical_nlp.py                 # Option B: Enhanced NLP processing
│       ├── validation.py
│       └── sanitization.py
├── test/            # Test files (mirrors app structure)
│   ├── config/
│   ├── models/
│   ├── routes/
│   ├── services/
│   └── utils/
├── venv/            # Virtual environment
├── .env             # Environment variables
├── .gitignore       # Git ignore rules
├── app.py           # Main application entry point
├── requirements.txt # Python dependencies
├── icd_10_codes.csv # ICD-10 data file
├── PMC_Patients_clean.csv # Patient data file
└── README.md        # This file

## Key Components

### Phase 2: Intelligence Layer 🧠

#### Clinical Analysis Service (`app/services/clinical_analysis_service.py`)
- **Claude AI Integration**: Uses Claude 3.5 Sonnet for clinical text analysis
- **Entity Extraction**: Identifies symptoms, conditions, medications, vital signs, procedures
- **Confidence Scoring**: Each extraction includes 0.0-1.0 confidence levels
- **Medical Context**: Handles negation, severity, temporal information
- **High-Priority Detection**: Flags critical findings requiring immediate attention
- **Batch Processing**: Analyzes multiple patient notes simultaneously

#### ICD-10 Vector Matcher (`app/services/icd10_vector_matcher.py`)
- **Vector Similarity**: Cosine similarity matching against ICD-10 embeddings
- **Text-Based Fallback**: Simple text matching when vectors unavailable
- **Entity Mapping**: Maps clinical entities to relevant ICD-10 codes
- **Hierarchy Analysis**: Provides ICD code category information
- **Confidence Weighting**: Combines extraction confidence with similarity scores

#### Analysis Storage Service (`app/services/analysis_storage_service.py`) 🆕
- **Session Management**: Create and track analysis sessions with complete audit trail
- **Entity Persistence**: Store clinical entities with confidence scores and metadata
- **ICD Mapping Storage**: Persistent storage of entity-to-ICD mappings with rankings
- **Smart Caching**: Automatic result caching with TTL and hit rate tracking
- **Priority Retrieval**: Query high-priority findings by note, patient, or risk level
- **Cache Maintenance**: Automatic cleanup of expired cache entries

#### Enhanced Clinical NLP (`app/utils/clinical_nlp.py`) 🆕
- **Medical Abbreviation Expansion**: 100+ medical abbreviations automatically expanded
- **Sophisticated Negation Detection**: 20+ negation patterns including medical-specific contexts
- **Temporal Relationship Extraction**: Onset, duration, frequency, and progression detection
- **Uncertainty Assessment**: Handles speculation markers and confidence adjustments
- **Clinical Context Awareness**: Entity relationships and medical coherence validation

#### Async Clinical Analysis (`app/services/async_clinical_analysis.py`) 🚀
- **High-Performance Batch Processing**: Up to 1000 notes with configurable concurrency
- **Priority Triage Scanning**: Rapid identification of high-risk cases (up to 2000 notes)
- **Intelligent Chunking**: Memory-efficient processing with automatic chunk management
- **Advanced Retry Logic**: Exponential backoff and error recovery mechanisms
- **Performance Monitoring**: Comprehensive metrics including cache hit rates and timing

#### Intelligence Layer Capabilities
- **Clinical Entity Types**: Symptoms, conditions, medications, vital signs, procedures, abnormal findings
- **Enhanced Accuracy**: Advanced NLP processing for clinical accuracy improvements
- **Risk Assessment**: Automatic risk level classification (low/moderate/high/critical)
- **Priority Flagging**: Identifies findings requiring immediate medical attention
- **Scalable Processing**: From single notes to large-scale batch operations
- **Medical Compliance**: Structured output suitable for clinical workflows
- **Persistent Storage**: Complete database persistence with session tracking
- **Performance Optimization**: Smart caching with automatic cache management and async processing
- **Comprehensive Testing**: Full test coverage with medical scenario validation

### Multi-Modal Integration 🧬

#### Multi-Modal Data Service (`app/services/multimodal_data_service.py`)
- **Data Ingestion Pipelines**: For MIMIC-IV, UK Biobank, FAERS, and Clinical Trials data.
- **Unified Patient Identity Management**: Centralized management of patient identities across diverse datasets.
- **Cross-Dataset Patient Resolution**: Resolves and links patient records from different sources.

#### Patient Identity Service (`app/services/patient_identity_service.py`)
- **Advanced Identity Resolution**: Uses probabilistic matching for accurate patient identification.
- **Demographic Feature Extraction**: Extracts and compares demographic features for matching.
- **Conflict Resolution**: Automated resolution of identity discrepancies.

#### Multi-Modal Vector Service (`app/services/multimodal_vector_service.py`)
- **Extends Faiss Infrastructure**: Leverages existing Faiss for cross-dataset similarity search.
- **Specialized Embeddings**: Generates embeddings tailored for different medical modalities.
- **High-Performance Vector Search**: Enables rapid similarity searches across millions of records.

#### Data Fusion Service (`app/services/data_fusion_service.py`)
- **Sophisticated Evidence Aggregation**: Combines evidence from multiple datasets with weighted scoring.
- **Temporal Decay**: Applies time-based weighting to clinical evidence for relevance.
- **Uncertainty Quantification**: Provides comprehensive analysis of data uncertainty.

#### Clinical Trials Matching Service (`app/services/clinical_trials_matching_service.py`)
- **Advanced Matching Algorithms**: Utilizes sophisticated algorithms for patient-to-trial matching.
- **Real-time Integration**: Connects with ClinicalTrials.gov API for up-to-date trial information.
- **Multi-dimensional Eligibility**: Assesses patient eligibility based on various criteria.

### Data Validation (`app/utils/validation.py`)
- Input validation for all API endpoints
- Schema validation for patient/note data
- Parameter validation with decorators

### Data Sanitization (`app/utils/sanitization.py`)
- XSS prevention with HTML sanitization
- SQL injection protection
- Input cleaning and normalization

### Security Middleware (`app/middleware/security.py`)
- Rate limiting per IP address
- Security headers (CSRF, XSS, etc.)
- Request logging and monitoring

### Database Utilities
- **ICD-10 Database**: `app/utils/create_icd10_db.py`
  - Batch processing for large CSV files
  - Progress tracking and resume capability
  - Error handling with retry mechanisms
- **Patient Notes Database**: `app/utils/create_patient_note_db.py`
  - Similar optimizations as ICD-10 script
  - Handles patient demographic data


## Running Tests

### Run All Tests
```bash
source venv/bin/activate
python -m pytest test/ -v
```

### Test Intelligence Layer (Phase 2 + Option B Enhancements)
```bash
source venv/bin/activate

# Core Intelligence Layer Tests
python test/test_intelligence_layer.py               # Complete workflow demonstration
python test/test_claude_service.py                   # Claude integration test
python test/test_simple_api.py                       # API integration test (no server needed)
python test/test_api_endpoints.py                    # Full API endpoint test (requires server)

# Database Persistence Tests
python test/test_analysis_storage_service.py         # Database persistence tests
python test/test_create_intelligence_db.py           # Database schema tests
python test/test_analysis_routes_persistence.py      # API persistence integration tests

# Enhanced NLP & Async Processing Tests (New!)
python test/test_clinical_nlp.py                     # Enhanced NLP features
python test/test_async_clinical_analysis.py          # Async batch processing
python test/test_enhanced_clinical_analysis.py       # NLP-enhanced clinical analysis

# Unit Tests
python -m pytest test/test_clinical_analysis_service.py -v
python -m pytest test/test_icd10_vector_matcher.py -v
```

### Run Tests for Specific Module
```bash
python -m pytest test/utils/          # Database utility tests
python -m pytest test/routes/         # API endpoint tests
python -m pytest test/services/       # Intelligence layer tests
python -m pytest test/utils/test_create_icd10_db.py
```

### Run Individual Test File
```bash
python -m unittest test.utils.test_create_icd10_db
python -m unittest test.utils.test_create_patient_note_db
python -m unittest test.routes.test_patient_routes
python -m unittest test.routes.test_note_routes
```

### Run Tests with Coverage Report
```bash
pip install pytest-cov
python -m pytest test/ --cov=app --cov-report=html
```

## Test Structure

```
test/
├── config/          # Tests for configuration files
├── models/          # Tests for data models
├── routes/          # Tests for API routes
│   ├── test_patient_routes.py
│   └── test_note_routes.py
├── services/        # Tests for business logic
│   ├── test_supabase_service.py
│   ├── test_clinical_analysis_service.py    # Phase 2: Claude integration
│   └── test_icd10_vector_matcher.py         # Phase 2: ICD-10 mapping
├── utils/           # Tests for utility functions
│   ├── test_create_icd10_db.py
│   ├── test_create_patient_note_db.py
│   ├── test_validation.py
│   └── test_sanitization.py
├── test_claude_service.py                 # Phase 2: Claude integration demo
├── test_intelligence_layer.py            # Phase 2: Complete workflow test
├── test_api_endpoints.py                 # Phase 2: Full API endpoint testing
├── test_simple_api.py                    # Phase 2: API integration test (no server)
├── test_analysis_storage_service.py      # Phase 2: Database persistence tests
├── test_create_intelligence_db.py        # Phase 2: Database schema tests
├── test_analysis_routes_persistence.py   # Phase 2: API persistence integration
├── test_clinical_nlp.py                  # Option B: Enhanced NLP features
├── test_async_clinical_analysis.py       # Option B: Async batch processing
├── test_enhanced_clinical_analysis.py    # Option B: NLP-enhanced analysis
└── test_api_manual.md                    # Phase 2: Manual API testing guide
```

## Development Guidelines

- Every new file in `app/` must have a corresponding test file in `test/`
- Tests must cover happy path, edge cases, and error handling
- All tests must pass before code changes are considered complete
- See `.claude/test_rules.md` for detailed testing requirements

## Project Structure

```
Patient-Analysis/
├── .claude/         # Claude Code configuration
├── .git/            # Git repository
├── app/
│   ├── __init__.py
│   ├── config/      # Configuration files
│   ├── middleware/  # Security middleware
│   │   └── security.py
│   ├── models/      # Data models
│   │   └── patient.py
│   ├── routes/      # API routes
│   │   ├── patient_routes.py
│   │   └── note_routes.py
│   ├── services/    # Business logic
│   │   ├── supabase_service.py
│   │   ├── clinical_analysis_service.py    # Phase 2: Claude AI integration (enhanced)
│   │   ├── icd10_vector_matcher.py         # Phase 2: ICD-10 mapping
│   │   ├── analysis_storage_service.py     # Phase 2: Database persistence
│   │   └── async_clinical_analysis.py      # Option B: High-performance async processing
│   └── utils/       # Utility functions
│       ├── create_icd10_db.py
│       ├── create_patient_note_db.py
│       ├── create_intelligence_db.py       # Phase 2: Intelligence layer tables
│       ├── clinical_nlp.py                 # Option B: Enhanced NLP processing
│       ├── validation.py
│       └── sanitization.py
├── test/            # Test files (mirrors app structure)
│   ├── config/
│   ├── models/
│   ├── routes/
│   ├── services/
│   └── utils/
├── venv/            # Virtual environment
├── .env             # Environment variables
├── .gitignore       # Git ignore rules
├── app.py           # Main application entry point
├── requirements.txt # Python dependencies
├── icd_10_codes.csv # ICD-10 data file
├── PMC_Patients_clean.csv # Patient data file
└── README.md        # This file
```

## Key Components

### Phase 2: Intelligence Layer 🧠

#### Clinical Analysis Service (`app/services/clinical_analysis_service.py`)
- **Claude AI Integration**: Uses Claude 3.5 Sonnet for clinical text analysis
- **Entity Extraction**: Identifies symptoms, conditions, medications, vital signs, procedures
- **Confidence Scoring**: Each extraction includes 0.0-1.0 confidence levels
- **Medical Context**: Handles negation, severity, temporal information
- **High-Priority Detection**: Flags critical findings requiring immediate attention
- **Batch Processing**: Analyzes multiple patient notes simultaneously

#### ICD-10 Vector Matcher (`app/services/icd10_vector_matcher.py`)
- **Vector Similarity**: Cosine similarity matching against ICD-10 embeddings
- **Text-Based Fallback**: Simple text matching when vectors unavailable
- **Entity Mapping**: Maps clinical entities to relevant ICD-10 codes
- **Hierarchy Analysis**: Provides ICD code category information
- **Confidence Weighting**: Combines extraction confidence with similarity scores

#### Analysis Storage Service (`app/services/analysis_storage_service.py`) 🆕
- **Session Management**: Create and track analysis sessions with complete audit trail
- **Entity Persistence**: Store clinical entities with confidence scores and metadata
- **ICD Mapping Storage**: Persistent storage of entity-to-ICD mappings with rankings
- **Smart Caching**: Automatic result caching with TTL and hit rate tracking
- **Priority Retrieval**: Query high-priority findings by note, patient, or risk level
- **Cache Maintenance**: Automatic cleanup of expired cache entries

#### Enhanced Clinical NLP (`app/utils/clinical_nlp.py`) 🆕
- **Medical Abbreviation Expansion**: 100+ medical abbreviations automatically expanded
- **Sophisticated Negation Detection**: 20+ negation patterns including medical-specific contexts
- **Temporal Relationship Extraction**: Onset, duration, frequency, and progression detection
- **Uncertainty Assessment**: Handles speculation markers and confidence adjustments
- **Clinical Context Awareness**: Entity relationships and medical coherence validation

#### Async Clinical Analysis (`app/services/async_clinical_analysis.py`) 🚀
- **High-Performance Batch Processing**: Up to 1000 notes with configurable concurrency
- **Priority Triage Scanning**: Rapid identification of high-risk cases (up to 2000 notes)
- **Intelligent Chunking**: Memory-efficient processing with automatic chunk management
- **Advanced Retry Logic**: Exponential backoff and error recovery mechanisms
- **Performance Monitoring**: Comprehensive metrics including cache hit rates and timing

#### Intelligence Layer Capabilities
- **Clinical Entity Types**: Symptoms, conditions, medications, vital signs, procedures, abnormal findings
- **Enhanced Accuracy**: Advanced NLP processing for clinical accuracy improvements
- **Risk Assessment**: Automatic risk level classification (low/moderate/high/critical)
- **Priority Flagging**: Identifies findings requiring immediate medical attention
- **Scalable Processing**: From single notes to large-scale batch operations
- **Medical Compliance**: Structured output suitable for clinical workflows
- **Persistent Storage**: Complete database persistence with session tracking
- **Performance Optimization**: Smart caching with automatic cache management and async processing
- **Comprehensive Testing**: Full test coverage with medical scenario validation

### Data Validation (`app/utils/validation.py`)
- Input validation for all API endpoints
- Schema validation for patient/note data
- Parameter validation with decorators

### Data Sanitization (`app/utils/sanitization.py`)
- XSS prevention with HTML sanitization
- SQL injection protection
- Input cleaning and normalization

### Security Middleware (`app/middleware/security.py`)
- Rate limiting per IP address
- Security headers (CSRF, XSS, etc.)
- Request logging and monitoring

### Database Utilities
- **ICD-10 Database**: `app/utils/create_icd10_db.py`
  - Batch processing for large CSV files
  - Progress tracking and resume capability
  - Error handling with retry mechanisms
- **Patient Notes Database**: `app/utils/create_patient_note_db.py`
  - Similar optimizations as ICD-10 script
  - Handles patient demographic data