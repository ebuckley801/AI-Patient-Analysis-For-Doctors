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

#### Intelligence Layer Routes (Phase 2)
- `POST /api/analysis/extract` - Extract clinical entities from patient notes
- `POST /api/analysis/diagnose` - Get ICD-10 mappings with confidence scores
- `POST /api/analysis/batch` - Process multiple notes for clinical insights
- `GET /api/analysis/priority/<note_id>` - Get high-priority findings for a note

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
```

## Database Setup

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

### Test Intelligence Layer (Phase 2)
```bash
source venv/bin/activate
python test_intelligence_layer.py     # Complete workflow demonstration
python test_claude_service.py         # Claude integration test
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
├── test_claude_service.py          # Phase 2: Demo script
└── test_intelligence_layer.py     # Phase 2: Complete workflow test
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
│   │   ├── clinical_analysis_service.py  # Phase 2: Claude AI integration
│   │   └── icd10_vector_matcher.py       # Phase 2: ICD-10 mapping
│   └── utils/       # Utility functions
│       ├── create_icd10_db.py
│       ├── create_patient_note_db.py
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

#### Intelligence Layer Capabilities
- **Clinical Entity Types**: Symptoms, conditions, medications, vital signs, procedures, abnormal findings
- **Risk Assessment**: Automatic risk level classification (low/moderate/high/critical)
- **Priority Flagging**: Identifies findings requiring immediate medical attention
- **Medical Compliance**: Structured output suitable for clinical workflows
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