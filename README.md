# Patient Analysis

A healthcare application for analyzing patient data and ICD-10 codes using Supabase.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables in `.env`:
```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
ROOT_DIR=path_to_your_csv_files
```

## Flask API

### Launch the API Server
```bash
python app.py
```

The API will be available at: `http://localhost:5000`

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
python -m pytest test/
```

### Run Tests with Verbose Output
```bash
python -m pytest test/ -v
```

### Run Tests for Specific Module
```bash
python -m pytest test/utils/          # Database utility tests
python -m pytest test/routes/         # API endpoint tests
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
├── services/        # Tests for business logic
└── utils/           # Tests for utility functions
    ├── test_create_icd10_db.py
    └── test_create_patient_note_db.py
```

## Development Guidelines

- Every new file in `app/` must have a corresponding test file in `test/`
- Tests must cover happy path, edge cases, and error handling
- All tests must pass before code changes are considered complete
- See `.claude/test_rules.md` for detailed testing requirements

## Project Structure

```
Patient-Analysis/
├── app/
│   ├── config/      # Configuration files
│   ├── models/      # Data models
│   ├── routes/      # API routes
│   ├── services/    # Business logic
│   └── utils/       # Utility functions
├── test/            # Test files (mirrors app structure)
├── .env             # Environment variables
└── README.md
```