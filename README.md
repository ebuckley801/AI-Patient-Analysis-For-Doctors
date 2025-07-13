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
python -m pytest test/utils/
python -m pytest test/utils/test_create_icd10_db.py
```

### Run Individual Test File
```bash
python -m unittest test.utils.test_create_icd10_db
python -m unittest test.utils.test_create_patient_note_db
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