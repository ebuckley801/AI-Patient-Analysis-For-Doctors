# Patient Analysis - Clinical Decision Support System Framework

## Project Structure & Architecture

### Core Components:

1. **Data Pipeline** - ETL for processing CSV files and incoming patient notes
2. **NLP Analysis Engine** - Claude integration for extracting clinical insights
3. **Medical Knowledge Base** - ICD-10 codes with structured relationships
4. **API Layer** - Flask endpoints for data ingestion and analysis
5. **Database Layer** - PostgreSQL with proper medical data schemas
6. **Validation & Compliance** - Data sanitization and medical accuracy checks

### Database Schema Design:

- **patients table** (demographics, unique identifiers)
- **clinical_notes table** (raw notes, timestamps, source info)
- **icd_codes table** (structured ICD-10 data with descriptions)
- **diagnoses table** (extracted diagnoses with confidence scores)
- **clinical_entities table** (symptoms, conditions, medications extracted)
- **audit_logs table** (for compliance and tracking)

### Flask Application Structure:
```
/app
  /models (SQLAlchemy models)
  /services (business logic, Claude integration)
  /routes (API endpoints)
  /utils (data processing, validation)
  /config (database, API configurations)
```

## Recommended Development Approach

### Phase 1: Foundation
- Set up PostgreSQL schemas and seed ICD-10 data
- Build basic Flask API with patient/note CRUD operations
- Create data validation and sanitization layer

### Phase 2: Intelligence Layer
- Integrate Claude for clinical entity extraction
- Build confidence scoring system for diagnoses
- Create mapping between extracted entities and ICD-10 codes

### Phase 3: Clinical Features
- Implement abnormality detection algorithms
- Build clinician-facing dashboard for review
- Add batch processing for large datasets

## Technical Implementation Guide

### Current Status (Phase 1 - Completed):
- ✅ Supabase database integration with patient_note and icd_codes tables
- ✅ Flask API with comprehensive CRUD operations
- ✅ Data validation and sanitization layer with security middleware
- ✅ Batch processing for large CSV files with progress tracking
- ✅ Comprehensive testing framework

### Next Steps (Phase 2 - Intelligence Layer):
- [ ] Integrate Claude API for clinical text analysis
- [ ] Create clinical entity extraction service
- [ ] Build confidence scoring system
- [ ] Map extracted entities to ICD-10 codes
- [ ] Add diagnoses table and related models

### Future Enhancements (Phase 3 - Clinical Features):
- [ ] Abnormality detection algorithms
- [ ] Clinician dashboard
- [ ] Advanced analytics and reporting
- [ ] Real-time processing capabilities

## Key Design Principles

1. **Medical Data Compliance** - HIPAA-compliant data handling and audit trails
2. **Scalability** - Batch processing, pagination, and efficient database queries
3. **Security** - Input validation, sanitization, and secure API endpoints
4. **Accuracy** - Confidence scoring and human review workflows
5. **Maintainability** - Clean architecture with separation of concerns

## Integration Points

### Claude API Integration:
- Clinical entity extraction from patient notes
- Symptom and condition identification
- Medication and treatment extraction
- Diagnostic suggestion generation

### Database Relationships:
- Patients ↔ Clinical Notes (one-to-many)
- Clinical Notes ↔ Diagnoses (one-to-many)
- Diagnoses ↔ ICD Codes (many-to-one)
- Clinical Entities ↔ Clinical Notes (many-to-many)

## Prompt Template for Claude Code

When implementing new features, use this structured approach:

```
"I'm building a clinical decision support system that analyzes patient notes to generate diagnostic insights for healthcare providers.

Current Architecture:
- Flask API with Supabase backend
- Comprehensive validation and sanitization layer
- Security middleware with rate limiting
- Batch processing for large datasets

Next Implementation Phase:
Intelligence Layer (Phase 2)

Technical Requirements:
- Maintain existing security and validation patterns
- Follow established project structure in /app
- Include comprehensive tests
- Ensure medical data compliance
- Implement proper error handling and logging

Focus on: Clean, resusable code. Do not introduce new libraries without talking through it with me to determine if it is necessary. Always build within the context of the current project to be interated upon. "
```

## Configuration Requirements

### Environment Variables:
```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_service_role_key
ANTHROPIC_KEY=your_anthropic_api_key
ROOT_DIR=path_to_your_csv_files
PORT=5000
```

### Dependencies (requirements.txt):
- Core: pandas, supabase, python-dotenv, flask
- AI: anthropic (for Claude integration)
- Security: bleach (XSS prevention)
- Testing: pytest, pytest-cov
- Processing: tqdm (progress tracking)

This framework should guide all future development decisions and ensure consistent, scalable architecture throughout the project evolution.