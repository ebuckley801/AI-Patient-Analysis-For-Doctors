# Manual API Testing Guide

## Intelligence Layer API Endpoints

The following endpoints are now available for testing the intelligence layer:

### 1. Health Check
```bash
curl -X GET http://localhost:5001/api/analysis/health
```

### 2. Extract Clinical Entities
```bash
curl -X POST http://localhost:5001/api/analysis/extract \
  -H "Content-Type: application/json" \
  -d '{
    "note_text": "Patient is a 58-year-old male with chest pain, shortness of breath, and elevated troponin. ECG shows ST elevation. Diagnosis: STEMI.",
    "patient_context": {
      "age": 58,
      "gender": "male",
      "medical_history": "hypertension, diabetes"
    }
  }'
```

### 3. Diagnose with ICD-10 Mapping
```bash
curl -X POST http://localhost:5001/api/analysis/diagnose \
  -H "Content-Type: application/json" \
  -d '{
    "note_text": "Patient diagnosed with acute myocardial infarction. Has chest pain radiating to left arm.",
    "patient_context": {
      "age": 65,
      "gender": "female"
    },
    "options": {
      "include_low_confidence": false,
      "max_icd_matches": 5
    }
  }'
```

### 4. Batch Analysis
```bash
curl -X POST http://localhost:5001/api/analysis/batch \
  -H "Content-Type: application/json" \
  -d '{
    "notes": [
      {
        "note_id": "note_1",
        "note_text": "Patient has fever and headache. Temperature 101.5F.",
        "patient_context": {"age": 30, "gender": "female"}
      },
      {
        "note_id": "note_2",
        "note_text": "Follow-up visit. Blood pressure controlled on medication.",
        "patient_context": {"age": 55, "gender": "male"}
      }
    ],
    "options": {
      "include_icd_mapping": true,
      "include_priority_analysis": true
    }
  }'
```

### 5. Priority Findings (Fully Implemented)
```bash
# Basic priority findings
curl -X GET http://localhost:5001/api/analysis/priority/test_note_123

# With specific risk threshold
curl -X GET "http://localhost:5001/api/analysis/priority/test_note_123?risk_threshold=critical"

# With detailed entity information
curl -X GET "http://localhost:5001/api/analysis/priority/test_note_123?include_details=true&risk_threshold=high"
```

## Starting the Server

1. Activate virtual environment:
```bash
source venv/bin/activate
```

2. Start Flask server:
```bash
python app.py
```

3. Server will be available at: `http://localhost:5001`

## Expected Responses

### Health Check Response:
```json
{
  "status": "healthy",
  "timestamp": "2025-07-13T20:30:00.000000",
  "services": {
    "clinical_analysis": "available",
    "icd_matcher": "available",
    "icd_cache": {
      "loaded": false,
      "total_codes": 0
    }
  }
}
```

### Extract Response:
```json
{
  "success": true,
  "data": {
    "symptoms": [
      {
        "entity": "chest pain",
        "severity": "severe",
        "confidence": 0.95,
        "text_span": "chest pain"
      }
    ],
    "conditions": [
      {
        "entity": "ST-elevation myocardial infarction",
        "status": "active",
        "confidence": 0.95
      }
    ],
    "overall_assessment": {
      "risk_level": "critical",
      "requires_immediate_attention": true,
      "summary": "Patient presenting with acute STEMI"
    }
  }
}
```

### 6. Enhanced NLP Features (New in Option B)
```bash
# Test enhanced negation detection
curl -X POST http://localhost:5001/api/analysis/extract \
  -H "Content-Type: application/json" \
  -d '{
    "note_text": "Patient denies chest pain but reports SOB. No fever present. Possible pneumonia.",
    "patient_context": {"age": 45, "gender": "male"},
    "note_id": "nlp_test_001"
  }'

# Test medical abbreviation expansion
curl -X POST http://localhost:5001/api/analysis/extract \
  -H "Content-Type: application/json" \
  -d '{
    "note_text": "Pt is 65 y/o M with h/o DM, HTN who presents with c/o SOB and CP. BP 160/90, HR 110 bpm.",
    "patient_context": {"age": 65, "gender": "male"}
  }'

# Test temporal relationship extraction
curl -X POST http://localhost:5001/api/analysis/extract \
  -H "Content-Type: application/json" \
  -d '{
    "note_text": "Chest pain started 3 days ago, worsening since yesterday. Intermittent episodes.",
    "patient_context": {"age": 55, "gender": "female"}
  }'
```

### 7. Async Batch Processing (High Performance)
```bash
# High-performance async batch processing (up to 1000 notes)
curl -X POST http://localhost:5001/api/analysis/batch-async \
  -H "Content-Type: application/json" \
  -d '{
    "notes": [
      {
        "note_id": "async_1",
        "note_text": "Patient has severe chest pain and SOB",
        "patient_context": {"age": 65, "gender": "male"},
        "patient_id": "patient_001"
      },
      {
        "note_id": "async_2", 
        "note_text": "Follow-up visit, no acute distress",
        "patient_context": {"age": 45, "gender": "female"},
        "patient_id": "patient_002"
      }
    ],
    "config": {
      "max_concurrent": 10,
      "timeout_seconds": 30,
      "include_icd_mapping": true,
      "include_storage": true,
      "chunk_size": 50
    }
  }'

# Priority scanning for rapid triage (up to 2000 notes)
curl -X POST http://localhost:5001/api/analysis/priority-scan \
  -H "Content-Type: application/json" \
  -d '{
    "notes": [
      {
        "note_id": "scan_1",
        "note_text": "Patient in severe distress with chest pain",
        "patient_context": {"age": 70, "gender": "male"}
      },
      {
        "note_id": "scan_2",
        "note_text": "Routine checkup, vitals stable",
        "patient_context": {"age": 35, "gender": "female"}
      }
    ],
    "risk_threshold": "high"
  }'
```

### 8. Cache and Storage Features (Phase 2)
```bash
# Create intelligence layer database tables (run once)
python app/utils/create_intelligence_db.py

# Test with persistent storage
curl -X POST http://localhost:5001/api/analysis/extract \
  -H "Content-Type: application/json" \
  -d '{
    "note_text": "Patient has chest pain and fever",
    "patient_context": {"age": 45, "gender": "male"},
    "note_id": "persistent_test_001",
    "patient_id": "patient_001"
  }'

# Check priority findings for that note
curl -X GET "http://localhost:5001/api/analysis/priority/persistent_test_001?include_details=true"
```

## Testing Notes

### Enhanced NLP Features (Option B)
- **Medical Abbreviation Expansion**: 100+ medical abbreviations automatically expanded
- **Negation Detection**: 20+ sophisticated negation patterns (denies, negative for, ruled out, etc.)
- **Temporal Extraction**: Onset, duration, frequency, and progression detection
- **Uncertainty Assessment**: Handles speculation markers (possible, suspected, likely)
- **Clinical Context Awareness**: Entity relationships and medical coherence

### Async Processing Performance
- **Batch Async**: Up to 1000 notes with configurable concurrency (max 20 concurrent)
- **Priority Scan**: Up to 2000 notes optimized for rapid triage
- **Chunking**: Automatic memory management with configurable chunk sizes
- **Retry Logic**: Automatic retry with exponential backoff for failed analyses
- **Performance Metrics**: Detailed timing and success rate statistics

### Phase 2 Persistence
- **Database Setup**: Run `python app/utils/create_intelligence_db.py` to set up database tables
- **Smart Caching**: Analysis results automatically cached for 7 days to improve performance
- **Session Tracking**: Each analysis creates a session record for tracking and retrieval
- **Graceful Degradation**: API works even if database storage fails

### General Testing
- The ICD-10 cache will be empty unless the `icd_codes` table exists in Supabase
- Clinical entity extraction will work regardless of database status
- All endpoints include comprehensive error handling and validation
- Request/response logging is enabled for debugging
- Enhanced entities include detailed NLP metadata and confidence adjustments