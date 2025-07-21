# Patient Analysis Platform - AI-Powered Healthcare Analytics

A full-stack healthcare analytics platform that combines advanced AI processing with secure patient data management. Features Claude 3.5 Sonnet integration for clinical entity extraction, FAISS-powered vector search for ICD-10 code matching, and comprehensive multimodal data integration.

## 🚀 Key Features

- **AI-Powered Clinical Analysis**: Claude 3.5 Sonnet integration for advanced NLP processing of medical documents
- **Vector-Based ICD-10 Matching**: FAISS implementation for high-performance medical diagnosis classification
- **Secure Full-Stack Architecture**: JWT authentication, Flask-RESTX API, React/Next.js frontend
- **Real-Time Data Processing**: Supabase integration with scalable batch processing capabilities
- **Multimodal Healthcare Integration**: Support for MIMIC-IV, UK Biobank, FAERS, and Clinical Trials data

## 🏗️ Tech Stack

**Backend**: Python, Flask, Flask-RESTX, Supabase  
**Frontend**: React, Next.js, TypeScript  
**AI/ML**: Claude 3.5 Sonnet (Anthropic API), FAISS Vector Search  
**Database**: PostgreSQL (Supabase), Vector Embeddings  
**Security**: JWT Authentication, Input Validation, Rate Limiting  

## 🎯 Core Intelligence Features

### Clinical Entity Extraction
- Automated identification of symptoms, conditions, medications, and vital signs
- Advanced negation detection and medical abbreviation expansion
- Confidence scoring and risk assessment (low/moderate/high/critical)
- Temporal relationship extraction for medical timeline analysis

### ICD-10 Code Mapping
- Vector similarity matching using FAISS for 50-100x faster classification
- Semantic search across medical diagnosis codes
- Confidence-weighted mapping with hierarchy analysis
- Smart caching with 7-day TTL for performance optimization

### Batch Processing & Analytics
- Async processing of up to 1000 patient notes concurrently
- Priority triage scanning for high-risk case identification
- Comprehensive performance monitoring and benchmarking
- Complete audit trail with session tracking

## 🔧 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Environment Setup
Create `.env` file with:
```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_service_role_key
ANTHROPIC_KEY=your_anthropic_api_key
PORT=5000
```

### Database Setup
```bash
# Set up core database tables
python app/utils/create_patient_note_db.py
python app/utils/create_icd10_db.py

# Initialize intelligence layer
python app/utils/create_intelligence_db.py
```

### Launch Application
```bash
python app.py
# API available at http://localhost:5000
```

## 🔍 API Examples

### Clinical Analysis
```bash
# Extract medical entities from patient notes
curl -X POST http://localhost:5000/api/analysis/extract \
  -H "Content-Type: application/json" \
  -d '{
    "note_text": "Patient presents with chest pain and shortness of breath",
    "patient_context": {"age": 45, "gender": "M"}
  }'
```

### ICD-10 Diagnosis Mapping
```bash
# Get ICD-10 code mappings with confidence scores
curl -X POST http://localhost:5000/api/analysis/diagnose \
  -H "Content-Type: application/json" \
  -d '{
    "note_text": "Patient diagnosed with acute myocardial infarction",
    "options": {"max_icd_matches": 5}
  }'
```

### Batch Processing
```bash
# Process multiple notes with async performance
curl -X POST http://localhost:5000/api/analysis/batch-async \
  -H "Content-Type: application/json" \
  -d '{
    "notes": [...],
    "config": {"max_concurrent": 10, "include_icd_mapping": true}
  }'
```

## 🧪 Testing

### Run Full Test Suite
```bash
python -m pytest test/ -v --cov=app --cov-report=html
```

### Intelligence Layer Tests
```bash
# Core AI functionality
python test/test_intelligence_layer.py
python test/test_claude_service.py

# Vector search and ICD mapping
python -m pytest test/test_icd10_vector_matcher.py -v

# Async processing and performance
python test/test_async_clinical_analysis.py
```

## 📊 Performance Metrics

- **Processing Speed**: Up to 1000 notes processed concurrently
- **Vector Search**: 50-100x faster ICD-10 matching with FAISS
- **Cache Performance**: Smart caching with automatic cleanup
- **Accuracy**: Enhanced NLP with medical abbreviation expansion
- **Scalability**: Priority triage scanning for 2000+ notes

## 🔒 Security Features

- JWT-based authentication with role-based access controls
- Input validation and SQL injection prevention
- XSS protection with security headers
- Rate limiting (100 requests/hour per IP)
- Comprehensive request logging and monitoring

## 📁 Project Structure

```
Patient-Analysis/
├── app/
│   ├── routes/          # Flask-RESTX API endpoints
│   ├── services/        # AI/ML and business logic
│   ├── utils/           # Database utilities and NLP processing
│   └── middleware/      # Security and validation
├── frontend/            # React/Next.js application
├── test/               # Comprehensive test suite
└── README.md
```

## 🎯 Use Cases

- **Clinical Decision Support**: AI-powered analysis of patient notes
- **Medical Coding Automation**: Automated ICD-10 code assignment
- **Risk Assessment**: Priority identification for patient triage
- **Healthcare Analytics**: Large-scale patient data processing
- **Research Applications**: Multimodal healthcare data integration

---

Built with modern healthcare technology standards and optimized for both clinical accuracy and system performance.