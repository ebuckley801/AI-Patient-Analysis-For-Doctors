# Unified Patient Access Implementation Guide

## Overview

This guide shows you how to implement cohesive patient data access using the Unified Patient Service. The service intelligently determines what multi-modal data is relevant and needed for each patient query, providing optimized access to all healthcare data sources.

## Key Concepts

### 🎯 Context-Aware Data Loading
The system automatically determines what data to load based on the **query context**:

- **Clinical Review**: Prioritizes recent clinical notes, diagnoses, medications
- **Emergency Triage**: Emphasizes vital signs, MIMIC patterns, critical alerts
- **Genetic Counseling**: Focuses on genetic data, family history, pharmacogenomics
- **Trial Matching**: Loads trial eligibility, matching criteria, research data
- **Risk Assessment**: Combines all modalities for comprehensive risk analysis

### ⚡ Intelligent Performance Optimization
- **Progressive Loading**: Critical data first, then relevant data based on context
- **Time Budgeting**: Respects response time limits with graceful degradation
- **Lazy Loading**: Expensive operations only when specifically needed
- **Caching**: Intelligent caching of computed results

### 🔗 Unified Patient Identity
- **Cross-Dataset Resolution**: Automatically resolves patient identities across all data sources
- **Data Availability Mapping**: Knows what data exists for each patient
- **Smart Prioritization**: Loads most clinically relevant data first

## Implementation Steps

### Step 1: Register the Unified Routes

Add to your Flask app initialization:

```python
from app.routes.unified_patient_routes import unified_bp

# Register unified patient routes
app.register_blueprint(unified_bp, url_prefix='/api/unified')
```

### Step 2: Database Setup

Run the multi-modal schema extensions:

```sql
-- Run in your Supabase SQL editor
\i setup/database/multimodal_integration_schema.sql
```

### Step 3: Basic Patient Access

```python
# Get comprehensive patient view for clinical review
GET /api/unified/patient/12345?context=clinical_review

# Response includes intelligently loaded data:
{
    "success": true,
    "data": {
        "patient_id": "unified-patient-uuid",
        "query_context": "clinical_review",
        "demographics": {
            "age": 45,
            "gender": "M",
            "ethnicity": "White"
        },
        "data_availability": {
            "demographics": true,
            "clinical_notes": true,
            "mimic_data": false,
            "genetic_data": true,
            "adverse_events": false,
            "trial_matches": true
        },
        "clinical_summary": {
            "recent_sessions": 3,
            "total_entities": 15,
            "entity_types": {
                "conditions": 4,
                "medications": 6,
                "procedures": 2
            },
            "recent_findings": [
                {
                    "type": "condition",
                    "text": "Type 2 diabetes mellitus",
                    "confidence": 0.95,
                    "severity": "moderate"
                }
            ]
        },
        "genetic_profile": {
            "available": true,
            "total_variants": 156,
            "high_risk_variants": 12,
            "risk_conditions": ["cardiovascular disease", "diabetes"]
        },
        "data_completeness_score": 0.75,
        "query_performance_ms": 1200,
        "recommendations": [
            "Consider genetic counseling consultation",
            "Review clinical trial opportunities"
        ]
    }
}
```

## Context-Specific Usage Examples

### 🏥 Clinical Review
```javascript
// Get optimized view for doctor reviewing patient
const response = await fetch('/api/unified/patient/12345/clinical-review');
const patientView = await response.json();

// Automatically includes:
// - Recent clinical notes and analyses
// - Current medications and conditions
// - Risk indicators and alerts
// - Relevant genetic factors
// - Clinical recommendations
```

### 🚨 Emergency Triage  
```javascript
// Get critical data for emergency department
const response = await fetch('/api/unified/patient/12345/emergency-triage');
const emergencyView = await response.json();

// Prioritizes:
// - Vital signs patterns from MIMIC-IV
// - Critical care history
// - Immediate risk factors
// - Adverse drug reactions
// - Emergency contact information
```

### 🧬 Genetic Counseling
```javascript
// Get comprehensive genetic profile
const response = await fetch('/api/unified/patient/12345/genetic-counseling');
const geneticView = await response.json();

// Focuses on:
// - Complete genetic variant profile
// - Pharmacogenomic markers
// - Family history patterns
// - Disease predisposition scores
// - Genetic counseling recommendations
```

### 🔬 Clinical Trial Matching
```javascript
// Get view optimized for trial recruitment
const response = await fetch('/api/unified/patient/12345/trial-matching?include_trials=true');
const trialView = await response.json();

// Includes:
// - Detailed eligibility assessment
// - Active trial matches with scores
// - Geographic feasibility analysis
// - Inclusion/exclusion criteria evaluation
// - Trial recommendation priorities
```

## Advanced Usage Patterns

### Multi-Patient Batch Operations
```javascript
// Get summaries for multiple patients efficiently
const batchRequest = {
    patient_ids: ["patient1", "patient2", "patient3"],
    max_patients: 50
};

const response = await fetch('/api/unified/patients/batch-summary', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(batchRequest)
});

const batchResults = await response.json();
// Returns optimized summaries for all patients
```

### Patient Data Refresh
```javascript
// Refresh patient data after new information added
const refreshRequest = {
    force_recompute: true,
    components: ["genetics", "trials", "fusion"]
};

const response = await fetch('/api/unified/patient/12345/refresh', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(refreshRequest)
});

// Triggers recomputation of specified components
```

### Custom Context with Time Limits
```javascript
// Get patient data with custom time budget
const url = '/api/unified/patient/12345?' + 
    'context=risk_assessment&' +
    'include_similar=true&' +
    'include_trials=true&' +
    'max_time=15';

const response = await fetch(url);
const customView = await response.json();

// System optimizes loading within 15-second budget
```

## Integration with Existing Services

### Extending Clinical Analysis
```python
from app.services.unified_patient_service import UnifiedPatientService, QueryContext

# In your existing clinical analysis workflow
unified_service = UnifiedPatientService()

# Get comprehensive patient context before analysis
patient_view = await unified_service.get_unified_patient_view(
    patient_identifier=patient_id,
    query_context=QueryContext.CLINICAL_REVIEW,
    max_response_time_seconds=10
)

# Use multi-modal context to enhance analysis
if patient_view.genetic_profile and patient_view.genetic_profile.get('available'):
    # Include genetic factors in clinical decision-making
    genetic_risks = patient_view.genetic_profile.get('risk_conditions', [])
    
if patient_view.mimic_profile and patient_view.mimic_profile.get('available'):
    # Consider critical care patterns
    icu_history = patient_view.mimic_profile.get('icu_stays', 0)
```

### Custom Query Contexts
```python
# Create custom context for specific use cases
class CustomQueryContext(Enum):
    MEDICATION_REVIEW = "medication_review"
    SURGICAL_PLANNING = "surgical_planning"
    DISCHARGE_PLANNING = "discharge_planning"

# Extend context relevance configuration
custom_relevance = {
    CustomQueryContext.MEDICATION_REVIEW: {
        'demographics': 0.8,
        'clinical_text': 0.9,
        'genetic': 0.7,  # High for pharmacogenomics
        'adverse_events': 0.9,  # Critical for drug safety
        'mimic': 0.3,
        'trials': 0.2
    }
}
```

### Performance Monitoring
```python
# Monitor query performance and data completeness
patient_view = await unified_service.get_unified_patient_view(patient_id)

print(f"Query completed in {patient_view.query_performance_ms}ms")
print(f"Data completeness: {patient_view.data_completeness_score:.1%}")
print(f"Recommendations: {patient_view.recommendations}")

# Log for analytics
logger.info(f"Patient {patient_id}: "
           f"completeness={patient_view.data_completeness_score:.2f}, "
           f"performance={patient_view.query_performance_ms}ms, "
           f"context={patient_view.query_context}")
```

## Best Practices

### 1. Choose Appropriate Context
```python
# Match context to use case
contexts = {
    'doctor_visit': QueryContext.CLINICAL_REVIEW,
    'er_admission': QueryContext.EMERGENCY_TRIAGE,
    'research_study': QueryContext.RESEARCH_ANALYSIS,
    'genetic_consult': QueryContext.GENETIC_COUNSELING,
    'trial_screening': QueryContext.TRIAL_MATCHING
}

context = contexts.get(use_case, QueryContext.CLINICAL_REVIEW)
```

### 2. Handle Missing Data Gracefully
```python
# Check data availability before using
if patient_view.genetic_profile and patient_view.genetic_profile.get('available'):
    # Use genetic data
    process_genetic_data(patient_view.genetic_profile)
else:
    # Fallback for missing genetic data
    logger.info(f"No genetic data available for patient {patient_id}")
```

### 3. Optimize for Common Patterns
```python
# Use quick summary for lists
patients_summary = await unified_service.get_patient_summary(patient_id)

# Use full view only when needed
if need_detailed_analysis:
    patient_view = await unified_service.get_unified_patient_view(
        patient_id, QueryContext.CLINICAL_REVIEW
    )
```

### 4. Implement Proper Error Handling
```python
try:
    patient_view = await unified_service.get_unified_patient_view(patient_id)
    return process_patient_view(patient_view)
    
except ValueError as e:
    # Patient not found
    return {"error": "Patient not found", "patient_id": patient_id}
    
except asyncio.TimeoutError:
    # Query timeout
    return {"error": "Query timeout", "patient_id": patient_id}
    
except Exception as e:
    # Other errors
    logger.error(f"Error getting patient view: {e}")
    return {"error": "Internal error", "patient_id": patient_id}
```

## API Reference

### Core Endpoints

| Endpoint | Method | Purpose | Response Time |
|----------|--------|---------|---------------|
| `/api/unified/patient/{id}` | GET | Complete unified view | 5-15s |
| `/api/unified/patient/{id}/summary` | GET | Quick summary | <2s |
| `/api/unified/patient/{id}/refresh` | POST | Refresh patient data | 10-60s |

### Context-Specific Endpoints

| Endpoint | Context | Optimized For |
|----------|---------|---------------|
| `/api/unified/patient/{id}/clinical-review` | Clinical Review | Recent notes, diagnoses, medications |
| `/api/unified/patient/{id}/emergency-triage` | Emergency Triage | Vital signs, critical care patterns |
| `/api/unified/patient/{id}/genetic-counseling` | Genetic Counseling | Genetic variants, pharmacogenomics |
| `/api/unified/patient/{id}/trial-matching` | Trial Matching | Eligibility, trial opportunities |
| `/api/unified/patient/{id}/risk-assessment` | Risk Assessment | Multi-modal risk stratification |

### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `context` | string | `clinical_review` | Query context for optimization |
| `include_similar` | boolean | `false` | Include similar patients analysis |
| `include_trials` | boolean | `false` | Include clinical trial matches |
| `max_time` | integer | `10` | Maximum response time in seconds |

## Troubleshooting

### Common Issues

**Slow Query Performance**
- Reduce `max_time` parameter for faster responses
- Use context-specific endpoints instead of generic endpoint
- Check `data_completeness_score` - missing data sources may cause delays

**Missing Data**
- Check `data_availability` in response to see what's available
- Use `refresh` endpoint to update patient data
- Verify data ingestion for specific modalities

**Context Not Optimizing Correctly**
- Ensure correct context enum value
- Check context relevance configuration
- Monitor `query_performance_ms` and `recommendations`

### Performance Optimization

1. **Use Appropriate Contexts**: Choose the most specific context for your use case
2. **Leverage Caching**: Repeated queries for same patient are cached
3. **Batch Operations**: Use batch endpoints for multiple patients
4. **Progressive Enhancement**: Start with summary, then get full view if needed

## Migration from Existing System

### Step-by-Step Migration

1. **Assess Current Usage Patterns**
   ```python
   # Identify common patient data access patterns
   current_patterns = analyze_existing_queries()
   ```

2. **Map to Unified Contexts**
   ```python
   # Map existing use cases to unified contexts
   context_mapping = {
       'patient_dashboard': QueryContext.CLINICAL_REVIEW,
       'triage_screen': QueryContext.EMERGENCY_TRIAGE,
       'research_portal': QueryContext.RESEARCH_ANALYSIS
   }
   ```

3. **Gradual Replacement**
   ```python
   # Replace existing endpoints gradually
   if USE_UNIFIED_SERVICE:
       return get_unified_patient_view(patient_id, context)
   else:
       return legacy_patient_query(patient_id)
   ```

4. **Performance Comparison**
   ```python
   # Compare performance between old and new systems
   old_time = time_legacy_query(patient_id)
   new_time = time_unified_query(patient_id)
   
   improvement = (old_time - new_time) / old_time * 100
   ```

This unified approach provides a cohesive way to access all patient data intelligently, ensuring that each query gets the most relevant information for its specific use case while maintaining optimal performance.