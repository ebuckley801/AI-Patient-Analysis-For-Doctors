# Phase 2 Framework - Intelligence Layer Implementation

## Phase 2 Strategy

### Key Components:

1. **Clinical Entity Extraction** - Use Claude to identify symptoms, conditions, medications, vital signs
2. **Vector-Based ICD-10 Matching** - Leverage your vectorized codes for semantic similarity
3. **Confidence Scoring** - Rate extraction quality and diagnostic certainty
4. **Structured Output** - Format results for clinical review

## Efficient ICD-10 Navigation with Vectors

### Vector Search Strategy:

- Use cosine similarity for semantic matching between extracted entities and ICD-10 descriptions
- Implement approximate nearest neighbor (ANN) search with libraries like Faiss or Annoy
- Create hierarchical clustering of ICD-10 codes for faster category-based filtering
- Build embedding cache for common clinical terms to speed up repeated lookups

### Optimization Techniques:

- Pre-filter by medical specialty (cardiology, respiratory, etc.) before vector search
- Multi-stage matching: exact text match → fuzzy matching → vector similarity
- Batch processing for multiple entity-to-code mappings simultaneously

## Implementation Requirements

### Current Setup:
- Flask API with Supabase database (Phase 1 complete)
- Patient notes stored in patient_note table
- ICD-10 codes are vectorized and stored in database
- Need to process medical notes and extract actionable clinical information

### Phase 2 Requirements:

#### 1. Clinical Entity Extraction Service:
- Extract symptoms, conditions, medications, vital signs, and procedures from patient notes
- Identify severity indicators and temporal information (onset, duration)
- Flag potential abnormalities or concerning findings
- Return structured JSON with entity types, confidence scores, and text positions

#### 2. Vector-Based ICD-10 Matching:
- Compare extracted entities to vectorized ICD-10 codes using cosine similarity
- Implement efficient similarity search (consider using numpy/scipy for vector operations)
- Return top 5 matching codes with similarity scores
- Include ICD-10 code descriptions and categories

#### 3. Diagnostic Confidence Scoring:
- Score extraction quality based on clinical context and specificity
- Weight matches based on patient demographics (age, gender)
- Flag high-confidence vs. suggested/possible diagnoses
- Consider multiple symptoms for differential diagnosis

#### 4. Integration Requirements:
- Service class that interfaces with Claude API for text analysis
- Vector operations service for ICD-10 similarity matching
- Database layer for caching frequent entity-code mappings
- API endpoints for batch processing multiple notes

## Technical Implementation Guidelines

### Core Technologies:
- Use efficient vector operations (numpy arrays, batch processing)
- Implement caching for repeated entity extractions
- Add logging for clinical decision tracking
- Handle edge cases (unclear notes, multiple conditions)

### Specific Deliverables:
- **ClinicalAnalysisService** class with Claude integration
- **ICD10VectorMatcher** class for efficient code matching
- Flask routes for note analysis and diagnosis retrieval
- Database models for storing extracted entities and mappings
- Utility functions for confidence scoring and result formatting

### Example Flow:
- **Input**: Patient note text
- **Output**: Structured JSON with extracted entities, matched ICD-10 codes, confidence scores, and flagged abnormalities

## Additional Considerations

### Performance Optimization:
- Precompute embeddings for common clinical terms
- Index frequently accessed ICD-10 codes in memory
- Implement async processing for large batch operations
- Use connection pooling for database operations

### Clinical Accuracy:
- Validate extractions against medical ontologies (SNOMED, UMLS)
- Implement negation detection ("no fever" vs "fever")
- Handle medical abbreviations and synonyms
- Consider context windows for proper entity relationships

## Implementation Approach

### Development Principles:
- Focus on accuracy, performance, and clinical relevance
- Include comprehensive error handling and validation for medical data
- Maintain existing security and validation patterns
- Follow established project structure in /app
- Include comprehensive tests
- Ensure medical data compliance
- Implement proper error handling and logging

### Architecture Integration:
- Build within the context of the current project to be iterated upon
- Use clean, reusable code
- Do not introduce new libraries without discussion and justification
- Leverage existing Supabase integration and security middleware

## Expected Database Schema Extensions

### New Tables for Phase 2:
- **clinical_entities** table (extracted entities with types and confidence)
- **entity_icd_mappings** table (entity to ICD-10 code relationships)
- **analysis_sessions** table (tracking analysis runs and results)
- **confidence_scores** table (storing scoring metadata and thresholds)

### Relationships:
- clinical_entities ↔ patient_note (many-to-one)
- entity_icd_mappings ↔ clinical_entities (many-to-one)
- entity_icd_mappings ↔ icd_codes (many-to-one)
- analysis_sessions ↔ patient_note (one-to-one)

This framework ensures systematic implementation of the intelligence layer while maintaining the existing project's security, validation, and architectural patterns.