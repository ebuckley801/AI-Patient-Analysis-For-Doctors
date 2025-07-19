"""
Multi-Modal Medical Data Integration Service

Handles data ingestion and integration from multiple healthcare datasets:
- MIMIC-IV (critical care database)
- UK Biobank (genetic predisposition data)
- FAERS (FDA adverse event reporting)
- Clinical Trials.gov API

Built on top of existing intelligence layer architecture.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import requests
import pandas as pd
from dataclasses import dataclass
from app.services.supabase_service import SupabaseService
from app.config.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PatientIdentity:
    """Represents a unified patient identity across datasets"""
    demographics: Dict[str, Any]
    confidence: float
    source_mappings: List[Dict[str, str]]

@dataclass
class DataIngestionResult:
    """Result of data ingestion operation"""
    success: bool
    records_processed: int
    errors: List[str]
    execution_time_ms: int
    metadata: Dict[str, Any]

class MultiModalDataService:
    """Service for multi-modal medical data integration"""
    
    def __init__(self):
        self.supabase = SupabaseService()
        self.clinical_trials_base_url = "https://clinicaltrials.gov/api/v2/studies"
        
    # ============================================================================
    # UNIFIED PATIENT IDENTITY MANAGEMENT
    # ============================================================================
    
    def create_unified_patient(self, demographics: Dict[str, Any], 
                             source_dataset: str, source_patient_id: str) -> str:
        """Create or find unified patient identity"""
        try:
            # Generate master record ID from demographics hash
            demographics_str = json.dumps(demographics, sort_keys=True)
            master_record_id = f"UPI_{hashlib.sha256(demographics_str.encode()).hexdigest()[:16]}"
            
            # Check if unified patient already exists
            existing = self.supabase.client.table('unified_patients')\
                .select('*')\
                .eq('master_record_id', master_record_id)\
                .execute()
            
            if existing.data:
                unified_patient_id = existing.data[0]['unified_patient_id']
            else:
                # Create new unified patient
                new_patient = {
                    'master_record_id': master_record_id,
                    'demographics': demographics,
                    'data_sources': [source_dataset],
                    'identity_confidence': 1.0
                }
                
                result = self.supabase.client.table('unified_patients')\
                    .insert(new_patient)\
                    .execute()
                
                unified_patient_id = result.data[0]['unified_patient_id']
            
            # Create identity mapping
            mapping = {
                'unified_patient_id': unified_patient_id,
                'source_dataset': source_dataset,
                'source_patient_id': source_patient_id,
                'confidence_score': 1.0,
                'matching_method': 'exact',
                'verified': True
            }
            
            self.supabase.client.table('patient_identity_mappings')\
                .upsert(mapping, on_conflict='source_dataset,source_patient_id')\
                .execute()
            
            return unified_patient_id
            
        except Exception as e:
            logger.error(f"Error creating unified patient: {str(e)}")
            raise
    
    def resolve_patient_identity(self, source_dataset: str, 
                               source_patient_id: str) -> Optional[str]:
        """Resolve source patient ID to unified patient ID"""
        try:
            result = self.supabase.client.table('patient_identity_mappings')\
                .select('unified_patient_id')\
                .eq('source_dataset', source_dataset)\
                .eq('source_patient_id', source_patient_id)\
                .execute()
            
            return result.data[0]['unified_patient_id'] if result.data else None
            
        except Exception as e:
            logger.error(f"Error resolving patient identity: {str(e)}")
            return None
    
    # ============================================================================
    # MIMIC-IV DATA INGESTION
    # ============================================================================
    
    async def ingest_mimic_admissions(self, admissions_data: List[Dict]) -> DataIngestionResult:
        """Ingest MIMIC-IV admissions data"""
        start_time = datetime.now()
        processed = 0
        errors = []
        
        try:
            for admission in admissions_data:
                try:
                    # Create demographics from MIMIC data
                    demographics = {
                        'age_at_admission': admission.get('age'),
                        'gender': admission.get('gender'),
                        'ethnicity': admission.get('ethnicity'),
                        'marital_status': admission.get('marital_status'),
                        'insurance': admission.get('insurance')
                    }
                    
                    # Create unified patient
                    unified_patient_id = self.create_unified_patient(
                        demographics, 'mimic', str(admission['subject_id'])
                    )
                    
                    # Insert admission record
                    admission_record = {
                        'unified_patient_id': unified_patient_id,
                        'mimic_subject_id': admission['subject_id'],
                        'mimic_hadm_id': admission['hadm_id'],
                        'admission_type': admission.get('admission_type'),
                        'admission_location': admission.get('admission_location'),
                        'discharge_location': admission.get('discharge_location'),
                        'insurance': admission.get('insurance'),
                        'language': admission.get('language'),
                        'marital_status': admission.get('marital_status'),
                        'ethnicity': admission.get('ethnicity'),
                        'hospital_expire_flag': admission.get('hospital_expire_flag'),
                        'admit_time': admission.get('admittime'),
                        'discharge_time': admission.get('dischtime'),
                        'deathtime': admission.get('deathtime'),
                        'diagnosis': admission.get('diagnosis')
                    }
                    
                    self.supabase.client.table('mimic_admissions')\
                        .upsert(admission_record, on_conflict='mimic_hadm_id')\
                        .execute()
                    
                    processed += 1
                    
                except Exception as e:
                    errors.append(f"Admission {admission.get('hadm_id', 'unknown')}: {str(e)}")
            
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return DataIngestionResult(
                success=len(errors) == 0,
                records_processed=processed,
                errors=errors,
                execution_time_ms=execution_time,
                metadata={'dataset': 'mimic_admissions', 'total_records': len(admissions_data)}
            )
            
        except Exception as e:
            logger.error(f"Error ingesting MIMIC admissions: {str(e)}")
            raise
    
    async def ingest_mimic_vitals(self, vitals_data: List[Dict]) -> DataIngestionResult:
        """Ingest MIMIC-IV vital signs data"""
        start_time = datetime.now()
        processed = 0
        errors = []
        
        try:
            # Group vitals by ICU stay for efficient processing
            vitals_by_stay = {}
            for vital in vitals_data:
                stay_id = vital.get('stay_id')
                if stay_id not in vitals_by_stay:
                    vitals_by_stay[stay_id] = []
                vitals_by_stay[stay_id].append(vital)
            
            for stay_id, stay_vitals in vitals_by_stay.items():
                try:
                    # Find ICU stay record
                    icu_stay = self.supabase.client.table('mimic_icu_stays')\
                        .select('icu_stay_id')\
                        .eq('mimic_stay_id', stay_id)\
                        .execute()
                    
                    if not icu_stay.data:
                        errors.append(f"ICU stay {stay_id} not found")
                        continue
                    
                    icu_stay_id = icu_stay.data[0]['icu_stay_id']
                    
                    # Process vitals for this stay
                    vital_records = []
                    for vital in stay_vitals:
                        vital_record = {
                            'icu_stay_id': icu_stay_id,
                            'charttime': vital.get('charttime'),
                            'vital_type': vital.get('itemid_label', 'unknown'),
                            'value': vital.get('value'),
                            'unit': vital.get('valueuom'),
                            'value_normalized': self._normalize_vital_value(
                                vital.get('value'), vital.get('itemid_label')
                            )
                        }
                        vital_records.append(vital_record)
                    
                    # Batch insert vitals
                    if vital_records:
                        self.supabase.client.table('mimic_vitals')\
                            .insert(vital_records)\
                            .execute()
                        
                        processed += len(vital_records)
                
                except Exception as e:
                    errors.append(f"Stay {stay_id}: {str(e)}")
            
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return DataIngestionResult(
                success=len(errors) == 0,
                records_processed=processed,
                errors=errors,
                execution_time_ms=execution_time,
                metadata={'dataset': 'mimic_vitals', 'total_records': len(vitals_data)}
            )
            
        except Exception as e:
            logger.error(f"Error ingesting MIMIC vitals: {str(e)}")
            raise
    
    # ============================================================================
    # UK BIOBANK DATA INGESTION
    # ============================================================================
    
    async def ingest_biobank_participants(self, participants_data: List[Dict]) -> DataIngestionResult:
        """Ingest UK Biobank participant data"""
        start_time = datetime.now()
        processed = 0
        errors = []
        
        try:
            for participant in participants_data:
                try:
                    # Create demographics from biobank data
                    demographics = {
                        'birth_year': participant.get('birth_year'),
                        'sex': participant.get('sex'),
                        'ethnic_background': participant.get('ethnic_background'),
                        'birth_country': participant.get('birth_country')
                    }
                    
                    # Create unified patient
                    unified_patient_id = self.create_unified_patient(
                        demographics, 'biobank', str(participant['eid'])
                    )
                    
                    # Insert participant record
                    participant_record = {
                        'unified_patient_id': unified_patient_id,
                        'biobank_eid': participant['eid'],
                        'assessment_centre': participant.get('assessment_centre'),
                        'genotyping_array': participant.get('genotyping_array'),
                        'genetic_ethnic_grouping': participant.get('genetic_ethnic_grouping'),
                        'birth_country': participant.get('birth_country')
                    }
                    
                    self.supabase.client.table('biobank_participants')\
                        .upsert(participant_record, on_conflict='biobank_eid')\
                        .execute()
                    
                    processed += 1
                    
                except Exception as e:
                    errors.append(f"Participant {participant.get('eid', 'unknown')}: {str(e)}")
            
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return DataIngestionResult(
                success=len(errors) == 0,
                records_processed=processed,
                errors=errors,
                execution_time_ms=execution_time,
                metadata={'dataset': 'biobank_participants', 'total_records': len(participants_data)}
            )
            
        except Exception as e:
            logger.error(f"Error ingesting biobank participants: {str(e)}")
            raise
    
    async def ingest_biobank_genetics(self, genetics_data: List[Dict]) -> DataIngestionResult:
        """Ingest UK Biobank genetic data"""
        start_time = datetime.now()
        processed = 0
        errors = []
        
        try:
            for genetic_record in genetics_data:
                try:
                    # Find participant
                    participant = self.supabase.client.table('biobank_participants')\
                        .select('participant_id')\
                        .eq('biobank_eid', genetic_record['eid'])\
                        .execute()
                    
                    if not participant.data:
                        errors.append(f"Participant {genetic_record['eid']} not found")
                        continue
                    
                    participant_id = participant.data[0]['participant_id']
                    
                    # Insert genetic record
                    genetic_data_record = {
                        'participant_id': participant_id,
                        'variant_type': genetic_record.get('variant_type', 'snp'),
                        'variant_id': genetic_record.get('variant_id'),
                        'chromosome': genetic_record.get('chromosome'),
                        'position': genetic_record.get('position'),
                        'allele_1': genetic_record.get('allele_1'),
                        'allele_2': genetic_record.get('allele_2'),
                        'genotype': genetic_record.get('genotype'),
                        'risk_score': genetic_record.get('risk_score'),
                        'confidence': genetic_record.get('confidence', 0.95),
                        'associated_conditions': genetic_record.get('associated_conditions', [])
                    }
                    
                    self.supabase.client.table('biobank_genetics')\
                        .insert(genetic_data_record)\
                        .execute()
                    
                    processed += 1
                    
                except Exception as e:
                    errors.append(f"Genetic record {genetic_record.get('variant_id', 'unknown')}: {str(e)}")
            
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return DataIngestionResult(
                success=len(errors) == 0,
                records_processed=processed,
                errors=errors,
                execution_time_ms=execution_time,
                metadata={'dataset': 'biobank_genetics', 'total_records': len(genetics_data)}
            )
            
        except Exception as e:
            logger.error(f"Error ingesting biobank genetics: {str(e)}")
            raise
    
    # ============================================================================
    # FAERS DATA INGESTION
    # ============================================================================
    
    async def ingest_faers_cases(self, faers_data: List[Dict]) -> DataIngestionResult:
        """Ingest FAERS case reports"""
        start_time = datetime.now()
        processed = 0
        errors = []
        
        try:
            for case in faers_data:
                try:
                    # Create demographics from FAERS data
                    demographics = {
                        'age': case.get('age'),
                        'sex': case.get('sex'),
                        'weight': case.get('weight'),
                        'country': case.get('country')
                    }
                    
                    # Create unified patient (if identifiable)
                    unified_patient_id = None
                    if case.get('patient_id'):
                        unified_patient_id = self.create_unified_patient(
                            demographics, 'faers', str(case['patient_id'])
                        )
                    
                    # Insert FAERS case
                    case_record = {
                        'unified_patient_id': unified_patient_id,
                        'faers_case_number': case['case_number'],
                        'case_version': case.get('case_version', 1),
                        'report_type': case.get('report_type', 'initial'),
                        'serious_adverse_event': case.get('serious', False),
                        'country': case.get('country'),
                        'report_date': case.get('report_date'),
                        'receive_date': case.get('receive_date'),
                        'reporter_type': case.get('reporter_qualification')
                    }
                    
                    self.supabase.client.table('faers_cases')\
                        .upsert(case_record, on_conflict='faers_case_number')\
                        .execute()
                    
                    processed += 1
                    
                except Exception as e:
                    errors.append(f"Case {case.get('case_number', 'unknown')}: {str(e)}")
            
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return DataIngestionResult(
                success=len(errors) == 0,
                records_processed=processed,
                errors=errors,
                execution_time_ms=execution_time,
                metadata={'dataset': 'faers_cases', 'total_records': len(faers_data)}
            )
            
        except Exception as e:
            logger.error(f"Error ingesting FAERS cases: {str(e)}")
            raise
    
    # ============================================================================
    # CLINICAL TRIALS API INTEGRATION
    # ============================================================================
    
    async def fetch_clinical_trials(self, conditions: List[str], 
                                  status: str = "RECRUITING") -> List[Dict]:
        """Fetch clinical trials from ClinicalTrials.gov API"""
        try:
            params = {
                'format': 'json',
                'query.cond': ' OR '.join(conditions),
                'query.status': status,
                'countTotal': 'true',
                'pageSize': 1000
            }
            
            response = requests.get(self.clinical_trials_base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            return data.get('studies', [])
            
        except Exception as e:
            logger.error(f"Error fetching clinical trials: {str(e)}")
            return []
    
    async def ingest_clinical_trials(self, trials_data: List[Dict]) -> DataIngestionResult:
        """Ingest clinical trials data"""
        start_time = datetime.now()
        processed = 0
        errors = []
        
        try:
            for trial in trials_data:
                try:
                    protocol_section = trial.get('protocolSection', {})
                    identification = protocol_section.get('identificationModule', {})
                    status_info = protocol_section.get('statusModule', {})
                    design_info = protocol_section.get('designModule', {})
                    conditions_info = protocol_section.get('conditionsModule', {})
                    eligibility_info = protocol_section.get('eligibilityModule', {})
                    
                    # Extract trial information
                    trial_record = {
                        'nct_id': identification.get('nctId'),
                        'title': identification.get('briefTitle'),
                        'brief_summary': identification.get('briefSummary'),
                        'detailed_description': identification.get('detailedDescription'),
                        'study_type': design_info.get('studyType'),
                        'phase': design_info.get('phases', [None])[0] if design_info.get('phases') else None,
                        'status': status_info.get('overallStatus'),
                        'enrollment_count': design_info.get('enrollmentInfo', {}).get('count'),
                        'primary_completion_date': status_info.get('primaryCompletionDateStruct', {}).get('date'),
                        'study_completion_date': status_info.get('completionDateStruct', {}).get('date'),
                        'sponsor_name': protocol_section.get('sponsorCollaboratorsModule', {}).get('leadSponsor', {}).get('name'),
                        'study_design': design_info,
                        'eligibility_criteria': eligibility_info.get('eligibilityCriteria'),
                        'conditions': conditions_info.get('conditions', []),
                        'interventions': protocol_section.get('armsInterventionsModule', {}),
                        'primary_outcomes': protocol_section.get('outcomesModule', {}).get('primaryOutcomes', []),
                        'secondary_outcomes': protocol_section.get('outcomesModule', {}).get('secondaryOutcomes', []),
                        'keywords': conditions_info.get('keywords', []),
                        'mesh_terms': []  # Would need additional processing
                    }
                    
                    self.supabase.client.table('clinical_trials')\
                        .upsert(trial_record, on_conflict='nct_id')\
                        .execute()
                    
                    processed += 1
                    
                except Exception as e:
                    errors.append(f"Trial {trial.get('nctId', 'unknown')}: {str(e)}")
            
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return DataIngestionResult(
                success=len(errors) == 0,
                records_processed=processed,
                errors=errors,
                execution_time_ms=execution_time,
                metadata={'dataset': 'clinical_trials', 'total_records': len(trials_data)}
            )
            
        except Exception as e:
            logger.error(f"Error ingesting clinical trials: {str(e)}")
            raise
    
    # ============================================================================
    # PATIENT-TRIAL MATCHING
    # ============================================================================
    
    async def match_patient_to_trials(self, unified_patient_id: str, 
                                    session_id: str) -> List[Dict]:
        """Match patient to relevant clinical trials"""
        try:
            # Get patient clinical entities from session
            entities = self.supabase.client.table('clinical_entities')\
                .select('*')\
                .eq('session_id', session_id)\
                .execute()
            
            if not entities.data:
                return []
            
            # Extract conditions for matching
            conditions = [
                entity['entity_text'] for entity in entities.data 
                if entity['entity_type'] == 'condition'
            ]
            
            if not conditions:
                return []
            
            # Find relevant trials
            trials = self.supabase.client.table('clinical_trials')\
                .select('*')\
                .eq('status', 'RECRUITING')\
                .execute()
            
            matches = []
            for trial in trials.data:
                trial_conditions = trial.get('conditions', [])
                
                # Simple matching based on condition overlap
                match_score = self._calculate_trial_match_score(conditions, trial_conditions)
                
                if match_score > 0.5:  # Threshold for relevance
                    match_record = {
                        'unified_patient_id': unified_patient_id,
                        'trial_id': trial['trial_id'],
                        'session_id': session_id,
                        'match_score': match_score,
                        'eligibility_assessment': self._assess_eligibility(entities.data, trial),
                        'matching_method': 'condition_overlap',
                        'recommendation_level': self._get_recommendation_level(match_score)
                    }
                    
                    # Insert match record
                    self.supabase.client.table('patient_trial_matches')\
                        .insert(match_record)\
                        .execute()
                    
                    matches.append(match_record)
            
            return matches
            
        except Exception as e:
            logger.error(f"Error matching patient to trials: {str(e)}")
            return []
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def _normalize_vital_value(self, value: float, vital_type: str) -> float:
        """Normalize vital sign values to standard ranges"""
        if not value or not vital_type:
            return value
        
        # Simple normalization - would implement sophisticated logic
        normalization_map = {
            'heart_rate': lambda x: max(0, min(300, x)),
            'blood_pressure_systolic': lambda x: max(0, min(300, x)),
            'temperature': lambda x: x if 30 <= x <= 45 else None
        }
        
        normalizer = normalization_map.get(vital_type.lower())
        return normalizer(value) if normalizer else value
    
    def _calculate_trial_match_score(self, patient_conditions: List[str], 
                                   trial_conditions: List[str]) -> float:
        """Calculate similarity score between patient and trial conditions"""
        if not patient_conditions or not trial_conditions:
            return 0.0
        
        # Simple overlap calculation - would implement sophisticated similarity
        patient_set = set(cond.lower() for cond in patient_conditions)
        trial_set = set(cond.lower() for cond in trial_conditions)
        
        overlap = len(patient_set & trial_set)
        union = len(patient_set | trial_set)
        
        return overlap / union if union > 0 else 0.0
    
    def _assess_eligibility(self, entities: List[Dict], trial: Dict) -> Dict:
        """Assess patient eligibility for clinical trial"""
        # Placeholder for sophisticated eligibility assessment
        return {
            'preliminary_eligible': True,
            'inclusion_criteria_met': [],
            'exclusion_criteria_violated': [],
            'assessment_notes': 'Preliminary assessment based on condition matching'
        }
    
    def _get_recommendation_level(self, match_score: float) -> str:
        """Convert match score to recommendation level"""
        if match_score >= 0.8:
            return 'high'
        elif match_score >= 0.6:
            return 'medium'
        elif match_score >= 0.4:
            return 'low'
        else:
            return 'exclude'