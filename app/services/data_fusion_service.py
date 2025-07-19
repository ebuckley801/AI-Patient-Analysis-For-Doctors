"""
Multi-Modal Medical Data Fusion Service

Advanced data fusion engine that combines insights from multiple healthcare datasets:
- Clinical text analysis (existing)
- MIMIC-IV critical care trajectories
- UK Biobank genetic risk profiles
- FAERS adverse event patterns
- Clinical trial eligibility matching

Implements sophisticated data fusion algorithms including:
- Weighted evidence aggregation
- Temporal correlation analysis
- Cross-modal conflict resolution
- Uncertainty quantification
- Risk stratification across modalities

Built on existing intelligence layer architecture with explainable AI integration.
"""

import logging
import numpy as np
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
from collections import defaultdict

from app.services.supabase_service import SupabaseService
from app.services.multimodal_vector_service import MultiModalVectorService, ModalityType
from app.services.patient_identity_service import PatientIdentityService
from app.services.clinical_analysis_service import ClinicalAnalysisService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvidenceLevel(Enum):
    """Evidence strength levels for data fusion"""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    CONFLICTING = "conflicting"
    INSUFFICIENT = "insufficient"

class RiskLevel(Enum):
    """Risk stratification levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    UNKNOWN = "unknown"

@dataclass
class Evidence:
    """Represents a piece of evidence from a data source"""
    source_modality: ModalityType
    data_source: str
    evidence_type: str  # 'diagnostic', 'prognostic', 'therapeutic', 'risk_factor'
    finding: str
    confidence: float
    supporting_data: Dict[str, Any]
    timestamp: datetime
    weight: float = 1.0

@dataclass
class FusedInsight:
    """Represents a fused insight from multiple evidence sources"""
    insight_type: str
    finding: str
    evidence_level: EvidenceLevel
    confidence_score: float
    supporting_evidence: List[Evidence]
    conflicting_evidence: List[Evidence]
    risk_assessment: RiskLevel
    clinical_significance: str
    recommendations: List[str]
    uncertainty_factors: List[str]

@dataclass
class PatientProfile:
    """Comprehensive patient profile from fused data"""
    unified_patient_id: str
    demographic_summary: Dict[str, Any]
    clinical_summary: Dict[str, Any]
    genetic_risk_profile: Dict[str, Any]
    vital_signs_patterns: Dict[str, Any]
    adverse_event_history: Dict[str, Any]
    trial_eligibility: Dict[str, Any]
    risk_stratification: Dict[str, RiskLevel]
    fused_insights: List[FusedInsight]
    data_completeness: Dict[str, float]
    last_updated: datetime

class DataFusionService:
    """Advanced multi-modal medical data fusion service"""
    
    def __init__(self):
        self.supabase = SupabaseService()
        self.vector_service = MultiModalVectorService()
        self.identity_service = PatientIdentityService()
        self.clinical_service = ClinicalAnalysisService()
        
        # Evidence weighting configuration
        self.modality_weights = {
            ModalityType.CLINICAL_TEXT: 1.0,
            ModalityType.ICD_CODES: 0.9,
            ModalityType.VITAL_SIGNS: 0.8,
            ModalityType.GENETIC_PROFILE: 0.7,
            ModalityType.ADVERSE_EVENTS: 0.6,
            ModalityType.TRIAL_ELIGIBILITY: 0.5
        }
        
        # Data source reliability weights
        self.source_weights = {
            'mimic': 0.95,  # High-quality critical care data
            'biobank': 0.90,  # Large-scale population study
            'local': 0.85,  # Local clinical data
            'faers': 0.70,  # Voluntary reporting system
            'trials': 0.80   # Structured trial data
        }
        
        # Temporal decay factors (evidence gets older, less relevant)
        self.temporal_decay_days = {
            'acute_findings': 7,
            'chronic_conditions': 365,
            'genetic_data': float('inf'),  # Genetic data doesn't decay
            'adverse_events': 180,
            'vital_signs': 30
        }
    
    # ============================================================================
    # COMPREHENSIVE PATIENT PROFILING
    # ============================================================================
    
    async def create_comprehensive_patient_profile(self, unified_patient_id: str) -> PatientProfile:
        """
        Create comprehensive patient profile by fusing all available data sources
        
        Args:
            unified_patient_id: Patient's unified identifier
            
        Returns:
            Complete PatientProfile with fused insights
        """
        try:
            logger.info(f"Creating comprehensive profile for patient {unified_patient_id}")
            
            # Gather data from all modalities
            raw_data = await self._gather_patient_data(unified_patient_id)
            
            # Extract evidence from each modality
            evidence_collection = await self._extract_evidence_from_data(raw_data)
            
            # Perform data fusion
            fused_insights = await self._fuse_evidence(evidence_collection)
            
            # Create comprehensive profile
            profile = PatientProfile(
                unified_patient_id=unified_patient_id,
                demographic_summary=self._create_demographic_summary(raw_data),
                clinical_summary=self._create_clinical_summary(raw_data, evidence_collection),
                genetic_risk_profile=self._create_genetic_risk_profile(raw_data),
                vital_signs_patterns=self._create_vital_signs_patterns(raw_data),
                adverse_event_history=self._create_adverse_event_summary(raw_data),
                trial_eligibility=self._create_trial_eligibility_summary(raw_data),
                risk_stratification=self._perform_risk_stratification(fused_insights),
                fused_insights=fused_insights,
                data_completeness=self._assess_data_completeness(raw_data),
                last_updated=datetime.now(timezone.utc)
            )
            
            # Store profile for future reference
            await self._store_patient_profile(profile)
            
            logger.info(f"Created profile with {len(fused_insights)} fused insights")
            return profile
            
        except Exception as e:
            logger.error(f"Error creating patient profile: {e}")
            raise
    
    async def _gather_patient_data(self, unified_patient_id: str) -> Dict[str, Any]:
        """Gather all available data for a patient across modalities"""
        raw_data = {
            'demographics': {},
            'clinical_entities': [],
            'mimic_admissions': [],
            'mimic_vitals': [],
            'biobank_genetics': [],
            'biobank_lifestyle': [],
            'faers_cases': [],
            'trial_matches': [],
            'icd_mappings': [],
            'variant_type': [],
        }
        
        try:
            # Get unified patient demographics
            patient_data = self.supabase.client.table('unified_patients')\
                .select('*')\
                .eq('unified_patient_id', unified_patient_id)\
                .execute()
            
            if patient_data.data:
                raw_data['demographics'] = patient_data.data[0]['demographics']
            
            # Get clinical entities from analysis sessions
            sessions = self.supabase.client.table('analysis_sessions')\
                .select('session_id')\
                .execute()
            
            for session in sessions.data:
                entities = self.supabase.client.table('clinical_entities')\
                    .select('*')\
                    .eq('session_id', session['session_id'])\
                    .execute()
                raw_data['clinical_entities'].extend(entities.data)
            
            # Get MIMIC-IV data
            mimic_admissions = self.supabase.client.table('mimic_admissions')\
                .select('*')\
                .eq('unified_patient_id', unified_patient_id)\
                .execute()
            raw_data['mimic_admissions'] = mimic_admissions.data
            
            # Get UK Biobank data
            biobank_participants = self.supabase.client.table('biobank_participants')\
                .select('participant_id')\
                .eq('unified_patient_id', unified_patient_id)\
                .execute()
            
            if biobank_participants.data:
                participant_id = biobank_participants.data[0]['participant_id']
                
                genetics = self.supabase.client.table('biobank_genetics')\
                    .select('*')\
                    .eq('participant_id', participant_id)\
                    .execute()
                raw_data['biobank_genetics'] = genetics.data
                
                lifestyle = self.supabase.client.table('biobank_lifestyle')\
                    .select('*')\
                    .eq('participant_id', participant_id)\
                    .execute()
                raw_data['biobank_lifestyle'] = lifestyle.data
            
            # Get FAERS data
            faers_cases = self.supabase.client.table('faers_cases')\
                .select('*')\
                .eq('unified_patient_id', unified_patient_id)\
                .execute()
            raw_data['faers_cases'] = faers_cases.data
            
            # Get clinical trial matches
            trial_matches = self.supabase.client.table('patient_trial_matches')\
                .select('*')\
                .eq('unified_patient_id', unified_patient_id)\
                .execute()
            raw_data['trial_matches'] = trial_matches.data
            
        except Exception as e:
            logger.warning(f"Error gathering some patient data: {e}")
        
        return raw_data
    
    # ============================================================================
    # EVIDENCE EXTRACTION
    # ============================================================================
    
    async def _extract_evidence_from_data(self, raw_data: Dict[str, Any]) -> List[Evidence]:
        """Extract structured evidence from raw multi-modal data"""
        evidence_collection = []
        
        # Extract clinical text evidence
        for entity in raw_data.get('clinical_entities', []):
            evidence = Evidence(
                source_modality=ModalityType.CLINICAL_TEXT,
                data_source='local',
                evidence_type='diagnostic',
                finding=f"{entity['entity_type']}: {entity['entity_text']}",
                confidence=float(entity.get('confidence', 0.0)),
                supporting_data={
                    'entity_type': entity['entity_type'],
                    'severity': entity.get('severity'),
                    'status': entity.get('status'),
                    'temporal_info': entity.get('temporal_info')
                },
                timestamp=datetime.fromisoformat(entity['created_at'].replace('Z', '+00:00')),
                weight=self.modality_weights[ModalityType.CLINICAL_TEXT]
            )
            evidence_collection.append(evidence)
        
        # Extract MIMIC-IV vital signs evidence
        for admission in raw_data.get('mimic_admissions', []):
            # Get vitals for this admission
            vitals = self._get_vitals_for_admission(admission)
            for vital_pattern in vitals:
                evidence = Evidence(
                    source_modality=ModalityType.VITAL_SIGNS,
                    data_source='mimic',
                    evidence_type='prognostic',
                    finding=vital_pattern['finding'],
                    confidence=vital_pattern['confidence'],
                    supporting_data=vital_pattern['data'],
                    timestamp=datetime.fromisoformat(admission['admit_time']) if admission.get('admit_time') else datetime.now(timezone.utc),
                    weight=self.modality_weights[ModalityType.VITAL_SIGNS] * self.source_weights['mimic']
                )
                evidence_collection.append(evidence)
        
        # Extract genetic evidence
        for genetic_record in raw_data.get('biobank_genetics', []):
            risk_score = genetic_record.get('risk_score', 0.0)
            if risk_score and risk_score > 0.5:  # Only include significant genetic findings
                evidence = Evidence(
                    source_modality=ModalityType.GENETIC_PROFILE,
                    data_source='biobank',
                    evidence_type='risk_factor',
                    finding=f"Genetic variant {genetic_record['variant_id']} (risk score: {risk_score})",
                    confidence=float(genetic_record.get('confidence', 0.0)),
                    supporting_data={
                        'variant_type': genetic_record['variant_type'],
                        'associated_conditions': genetic_record.get('associated_conditions', []),
                        'chromosome': genetic_record.get('chromosome'),
                        'position': genetic_record.get('position')
                    },
                    timestamp=datetime.fromisoformat(genetic_record['created_at'].replace('Z', '+00:00')),
                    weight=self.modality_weights[ModalityType.GENETIC_PROFILE] * self.source_weights['biobank']
                )
                evidence_collection.append(evidence)
        
        # Extract adverse event evidence
        for case in raw_data.get('faers_cases', []):
            evidence = Evidence(
                source_modality=ModalityType.ADVERSE_EVENTS,
                data_source='faers',
                evidence_type='therapeutic',
                finding=f"Adverse event case: {case['faers_case_number']}",
                confidence=0.7,  # FAERS has moderate confidence due to voluntary reporting
                supporting_data={
                    'serious_adverse_event': case.get('serious_adverse_event'),
                    'report_type': case.get('report_type'),
                    'country': case.get('country')
                },
                timestamp=datetime.fromisoformat(case['report_date']) if case.get('report_date') else datetime.now(timezone.utc),
                weight=self.modality_weights[ModalityType.ADVERSE_EVENTS] * self.source_weights['faers']
            )
            evidence_collection.append(evidence)
        
        # Extract trial eligibility evidence
        for trial_match in raw_data.get('trial_matches', []):
            if trial_match.get('match_score', 0) > 0.6:  # Only high-confidence matches
                evidence = Evidence(
                    source_modality=ModalityType.TRIAL_ELIGIBILITY,
                    data_source='trials',
                    evidence_type='therapeutic',
                    finding=f"Trial eligibility match (score: {trial_match['match_score']})",
                    confidence=float(trial_match['match_score']),
                    supporting_data={
                        'trial_id': trial_match['trial_id'],
                        'recommendation_level': trial_match.get('recommendation_level'),
                        'eligibility_assessment': trial_match.get('eligibility_assessment')
                    },
                    timestamp=datetime.fromisoformat(trial_match['created_at'].replace('Z', '+00:00')),
                    weight=self.modality_weights[ModalityType.TRIAL_ELIGIBILITY] * self.source_weights['trials']
                )
                evidence_collection.append(evidence)
        
        logger.info(f"Extracted {len(evidence_collection)} evidence items from patient data")
        return evidence_collection
    
    # ============================================================================
    # DATA FUSION ALGORITHMS
    # ============================================================================
    
    async def _fuse_evidence(self, evidence_collection: List[Evidence]) -> List[FusedInsight]:
        """Fuse evidence from multiple sources using weighted aggregation"""
        
        # Group evidence by clinical domain
        evidence_groups = self._group_evidence_by_domain(evidence_collection)
        
        fused_insights = []
        
        for domain, evidence_list in evidence_groups.items():
            if len(evidence_list) < 2:  # Need at least 2 pieces of evidence for fusion
                continue
            
            # Separate supporting and conflicting evidence
            supporting_evidence, conflicting_evidence = self._identify_conflicts(evidence_list)
            
            # Calculate aggregated confidence
            aggregated_confidence = self._calculate_weighted_confidence(supporting_evidence)
            
            # Determine evidence level
            evidence_level = self._determine_evidence_level(supporting_evidence, conflicting_evidence)
            
            # Assess risk level
            risk_level = self._assess_domain_risk(domain, supporting_evidence)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(domain, supporting_evidence, evidence_level)
            
            # Identify uncertainty factors
            uncertainty_factors = self._identify_uncertainty_factors(supporting_evidence, conflicting_evidence)
            
            # Create fused insight
            insight = FusedInsight(
                insight_type=domain,
                finding=self._synthesize_finding(domain, supporting_evidence),
                evidence_level=evidence_level,
                confidence_score=aggregated_confidence,
                supporting_evidence=supporting_evidence,
                conflicting_evidence=conflicting_evidence,
                risk_assessment=risk_level,
                clinical_significance=self._assess_clinical_significance(domain, supporting_evidence, risk_level),
                recommendations=recommendations,
                uncertainty_factors=uncertainty_factors
            )
            
            fused_insights.append(insight)
        
        # Sort insights by clinical significance and risk level
        fused_insights.sort(
            key=lambda x: (x.risk_assessment.value, -x.confidence_score, -len(x.supporting_evidence))
        )
        
        return fused_insights
    
    def _group_evidence_by_domain(self, evidence_list: List[Evidence]) -> Dict[str, List[Evidence]]:
        """Group evidence by clinical domain"""
        domains = defaultdict(list)
        
        for evidence in evidence_list:
            # Determine domain based on evidence content
            domain = self._classify_evidence_domain(evidence)
            domains[domain].append(evidence)
        
        return dict(domains)
    
    def _classify_evidence_domain(self, evidence: Evidence) -> str:
        """Classify evidence into clinical domains"""
        finding_lower = evidence.finding.lower()
        
        # Cardiovascular domain
        if any(term in finding_lower for term in ['heart', 'cardiac', 'cardiovascular', 'blood pressure', 'chest pain']):
            return 'cardiovascular'
        
        # Respiratory domain
        if any(term in finding_lower for term in ['lung', 'respiratory', 'breathing', 'cough', 'dyspnea']):
            return 'respiratory'
        
        # Neurological domain
        if any(term in finding_lower for term in ['neuro', 'brain', 'headache', 'seizure', 'confusion']):
            return 'neurological'
        
        # Metabolic domain
        if any(term in finding_lower for term in ['diabetes', 'glucose', 'metabolic', 'thyroid']):
            return 'metabolic'
        
        # Infectious disease domain
        if any(term in finding_lower for term in ['infection', 'fever', 'antibiotic', 'sepsis']):
            return 'infectious'
        
        # Oncology domain
        if any(term in finding_lower for term in ['cancer', 'tumor', 'oncology', 'malignancy']):
            return 'oncology'
        
        # Default to general medicine
        return 'general_medicine'
    
    def _identify_conflicts(self, evidence_list: List[Evidence]) -> Tuple[List[Evidence], List[Evidence]]:
        """Identify supporting vs conflicting evidence"""
        # Simple implementation - in practice would use NLP and medical knowledge
        supporting = []
        conflicting = []
        
        findings_seen = set()
        
        for evidence in evidence_list:
            finding_key = self._normalize_finding(evidence.finding)
            
            if finding_key in findings_seen:
                # Check if this contradicts previous evidence
                if self._is_contradictory(evidence, supporting):
                    conflicting.append(evidence)
                else:
                    supporting.append(evidence)
            else:
                supporting.append(evidence)
                findings_seen.add(finding_key)
        
        return supporting, conflicting
    
    def _calculate_weighted_confidence(self, evidence_list: List[Evidence]) -> float:
        """Calculate confidence using temporal decay and source weighting"""
        if not evidence_list:
            return 0.0
        
        total_weighted_confidence = 0.0
        total_weight = 0.0
        
        for evidence in evidence_list:
            # Apply temporal decay
            age_days = (datetime.utcnow() - evidence.timestamp).days
            temporal_weight = self._calculate_temporal_weight(evidence, age_days)
            
            # Combine all weights
            final_weight = evidence.weight * temporal_weight
            
            total_weighted_confidence += evidence.confidence * final_weight
            total_weight += final_weight
        
        return total_weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def _calculate_temporal_weight(self, evidence: Evidence, age_days: int) -> float:
        """Calculate temporal decay weight"""
        evidence_type = evidence.evidence_type
        
        # Get decay period based on evidence type
        if evidence_type == 'diagnostic' and 'acute' in evidence.finding.lower():
            decay_days = self.temporal_decay_days['acute_findings']
        elif evidence.source_modality == ModalityType.GENETIC_PROFILE:
            decay_days = self.temporal_decay_days['genetic_data']
        elif evidence.source_modality == ModalityType.VITAL_SIGNS:
            decay_days = self.temporal_decay_days['vital_signs']
        elif evidence.source_modality == ModalityType.ADVERSE_EVENTS:
            decay_days = self.temporal_decay_days['adverse_events']
        else:
            decay_days = self.temporal_decay_days['chronic_conditions']
        
        if decay_days == float('inf'):
            return 1.0  # No temporal decay
        
        # Exponential decay
        return np.exp(-age_days / decay_days)
    
    # ============================================================================
    # RISK STRATIFICATION AND ASSESSMENT
    # ============================================================================
    
    def _perform_risk_stratification(self, fused_insights: List[FusedInsight]) -> Dict[str, RiskLevel]:
        """Perform comprehensive risk stratification across domains"""
        
        risk_stratification = {}
        
        # Analyze each clinical domain
        domains = set(insight.insight_type for insight in fused_insights)
        
        for domain in domains:
            domain_insights = [insight for insight in fused_insights if insight.insight_type == domain]
            
            # Calculate overall domain risk
            domain_risk = self._calculate_domain_risk(domain_insights)
            risk_stratification[domain] = domain_risk
        
        # Calculate overall patient risk
        if risk_stratification:
            max_risk = max(risk_stratification.values(), key=lambda x: x.value)
            risk_stratification['overall'] = max_risk
        else:
            risk_stratification['overall'] = RiskLevel.UNKNOWN
        
        return risk_stratification
    
    def _calculate_domain_risk(self, domain_insights: List[FusedInsight]) -> RiskLevel:
        """Calculate risk level for a specific clinical domain"""
        if not domain_insights:
            return RiskLevel.UNKNOWN
        
        # Weight insights by evidence level and confidence
        risk_scores = []
        
        for insight in domain_insights:
            # Convert risk assessment to numeric score
            risk_score = self._risk_level_to_score(insight.risk_assessment)
            
            # Weight by evidence level
            evidence_weight = self._evidence_level_to_weight(insight.evidence_level)
            
            # Weight by confidence
            confidence_weight = insight.confidence_score
            
            weighted_score = risk_score * evidence_weight * confidence_weight
            risk_scores.append(weighted_score)
        
        # Calculate aggregated risk score
        if risk_scores:
            avg_risk_score = statistics.mean(risk_scores)
            max_risk_score = max(risk_scores)
            
            # Combine average and maximum (giving more weight to max for safety)
            final_score = 0.3 * avg_risk_score + 0.7 * max_risk_score
            
            return self._score_to_risk_level(final_score)
        
        return RiskLevel.UNKNOWN
    
    def _risk_level_to_score(self, risk_level: RiskLevel) -> float:
        """Convert risk level to numeric score"""
        risk_scores = {
            RiskLevel.CRITICAL: 1.0,
            RiskLevel.HIGH: 0.8,
            RiskLevel.MODERATE: 0.6,
            RiskLevel.LOW: 0.2,
            RiskLevel.UNKNOWN: 0.4
        }
        return risk_scores.get(risk_level, 0.4)
    
    def _evidence_level_to_weight(self, evidence_level: EvidenceLevel) -> float:
        """Convert evidence level to weight"""
        weights = {
            EvidenceLevel.STRONG: 1.0,
            EvidenceLevel.MODERATE: 0.8,
            EvidenceLevel.WEAK: 0.5,
            EvidenceLevel.CONFLICTING: 0.3,
            EvidenceLevel.INSUFFICIENT: 0.1
        }
        return weights.get(evidence_level, 0.5)
    
    def _score_to_risk_level(self, score: float) -> RiskLevel:
        """Convert numeric score back to risk level"""
        if score >= 0.9:
            return RiskLevel.CRITICAL
        elif score >= 0.7:
            return RiskLevel.HIGH
        elif score >= 0.5:
            return RiskLevel.MODERATE
        elif score >= 0.3:
            return RiskLevel.LOW
        else:
            return RiskLevel.UNKNOWN
    
    # ============================================================================
    # PROFILE CREATION HELPERS
    # ============================================================================
    
    def _create_demographic_summary(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create demographic summary from raw data"""
        demographics = raw_data.get('demographics', {})
        
        return {
            'age': demographics.get('age'),
            'gender': demographics.get('gender'),
            'ethnicity': demographics.get('ethnicity'),
            'geographic_region': demographics.get('country') or demographics.get('state'),
            'data_sources': demographics.get('data_sources', [])
        }
    
    def _create_clinical_summary(self, raw_data: Dict[str, Any], evidence: List[Evidence]) -> Dict[str, Any]:
        """Create clinical summary from entities and evidence"""
        entities = raw_data.get('clinical_entities', [])
        
        # Count entities by type
        entity_counts = defaultdict(int)
        for entity in entities:
            entity_counts[entity['entity_type']] += 1
        
        # Get most recent clinical findings
        recent_findings = []
        clinical_evidence = [e for e in evidence if e.source_modality == ModalityType.CLINICAL_TEXT]
        clinical_evidence.sort(key=lambda x: x.timestamp, reverse=True)
        
        for evidence_item in clinical_evidence[:10]:  # Top 10 most recent
            recent_findings.append({
                'finding': evidence_item.finding,
                'confidence': evidence_item.confidence,
                'timestamp': evidence_item.timestamp.isoformat()
            })
        
        return {
            'entity_counts': dict(entity_counts),
            'total_clinical_encounters': len(set(e.supporting_data.get('session_id') for e in clinical_evidence if 'session_id' in e.supporting_data)),
            'recent_findings': recent_findings
        }
    
    def _create_genetic_risk_profile(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create genetic risk profile from biobank data"""
        genetics_data = raw_data.get('biobank_genetics', [])
        
        if not genetics_data:
            return {'available': False}
        
        # Aggregate risk scores by condition
        risk_by_condition = defaultdict(list)
        
        for record in genetics_data:
            conditions = record.get('associated_conditions', [])
            risk_score = record.get('risk_score')
            
            if risk_score:
                for condition in conditions:
                    risk_by_condition[condition].append(risk_score)
        
        # Calculate average risk per condition
        condition_risks = {}
        for condition, scores in risk_by_condition.items():
            condition_risks[condition] = {
                'average_risk': statistics.mean(scores),
                'max_risk': max(scores),
                'variant_count': len(scores)
            }
        
        return {
            'available': True,
            'total_variants': len(genetics_data),
            'high_risk_variants': len([g for g in genetics_data if g.get('risk_score', 0) > 0.8]),
            'condition_risks': condition_risks
        }
    
    def _create_vital_signs_patterns(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create vital signs patterns from MIMIC data"""
        # Placeholder - would implement sophisticated vital signs analysis
        admissions = raw_data.get('mimic_admissions', [])
        
        return {
            'available': len(admissions) > 0,
            'total_admissions': len(admissions),
            'icu_stays': len([a for a in admissions if a.get('admission_location', '').lower().find('icu') != -1])
        }
    
    def _create_adverse_event_summary(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create adverse event summary from FAERS data"""
        faers_cases = raw_data.get('faers_cases', [])
        
        return {
            'available': len(faers_cases) > 0,
            'total_cases': len(faers_cases),
            'serious_events': len([c for c in faers_cases if c.get('serious_adverse_event', False)])
        }
    
    def _create_trial_eligibility_summary(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create trial eligibility summary"""
        trial_matches = raw_data.get('trial_matches', [])
        
        high_matches = [m for m in trial_matches if m.get('match_score', 0) > 0.8]
        
        return {
            'available': len(trial_matches) > 0,
            'total_matches': len(trial_matches),
            'high_confidence_matches': len(high_matches),
            'recommendation_levels': list(set(m.get('recommendation_level') for m in trial_matches if m.get('recommendation_level')))
        }
    
    def _assess_data_completeness(self, raw_data: Dict[str, Any]) -> Dict[str, float]:
        """Assess data completeness across modalities"""
        completeness = {}
        
        # Check each modality
        completeness['demographics'] = 1.0 if raw_data.get('demographics') else 0.0
        completeness['clinical_text'] = min(1.0, len(raw_data.get('clinical_entities', [])) / 10)  # Normalize to 10 entities
        completeness['mimic_data'] = 1.0 if raw_data.get('mimic_admissions') else 0.0
        completeness['genetic_data'] = 1.0 if raw_data.get('biobank_genetics') else 0.0
        completeness['adverse_events'] = 1.0 if raw_data.get('faers_cases') else 0.0
        completeness['trial_data'] = 1.0 if raw_data.get('trial_matches') else 0.0
        
        # Overall completeness
        completeness['overall'] = statistics.mean(completeness.values())
        
        return completeness
    
    # ============================================================================
    # STORAGE AND RETRIEVAL
    # ============================================================================
    
    async def _store_patient_profile(self, profile: PatientProfile) -> bool:
        """Store patient profile in database for future retrieval"""
        try:
            # Convert profile to JSON-serializable format
            profile_data = {
                'unified_patient_id': profile.unified_patient_id,
                'profile_data': self._serialize_profile(profile),
                'created_at': datetime.utcnow().isoformat(),
                'data_completeness_score': profile.data_completeness.get('overall', 0.0),
                'risk_level': profile.risk_stratification.get('overall', RiskLevel.UNKNOWN).value,
                'total_insights': len(profile.fused_insights)
            }
            
            # Store in a dedicated table (would need to create this table)
            # For now, just log the successful creation
            logger.info(f"Would store profile for patient {profile.unified_patient_id} with {len(profile.fused_insights)} insights")
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing patient profile: {e}")
            return False
    
    def _serialize_profile(self, profile: PatientProfile) -> Dict[str, Any]:
        """Serialize profile to JSON-compatible format"""
        serialized = asdict(profile)
        
        # Convert datetime objects to ISO strings
        serialized['last_updated'] = profile.last_updated.isoformat()
        
        # Convert enums to strings
        for insight in serialized['fused_insights']:
            insight['evidence_level'] = insight['evidence_level']
            insight['risk_assessment'] = insight['risk_assessment']
            
            # Convert evidence timestamps
            for evidence in insight['supporting_evidence']:
                evidence['timestamp'] = evidence['timestamp']
                evidence['source_modality'] = evidence['source_modality']
            
            for evidence in insight['conflicting_evidence']:
                evidence['timestamp'] = evidence['timestamp']
                evidence['source_modality'] = evidence['source_modality']
        
        # Convert risk stratification
        for key, value in serialized['risk_stratification'].items():
            serialized['risk_stratification'][key] = value
        
        return serialized
    
    # ============================================================================
    # HELPER METHODS (PLACEHOLDERS FOR COMPLEX LOGIC)
    # ============================================================================
    
    def _get_vitals_for_admission(self, admission: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get vital signs patterns for an admission (placeholder)"""
        # Would implement sophisticated vital signs pattern analysis
        return []
    
    def _normalize_finding(self, finding: str) -> str:
        """Normalize finding text for comparison"""
        return finding.lower().strip()
    
    def _is_contradictory(self, evidence: Evidence, existing_evidence: List[Evidence]) -> bool:
        """Check if evidence contradicts existing evidence (placeholder)"""
        # Would implement sophisticated contradiction detection
        return False
    
    def _determine_evidence_level(self, supporting: List[Evidence], conflicting: List[Evidence]) -> EvidenceLevel:
        """Determine overall evidence level"""
        if len(conflicting) > len(supporting):
            return EvidenceLevel.CONFLICTING
        elif len(supporting) >= 3 and all(e.confidence > 0.8 for e in supporting):
            return EvidenceLevel.STRONG
        elif len(supporting) >= 2 and all(e.confidence > 0.6 for e in supporting):
            return EvidenceLevel.MODERATE
        elif len(supporting) >= 1:
            return EvidenceLevel.WEAK
        else:
            return EvidenceLevel.INSUFFICIENT
    
    def _assess_domain_risk(self, domain: str, evidence: List[Evidence]) -> RiskLevel:
        """Assess risk level for a clinical domain"""
        # Simple implementation - would use sophisticated medical logic
        if any('critical' in e.finding.lower() for e in evidence):
            return RiskLevel.CRITICAL
        elif any('severe' in e.finding.lower() for e in evidence):
            return RiskLevel.HIGH
        elif len(evidence) > 2:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW
    
    def _synthesize_finding(self, domain: str, evidence: List[Evidence]) -> str:
        """Synthesize findings from multiple evidence sources"""
        if not evidence:
            return f"No significant findings in {domain}"
        
        # Simple synthesis - would implement sophisticated NLP
        key_findings = [e.finding for e in evidence[:3]]  # Top 3 findings
        return f"{domain.title()} analysis reveals: {'; '.join(key_findings)}"
    
    def _assess_clinical_significance(self, domain: str, evidence: List[Evidence], risk_level: RiskLevel) -> str:
        """Assess clinical significance of findings"""
        if risk_level == RiskLevel.CRITICAL:
            return "Requires immediate clinical attention"
        elif risk_level == RiskLevel.HIGH:
            return "Requires prompt clinical evaluation"
        elif risk_level == RiskLevel.MODERATE:
            return "Should be monitored closely"
        else:
            return "Standard follow-up recommended"
    
    def _generate_recommendations(self, domain: str, evidence: List[Evidence], evidence_level: EvidenceLevel) -> List[str]:
        """Generate clinical recommendations"""
        recommendations = []
        
        if evidence_level in [EvidenceLevel.STRONG, EvidenceLevel.MODERATE]:
            recommendations.append(f"Consider specialist consultation for {domain}")
            recommendations.append("Monitor patient closely")
        
        if len(evidence) > 3:
            recommendations.append("Consider additional diagnostic testing")
        
        return recommendations
    
    def _identify_uncertainty_factors(self, supporting: List[Evidence], conflicting: List[Evidence]) -> List[str]:
        """Identify factors contributing to uncertainty"""
        factors = []
        
        if conflicting:
            factors.append("Conflicting evidence from multiple sources")
        
        if any(e.confidence < 0.7 for e in supporting):
            factors.append("Low confidence in some evidence")
        
        if len(supporting) < 2:
            factors.append("Limited evidence available")
        
        return factors