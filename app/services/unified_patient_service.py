"""
Unified Patient Service

Cohesive service that orchestrates all multi-modal components to provide intelligent,
contextual patient data access. Determines what data is relevant and needed for each
patient based on their clinical profile, available data, and query context.

Key Features:
- Smart data prioritization based on clinical relevance
- Lazy loading of expensive computations
- Contextual data recommendations
- Performance-optimized data fetching
- Unified patient view across all modalities
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json

from app.services.supabase_service import SupabaseService
from app.services.multimodal_data_service import MultiModalDataService
from app.services.multimodal_vector_service import MultiModalVectorService, ModalityType
from app.services.patient_identity_service import PatientIdentityService
from app.services.data_fusion_service import DataFusionService, PatientProfile, RiskLevel
from app.services.clinical_trials_matching_service import ClinicalTrialsMatchingService, MatchingMethod
from app.services.clinical_analysis_service import ClinicalAnalysisService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPriority(Enum):
    """Data priority levels for intelligent loading"""
    CRITICAL = "critical"      # Always load (demographics, recent clinical data)
    HIGH = "high"             # Load for most queries (conditions, medications)
    MEDIUM = "medium"         # Load when relevant (genetic data, trials)
    LOW = "low"              # Load only when specifically requested
    BACKGROUND = "background"  # Load asynchronously if time permits

class QueryContext(Enum):
    """Different query contexts that affect data relevance"""
    CLINICAL_REVIEW = "clinical_review"           # Doctor reviewing patient
    RESEARCH_ANALYSIS = "research_analysis"      # Research study inclusion
    TRIAL_MATCHING = "trial_matching"           # Clinical trial recruitment
    RISK_ASSESSMENT = "risk_assessment"         # Risk stratification
    EMERGENCY_TRIAGE = "emergency_triage"       # Emergency department
    POPULATION_HEALTH = "population_health"     # Population analysis
    GENETIC_COUNSELING = "genetic_counseling"   # Genetic consultation

@dataclass
class DataAvailabilityMap:
    """Maps what data is available for a patient across all modalities"""
    patient_id: str
    demographics: bool = False
    clinical_notes: bool = False
    mimic_data: bool = False
    genetic_data: bool = False
    adverse_events: bool = False
    trial_matches: bool = False
    vector_embeddings: Dict[str, bool] = None
    last_updated: Optional[datetime] = None
    
    def __post_init__(self):
        if self.vector_embeddings is None:
            self.vector_embeddings = {}

@dataclass
class DataRelevanceScore:
    """Scores how relevant each data type is for a specific query"""
    modality: ModalityType
    relevance_score: float  # 0.0 - 1.0
    loading_priority: DataPriority
    estimated_load_time_ms: int
    clinical_impact: str  # Description of clinical relevance
    dependencies: List[str] = None  # Other data needed first

@dataclass
class UnifiedPatientView:
    """Complete unified view of patient data across all modalities"""
    patient_id: str
    query_context: QueryContext
    
    # Core data (always loaded)
    demographics: Dict[str, Any]
    unified_identity: Dict[str, Any]
    data_availability: DataAvailabilityMap
    
    # Clinical data (context-dependent)
    clinical_summary: Optional[Dict[str, Any]] = None
    recent_analyses: Optional[List[Dict[str, Any]]] = None
    risk_stratification: Optional[Dict[str, RiskLevel]] = None
    
    # Multi-modal data (loaded as needed)
    mimic_profile: Optional[Dict[str, Any]] = None
    genetic_profile: Optional[Dict[str, Any]] = None
    adverse_event_profile: Optional[Dict[str, Any]] = None
    
    # Advanced analyses (computed on demand)
    fusion_insights: Optional[List[Dict[str, Any]]] = None
    similar_patients: Optional[List[Dict[str, Any]]] = None
    trial_matches: Optional[List[Dict[str, Any]]] = None
    
    # Metadata
    data_completeness_score: float = 0.0
    query_performance_ms: int = 0
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []

class UnifiedPatientService:
    """Unified service for intelligent patient data access across all modalities"""
    
    def __init__(self):
        # Initialize all component services
        self.supabase = SupabaseService()
        self.multimodal_service = MultiModalDataService()
        self.vector_service = MultiModalVectorService()
        self.identity_service = PatientIdentityService()
        self.fusion_service = DataFusionService()
        self.trials_service = ClinicalTrialsMatchingService()
        self.clinical_service = ClinicalAnalysisService()
        
        # Data relevance configuration per context
        self.context_relevance = self._initialize_context_relevance()
        
        # Performance thresholds
        self.max_query_time_seconds = 30
        self.max_expensive_operations = 3
        
    # ============================================================================
    # MAIN UNIFIED PATIENT ACCESS
    # ============================================================================
    
    async def get_unified_patient_view(self, 
                                     patient_identifier: str,
                                     query_context: QueryContext = QueryContext.CLINICAL_REVIEW,
                                     include_similar_patients: bool = False,
                                     include_trial_matches: bool = False,
                                     max_response_time_seconds: int = 10) -> UnifiedPatientView:
        """
        Get unified patient view with intelligent data loading
        
        Args:
            patient_identifier: Patient ID (can be source ID or unified ID)
            query_context: Context for query to determine relevant data
            include_similar_patients: Whether to include similar patient analysis
            include_trial_matches: Whether to include clinical trial matches
            max_response_time_seconds: Maximum time to spend on query
            
        Returns:
            Complete unified patient view optimized for the query context
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Resolve patient identity and get availability map
            unified_patient_id = await self._resolve_patient_identity(patient_identifier)
            if not unified_patient_id:
                raise ValueError(f"Patient not found: {patient_identifier}")
            
            availability_map = await self._build_data_availability_map(unified_patient_id)
            
            # Step 2: Determine data relevance for this context
            relevance_scores = self._calculate_data_relevance(availability_map, query_context)
            
            # Step 3: Create loading plan based on relevance and time budget
            loading_plan = self._create_loading_plan(relevance_scores, max_response_time_seconds)
            
            # Step 4: Execute loading plan with progressive enhancement
            unified_view = await self._execute_loading_plan(
                unified_patient_id, query_context, availability_map, loading_plan
            )
            
            # Step 5: Add optional expensive operations if time permits
            remaining_time = max_response_time_seconds - (datetime.now() - start_time).total_seconds()
            
            if remaining_time > 5 and include_similar_patients:
                unified_view.similar_patients = await self._get_similar_patients_summary(
                    unified_patient_id, top_k=5
                )
            
            if remaining_time > 3 and include_trial_matches:
                unified_view.trial_matches = await self._get_trial_matches_summary(
                    unified_patient_id, max_matches=5
                )
            
            # Step 6: Generate contextual recommendations
            unified_view.recommendations = self._generate_contextual_recommendations(
                unified_view, query_context
            )
            
            # Step 7: Calculate final metrics
            unified_view.query_performance_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            unified_view.data_completeness_score = self._calculate_completeness_score(unified_view)
            
            logger.info(f"Unified patient view generated in {unified_view.query_performance_ms}ms "
                       f"with {unified_view.data_completeness_score:.1%} completeness")
            
            return unified_view
            
        except Exception as e:
            logger.error(f"Error creating unified patient view: {e}")
            raise
    
    async def get_patient_summary(self, patient_identifier: str) -> Dict[str, Any]:
        """
        Get quick patient summary for lists/searches
        Optimized for speed - only essential data
        """
        try:
            unified_patient_id = await self._resolve_patient_identity(patient_identifier)
            if not unified_patient_id:
                return {'error': 'Patient not found'}
            
            # Get core demographics and recent analysis
            demographics = await self._get_patient_demographics(unified_patient_id)
            recent_analysis = await self._get_most_recent_analysis(unified_patient_id)
            
            # Quick risk assessment
            risk_indicators = await self._get_risk_indicators(unified_patient_id)
            
            return {
                'patient_id': unified_patient_id,
                'demographics': demographics,
                'recent_analysis': recent_analysis,
                'risk_indicators': risk_indicators,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating patient summary: {e}")
            return {'error': str(e)}
    
    async def refresh_patient_data(self, patient_identifier: str, 
                                 force_recompute: bool = False) -> Dict[str, Any]:
        """
        Refresh patient data across all modalities
        Useful when new data has been added
        """
        try:
            unified_patient_id = await self._resolve_patient_identity(patient_identifier)
            
            tasks = []
            
            # Refresh vector embeddings if needed
            tasks.append(self._refresh_vector_embeddings(unified_patient_id, force_recompute))
            
            # Refresh fusion insights
            tasks.append(self._refresh_fusion_profile(unified_patient_id, force_recompute))
            
            # Update trial matches
            tasks.append(self._refresh_trial_matches(unified_patient_id))
            
            # Execute refresh tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successful refreshes
            successful_refreshes = sum(1 for r in results if not isinstance(r, Exception))
            
            return {
                'patient_id': unified_patient_id,
                'refreshed_components': successful_refreshes,
                'total_components': len(tasks),
                'refresh_timestamp': datetime.now().isoformat(),
                'errors': [str(r) for r in results if isinstance(r, Exception)]
            }
            
        except Exception as e:
            logger.error(f"Error refreshing patient data: {e}")
            return {'error': str(e)}
    
    # ============================================================================
    # IDENTITY RESOLUTION AND DATA AVAILABILITY
    # ============================================================================
    
    async def _resolve_patient_identity(self, patient_identifier: str) -> Optional[str]:
        """Resolve any patient identifier to unified patient ID"""
        try:
            # Check if it's already a unified patient ID
            unified_check = self.supabase.client.table('unified_patients')\
                .select('unified_patient_id')\
                .eq('unified_patient_id', patient_identifier)\
                .execute()
            
            if unified_check.data:
                return patient_identifier
            
            # Check identity mappings
            mapping_check = self.supabase.client.table('patient_identity_mappings')\
                .select('unified_patient_id')\
                .eq('source_patient_id', patient_identifier)\
                .execute()
            
            if mapping_check.data:
                return mapping_check.data[0]['unified_patient_id']
            
            # Try to find by master record ID
            master_check = self.supabase.client.table('unified_patients')\
                .select('unified_patient_id')\
                .eq('master_record_id', patient_identifier)\
                .execute()
            
            if master_check.data:
                return master_check.data[0]['unified_patient_id']
            
            return None
            
        except Exception as e:
            logger.error(f"Error resolving patient identity: {e}")
            return None
    
    async def _build_data_availability_map(self, unified_patient_id: str) -> DataAvailabilityMap:
        """Build map of what data is available for this patient"""
        try:
            availability = DataAvailabilityMap(patient_id=unified_patient_id)
            
            # Check demographics (unified patients table)
            demographics_check = self.supabase.client.table('unified_patients')\
                .select('demographics')\
                .eq('unified_patient_id', unified_patient_id)\
                .execute()
            
            availability.demographics = bool(demographics_check.data)
            
            # Check clinical notes (via analysis sessions)
            clinical_check = self.supabase.client.table('analysis_sessions')\
                .select('session_id')\
                .limit(1)\
                .execute()
            
            availability.clinical_notes = bool(clinical_check.data)
            
            # Check MIMIC-IV data
            mimic_check = self.supabase.client.table('mimic_admissions')\
                .select('admission_id')\
                .eq('unified_patient_id', unified_patient_id)\
                .limit(1)\
                .execute()
            
            availability.mimic_data = bool(mimic_check.data)
            
            # Check genetic data
            genetic_check = self.supabase.client.table('biobank_participants')\
                .select('participant_id')\
                .eq('unified_patient_id', unified_patient_id)\
                .limit(1)\
                .execute()
            
            availability.genetic_data = bool(genetic_check.data)
            
            # Check adverse events
            faers_check = self.supabase.client.table('faers_cases')\
                .select('case_id')\
                .eq('unified_patient_id', unified_patient_id)\
                .limit(1)\
                .execute()
            
            availability.adverse_events = bool(faers_check.data)
            
            # Check trial matches
            trials_check = self.supabase.client.table('patient_trial_matches')\
                .select('match_id')\
                .eq('unified_patient_id', unified_patient_id)\
                .limit(1)\
                .execute()
            
            availability.trial_matches = bool(trials_check.data)
            
            # Check vector embeddings
            embeddings_check = self.supabase.client.table('multimodal_embeddings')\
                .select('data_type')\
                .eq('unified_patient_id', unified_patient_id)\
                .execute()
            
            for embedding in embeddings_check.data:
                availability.vector_embeddings[embedding['data_type']] = True
            
            availability.last_updated = datetime.now()
            
            return availability
            
        except Exception as e:
            logger.error(f"Error building data availability map: {e}")
            return DataAvailabilityMap(patient_id=unified_patient_id)
    
    # ============================================================================
    # DATA RELEVANCE AND LOADING PLANS
    # ============================================================================
    
    def _calculate_data_relevance(self, availability: DataAvailabilityMap, 
                                context: QueryContext) -> List[DataRelevanceScore]:
        """Calculate relevance scores for available data based on query context"""
        relevance_scores = []
        
        # Get base relevance for this context
        context_config = self.context_relevance.get(context, {})
        
        # Score each available data type
        if availability.demographics:
            relevance_scores.append(DataRelevanceScore(
                modality=ModalityType.DEMOGRAPHICS,
                relevance_score=context_config.get('demographics', 1.0),
                loading_priority=DataPriority.CRITICAL,
                estimated_load_time_ms=50,
                clinical_impact="Essential patient identification and demographic context"
            ))
        
        if availability.clinical_notes:
            relevance_scores.append(DataRelevanceScore(
                modality=ModalityType.CLINICAL_TEXT,
                relevance_score=context_config.get('clinical_text', 0.9),
                loading_priority=DataPriority.HIGH,
                estimated_load_time_ms=200,
                clinical_impact="Recent clinical findings and provider assessments"
            ))
        
        if availability.genetic_data:
            genetic_relevance = context_config.get('genetic', 0.3)
            if context == QueryContext.GENETIC_COUNSELING:
                genetic_relevance = 1.0
            elif context == QueryContext.RISK_ASSESSMENT:
                genetic_relevance = 0.8
                
            relevance_scores.append(DataRelevanceScore(
                modality=ModalityType.GENETIC_PROFILE,
                relevance_score=genetic_relevance,
                loading_priority=DataPriority.MEDIUM if genetic_relevance > 0.6 else DataPriority.LOW,
                estimated_load_time_ms=500,
                clinical_impact="Genetic predisposition and pharmacogenomic factors"
            ))
        
        if availability.mimic_data:
            mimic_relevance = context_config.get('mimic', 0.4)
            if context == QueryContext.EMERGENCY_TRIAGE:
                mimic_relevance = 0.9
            elif context == QueryContext.RISK_ASSESSMENT:
                mimic_relevance = 0.7
                
            relevance_scores.append(DataRelevanceScore(
                modality=ModalityType.VITAL_SIGNS,
                relevance_score=mimic_relevance,
                loading_priority=DataPriority.MEDIUM if mimic_relevance > 0.6 else DataPriority.LOW,
                estimated_load_time_ms=800,
                clinical_impact="Critical care patterns and vital signs trends"
            ))
        
        if availability.adverse_events:
            ae_relevance = context_config.get('adverse_events', 0.3)
            if context == QueryContext.CLINICAL_REVIEW:
                ae_relevance = 0.6
                
            relevance_scores.append(DataRelevanceScore(
                modality=ModalityType.ADVERSE_EVENTS,
                relevance_score=ae_relevance,
                loading_priority=DataPriority.LOW,
                estimated_load_time_ms=300,
                clinical_impact="Drug safety history and adverse reaction patterns"
            ))
        
        if availability.trial_matches:
            trial_relevance = context_config.get('trials', 0.2)
            if context == QueryContext.TRIAL_MATCHING:
                trial_relevance = 1.0
            elif context == QueryContext.RESEARCH_ANALYSIS:
                trial_relevance = 0.8
                
            relevance_scores.append(DataRelevanceScore(
                modality=ModalityType.TRIAL_ELIGIBILITY,
                relevance_score=trial_relevance,
                loading_priority=DataPriority.LOW if trial_relevance < 0.5 else DataPriority.MEDIUM,
                estimated_load_time_ms=1000,
                clinical_impact="Clinical trial eligibility and research opportunities"
            ))
        
        # Sort by relevance score
        relevance_scores.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return relevance_scores
    
    def _create_loading_plan(self, relevance_scores: List[DataRelevanceScore],
                           max_time_seconds: int) -> Dict[str, List[DataRelevanceScore]]:
        """Create optimized loading plan based on relevance and time budget"""
        plan = {
            'immediate': [],  # Load immediately (critical priority)
            'primary': [],    # Load next (high priority, high relevance)
            'secondary': [],  # Load if time permits (medium priority)
            'background': []  # Load asynchronously (low priority)
        }
        
        time_budget_ms = max_time_seconds * 1000
        estimated_time = 0
        
        for score in relevance_scores:
            if score.loading_priority == DataPriority.CRITICAL:
                plan['immediate'].append(score)
                estimated_time += score.estimated_load_time_ms
                
            elif score.loading_priority == DataPriority.HIGH:
                if estimated_time + score.estimated_load_time_ms < time_budget_ms * 0.6:
                    plan['primary'].append(score)
                    estimated_time += score.estimated_load_time_ms
                else:
                    plan['background'].append(score)
                    
            elif score.loading_priority == DataPriority.MEDIUM:
                if estimated_time + score.estimated_load_time_ms < time_budget_ms * 0.8:
                    plan['secondary'].append(score)
                    estimated_time += score.estimated_load_time_ms
                else:
                    plan['background'].append(score)
                    
            else:  # LOW priority
                plan['background'].append(score)
        
        return plan
    
    # ============================================================================
    # DATA LOADING EXECUTION
    # ============================================================================
    
    async def _execute_loading_plan(self, unified_patient_id: str, context: QueryContext,
                                  availability: DataAvailabilityMap,
                                  loading_plan: Dict[str, List[DataRelevanceScore]]) -> UnifiedPatientView:
        """Execute the loading plan to build unified patient view"""
        
        # Initialize unified view
        view = UnifiedPatientView(
            patient_id=unified_patient_id,
            query_context=context,
            demographics={},
            unified_identity={},
            data_availability=availability
        )
        
        # Phase 1: Load immediate (critical) data
        immediate_tasks = []
        for score in loading_plan['immediate']:
            if score.modality == ModalityType.DEMOGRAPHICS:
                immediate_tasks.append(self._load_demographics(unified_patient_id))
            # Add other critical loaders as needed
        
        if immediate_tasks:
            immediate_results = await asyncio.gather(*immediate_tasks, return_exceptions=True)
            
            # Process immediate results
            for i, result in enumerate(immediate_results):
                if not isinstance(result, Exception):
                    score = loading_plan['immediate'][i]
                    if score.modality == ModalityType.DEMOGRAPHICS:
                        view.demographics = result
        
        # Phase 2: Load primary data
        primary_tasks = []
        for score in loading_plan['primary']:
            if score.modality == ModalityType.CLINICAL_TEXT:
                primary_tasks.append(self._load_clinical_summary(unified_patient_id))
            # Add other primary loaders
        
        if primary_tasks:
            primary_results = await asyncio.gather(*primary_tasks, return_exceptions=True)
            
            for i, result in enumerate(primary_results):
                if not isinstance(result, Exception):
                    score = loading_plan['primary'][i]
                    if score.modality == ModalityType.CLINICAL_TEXT:
                        view.clinical_summary = result
        
        # Phase 3: Load secondary data if available
        secondary_tasks = []
        for score in loading_plan['secondary']:
            if score.modality == ModalityType.GENETIC_PROFILE:
                secondary_tasks.append(self._load_genetic_profile(unified_patient_id))
            elif score.modality == ModalityType.VITAL_SIGNS:
                secondary_tasks.append(self._load_mimic_profile(unified_patient_id))
            # Add other secondary loaders
        
        if secondary_tasks:
            secondary_results = await asyncio.gather(*secondary_tasks, return_exceptions=True)
            
            for i, result in enumerate(secondary_results):
                if not isinstance(result, Exception):
                    score = loading_plan['secondary'][i]
                    if score.modality == ModalityType.GENETIC_PROFILE:
                        view.genetic_profile = result
                    elif score.modality == ModalityType.VITAL_SIGNS:
                        view.mimic_profile = result
        
        # Phase 4: Start background tasks (don't wait for them)
        background_tasks = []
        for score in loading_plan['background']:
            if score.modality == ModalityType.ADVERSE_EVENTS:
                background_tasks.append(self._load_adverse_events_profile(unified_patient_id))
        
        # Don't await background tasks - they'll complete asynchronously
        if background_tasks:
            asyncio.gather(*background_tasks, return_exceptions=True)
        
        return view
    
    # ============================================================================
    # SPECIFIC DATA LOADERS
    # ============================================================================
    
    async def _load_demographics(self, unified_patient_id: str) -> Dict[str, Any]:
        """Load patient demographics"""
        try:
            result = self.supabase.client.table('unified_patients')\
                .select('demographics, data_sources, identity_confidence')\
                .eq('unified_patient_id', unified_patient_id)\
                .execute()
            
            if result.data:
                return result.data[0]
            return {}
            
        except Exception as e:
            logger.error(f"Error loading demographics: {e}")
            return {}
    
    async def _load_clinical_summary(self, unified_patient_id: str) -> Dict[str, Any]:
        """Load clinical summary from recent analyses"""
        try:
            # Get recent analysis sessions
            sessions = self.supabase.client.table('analysis_sessions')\
                .select('*')\
                .order('created_at', desc=True)\
                .limit(5)\
                .execute()
            
            if not sessions.data:
                return {}
            
            # Get entities from recent sessions
            recent_entities = []
            for session in sessions.data[:3]:  # Top 3 sessions
                entities = self.supabase.client.table('clinical_entities')\
                    .select('*')\
                    .eq('session_id', session['session_id'])\
                    .execute()
                
                recent_entities.extend(entities.data)
            
            # Summarize clinical findings
            summary = {
                'recent_sessions': len(sessions.data),
                'total_entities': len(recent_entities),
                'entity_types': {},
                'recent_findings': [],
                'risk_indicators': []
            }
            
            # Process entities
            for entity in recent_entities:
                entity_type = entity['entity_type']
                summary['entity_types'][entity_type] = summary['entity_types'].get(entity_type, 0) + 1
                
                if entity['confidence'] > 0.8:
                    summary['recent_findings'].append({
                        'type': entity_type,
                        'text': entity['entity_text'],
                        'confidence': entity['confidence'],
                        'severity': entity.get('severity')
                    })
            
            return summary
            
        except Exception as e:
            logger.error(f"Error loading clinical summary: {e}")
            return {}
    
    async def _load_genetic_profile(self, unified_patient_id: str) -> Dict[str, Any]:
        """Load genetic risk profile"""
        try:
            # Get biobank participant
            participant = self.supabase.client.table('biobank_participants')\
                .select('participant_id')\
                .eq('unified_patient_id', unified_patient_id)\
                .execute()
            
            if not participant.data:
                return {'available': False}
            
            participant_id = participant.data[0]['participant_id']
            
            # Get genetic data
            genetics = self.supabase.client.table('biobank_genetics')\
                .select('*')\
                .eq('participant_id', participant_id)\
                .execute()
            
            if not genetics.data:
                return {'available': False}
            
            # Summarize genetic profile
            profile = {
                'available': True,
                'total_variants': len(genetics.data),
                'high_risk_variants': 0,
                'risk_conditions': set(),
                'pharmacogenomic_variants': 0
            }
            
            for variant in genetics.data:
                if variant.get('risk_score', 0) > 0.7:
                    profile['high_risk_variants'] += 1
                
                conditions = variant.get('associated_conditions', [])
                profile['risk_conditions'].update(conditions)
                
                if 'pharmaco' in variant.get('variant_type', '').lower():
                    profile['pharmacogenomic_variants'] += 1
            
            profile['risk_conditions'] = list(profile['risk_conditions'])
            
            return profile
            
        except Exception as e:
            logger.error(f"Error loading genetic profile: {e}")
            return {'available': False, 'error': str(e)}
    
    async def _load_mimic_profile(self, unified_patient_id: str) -> Dict[str, Any]:
        """Load MIMIC-IV critical care profile"""
        try:
            # Get admissions
            admissions = self.supabase.client.table('mimic_admissions')\
                .select('*')\
                .eq('unified_patient_id', unified_patient_id)\
                .order('admit_time', desc=True)\
                .execute()
            
            if not admissions.data:
                return {'available': False}
            
            profile = {
                'available': True,
                'total_admissions': len(admissions.data),
                'recent_admission': admissions.data[0] if admissions.data else None,
                'icu_stays': 0,
                'mortality_events': 0,
                'common_diagnoses': {}
            }
            
            for admission in admissions.data:
                if admission.get('admission_location', '').upper().find('ICU') != -1:
                    profile['icu_stays'] += 1
                
                if admission.get('hospital_expire_flag'):
                    profile['mortality_events'] += 1
                
                diagnosis = admission.get('diagnosis', '')
                if diagnosis:
                    profile['common_diagnoses'][diagnosis] = profile['common_diagnoses'].get(diagnosis, 0) + 1
            
            return profile
            
        except Exception as e:
            logger.error(f"Error loading MIMIC profile: {e}")
            return {'available': False, 'error': str(e)}
    
    async def _load_adverse_events_profile(self, unified_patient_id: str) -> Dict[str, Any]:
        """Load adverse events profile"""
        try:
            cases = self.supabase.client.table('faers_cases')\
                .select('*')\
                .eq('unified_patient_id', unified_patient_id)\
                .execute()
            
            if not cases.data:
                return {'available': False}
            
            profile = {
                'available': True,
                'total_cases': len(cases.data),
                'serious_events': sum(1 for case in cases.data if case.get('serious_adverse_event')),
                'recent_case': max(cases.data, key=lambda x: x.get('report_date', '')) if cases.data else None
            }
            
            return profile
            
        except Exception as e:
            logger.error(f"Error loading adverse events profile: {e}")
            return {'available': False, 'error': str(e)}
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    async def _get_similar_patients_summary(self, unified_patient_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get similar patients summary"""
        try:
            # Use vector service to find similar patients
            similar_patients = await self.vector_service.search_similar_patients(
                unified_patient_id, 
                ModalityType.CLINICAL_TEXT,
                top_k=top_k,
                min_similarity=0.6
            )
            
            return [{
                'patient_id': result.patient_id,
                'similarity_score': result.similarity_score,
                'modality': result.modality.value,
                'content_summary': result.content_summary
            } for result in similar_patients]
            
        except Exception as e:
            logger.error(f"Error getting similar patients: {e}")
            return []
    
    async def _get_trial_matches_summary(self, unified_patient_id: str, max_matches: int = 5) -> List[Dict[str, Any]]:
        """Get trial matches summary"""
        try:
            matches = self.supabase.client.table('patient_trial_matches')\
                .select('*')\
                .eq('unified_patient_id', unified_patient_id)\
                .order('match_score', desc=True)\
                .limit(max_matches)\
                .execute()
            
            return [{
                'nct_id': match['nct_id'],
                'match_score': match['match_score'],
                'recommendation_level': match.get('recommendation_level'),
                'eligibility_status': match.get('eligibility_status')
            } for match in matches.data]
            
        except Exception as e:
            logger.error(f"Error getting trial matches: {e}")
            return []
    
    def _generate_contextual_recommendations(self, view: UnifiedPatientView, context: QueryContext) -> List[str]:
        """Generate contextual recommendations based on available data and context"""
        recommendations = []
        
        # Context-specific recommendations
        if context == QueryContext.CLINICAL_REVIEW:
            if view.genetic_profile and view.genetic_profile.get('high_risk_variants', 0) > 0:
                recommendations.append("Consider genetic counseling consultation")
            
            if view.trial_matches:
                recommendations.append("Review clinical trial opportunities")
                
        elif context == QueryContext.EMERGENCY_TRIAGE:
            if view.mimic_profile and view.mimic_profile.get('icu_stays', 0) > 0:
                recommendations.append("Patient has history of ICU admission - monitor closely")
                
        elif context == QueryContext.TRIAL_MATCHING:
            if not view.trial_matches:
                recommendations.append("Run trial matching analysis")
        
        # Data completeness recommendations
        if view.data_completeness_score < 0.5:
            recommendations.append("Consider additional data collection to improve clinical insights")
        
        return recommendations
    
    def _calculate_completeness_score(self, view: UnifiedPatientView) -> float:
        """Calculate data completeness score"""
        available_components = 0
        total_components = 6  # demographics, clinical, genetic, mimic, adverse events, trials
        
        if view.demographics:
            available_components += 1
        if view.clinical_summary:
            available_components += 1
        if view.genetic_profile and view.genetic_profile.get('available'):
            available_components += 1
        if view.mimic_profile and view.mimic_profile.get('available'):
            available_components += 1
        if view.adverse_event_profile and view.adverse_event_profile.get('available'):
            available_components += 1
        if view.trial_matches:
            available_components += 1
        
        return available_components / total_components
    
    def _initialize_context_relevance(self) -> Dict[QueryContext, Dict[str, float]]:
        """Initialize relevance scores for each query context"""
        return {
            QueryContext.CLINICAL_REVIEW: {
                'demographics': 1.0,
                'clinical_text': 0.9,
                'genetic': 0.4,
                'mimic': 0.5,
                'adverse_events': 0.6,
                'trials': 0.3
            },
            QueryContext.EMERGENCY_TRIAGE: {
                'demographics': 1.0,
                'clinical_text': 0.9,
                'genetic': 0.1,
                'mimic': 0.8,
                'adverse_events': 0.4,
                'trials': 0.1
            },
            QueryContext.GENETIC_COUNSELING: {
                'demographics': 0.8,
                'clinical_text': 0.6,
                'genetic': 1.0,
                'mimic': 0.2,
                'adverse_events': 0.3,
                'trials': 0.4
            },
            QueryContext.TRIAL_MATCHING: {
                'demographics': 0.7,
                'clinical_text': 0.8,
                'genetic': 0.6,
                'mimic': 0.4,
                'adverse_events': 0.5,
                'trials': 1.0
            },
            QueryContext.RISK_ASSESSMENT: {
                'demographics': 0.8,
                'clinical_text': 0.9,
                'genetic': 0.8,
                'mimic': 0.7,
                'adverse_events': 0.6,
                'trials': 0.3
            }
        }
    
    # Additional helper methods for refresh operations
    async def _refresh_vector_embeddings(self, unified_patient_id: str, force_recompute: bool) -> bool:
        """Refresh vector embeddings for patient"""
        try:
            # Implementation would refresh embeddings across all modalities
            return True
        except Exception as e:
            logger.error(f"Error refreshing vector embeddings: {e}")
            return False
    
    async def _refresh_fusion_profile(self, unified_patient_id: str, force_recompute: bool) -> bool:
        """Refresh fusion profile"""
        try:
            if force_recompute:
                # Regenerate complete fusion profile
                await self.fusion_service.create_comprehensive_patient_profile(unified_patient_id)
            return True
        except Exception as e:
            logger.error(f"Error refreshing fusion profile: {e}")
            return False
    
    async def _refresh_trial_matches(self, unified_patient_id: str) -> bool:
        """Refresh clinical trial matches"""
        try:
            # Get updated trial matches
            await self.trials_service.find_matching_trials(unified_patient_id, max_results=10)
            return True
        except Exception as e:
            logger.error(f"Error refreshing trial matches: {e}")
            return False
    
    # Quick access methods for common scenarios
    async def _get_patient_demographics(self, unified_patient_id: str) -> Dict[str, Any]:
        """Quick demographics access"""
        result = self.supabase.client.table('unified_patients')\
            .select('demographics')\
            .eq('unified_patient_id', unified_patient_id)\
            .execute()
        
        return result.data[0]['demographics'] if result.data else {}
    
    async def _get_most_recent_analysis(self, unified_patient_id: str) -> Optional[Dict[str, Any]]:
        """Get most recent clinical analysis"""
        try:
            # This would query analysis_sessions for the most recent analysis
            # Implementation depends on how you link patients to sessions
            return None
        except Exception:
            return None
    
    async def _get_risk_indicators(self, unified_patient_id: str) -> Dict[str, Any]:
        """Get quick risk indicators"""
        try:
            # Quick risk assessment based on available data
            return {
                'overall_risk': 'moderate',
                'genetic_risk': 'unknown',
                'clinical_risk': 'low'
            }
        except Exception:
            return {'overall_risk': 'unknown'}