import axios, { AxiosInstance, AxiosResponse } from 'axios';
import Cookies from 'js-cookie';
import { ApiResponse, AuthResponse, LoginRequest, RegisterRequest, AnalysisRequest, AnalysisResult } from './types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:5000';
console.log('API_BASE_URL:', API_BASE_URL); // Debug log

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor to add auth token
    this.client.interceptors.request.use((config) => {
      const token = Cookies.get('access_token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response: AxiosResponse) => response,
      (error) => {
        if (error.response?.status === 401) {
          // Clear invalid token
          Cookies.remove('access_token');
          Cookies.remove('user');
          // Redirect to login if needed
          if (typeof window !== 'undefined') {
            window.location.href = '/login';
          }
        }
        return Promise.reject(error);
      }
    );
  }

  // Auth methods
  async login(credentials: LoginRequest): Promise<AuthResponse> {
    const response = await this.client.post<AuthResponse>('/api/auth/login', credentials);
    return response.data;
  }

  async register(userData: RegisterRequest): Promise<AuthResponse> {
    const response = await this.client.post<AuthResponse>('/api/auth/register', userData);
    return response.data;
  }

  async verifyToken(): Promise<{ user: any }> {
    const response = await this.client.get('/api/auth/protected');
    return response.data;
  }

  // Analysis methods
  async analyzeText(data: AnalysisRequest): Promise<AnalysisResult> {
    // Transform frontend data format to backend expected format
    const requestData = {
      note_text: data.text,
      patient_context: data.context ? { medical_history: data.context } : {},
      include_icd_mapping: true,
      icd_top_k: 5,
      enable_nlp_preprocessing: true
    };
    
    const response = await this.client.post('/api/analysis/extract-enhanced', requestData);
    
    // Transform backend response to frontend expected structure
    const backendData = response.data;
    
    // Combine all entity types into a single entities array
    const entities: any[] = [];
    const entityTypes = ['symptoms', 'conditions', 'medications', 'vital_signs', 'procedures', 'abnormal_findings'];
    
    entityTypes.forEach(entityType => {
      if (backendData[entityType] && Array.isArray(backendData[entityType])) {
        backendData[entityType].forEach((entity: any) => {
          entities.push({
            text: entity.text || entity.entity_text || entity.name || '',
            label: entityType.slice(0, -1), // Remove 's' from plural
            start: entity.start || entity.start_position || 0,
            end: entity.end || entity.end_position || 0,
            confidence: entity.confidence || entity.score || 0
          });
        });
      }
    });
    
    // Transform ICD mappings to frontend expected structure
    const icdMappings: any[] = [];
    
    // Debug: Log ICD mappings structure
    console.log('ICD mappings from backend:', backendData.icd_mappings);
    
    if (backendData.icd_mappings && Array.isArray(backendData.icd_mappings)) {
      backendData.icd_mappings.forEach((mapping: any) => {
        // Transform each mapping to frontend structure
        const transformedMapping: any = {
          entity: mapping.entity || '',
          entity_type: mapping.entity_type || '',
          best_match: null,
          icd_matches: []
        };
        
        // Transform best_match
        if (mapping.best_match) {
          transformedMapping.best_match = {
            code: mapping.best_match.code || '',
            description: mapping.best_match.description || '',
            similarity: mapping.best_match.similarity || 0
          };
        }
        
        // Transform icd_matches array
        if (mapping.icd_matches && Array.isArray(mapping.icd_matches)) {
          transformedMapping.icd_matches = mapping.icd_matches.map((match: any) => ({
            code: match.code || '',
            description: match.description || '',
            similarity: match.similarity || 0
          }));
        }
        
        icdMappings.push(transformedMapping);
      });
    }
    
    return {
      id: backendData.session_id || Date.now().toString(),
      entities,
      icd_mappings: icdMappings,
      icd_search_method: backendData.icd_search_method || 'faiss',
      performance_metrics: backendData.performance_metrics || { total_time_ms: 0 },
      created_at: backendData.analysis_timestamp || new Date().toISOString()
    };
  }

  async getAnalysisHistory(): Promise<AnalysisResult[]> {
    const response = await this.client.get<AnalysisResult[]>('/api/analysis/history');
    return response.data;
  }

  async getPerformanceStats(): Promise<any> {
    const response = await this.client.get('/api/analysis/performance-stats');
    return response.data;
  }
}

export const apiClient = new ApiClient();