// User and authentication types
export interface User {
  id: string;
  email: string;
  role: string;
  created_at: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  password: string;
}

export interface AuthResponse {
  access_token: string;
  user: User;
}

// Analysis types
export interface AnalysisRequest {
  text: string;
  context?: string;
}

export interface ClinicalEntity {
  text: string;
  label: string;
  start: number;
  end: number;
  confidence: number;
}

export interface ICDMapping {
  code: string;
  description: string;
  similarity_score: number;
  entity_text: string;
}

export interface AnalysisResult {
  id: string;
  entities: ClinicalEntity[];
  icd_mappings: ICDMapping[];
  analysis_time: number;
  search_method: 'faiss' | 'numpy';
  created_at: string;
}

// API response wrapper
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  code?: string;
}