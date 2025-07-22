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

export interface ICDMatch {
  code: string;
  description: string;
  similarity: number;
}

export interface ICDMapping {
  entity: string;
  entity_type: string;
  best_match: ICDMatch | null;
  icd_matches: ICDMatch[];
}

export interface AnalysisResult {
  id: string;
  entities: ClinicalEntity[];
  icd_mappings: ICDMapping[];
  icd_search_method: 'faiss' | 'numpy';
  created_at: string;
  performance_metrics?: {
    total_time_ms: number;
    [key: string]: any;
  };
  [key: string]: any;
}

// API response wrapper
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  code?: string;
}
