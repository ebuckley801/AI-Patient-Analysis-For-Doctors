'use client';

import { useEffect } from 'react';
import { useAuthStore } from '@/lib/auth-store';

interface AuthProviderProps {
  children: React.ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const initialize = useAuthStore((state) => state.initialize);
  
  useEffect(() => {
    // Initialize auth state on app load
    initialize();
  }, [initialize]);

  return <>{children}</>;
}