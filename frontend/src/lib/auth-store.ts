import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import Cookies from 'js-cookie';
import { User } from './types';

interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isInitialized: boolean;
  setUser: (user: User) => void;
  setToken: (token: string) => void;
  logout: () => void;
  initialize: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      isAuthenticated: false,
      isInitialized: false,
      
      setUser: (user: User) => {
        set({ user, isAuthenticated: true, isInitialized: true });
        // Store user data in cookie for SSR
        Cookies.set('user', JSON.stringify(user), { expires: 7 }); // 7 days
      },
      
      setToken: (token: string) => {
        Cookies.set('access_token', token, { expires: 7 }); // 7 days
      },
      
      logout: () => {
        set({ user: null, isAuthenticated: false, isInitialized: true });
        Cookies.remove('access_token');
        Cookies.remove('user');
      },
      
      initialize: () => {
        try {
          const userCookie = Cookies.get('user');
          const token = Cookies.get('access_token');
          
          if (userCookie && token) {
            const user = JSON.parse(userCookie);
            set({ user, isAuthenticated: true, isInitialized: true });
          } else {
            set({ isInitialized: true });
          }
        } catch (error) {
          console.error('Failed to initialize auth state:', error);
          set({ user: null, isAuthenticated: false, isInitialized: true });
        }
      },
    }),
    {
      name: 'auth-storage',
      // Only persist user data, tokens are handled by cookies
      partialize: (state) => ({ user: state.user, isAuthenticated: state.isAuthenticated, isInitialized: state.isInitialized }),
    }
  )
);