'use client';

import { ProtectedRoute } from '@/components/auth/protected-route';
import { Header } from './header';
import { Sidebar } from './sidebar';

interface DashboardLayoutProps {
  children: React.ReactNode;
}

export function DashboardLayout({ children }: DashboardLayoutProps) {
  return (
    <ProtectedRoute>
      <div className="min-h-screen bg-background">
        <Header />
        <div className="flex">
          {/* Desktop sidebar */}
          <div className="hidden md:flex md:w-64 md:flex-col">
            <Sidebar />
          </div>
          
          {/* Main content */}
          <main className="flex-1 p-6 md:p-8">
            {children}
          </main>
        </div>
      </div>
    </ProtectedRoute>
  );
}