'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuthStore } from '@/lib/auth-store';
import { Button } from '@/components/ui/button';
import { Sparkles, ArrowRight, Shield, Zap } from 'lucide-react';

export default function Home() {
  const { isAuthenticated } = useAuthStore();
  const router = useRouter();

  useEffect(() => {
    if (isAuthenticated) {
      router.push('/dashboard');
    }
  }, [isAuthenticated, router]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className="container flex h-16 items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="h-8 w-8 rounded-lg bg-primary flex items-center justify-center">
              <span className="text-primary-foreground font-bold text-sm">CA</span>
            </div>
            <span className="font-bold text-xl">Clinical Analysis</span>
          </div>
          <div className="flex items-center space-x-4">
            <Button variant="ghost" onClick={() => router.push('/login')}>
              Sign In
            </Button>
            <Button onClick={() => router.push('/register')}>
              Get Started
            </Button>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <main className="container py-16 lg:py-24">
        <div className="mx-auto max-w-4xl text-center">
          <div className="mb-8">
            <div className="inline-flex items-center rounded-lg bg-primary/10 px-3 py-1 text-sm font-medium text-primary mb-6">
              <Sparkles className="mr-2 h-4 w-4" />
              AI-Powered Clinical Analysis
            </div>
            <h1 className="text-4xl font-bold tracking-tight text-gray-900 sm:text-6xl mb-6">
              Transform Patient Notes into{' '}
              <span className="text-primary">Clinical Insights</span>
            </h1>
            <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
              Advanced AI-powered analysis of patient notes with automatic entity extraction, 
              ICD-10 mapping, and clinical intelligence for healthcare professionals.
            </p>
          </div>

          <div className="flex items-center justify-center space-x-4 mb-16">
            <Button size="lg" onClick={() => router.push('/register')}>
              Start Analyzing
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
            <Button variant="outline" size="lg" onClick={() => router.push('/login')}>
              Sign In
            </Button>
          </div>

          {/* Features */}
          <div className="grid gap-8 md:grid-cols-3 mb-16">
            <div className="text-center">
              <div className="mx-auto mb-4 h-12 w-12 rounded-lg bg-green-100 flex items-center justify-center">
                <Sparkles className="h-6 w-6 text-green-600" />
              </div>
              <h3 className="text-lg font-semibold mb-2">Entity Extraction</h3>
              <p className="text-gray-600">
                Automatically identify symptoms, conditions, medications, and procedures from clinical notes
              </p>
            </div>
            
            <div className="text-center">
              <div className="mx-auto mb-4 h-12 w-12 rounded-lg bg-blue-100 flex items-center justify-center">
                <Shield className="h-6 w-6 text-blue-600" />
              </div>
              <h3 className="text-lg font-semibold mb-2">ICD-10 Mapping</h3>
              <p className="text-gray-600">
                Precise mapping to ICD-10 codes with confidence scores and similarity analysis
              </p>
            </div>
            
            <div className="text-center">
              <div className="mx-auto mb-4 h-12 w-12 rounded-lg bg-purple-100 flex items-center justify-center">
                <Zap className="h-6 w-6 text-purple-600" />
              </div>
              <h3 className="text-lg font-semibold mb-2">Fast Processing</h3>
              <p className="text-gray-600">
                Lightning-fast analysis with optimized vector search and intelligent caching
              </p>
            </div>
          </div>

          {/* Stats */}
          <div className="rounded-2xl bg-white border p-8">
            <div className="grid gap-8 md:grid-cols-3">
              <div className="text-center">
                <div className="text-3xl font-bold text-gray-900 mb-2">99.5%</div>
                <div className="text-gray-600">Accuracy Rate</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-gray-900 mb-2">&lt;3s</div>
                <div className="text-gray-600">Average Processing</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-gray-900 mb-2">72K+</div>
                <div className="text-gray-600">ICD-10 Codes</div>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t bg-white">
        <div className="container py-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className="h-6 w-6 rounded bg-primary flex items-center justify-center">
                <span className="text-primary-foreground font-bold text-xs">CA</span>
              </div>
              <span className="font-medium">Clinical Analysis</span>
            </div>
            <p className="text-sm text-gray-500">
              © 2024 Clinical Analysis. AI-powered medical analysis platform.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
