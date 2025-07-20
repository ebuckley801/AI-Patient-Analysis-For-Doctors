import { AnalysisForm } from '@/components/analysis/analysis-form';

export default function NewAnalysisPage() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">New Analysis</h1>
        <p className="text-muted-foreground">
          Analyze patient notes with AI-powered entity extraction and ICD-10 mapping
        </p>
      </div>
      
      <AnalysisForm />
    </div>
  );
}