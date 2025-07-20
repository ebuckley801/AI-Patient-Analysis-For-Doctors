'use client';

import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Alert } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { Loader2, FileText, Sparkles, Clock } from 'lucide-react';
import { apiClient } from '@/lib/api';
import { analysisSchema, AnalysisFormData } from '@/lib/validations';
import { AnalysisResult } from '@/lib/types';
import { AnalysisResults } from './analysis-results';

export function AnalysisForm() {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState('');
  const [progress, setProgress] = useState(0);
  const [estimatedTime, setEstimatedTime] = useState(0);
  const [results, setResults] = useState<AnalysisResult | null>(null);

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset,
  } = useForm<AnalysisFormData>({
    resolver: zodResolver(analysisSchema),
  });

  const onSubmit = async (data: AnalysisFormData) => {
    try {
      setIsAnalyzing(true);
      setError('');
      setResults(null);
      setProgress(0);
      setEstimatedTime(3.5); // Estimated time in seconds

      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) return prev;
          return prev + 10;
        });
        setEstimatedTime((prev) => Math.max(0, prev - 0.4));
      }, 400);

      const result = await apiClient.analyzeText(data);
      
      clearInterval(progressInterval);
      setProgress(100);
      setEstimatedTime(0);
      setResults(result);
      
      // Reset form after successful analysis
      reset();
      
    } catch (err: any) {
      setError(
        err.response?.data?.message || 
        err.response?.data?.error || 
        'Analysis failed. Please try again.'
      );
    } finally {
      setIsAnalyzing(false);
      setProgress(0);
      setEstimatedTime(0);
    }
  };

  const handleNewAnalysis = () => {
    setResults(null);
    setError('');
  };

  if (results) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-bold">Analysis Results</h2>
          <Button onClick={handleNewAnalysis} variant="outline">
            <FileText className="mr-2 h-4 w-4" />
            New Analysis
          </Button>
        </div>
        <AnalysisResults results={results} />
      </div>
    );
  }

  return (
    <div className="max-w-4xl">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Sparkles className="h-5 w-5 text-primary" />
            <span>Clinical Analysis</span>
          </CardTitle>
        </CardHeader>
        
        <form onSubmit={handleSubmit(onSubmit)}>
          <CardContent className="space-y-6">
            {error && (
              <Alert variant="destructive">
                {error}
              </Alert>
            )}

            {isAnalyzing && (
              <Alert>
                <Loader2 className="h-4 w-4 animate-spin" />
                <div className="ml-2 space-y-2 w-full">
                  <div className="flex items-center justify-between">
                    <span>Analyzing patient note...</span>
                    <span className="text-sm text-muted-foreground">
                      {estimatedTime > 0 && (
                        <span className="flex items-center">
                          <Clock className="h-3 w-3 mr-1" />
                          ~{Math.ceil(estimatedTime)}s remaining
                        </span>
                      )}
                    </span>
                  </div>
                  <Progress value={progress} className="w-full" />
                </div>
              </Alert>
            )}
            
            <div className="space-y-2">
              <Label htmlFor="text">Patient Note *</Label>
              <Textarea
                id="text"
                placeholder="Enter the patient note to be analyzed. Include symptoms, conditions, medications, and any relevant clinical information..."
                className="min-h-[200px] resize-none"
                disabled={isAnalyzing}
                {...register('text')}
              />
              {errors.text && (
                <p className="text-sm text-destructive">{errors.text.message}</p>
              )}
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="context">Additional Context (Optional)</Label>
              <Input
                id="context"
                placeholder="e.g., Emergency department visit, follow-up appointment, etc."
                disabled={isAnalyzing}
                {...register('context')}
              />
              <p className="text-xs text-muted-foreground">
                Provide additional context to improve analysis accuracy
              </p>
            </div>

            <div className="flex justify-end space-x-4">
              <Button 
                type="button" 
                variant="outline" 
                onClick={() => reset()}
                disabled={isAnalyzing}
              >
                Clear
              </Button>
              <Button type="submit" disabled={isAnalyzing}>
                {isAnalyzing ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Sparkles className="mr-2 h-4 w-4" />
                    Analyze Note
                  </>
                )}
              </Button>
            </div>

            <div className="border-t pt-4 space-y-2">
              <h4 className="font-medium text-sm">Analysis Features</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs text-muted-foreground">
                <div>✨ Entity Extraction</div>
                <div>🏥 ICD-10 Mapping</div>
                <div>🧠 NLP Enhancement</div>
                <div>⚡ Fast Processing</div>
              </div>
            </div>
          </CardContent>
        </form>
      </Card>
    </div>
  );
}