'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { 
  FileText, 
  Hash, 
  Clock, 
  Zap, 
  Download,
  Eye,
  Activity,
  Stethoscope,
  Pill,
  AlertTriangle
} from 'lucide-react';
import { AnalysisResult } from '@/lib/types';

interface AnalysisResultsProps {
  results: AnalysisResult;
}

export function AnalysisResults({ results }: AnalysisResultsProps) {
  const getEntityIcon = (label: string) => {
    switch (label.toLowerCase()) {
      case 'symptom':
        return <Activity className="h-3 w-3" />;
      case 'condition':
        return <Stethoscope className="h-3 w-3" />;
      case 'medication':
        return <Pill className="h-3 w-3" />;
      default:
        return <Hash className="h-3 w-3" />;
    }
  };

  const getEntityColor = (label: string) => {
    switch (label.toLowerCase()) {
      case 'symptom':
        return 'bg-orange-100 text-orange-800 border-orange-200';
      case 'condition':
        return 'bg-red-100 text-red-800 border-red-200';
      case 'medication':
        return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'procedure':
        return 'bg-purple-100 text-purple-800 border-purple-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="space-y-6">
      {/* Analysis Summary */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Entities Found</CardTitle>
            <Hash className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{results.entities?.length || 0}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">ICD Mappings</CardTitle>
            <FileText className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{results.icd_mappings?.length || 0}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Processing Time</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{results.analysis_time || 0}s</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Search Method</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-sm font-bold capitalize">
              <Badge variant={results.search_method === 'faiss' ? 'default' : 'secondary'}>
                {results.search_method || 'N/A'}
              </Badge>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Tabbed Results */}
      <Tabs defaultValue="entities" className="w-full">
        <div className="flex items-center justify-between mb-4">
          <TabsList>
            <TabsTrigger value="entities">Clinical Entities</TabsTrigger>
            <TabsTrigger value="icd">ICD-10 Mappings</TabsTrigger>
            <TabsTrigger value="insights">Insights</TabsTrigger>
          </TabsList>
          
          <div className="flex space-x-2">
            <Button variant="outline" size="sm">
              <Eye className="mr-2 h-4 w-4" />
              View Details
            </Button>
            <Button variant="outline" size="sm">
              <Download className="mr-2 h-4 w-4" />
              Export
            </Button>
          </div>
        </div>

        <TabsContent value="entities" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Extracted Clinical Entities</CardTitle>
              <CardDescription>
                Named entities identified from the patient note with confidence scores
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {(results.entities || []).map((entity, index) => (
                  <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center space-x-3">
                      <Badge className={`${getEntityColor(entity.label)} border`}>
                        {getEntityIcon(entity.label)}
                        <span className="ml-1">{entity.label}</span>
                      </Badge>
                      <div>
                        <p className="font-medium">{entity.text}</p>
                        <p className="text-sm text-muted-foreground">
                          Position: {entity.start}-{entity.end}
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className={`text-sm font-medium ${getConfidenceColor(entity.confidence)}`}>
                        {Math.round(entity.confidence * 100)}%
                      </p>
                      <div className="w-16">
                        <Progress value={entity.confidence * 100} className="h-2" />
                      </div>
                    </div>
                  </div>
                ))}
                
                {(!results.entities || results.entities.length === 0) && (
                  <div className="text-center py-8 text-muted-foreground">
                    <AlertTriangle className="h-8 w-8 mx-auto mb-2" />
                    <p>No clinical entities found in the provided text.</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="icd" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>ICD-10 Code Mappings</CardTitle>
              <CardDescription>
                Relevant ICD-10 codes mapped to identified entities with similarity scores
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {(results.icd_mappings || []).map((mapping, index) => (
                  <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-1">
                        <Badge variant="outline" className="font-mono">
                          {mapping.code}
                        </Badge>
                        <p className="font-medium">{mapping.description}</p>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Mapped from: <span className="font-medium">{mapping.entity_text}</span>
                      </p>
                    </div>
                    <div className="text-right">
                      <p className={`text-sm font-medium ${getConfidenceColor(mapping.similarity_score)}`}>
                        {Math.round(mapping.similarity_score * 100)}%
                      </p>
                      <div className="w-16">
                        <Progress value={mapping.similarity_score * 100} className="h-2" />
                      </div>
                    </div>
                  </div>
                ))}
                
                {(!results.icd_mappings || results.icd_mappings.length === 0) && (
                  <div className="text-center py-8 text-muted-foreground">
                    <AlertTriangle className="h-8 w-8 mx-auto mb-2" />
                    <p>No ICD-10 mappings found for the identified entities.</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="insights" className="space-y-4">
          <div className="grid gap-4 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Analysis Summary</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <p className="text-sm"><strong>Entity Types Found:</strong></p>
                  <div className="flex flex-wrap gap-2">
                    {Array.from(new Set((results.entities || []).map(e => e.label))).map((label) => (
                      <Badge key={label} className={getEntityColor(label)}>
                        {label}
                      </Badge>
                    ))}
                  </div>
                </div>
                
                <div className="space-y-2">
                  <p className="text-sm"><strong>Confidence Distribution:</strong></p>
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span>High (80%+)</span>
                      <span>{(results.entities || []).filter(e => e.confidence >= 0.8).length}</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span>Medium (60-80%)</span>
                      <span>{(results.entities || []).filter(e => e.confidence >= 0.6 && e.confidence < 0.8).length}</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span>Low (&lt;60%)</span>
                      <span>{(results.entities || []).filter(e => e.confidence < 0.6).length}</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Performance Metrics</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm">Processing Time</span>
                    <span className="text-sm font-medium">{results.analysis_time || 0}s</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Search Method</span>
                    <Badge variant={results.search_method === 'faiss' ? 'default' : 'secondary'}>
                      {(results.search_method || 'N/A').toUpperCase()}
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Analysis Date</span>
                    <span className="text-sm font-medium">
                      {results.created_at ? new Date(results.created_at).toLocaleString() : 'N/A'}
                    </span>
                  </div>
                </div>

                <div className="pt-2 border-t">
                  <p className="text-xs text-muted-foreground">
                    Analysis completed successfully with {results.search_method === 'faiss' ? 'optimized vector' : 'standard'} search method.
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}