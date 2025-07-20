import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  Activity, 
  BarChart3, 
  FileText, 
  Clock,
  TrendingUp,
  Database,
  Zap,
  CheckCircle
} from 'lucide-react';

export default function DashboardPage() {
  // Mock data - will be replaced with real API calls
  const stats = {
    totalAnalyses: 1247,
    todayAnalyses: 23,
    avgProcessingTime: 2.4,
    successRate: 98.5
  };

  const recentAnalyses = [
    {
      id: '1',
      timestamp: '2024-01-19 10:30 AM',
      entities: 12,
      icdCodes: 8,
      processingTime: 2.1
    },
    {
      id: '2',
      timestamp: '2024-01-19 09:45 AM', 
      entities: 7,
      icdCodes: 5,
      processingTime: 1.8
    },
    {
      id: '3',
      timestamp: '2024-01-19 09:12 AM',
      entities: 15,
      icdCodes: 11,
      processingTime: 3.2
    }
  ];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-muted-foreground">
          Monitor your clinical analysis performance and system metrics
        </p>
      </div>

      {/* Stats Overview */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Analyses</CardTitle>
            <FileText className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.totalAnalyses.toLocaleString()}</div>
            <p className="text-xs text-muted-foreground">
              +{stats.todayAnalyses} today
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Processing Time</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.avgProcessingTime}s</div>
            <p className="text-xs text-muted-foreground">
              <TrendingUp className="inline h-3 w-3 mr-1" />
              12% faster this week
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
            <CheckCircle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.successRate}%</div>
            <Progress value={stats.successRate} className="mt-2" />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">System Status</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="flex items-center space-x-2">
              <Badge variant="default" className="bg-green-500">
                <Zap className="h-3 w-3 mr-1" />
                Optimal
              </Badge>
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              All systems operational
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Recent Activity and Quick Actions */}
      <div className="grid gap-4 lg:grid-cols-3">
        {/* Recent Analyses */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Recent Analyses</CardTitle>
            <CardDescription>
              Latest patient note analyses processed by the system
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {recentAnalyses.map((analysis) => (
                <div key={analysis.id} className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex flex-col space-y-1">
                    <p className="text-sm font-medium">Analysis #{analysis.id}</p>
                    <p className="text-xs text-muted-foreground">{analysis.timestamp}</p>
                  </div>
                  <div className="flex items-center space-x-4 text-sm">
                    <div className="text-center">
                      <p className="font-medium">{analysis.entities}</p>
                      <p className="text-xs text-muted-foreground">Entities</p>
                    </div>
                    <div className="text-center">
                      <p className="font-medium">{analysis.icdCodes}</p>
                      <p className="text-xs text-muted-foreground">ICD Codes</p>
                    </div>
                    <div className="text-center">
                      <p className="font-medium">{analysis.processingTime}s</p>
                      <p className="text-xs text-muted-foreground">Time</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* System Overview */}
        <Card>
          <CardHeader>
            <CardTitle>System Overview</CardTitle>
            <CardDescription>
              Current system performance and status
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span>Faiss Index</span>
                <Badge variant="default" className="bg-green-500">Active</Badge>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span>Vector Search</span>
                <Badge variant="default" className="bg-green-500">Optimal</Badge>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span>Cache Hit Rate</span>
                <span className="font-medium">94.2%</span>
              </div>
            </div>

            <div className="space-y-2">
              <p className="text-sm font-medium">Processing Queue</p>
              <div className="flex items-center space-x-2">
                <Progress value={15} className="flex-1" />
                <span className="text-xs text-muted-foreground">2/15</span>
              </div>
            </div>

            <div className="pt-2 border-t">
              <div className="flex items-center space-x-2">
                <Database className="h-4 w-4 text-muted-foreground" />
                <div className="text-xs text-muted-foreground">
                  <p>ICD-10 Database: 72,184 codes</p>
                  <p>Last updated: Jan 15, 2024</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}