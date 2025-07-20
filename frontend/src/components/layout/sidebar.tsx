'use client';

import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { 
  LayoutDashboard, 
  FileText, 
  BarChart3, 
  History,
  Plus,
  Activity,
  Database
} from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

const navigation = [
  {
    name: 'Dashboard',
    href: '/dashboard',
    icon: LayoutDashboard,
  },
  {
    name: 'New Analysis',
    href: '/analysis/new',
    icon: Plus,
  },
  {
    name: 'Analysis History',
    href: '/analysis',
    icon: History,
  },
  {
    name: 'Performance',
    href: '/performance',
    icon: BarChart3,
  },
];

const secondaryNavigation = [
  {
    name: 'System Status',
    href: '/system',
    icon: Activity,
  },
  {
    name: 'Data Management',
    href: '/data',
    icon: Database,
  },
];

interface SidebarProps {
  className?: string;
}

export function Sidebar({ className }: SidebarProps) {
  const pathname = usePathname();

  return (
    <div className={cn('flex h-full w-64 flex-col bg-background border-r', className)}>
      {/* Logo section */}
      <div className="flex h-16 items-center border-b px-6">
        <div className="flex items-center space-x-2">
          <div className="h-8 w-8 rounded-lg bg-primary flex items-center justify-center">
            <span className="text-primary-foreground font-bold text-sm">CA</span>
          </div>
          <div>
            <h2 className="font-semibold text-lg">Clinical Analysis</h2>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <ScrollArea className="flex-1 px-3 py-4">
        <div className="space-y-1">
          <div className="px-3 py-2">
            <h3 className="mb-2 px-4 text-sm font-semibold tracking-tight text-muted-foreground">
              Main
            </h3>
            <div className="space-y-1">
              {navigation.map((item) => {
                const isActive = pathname === item.href || 
                  (item.href === '/analysis' && pathname.startsWith('/analysis') && pathname !== '/analysis/new');
                
                return (
                  <Button
                    key={item.href}
                    variant={isActive ? 'secondary' : 'ghost'}
                    className={cn(
                      'w-full justify-start',
                      isActive && 'bg-secondary'
                    )}
                    asChild
                  >
                    <Link href={item.href}>
                      <item.icon className="mr-2 h-4 w-4" />
                      {item.name}
                    </Link>
                  </Button>
                );
              })}
            </div>
          </div>

          <div className="px-3 py-2">
            <h3 className="mb-2 px-4 text-sm font-semibold tracking-tight text-muted-foreground">
              System
            </h3>
            <div className="space-y-1">
              {secondaryNavigation.map((item) => {
                const isActive = pathname === item.href;
                
                return (
                  <Button
                    key={item.href}
                    variant={isActive ? 'secondary' : 'ghost'}
                    className={cn(
                      'w-full justify-start',
                      isActive && 'bg-secondary'
                    )}
                    asChild
                  >
                    <Link href={item.href}>
                      <item.icon className="mr-2 h-4 w-4" />
                      {item.name}
                    </Link>
                  </Button>
                );
              })}
            </div>
          </div>
        </div>
      </ScrollArea>

      {/* Footer */}
      <div className="border-t p-4">
        <div className="text-xs text-muted-foreground">
          <p>Clinical Analysis v1.0</p>
          <p>AI-powered medical analysis</p>
        </div>
      </div>
    </div>
  );
}