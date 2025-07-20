import { DashboardLayout } from '@/components/layout/dashboard-layout';

interface DashboardRootLayoutProps {
  children: React.ReactNode;
}

export default function DashboardRootLayout({ children }: DashboardRootLayoutProps) {
  return (
    <DashboardLayout>
      {children}
    </DashboardLayout>
  );
}