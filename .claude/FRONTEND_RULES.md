# Frontend Development Rules & Conventions

## 🎯 **Project Overview**
Clinical Analysis Dashboard built with Next.js, React, Tailwind CSS, and shadcn/ui for analyzing patient notes with AI-powered entity extraction and ICD-10 mapping.

## 🛠 **Technology Stack**

### **Core Framework**
- **Next.js 14+** - App Router, Server Components, TypeScript
- **React 18+** - Functional components, hooks, modern patterns
- **TypeScript** - Strict typing, interfaces, proper error handling

### **Styling & UI**
- **Tailwind CSS** - Utility-first styling, responsive design
- **shadcn/ui** - High-quality, accessible component library
- **Lucide React** - Consistent iconography
- **Class Variance Authority (CVA)** - Component variant management

### **Data & State Management**
- **React Query (TanStack Query)** - Server state management, caching
- **Zustand** - Client state management (if needed)
- **React Hook Form** - Form handling with validation
- **Zod** - Runtime validation and TypeScript inference

### **Development Tools**
- **ESLint + Prettier** - Code formatting and linting
- **Husky** - Git hooks for quality control
- **TypeScript strict mode** - Maximum type safety

## 🏗 **File Structure**

```
frontend/
├── src/
│   ├── app/                          # Next.js App Router
│   │   ├── (dashboard)/              # Route groups
│   │   │   ├── dashboard/
│   │   │   │   ├── page.tsx          # Dashboard overview
│   │   │   │   └── loading.tsx       # Loading UI
│   │   │   ├── analysis/
│   │   │   │   ├── page.tsx          # Analysis interface
│   │   │   │   ├── new/page.tsx      # New analysis form
│   │   │   │   └── [id]/page.tsx     # Analysis results
│   │   │   └── performance/
│   │   │       └── page.tsx          # Performance dashboard
│   │   ├── api/                      # API route handlers (proxy)
│   │   │   └── analysis/
│   │   │       └── route.ts          # Backend API proxy
│   │   ├── globals.css               # Global styles + Tailwind
│   │   ├── layout.tsx                # Root layout
│   │   ├── page.tsx                  # Landing/home page
│   │   ├── loading.tsx               # Global loading
│   │   └── error.tsx                 # Global error boundary
│   ├── components/                   # Reusable components
│   │   ├── ui/                       # shadcn/ui components
│   │   │   ├── button.tsx
│   │   │   ├── card.tsx
│   │   │   ├── input.tsx
│   │   │   ├── badge.tsx
│   │   │   ├── progress.tsx
│   │   │   ├── skeleton.tsx
│   │   │   ├── toast.tsx
│   │   │   ├── dialog.tsx
│   │   │   ├── tabs.tsx
│   │   │   └── ...
│   │   ├── analysis/                 # Analysis-specific components
│   │   │   ├── analysis-form.tsx     # Patient note input form
│   │   │   ├── entity-display.tsx    # Clinical entities viewer
│   │   │   ├── icd-mapping.tsx       # ICD code results
│   │   │   ├── nlp-insights.tsx      # NLP enhancements display
│   │   │   ├── performance-metrics.tsx # Analysis performance
│   │   │   └── analysis-history.tsx  # Previous analyses
│   │   ├── dashboard/                # Dashboard components
│   │   │   ├── stats-overview.tsx    # Performance statistics
│   │   │   ├── recent-analyses.tsx   # Recent analysis list
│   │   │   ├── performance-chart.tsx # Performance visualization
│   │   │   └── system-status.tsx     # Faiss/system status
│   │   ├── layout/                   # Layout components
│   │   │   ├── header.tsx            # Main navigation
│   │   │   ├── sidebar.tsx           # Side navigation
│   │   │   ├── breadcrumbs.tsx       # Navigation breadcrumbs
│   │   │   └── footer.tsx            # Footer component
│   │   └── common/                   # Common components
│   │       ├── loading-spinner.tsx   # Loading states
│   │       ├── error-boundary.tsx    # Error handling
│   │       ├── confirmation-dialog.tsx
│   │       └── data-table.tsx        # Reusable table
│   ├── lib/                          # Utilities and configurations
│   │   ├── api.ts                    # API client configuration
│   │   ├── utils.ts                  # Utility functions
│   │   ├── validations.ts            # Zod schemas
│   │   ├── constants.ts              # App constants
│   │   └── types.ts                  # TypeScript types
│   ├── hooks/                        # Custom React hooks
│   │   ├── use-analysis.ts           # Analysis API hooks
│   │   ├── use-performance.ts        # Performance data hooks
│   │   ├── use-debounce.ts          # Debounce hook
│   │   └── use-local-storage.ts     # Local storage hook
│   └── styles/                       # Additional styles
│       └── globals.css               # Extended global styles
├── public/                           # Static assets
│   ├── favicon.ico
│   └── logo.svg
├── .env.local                        # Environment variables
├── .env.example                      # Environment template
├── .eslintrc.json                    # ESLint configuration
├── .prettierrc                       # Prettier configuration
├── tailwind.config.js                # Tailwind configuration
├── tsconfig.json                     # TypeScript configuration
├── next.config.js                    # Next.js configuration
├── package.json                      # Dependencies
├── components.json                   # shadcn/ui configuration
└── README.md                         # Project documentation
```

## 🎨 **Design Principles**

### **Visual Design**
- **Clean & Minimal** - Embrace whitespace, reduce visual clutter
- **Modern Interface** - Contemporary design patterns, subtle animations
- **Medical Professional** - Professional color scheme, clinical feel
- **Accessibility First** - WCAG 2.1 AA compliance, keyboard navigation

### **Color Palette**
```css
/* Primary Colors */
--primary: 210 100% 50%;          /* Medical blue */
--primary-foreground: 0 0% 98%;   

/* Accent Colors */
--success: 142 76% 36%;           /* Clinical green */
--warning: 38 92% 50%;            /* Attention amber */
--destructive: 0 84% 60%;         /* Error red */

/* Neutral Colors */
--background: 0 0% 100%;          /* Clean white */
--foreground: 222 84% 5%;         /* Near black */
--muted: 210 40% 98%;             /* Light gray */
--border: 214 32% 91%;            /* Subtle borders */
```

### **Typography**
- **Font Family**: Inter (system fallback)
- **Scale**: Tailwind's default scale (text-sm to text-4xl)
- **Hierarchy**: Clear heading structure, consistent line heights

## 📋 **Component Conventions**

### **Component Structure**
```typescript
// Example component structure
interface ComponentProps {
  // Props interface
}

export function ComponentName({ prop1, prop2 }: ComponentProps) {
  // Hooks at the top
  // Event handlers
  // Render logic
  
  return (
    <div className="component-wrapper">
      {/* JSX */}
    </div>
  );
}
```

### **Naming Conventions**
- **Components**: PascalCase (`AnalysisForm`, `EntityDisplay`)
- **Files**: kebab-case (`analysis-form.tsx`, `entity-display.tsx`)
- **Functions**: camelCase (`handleSubmit`, `formatDate`)
- **Constants**: SCREAMING_SNAKE_CASE (`API_BASE_URL`)

### **CSS Classes**
- **Tailwind First** - Use utility classes primarily
- **Component Classes** - For complex repeated patterns
- **Responsive Design** - Mobile-first approach
- **Dark Mode Ready** - Use CSS variables for theming

## 🔌 **API Integration**

### **Backend Endpoints**
```typescript
// Primary endpoints to implement
const API_ENDPOINTS = {
  // Enhanced Analysis (PRIMARY)
  EXTRACT_ENHANCED: '/api/analysis/extract-enhanced',
  PERFORMANCE_STATS: '/api/analysis/performance-stats', 
  BENCHMARK: '/api/analysis/benchmark',
  
  // Standard Analysis (FALLBACK)
  EXTRACT: '/api/analysis/extract',
  EXTRACT_WITH_ICD: '/api/analysis/extract-with-icd'
} as const;
```

### **Data Flow**
1. **Form Input** → Patient note + context
2. **API Call** → Enhanced analysis endpoint
3. **Loading State** → Show progress with estimated time
4. **Results Display** → Entities, ICD codes, NLP insights
5. **Performance Feedback** → Analysis time, search method used

## 🧩 **Key Features to Implement**

### **1. Analysis Interface**
- **Input Form**: Patient note textarea, context fields
- **Real-time Validation**: Form validation with Zod
- **Progress Indicator**: Analysis progress with time estimates
- **Results View**: Tabbed interface for different result types

### **2. Entity Display**
- **Structured Layout**: Cards for symptoms, conditions, medications
- **NLP Enhancements**: Badges for negation, temporal, uncertainty
- **Confidence Scores**: Visual indicators (progress bars/colors)
- **Interactive Elements**: Expandable details, tooltips

### **3. ICD Mapping Results**
- **Searchable Table**: ICD codes with similarity scores
- **Filter Options**: By entity type, confidence level
- **Hierarchy Display**: ICD code categories and subcategories
- **Export Options**: CSV, JSON download

### **4. Performance Dashboard**
- **Real-time Metrics**: Analysis speed, success rates
- **Search Method Indicator**: Faiss vs numpy usage
- **Historical Charts**: Performance trends over time
- **System Status**: Backend health, cache status

### **5. Analysis History**
- **Previous Analyses**: Saved analysis results
- **Search/Filter**: By date, patient, entity types
- **Comparison View**: Side-by-side analysis comparison
- **Export/Share**: Analysis report generation

## ⚡ **Performance Requirements**

### **Loading States**
- **Immediate Feedback** - Loading states for all async operations
- **Skeleton Loading** - Content placeholders while loading
- **Progressive Enhancement** - Show data as it becomes available

### **Optimization**
- **Code Splitting** - Route-based and component-based
- **Image Optimization** - Next.js Image component
- **Bundle Analysis** - Regular bundle size monitoring
- **Caching Strategy** - React Query for server state

## 🔒 **Security & Validation**

### **Input Validation**
- **Client-side**: Zod schemas for form validation
- **Sanitization**: Prevent XSS in patient notes
- **Rate Limiting**: Prevent API abuse
- **Error Boundaries**: Graceful error handling

### **Data Handling**
- **No PHI Storage** - No persistent patient data
- **Session Management** - Temporary analysis storage
- **Secure API Calls** - Proper headers and validation

## 🧪 **Testing Strategy**

### **Testing Levels**
- **Unit Tests**: Jest + React Testing Library
- **Integration Tests**: API integration testing
- **E2E Tests**: Playwright for critical user flows
- **Accessibility Tests**: axe-core integration

### **Coverage Requirements**
- **Components**: 80%+ test coverage
- **Utilities**: 90%+ test coverage
- **Critical Paths**: 100% E2E coverage

## 📱 **Responsive Design**

### **Breakpoints**
- **Mobile**: < 768px (single column, stacked layout)
- **Tablet**: 768px - 1024px (adaptive grid)
- **Desktop**: > 1024px (full layout with sidebar)

### **Mobile-First Approach**
- **Touch-Friendly**: 44px minimum touch targets
- **Readable Text**: 16px base font size
- **Optimized Navigation**: Collapsible menus, bottom navigation

## 🎯 **Success Metrics**

### **User Experience**
- **First Paint**: < 1.5s
- **Time to Interactive**: < 3s
- **Analysis Display**: < 500ms after API response
- **Accessibility Score**: 95%+

### **Technical Metrics**
- **Bundle Size**: < 250KB initial
- **Lighthouse Score**: 90%+ across all metrics
- **Error Rate**: < 1%
- **API Success Rate**: 99%+

---

## 📝 **Development Workflow**

1. **Feature Planning** - Review requirements with backend capabilities
2. **Component Design** - Create reusable, tested components
3. **API Integration** - Implement with proper error handling
4. **Testing** - Unit, integration, and E2E tests
5. **Performance Review** - Bundle analysis and optimization
6. **Accessibility Audit** - WCAG compliance verification
7. **Code Review** - Peer review focusing on conventions
8. **Documentation** - Update component docs and README

This frontend will provide a modern, efficient, and user-friendly interface for the clinical analysis system while maintaining the highest standards of code quality and user experience.