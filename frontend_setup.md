🚀 Frontend Setup Commands

  Run these commands in sequence from your project root:

  # 1. Create Next.js app with TypeScript and Tailwind CSS
  npx create-next-app@latest frontend --typescript --tailwind --eslint --app --src-dir --import-alias "@/*"

  # 2. Navigate to frontend directory
  cd frontend

  # 3. Install core dependencies
  npm install @tanstack/react-query @tanstack/react-query-devtools react-hook-form @hookform/resolvers zod lucide-react class-variance-authority clsx
  tailwind-merge @radix-ui/react-slot @radix-ui/react-toast @radix-ui/react-dialog @radix-ui/react-dropdown-menu @radix-ui/react-tabs @radix-ui/react-progress
   recharts date-fns zustand

  # 4. Install development dependencies
  npm install -D @types/node prettier prettier-plugin-tailwindcss husky lint-staged @typescript-eslint/eslint-plugin

  # 5. Initialize shadcn/ui (follow prompts - choose defaults)
  npx shadcn-ui@latest init

  # 6. Install essential shadcn/ui components
  npx shadcn@latest add button card input label textarea badge progress skeleton toast dialog dropdown-menu tabs table alert separator sheet select form

  # 7. Setup Git hooks
  npx husky init

  ⚙️ Create Configuration Files

  After running the above commands, you'll need to:

  1. Create environment files:
  # Create .env.local
  echo "NEXT_PUBLIC_API_BASE_URL=http://localhost:8000" > .env.local
  echo "NEXT_PUBLIC_APP_ENV=development" >> .env.local

  # Create .env.example  
  echo "NEXT_PUBLIC_API_BASE_URL=http://localhost:8000" > .env.example
  echo "NEXT_PUBLIC_APP_ENV=development" >> .env.example

  2. Create the directory structure:
  mkdir -p src/components/{ui,analysis,dashboard,layout,common}
  mkdir -p src/{lib,hooks,styles}
  mkdir -p src/app/\(dashboard\)/{dashboard,analysis/new,performance}
  mkdir -p src/app/api/analysis

  3. Verify everything works:
  npm run dev

  🎯 What This Setup Includes

  ✅ Next.js 14 with App Router and TypeScript✅ Tailwind CSS for styling✅ shadcn/ui component library✅ React Query for server state management✅ React Hook
   Form + Zod for form validation✅ Lucide React for icons✅ Recharts for performance visualizations✅ ESLint + Prettier for code quality✅ Husky for git
  hooks

  🚀 Next Steps

  1. Start the development server: npm run dev
  2. Visit http://localhost:3000 to see your app
  3. Begin building components following the structure in FRONTEND_RULES.md
  4. Start with the layout components and basic routing
  5. Integrate with your backend API running on port 8000

  Your frontend is now fully configured and ready for development according to modern React/Next.js best practices! 🎉