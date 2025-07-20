import { RegisterForm } from '@/components/auth/register-form';

export default function RegisterPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100 px-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Clinical Analysis
          </h1>
          <p className="text-gray-600">
            AI-powered patient note analysis and ICD-10 mapping
          </p>
        </div>
        <RegisterForm />
      </div>
    </div>
  );
}