import React, { useState } from 'react';
import { ShieldCheck, Mail, Lock, ArrowRight, AlertCircle } from 'lucide-react';
import { useAuth } from '../../services/authContext';
import { NavItem } from '../../types';

interface LoginScreenProps {
  setActiveTab: (tab: NavItem) => void;
  onSuccess?: () => void;
}

export const LoginScreen: React.FC<LoginScreenProps> = ({ setActiveTab, onSuccess }) => {
  const { login } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    if (!email || !password) {
      setError('Please provide your email and password.');
      return;
    }

    setLoading(true);
    const res = await login(email, password);
    setLoading(false);

    if (res.success) {
      if (onSuccess) onSuccess();
      setActiveTab('home');
    } else {
      setError(res.error || 'Invalid email or password.');
    }
  };

  return (
    <div className="max-w-md mx-auto py-12 px-4">
      <div className="glass-panel rounded-2xl p-8 shadow-xl space-y-6">
        {/* Brand Header */}
        <div className="text-center space-y-2">
          <div className="inline-flex w-12 h-12 rounded-2xl items-center justify-center text-amber-200 shadow-md mb-2"
            style={{ background: 'rgba(20, 18, 15, 0.85)' }}>
            <ShieldCheck className="w-7 h-7 text-red-500" />
          </div>
          <h2 className="text-2xl font-serif font-black text-stone-950 tracking-tight">Sign In to ForensIQ</h2>
          <p className="text-xs text-stone-600 leading-relaxed">
            Enter your credentials to access your investigations, case files, and forensic custody records.
          </p>
        </div>

        {error && (
          <div className="p-3.5 rounded-xl border border-red-300 bg-red-50/80 text-red-900 text-xs flex items-center gap-2.5">
            <AlertCircle className="w-4 h-4 text-red-600 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-stone-800 mb-1">Email Address</label>
            <div className="relative">
              <Mail className="w-4 h-4 absolute left-3 top-3 text-stone-400" />
              <input
                type="email"
                required
                placeholder="investigator@agency.gov"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full pl-9 pr-3 py-2 text-xs rounded-xl glass-input text-stone-900 placeholder-stone-400 focus:outline-none"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-stone-800 mb-1">Password</label>
            <div className="relative">
              <Lock className="w-4 h-4 absolute left-3 top-3 text-stone-400" />
              <input
                type="password"
                required
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full pl-9 pr-3 py-2 text-xs rounded-xl glass-input text-stone-900 placeholder-stone-400 focus:outline-none"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl text-xs font-bold text-white transition-all cursor-pointer shadow-sm disabled:opacity-50"
            style={{ background: 'rgba(20, 18, 15, 0.88)' }}
          >
            <span>{loading ? 'Authenticating...' : 'Sign In'}</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </form>

        <div className="text-center pt-2 border-t border-[rgba(200,185,155,0.25)] text-xs text-stone-600">
          Don't have an account yet?{' '}
          <button
            onClick={() => setActiveTab('signup')}
            className="font-bold text-stone-900 hover:text-red-700 underline cursor-pointer"
          >
            Create an account
          </button>
        </div>
      </div>
    </div>
  );
};
