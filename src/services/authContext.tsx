import React, { createContext, useContext, useState, useEffect } from 'react';
import { AuthUser, InvestigatorAppInfo, OrganizationInfo } from '../types';

interface AuthContextType {
  user: AuthUser | null;
  token: string | null;
  isLoading: boolean;
  login: (email: string, pass: string) => Promise<{ success: boolean; error?: string }>;
  signup: (email: string, pass: string, displayName?: string) => Promise<{ success: boolean; error?: string }>;
  logout: () => void;
  applyInvestigator: (data: {
    organization_name?: string;
    designation?: string;
    work_email: string;
    org_website?: string;
    linkedin_url?: string;
    reason?: string;
  }) => Promise<{ success: boolean; error?: string; status?: string }>;
  refreshUser: () => Promise<void>;
  currentOrg: OrganizationInfo | null;
  investigatorApplication: InvestigatorAppInfo | null;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [token, setToken] = useState<string | null>(() => localStorage.getItem('forensiq_token'));
  const [user, setUser] = useState<AuthUser | null>(null);
  const [currentOrg, setCurrentOrg] = useState<OrganizationInfo | null>(null);
  const [investigatorApplication, setInvestigatorApplication] = useState<InvestigatorAppInfo | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  const fetchCurrentUser = async (authToken: string) => {
    try {
      const res = await fetch('/api/auth/me', {
        headers: {
          Authorization: `Bearer ${authToken}`,
        },
      });
      if (res.ok) {
        const userData = await res.json();
        setUser(userData);
        // Also fetch investigator status
        const statusRes = await fetch('/api/auth/investigator/status', {
          headers: {
            Authorization: `Bearer ${authToken}`,
          },
        });
        if (statusRes.ok) {
          const statusData = await statusRes.json();
          if (statusData.organization) setCurrentOrg(statusData.organization);
          if (statusData.application) setInvestigatorApplication(statusData.application);
        }
      } else {
        // Token expired or invalid
        localStorage.removeItem('forensiq_token');
        setToken(null);
        setUser(null);
      }
    } catch (err) {
      console.error('Failed to authenticate token:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (token) {
      fetchCurrentUser(token);
    } else {
      setIsLoading(false);
    }
  }, [token]);

  const login = async (email: string, pass: string) => {
    try {
      const res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password: pass }),
      });
      const data = await res.json();
      if (!res.ok) {
        return { success: false, error: data.detail || 'Login failed' };
      }
      localStorage.setItem('forensiq_token', data.access_token);
      setToken(data.access_token);
      setUser(data.user);
      await fetchCurrentUser(data.access_token);
      return { success: true };
    } catch (err: any) {
      return { success: false, error: err.message || 'Network error' };
    }
  };

  const signup = async (email: string, pass: string, displayName?: string) => {
    try {
      const res = await fetch('/api/auth/signup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password: pass, display_name: displayName }),
      });
      const data = await res.json();
      if (!res.ok) {
        return { success: false, error: data.detail || 'Signup failed' };
      }
      localStorage.setItem('forensiq_token', data.access_token);
      setToken(data.access_token);
      setUser(data.user);
      await fetchCurrentUser(data.access_token);
      return { success: true };
    } catch (err: any) {
      return { success: false, error: err.message || 'Network error' };
    }
  };

  const logout = () => {
    localStorage.removeItem('forensiq_token');
    setToken(null);
    setUser(null);
    setCurrentOrg(null);
    setInvestigatorApplication(null);
  };

  const applyInvestigator = async (formData: any) => {
    if (!token) return { success: false, error: 'Not authenticated' };
    try {
      const res = await fetch('/api/auth/investigator/apply', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(formData),
      });
      const data = await res.json();
      if (!res.ok) {
        return { success: false, error: data.detail || 'Application failed' };
      }
      if (data.user) setUser(data.user);
      if (data.organization) setCurrentOrg(data.organization);
      if (data.application) setInvestigatorApplication(data.application);
      return { success: true, status: data.application?.status };
    } catch (err: any) {
      return { success: false, error: err.message || 'Network error' };
    }
  };

  const refreshUser = async () => {
    if (token) {
      await fetchCurrentUser(token);
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isLoading,
        login,
        signup,
        logout,
        applyInvestigator,
        refreshUser,
        currentOrg,
        investigatorApplication,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
