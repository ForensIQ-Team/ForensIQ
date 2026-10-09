import React, { useState } from 'react';
import {
  Shield,
  UserCheck,
  Building,
  Mail,
  Globe,
  Briefcase,
  FileText,
  CheckCircle2,
  Clock,
  XCircle,
  ArrowRight,
  LogOut,
  AlertCircle,
  ExternalLink,
} from 'lucide-react';
import { useAuth } from '../../services/authContext';
import { NavItem, UserRole } from '../../types';

interface ProfileScreenProps {
  setActiveTab: (tab: NavItem) => void;
  setUserRole: (role: UserRole) => void;
}

export const ProfileScreen: React.FC<ProfileScreenProps> = ({ setActiveTab, setUserRole }) => {
  const { user, logout, applyInvestigator, currentOrg, investigatorApplication, refreshUser } = useAuth();

  const [showApplyForm, setShowApplyForm] = useState(false);
  const [formData, setFormData] = useState({
    organization_name: '',
    designation: '',
    work_email: user?.email || '',
    org_website: '',
    linkedin_url: '',
    reason: '',
    reference_email: '',
  });
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<{ text: string; type: 'success' | 'error' } | null>(null);

  if (!user) {
    return (
      <div className="max-w-md mx-auto py-16 text-center space-y-4">
        <div className="glass-panel p-8 rounded-2xl space-y-4">
          <p className="text-sm text-stone-700 font-semibold">You are not currently signed in.</p>
          <div className="flex justify-center gap-3">
            <button
              onClick={() => setActiveTab('login')}
              className="px-4 py-2 rounded-xl text-xs font-bold text-white bg-stone-900 shadow-sm cursor-pointer"
            >
              Sign In
            </button>
            <button
              onClick={() => setActiveTab('signup')}
              className="px-4 py-2 rounded-xl text-xs font-bold text-stone-800 glass-card cursor-pointer"
            >
              Create Account
            </button>
          </div>
        </div>
      </div>
    );
  }

  const handleApply = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setMessage(null);

    const res = await applyInvestigator(formData);
    setLoading(false);

    if (res.success) {
      setMessage({
        text: res.status === 'approved'
          ? 'Application approved! Organization verified and investigator tier unlocked.'
          : 'Application submitted! Your credentials are under manual review.',
        type: 'success',
      });
      setShowApplyForm(false);
      await refreshUser();
      if (res.status === 'approved') {
        setUserRole('investigator');
      }
    } else {
      setMessage({ text: res.error || 'Failed to submit application', type: 'error' });
    }
  };

  const isInvestigator = user.role === 'investigator' || user.role === 'admin';
  const appStatus = investigatorApplication?.status || (isInvestigator ? 'approved' : 'none');

  return (
    <div className="max-w-3xl mx-auto py-8 px-4 space-y-6">
      {/* Profile Header Card */}
      <div className="glass-panel rounded-2xl p-6 shadow-md flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <div
            className="w-14 h-14 rounded-2xl flex items-center justify-center text-amber-200 shadow-sm"
            style={{ background: 'rgba(20, 18, 15, 0.85)' }}
          >
            {isInvestigator ? <Shield className="w-7 h-7 text-red-500" /> : <UserCheck className="w-7 h-7 text-stone-300" />}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-xl font-serif font-black text-stone-950">{user.display_name || 'ForensIQ User'}</h2>
              <span
                className="text-[10px] font-mono font-bold px-2 py-0.5 rounded uppercase"
                style={{
                  background: isInvestigator ? 'rgba(185, 28, 28, 0.15)' : 'rgba(200, 185, 155, 0.35)',
                  color: isInvestigator ? '#991b1b' : '#44403c',
                  border: isInvestigator ? '1px solid rgba(185, 28, 28, 0.3)' : '1px solid rgba(200, 185, 155, 0.4)',
                }}
              >
                {user.role} {user.tier > 0 && `(Tier ${user.tier})`}
              </span>
            </div>
            <p className="text-xs text-stone-600 font-mono mt-0.5">{user.email}</p>
            {currentOrg && (
              <p className="text-xs text-stone-700 font-semibold mt-1 flex items-center gap-1.5">
                <Building className="w-3.5 h-3.5 text-stone-500" />
                <span>{currentOrg.name}</span>
                {currentOrg.domain && <span className="text-[10px] text-stone-500">(@{currentOrg.domain})</span>}
              </p>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {isInvestigator && (
            <button
              onClick={() => setActiveTab('investigator-workspace')}
              className="px-4 py-2 rounded-xl text-xs font-bold text-white shadow-sm flex items-center gap-1.5 cursor-pointer"
              style={{ background: 'rgba(185, 28, 28, 0.9)' }}
            >
              <Shield className="w-3.5 h-3.5" />
              <span>Workspace</span>
            </button>
          )}
          <button
            onClick={() => {
              logout();
              setActiveTab('home');
            }}
            className="p-2 text-stone-600 hover:text-red-700 glass-card rounded-xl text-xs font-semibold flex items-center gap-1.5 cursor-pointer"
            title="Sign Out"
          >
            <LogOut className="w-4 h-4" />
            <span className="hidden sm:inline">Sign Out</span>
          </button>
        </div>
      </div>

      {message && (
        <div
          className={`p-4 rounded-xl text-xs flex items-center gap-2.5 border ${
            message.type === 'success'
              ? 'bg-emerald-50/80 border-emerald-200 text-emerald-900'
              : 'bg-red-50/80 border-red-200 text-red-900'
          }`}
        >
          {message.type === 'success' ? <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" /> : <AlertCircle className="w-4 h-4 text-red-600 shrink-0" />}
          <span>{message.text}</span>
        </div>
      )}

      {/* Application Status Banner */}
      {appStatus !== 'none' && (
        <div
          className="glass-panel rounded-2xl p-5 shadow-xs flex items-start justify-between gap-4"
          style={{
            borderLeft: `4px solid ${
              appStatus === 'approved' ? '#059669' : appStatus === 'pending' ? '#d97706' : '#dc2626'
            }`,
          }}
        >
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              {appStatus === 'approved' && <CheckCircle2 className="w-4 h-4 text-emerald-600" />}
              {appStatus === 'pending' && <Clock className="w-4 h-4 text-amber-600" />}
              {appStatus === 'rejected' && <XCircle className="w-4 h-4 text-red-600" />}
              <span className="font-bold text-xs uppercase tracking-wider text-stone-900">
                Investigator Verification Status: {appStatus.toUpperCase()}
              </span>
            </div>
            <p className="text-xs text-stone-600 leading-relaxed">
              {appStatus === 'approved' &&
                'Your verified investigator credentials are valid. You have unrestricted access to full case management, evidence locking, custody records, and monitoring watchlists.'}
              {appStatus === 'pending' &&
                'Your application has been received and is in the review queue. Standard public email domains require manual credential checks.'}
              {appStatus === 'rejected' &&
                'Your application was not approved. You can submit additional organization credentials for re-evaluation.'}
            </p>
          </div>

          {appStatus === 'approved' && (
            <button
              onClick={() => setActiveTab('investigator-workspace')}
              className="px-3 py-1.5 rounded-xl text-xs font-bold text-white bg-stone-900 shrink-0 shadow-xs cursor-pointer flex items-center gap-1"
            >
              <span>Go to Workspace</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      )}

      {/* Become an Investigator Box (if not approved) */}
      {!isInvestigator && (
        <div className="glass-panel rounded-2xl p-6 shadow-md space-y-4">
          <div className="flex items-start justify-between">
            <div className="space-y-1">
              <h3 className="text-base font-serif font-black text-stone-950 flex items-center gap-2">
                <Shield className="w-4 h-4 text-red-600" />
                <span>Become a Verified Investigator</span>
              </h3>
              <p className="text-xs text-stone-600 leading-relaxed max-w-xl">
                Investigators unlock the full forensic command center: legal evidence custody chain, case management,
                organization watchlists, source reputation index, and certified dossier exports.
              </p>
            </div>
            {!showApplyForm && (
              <button
                onClick={() => setShowApplyForm(true)}
                className="px-4 py-2 rounded-xl text-xs font-bold text-white shadow-sm cursor-pointer shrink-0"
                style={{ background: 'rgba(20, 18, 15, 0.88)' }}
              >
                Apply Now
              </button>
            )}
          </div>

          {showApplyForm && (
            <form onSubmit={handleApply} className="pt-4 border-t border-[rgba(200,185,155,0.3)] space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
                <div>
                  <label className="block font-semibold text-stone-800 mb-1">Organization Name *</label>
                  <input
                    type="text"
                    required
                    placeholder="Bellingcat / Reuters / Interpol"
                    value={formData.organization_name}
                    onChange={(e) => setFormData({ ...formData, organization_name: e.target.value })}
                    className="w-full p-2 rounded-xl glass-input text-stone-900 focus:outline-none"
                  />
                </div>

                <div>
                  <label className="block font-semibold text-stone-800 mb-1">Professional Designation *</label>
                  <input
                    type="text"
                    required
                    placeholder="Senior OSINT Analyst"
                    value={formData.designation}
                    onChange={(e) => setFormData({ ...formData, designation: e.target.value })}
                    className="w-full p-2 rounded-xl glass-input text-stone-900 focus:outline-none"
                  />
                </div>

                <div>
                  <label className="block font-semibold text-stone-800 mb-1">Work Email (Organization Domain) *</label>
                  <input
                    type="email"
                    required
                    placeholder="analyst@agency.org"
                    value={formData.work_email}
                    onChange={(e) => setFormData({ ...formData, work_email: e.target.value })}
                    className="w-full p-2 rounded-xl glass-input text-stone-900 focus:outline-none"
                  />
                </div>

                <div>
                  <label className="block font-semibold text-stone-800 mb-1">Organization Website</label>
                  <input
                    type="url"
                    placeholder="https://agency.org"
                    value={formData.org_website}
                    onChange={(e) => setFormData({ ...formData, org_website: e.target.value })}
                    className="w-full p-2 rounded-xl glass-input text-stone-900 focus:outline-none"
                  />
                </div>

                <div>
                  <label className="block font-semibold text-stone-800 mb-1">LinkedIn / Credential URL</label>
                  <input
                    type="url"
                    placeholder="https://linkedin.com/in/analyst"
                    value={formData.linkedin_url}
                    onChange={(e) => setFormData({ ...formData, linkedin_url: e.target.value })}
                    className="w-full p-2 rounded-xl glass-input text-stone-900 focus:outline-none"
                  />
                </div>

                <div>
                  <label className="block font-semibold text-stone-800 mb-1">Reference Email</label>
                  <input
                    type="email"
                    placeholder="supervisor@agency.org"
                    value={formData.reference_email}
                    onChange={(e) => setFormData({ ...formData, reference_email: e.target.value })}
                    className="w-full p-2 rounded-xl glass-input text-stone-900 focus:outline-none"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-stone-800 mb-1">Investigation Scope &amp; Reason</label>
                <textarea
                  rows={3}
                  placeholder="Detail your use-case (e.g. brand impersonation monitoring, election integrity forensic verification, legal evidence preservation)..."
                  value={formData.reason}
                  onChange={(e) => setFormData({ ...formData, reason: e.target.value })}
                  className="w-full p-2.5 rounded-xl glass-input text-stone-900 text-xs focus:outline-none"
                />
              </div>

              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setShowApplyForm(false)}
                  className="px-4 py-2 rounded-xl text-xs font-semibold text-stone-700 glass-card cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={loading}
                  className="px-5 py-2 rounded-xl text-xs font-bold text-white bg-stone-950 shadow-sm cursor-pointer disabled:opacity-50"
                >
                  {loading ? 'Submitting...' : 'Submit Application'}
                </button>
              </div>
            </form>
          )}
        </div>
      )}
    </div>
  );
};
