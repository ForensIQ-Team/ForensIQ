import React from 'react';
import { UserRole } from '../types';
import { Shield, UserCheck, Key, Lock, Bell, Sparkles } from 'lucide-react';

interface AccountSettingsProps {
  userRole: UserRole;
  setUserRole: (role: UserRole) => void;
}

export const AccountSettingsModal: React.FC<AccountSettingsProps> = ({
  userRole,
  setUserRole,
}) => {
  return (
    <div className="space-y-8 max-w-4xl mx-auto pb-12">
      <div>
        <h2 className="text-2xl font-bold text-stone-950 tracking-tight">Account &amp; Security Settings</h2>
        <p className="text-stone-600 text-xs sm:text-sm mt-1">
          Manage your account role, security credentials, and trust preferences.
        </p>
      </div>

      {/* Role Selection Box */}
      <div className="glass-panel rounded-xl p-6 shadow-xs space-y-4">
        <h3 className="font-bold text-stone-950 text-sm uppercase tracking-wider border-b border-[#e2d8c3]/80 pb-2 flex items-center gap-2">
          <UserCheck className="w-4 h-4 text-red-700" />
          <span>ACCOUNT ROLE &amp; ACCESS LEVEL</span>
        </h3>

        <p className="text-xs text-stone-600 leading-relaxed">
          ForensIQ utilizes role-based access control within a unified platform architecture. You can toggle between Standard User and Verified Investigator modes below:
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
          {/* Option 1: Standard User */}
          <div
            onClick={() => setUserRole('normal')}
            className={`p-5 rounded-xl border transition-all cursor-pointer flex flex-col justify-between space-y-3 ${
              userRole === 'normal'
                ? 'border-stone-950 bg-[#1e1b18] text-white shadow-xs'
                : 'border-[#e2d8c3]/80 glass-card text-stone-800 hover:bg-[#faf7f2]/60'
            }`}
          >
            <div className="space-y-1">
              <div className="flex items-center justify-between">
                <span className="font-bold text-sm">Standard User</span>
                {userRole === 'normal' && (
                  <span className="px-2 py-0.5 rounded bg-white/20 text-amber-200 text-[10px] font-bold">
                    Active
                  </span>
                )}
              </div>
              <p
                className={`text-xs leading-relaxed ${
                  userRole === 'normal' ? 'text-stone-300' : 'text-stone-500'
                }`}
              >
                Essential forensic suite: Check media authenticity, protect personal images, and search web misuse.
              </p>
            </div>
            <div className="text-[11px] font-semibold text-amber-300">
              ✓ Clean, simplified interface
            </div>
          </div>

          {/* Option 2: Verified Investigator */}
          <div
            onClick={() => setUserRole('investigator')}
            className={`p-5 rounded-xl border transition-all cursor-pointer flex flex-col justify-between space-y-3 ${
              userRole === 'investigator'
                ? 'border-stone-950 bg-[#1e1b18] text-white shadow-xs'
                : 'border-[#e2d8c3]/80 glass-card text-stone-800 hover:bg-[#faf7f2]/60'
            }`}
          >
            <div className="space-y-1">
              <div className="flex items-center justify-between">
                <span className="font-bold text-sm flex items-center gap-1.5">
                  <Shield className="w-4 h-4 text-red-400" />
                  <span>Verified Investigator</span>
                </span>
                {userRole === 'investigator' && (
                  <span className="px-2 py-0.5 rounded bg-white/20 text-amber-200 text-[10px] font-bold">
                    Active
                  </span>
                )}
              </div>
              <p
                className={`text-xs leading-relaxed ${
                  userRole === 'investigator' ? 'text-stone-300' : 'text-stone-500'
                }`}
              >
                Full forensic suite: Interactive spectral viewer, ELA residual maps, compare workspace, and audit report generator.
              </p>
            </div>
            <div className="text-[11px] font-semibold text-amber-300">
              ✓ Advanced evidence &amp; diagnostic suite unlocked
            </div>
          </div>
        </div>
      </div>

      {/* Account Profile Card */}
      <div className="glass-panel rounded-xl p-6 shadow-xs space-y-4">
        <h3 className="font-bold text-stone-950 text-sm uppercase tracking-wider border-b border-[#e2d8c3]/80 pb-2">
          PROFILE INFORMATION
        </h3>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
          <div>
            <label className="font-semibold text-stone-700 block mb-1">Account Name:</label>
            <input
              type="text"
              readOnly
              value="Digital Media Professional"
              className="w-full p-2 glass-input rounded-lg text-stone-900 font-medium"
            />
          </div>

          <div>
            <label className="font-semibold text-stone-700 block mb-1">Email Address:</label>
            <input
              type="text"
              readOnly
              value="investigator@forensiq.org"
              className="w-full p-2 glass-input rounded-lg text-stone-900 font-medium"
            />
          </div>
        </div>
      </div>
    </div>
  );
};
