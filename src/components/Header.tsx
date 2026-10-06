import React, { useState } from 'react';
import {
  Menu,
  X,
  Shield,
  UserCheck,
  HelpCircle,
  Bell,
  Settings,
  Layers,
  FileSearch,
  Lock,
  Search,
  History,
  FileText,
} from 'lucide-react';
import { NavItem, UserRole } from '../types';

interface HeaderProps {
  activeTab: NavItem;
  setActiveTab: (tab: NavItem) => void;
  userRole: UserRole;
  setUserRole: (role: UserRole) => void;
  isMobileOpen: boolean;
  setIsMobileOpen: (open: boolean) => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  userRole,
  setUserRole,
  isMobileOpen,
  setIsMobileOpen,
}) => {
  const [showTrustModal, setShowTrustModal] = useState(false);

  const handleNavClick = (tab: NavItem) => {
    setActiveTab(tab);
    setIsMobileOpen(false);
  };

  const navItems: { id: NavItem; label: string; icon: any }[] = [
    { id: 'home', label: 'Overview', icon: Layers },
    { id: 'check-media', label: 'Check Media', icon: FileSearch },
    { id: 'protect-image', label: 'Protect Image', icon: Lock },
    { id: 'find-misuse', label: 'Find Misuse', icon: Search },
    { id: 'history', label: 'History', icon: History },
  ];

  return (
    <>
      {/* Floating Compact Glass Navbar */}
      <header className="sticky top-3 z-50 max-w-[1700px] mx-auto w-full px-2 sm:px-4 mb-4">
        <div
          className="rounded-2xl px-4 py-2.5 flex items-center justify-between transition-all duration-200"
          style={{
            background: 'rgba(255, 252, 244, 0.75)',
            backdropFilter: 'blur(20px) saturate(130%)',
            WebkitBackdropFilter: 'blur(20px) saturate(130%)',
            border: '1px solid rgba(255, 255, 255, 0.65)',
            boxShadow: '0 8px 32px rgba(40, 30, 15, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.5)',
          }}
        >
          {/* Brand Logo & V2.4 Badge */}
          <div
            className="flex items-center gap-2 cursor-pointer group shrink-0"
            onClick={() => handleNavClick('home')}
          >
            <div className="w-8 h-8 rounded-xl bg-stone-900 flex items-center justify-center shadow-xs transition-transform group-hover:scale-105">
              <Shield className="w-4 h-4 text-red-500 fill-red-500/20" />
            </div>
            <span className="font-serif font-black tracking-wider text-stone-900 text-sm sm:text-base uppercase">
              FORENSIQ
            </span>
            <span className="px-1.5 py-0.5 text-[9px] font-mono font-bold text-amber-900/80 bg-amber-200/50 rounded-md border border-amber-300/40">
              V2.4
            </span>
          </div>

          {/* Desktop Navigation Links */}
          <nav className="hidden lg:flex items-center gap-1 bg-[#eae0cf]/40 p-1 rounded-xl border border-stone-300/30">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => handleNavClick(item.id)}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-bold transition-all cursor-pointer ${
                    isActive
                      ? 'bg-[#1e1b18] text-white shadow-xs'
                      : 'text-stone-800 hover:bg-[#ded1bc]/50 hover:text-stone-950'
                  }`}
                >
                  <Icon
                    className={`w-3.5 h-3.5 ${
                      isActive ? 'text-red-400' : 'text-stone-600'
                    }`}
                  />
                  <span>{item.label}</span>
                </button>
              );
            })}

            {userRole === 'investigator' && (
              <button
                onClick={() => handleNavClick('professional-analysis')}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-bold transition-all cursor-pointer ${
                  activeTab === 'professional-analysis' || activeTab === 'forensic-viewer'
                    ? 'bg-[#1e1b18] text-white shadow-xs'
                    : 'text-stone-800 hover:bg-[#ded1bc]/50 hover:text-stone-950'
                }`}
              >
                <Shield className="w-3.5 h-3.5 text-red-600" />
                <span>Investigator Suite</span>
              </button>
            )}

            <button
              onClick={() => handleNavClick('report')}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-bold transition-all cursor-pointer ${
                activeTab === 'report'
                  ? 'bg-[#1e1b18] text-white shadow-xs'
                  : 'text-stone-800 hover:bg-[#ded1bc]/50 hover:text-stone-950'
              }`}
            >
              <FileText className="w-3.5 h-3.5 text-amber-600" />
              <span>Reports</span>
            </button>
          </nav>

          {/* Right Utilities Icons */}
          <div className="flex items-center gap-2 sm:gap-2.5">
            {/* User Role Toggle Button */}
            <button
              onClick={() => setUserRole(userRole === 'normal' ? 'investigator' : 'normal')}
              className="p-2.5 rounded-xl text-stone-800 transition-all cursor-pointer shrink-0 flex items-center gap-1.5 text-xs font-bold"
              style={{
                background: userRole === 'investigator' ? 'rgba(20, 18, 15, 0.85)' : 'rgba(230, 220, 200, 0.45)',
                color: userRole === 'investigator' ? '#fef3c7' : '#292524',
                border: '1px solid rgba(200, 180, 140, 0.40)',
              }}
              title={`Role: ${userRole === 'investigator' ? 'Investigator' : 'Standard User'}`}
            >
              {userRole === 'investigator' ? (
                <>
                  <Shield className="w-4 h-4 text-red-400" />
                  <span className="hidden sm:inline">Investigator</span>
                </>
              ) : (
                <>
                  <UserCheck className="w-4 h-4 text-stone-700" />
                  <span className="hidden sm:inline">Standard</span>
                </>
              )}
            </button>

            {/* Notifications Button */}
            <button
              className="p-2.5 text-stone-700 hover:text-stone-950 rounded-xl transition-colors cursor-pointer relative"
              style={{ background: 'rgba(230, 220, 200, 0.45)', border: '1px solid rgba(200, 185, 155, 0.40)' }}
              title="Notifications"
            >
              <Bell className="w-4 h-4" />
              <span className="absolute top-2 right-2 w-2 h-2 rounded-full bg-red-600" />
            </button>

            {/* Account Settings Button */}
            <button
              onClick={() => handleNavClick('settings')}
              className="p-2.5 text-stone-700 hover:text-stone-950 rounded-xl transition-colors cursor-pointer"
              style={{
                background: activeTab === 'settings' ? 'rgba(20, 18, 15, 0.85)' : 'rgba(230, 220, 200, 0.45)',
                color: activeTab === 'settings' ? '#ffffff' : undefined,
                border: '1px solid rgba(200, 185, 155, 0.40)',
              }}
              title="Account Settings"
            >
              <Settings className="w-4 h-4" />
            </button>

            {/* Help / Trust Disclaimer Button */}
            <button
              onClick={() => setShowTrustModal(true)}
              className="p-2.5 text-stone-700 hover:text-stone-950 rounded-xl transition-colors cursor-pointer"
              style={{ background: 'rgba(230, 220, 200, 0.45)', border: '1px solid rgba(200, 185, 155, 0.40)' }}
              title="Trust & Forensic Principles"
            >
              <HelpCircle className="w-4 h-4" />
            </button>

            {/* Menu Toggle Button (For Mobile/Tablet) */}
            <button
              onClick={() => setIsMobileOpen(!isMobileOpen)}
              className="p-2.5 rounded-xl text-stone-800 cursor-pointer lg:hidden"
              style={{ background: 'rgba(230, 220, 200, 0.45)', border: '1px solid rgba(200, 185, 155, 0.40)' }}
              aria-label="Toggle Menu"
            >
              {isMobileOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
            </button>
          </div>
        </div>

        {/* Dropdown Navigation Menu for Mobile */}
        {isMobileOpen && (
          <div
            className="mt-2 rounded-2xl p-3 shadow-xl space-y-1.5 transition-all lg:hidden"
            style={{
              background: 'rgba(255, 252, 244, 0.95)',
              backdropFilter: 'blur(22px) saturate(130%)',
              WebkitBackdropFilter: 'blur(22px) saturate(130%)',
              border: '1px solid rgba(255, 255, 255, 0.60)',
            }}
          >
            <div className="px-2 py-1 text-[10px] font-mono font-bold text-stone-500 uppercase tracking-wider">
              Navigation Menu
            </div>
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => handleNavClick(item.id)}
                  className={`w-full flex items-center gap-3 px-3 py-2 rounded-xl text-xs font-bold transition-all cursor-pointer ${
                    isActive ? 'text-white' : 'text-stone-800 hover:bg-[#ede4d4]/60'
                  }`}
                  style={
                    isActive ? { background: 'rgba(20, 18, 15, 0.85)' } : {}
                  }
                >
                  <Icon
                    className={`w-4 h-4 ${
                      isActive ? 'text-red-400' : 'text-stone-600'
                    }`}
                  />
                  <span>{item.label}</span>
                </button>
              );
            })}

            {userRole === 'investigator' && (
              <button
                onClick={() => handleNavClick('professional-analysis')}
                className={`w-full flex items-center gap-3 px-3 py-2 rounded-xl text-xs font-bold transition-all cursor-pointer ${
                  activeTab === 'professional-analysis' || activeTab === 'forensic-viewer'
                    ? 'text-white'
                    : 'text-stone-800 hover:bg-[#ede4d4]/60'
                }`}
                style={
                  activeTab === 'professional-analysis' || activeTab === 'forensic-viewer'
                    ? { background: 'rgba(20, 18, 15, 0.85)' }
                    : {}
                }
              >
                <Shield className="w-4 h-4 text-red-700" />
                <span>Professional Investigator Suite</span>
              </button>
            )}

            <button
              onClick={() => handleNavClick('report')}
              className={`w-full flex items-center gap-3 px-3 py-2 rounded-xl text-xs font-bold transition-all cursor-pointer ${
                activeTab === 'report' ? 'text-white' : 'text-stone-800 hover:bg-[#ede4d4]/60'
              }`}
              style={activeTab === 'report' ? { background: 'rgba(20, 18, 15, 0.85)' } : {}}
            >
              <FileText className="w-4 h-4 text-amber-600" />
              <span>Forensic Reports</span>
            </button>
          </div>
        )}
      </header>

      {/* Trust Principles Modal */}
      {showTrustModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 glass-modal-bg">
          <div
            className="rounded-2xl max-w-lg w-full p-6 shadow-2xl space-y-4"
            style={{
              background: 'rgba(250, 245, 232, 0.92)',
              backdropFilter: 'blur(22px) saturate(120%)',
              WebkitBackdropFilter: 'blur(22px) saturate(120%)',
              border: '1px solid rgba(255, 255, 255, 0.50)',
            }}
          >
            <div
              className="flex items-center justify-between pb-3"
              style={{ borderBottom: '1px solid rgba(200, 185, 155, 0.30)' }}
            >
              <div className="flex items-center gap-2">
                <Shield className="w-5 h-5 text-red-700" />
                <h3 className="font-serif font-bold text-stone-950 text-base">
                  ForensIQ Trust &amp; Provenance Principles
                </h3>
              </div>
              <button
                onClick={() => setShowTrustModal(false)}
                className="text-stone-400 hover:text-stone-700 text-sm font-bold cursor-pointer"
              >
                ✕
              </button>
            </div>

            <div className="space-y-3 text-xs text-stone-700 leading-relaxed">
              <p
                className="p-3 rounded-xl text-stone-800"
                style={{
                  background: 'rgba(240, 228, 200, 0.45)',
                  border: '1px solid rgba(200, 180, 140, 0.35)',
                }}
              >
                <strong>Probabilistic Intelligence:</strong> AI detection and media forensics are
                probabilistic signals. ForensIQ does not issue absolute 100% guarantees.
              </p>
            </div>

            <div
              className="pt-2 flex justify-end"
              style={{ borderTop: '1px solid rgba(200, 185, 155, 0.30)' }}
            >
              <button
                onClick={() => setShowTrustModal(false)}
                className="px-4 py-2 rounded-xl text-xs font-bold cursor-pointer text-white transition-colors hover:opacity-90"
                style={{ background: 'rgba(20, 18, 15, 0.85)' }}
              >
                I Understand
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};