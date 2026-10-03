import React, { useState } from 'react';
import {
  Menu,
  X,
  Search,
  Shield,
  UserCheck,
  HelpCircle,
  Bell,
  Layers,
  FileSearch,
  Lock,
  History,
  Settings,
  ShieldCheck,
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
  onQuickSearch?: (query: string) => void;
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
  const [searchQuery, setSearchQuery] = useState('');

  const handleNavClick = (tab: NavItem) => {
    setActiveTab(tab);
    setIsMobileOpen(false);
  };

  const navItems: { id: NavItem; label: string; icon: any }[] = [
    { id: 'home', label: 'Platform Overview', icon: Layers },
    { id: 'check-media', label: 'Check Media', icon: FileSearch },
    { id: 'protect-image', label: 'Protect Image', icon: Lock },
    { id: 'find-misuse', label: 'Find Misuse', icon: Search },
    { id: 'history', label: 'geography', icon: History },
  ];

  return (
    <>
      {/* Floating Top Glass Navbar Wrapper */}
      <header className="sticky top-3 z-40 max-w-[1700px] mx-auto w-full px-2 sm:px-4 mb-4">
        <div
          className="rounded-2xl px-4 py-2.5 flex items-center justify-between transition-all duration-200"
          style={{
            background: 'rgba(255, 252, 244, 0.32)',
            backdropFilter: 'blur(20px) saturate(130%)',
            WebkitBackdropFilter: 'blur(20px) saturate(130%)',
            border: '1px solid rgba(255, 255, 255, 0.42)',
            boxShadow: '0 8px 32px rgba(40, 30, 15, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.5)',
          }}
        >
          {/* Brand Logo & Title */}
          <div
            className="flex items-center gap-2.5 cursor-pointer group shrink-0"
            onClick={() => handleNavClick('home')}
          >
            <div
              className="w-9 h-9 rounded-xl flex items-center justify-center text-amber-200 shadow-sm transition-transform group-hover:scale-105"
              style={{ background: 'rgba(20, 18, 15, 0.85)' }}
            >
              <ShieldCheck className="w-5 h-5 text-red-500" />
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <span className="font-serif font-black text-lg text-stone-950 tracking-tight leading-none">
                  FORENSIQ
                </span>
                <span
                  className="text-[9px] font-mono font-bold px-1.5 py-0.5 rounded uppercase leading-none"
                  style={{
                    background: 'rgba(220, 200, 155, 0.40)',
                    border: '1px solid rgba(180, 155, 100, 0.35)',
                    color: '#4a3a20',
                  }}
                >
                  v2.4
                </span>
              </div>
              <p className="hidden sm:block text-[10px] text-stone-600 font-semibold leading-none mt-0.5">
                Forensic Workstation
              </p>
            </div>
          </div>

          {/* Desktop Center Navigation Pills */}
          <nav className="hidden lg:flex items-center gap-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => handleNavClick(item.id)}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-bold transition-all cursor-pointer ${isActive
                    ? 'text-white shadow-xs'
                    : 'text-stone-800 hover:text-stone-950'
                    }`}
                  style={
                    isActive
                      ? {
                        background: 'rgba(20, 18, 15, 0.85)',
                        boxShadow: '0 2px 10px rgba(0, 0, 0, 0.15)',
                      }
                      : { background: 'transparent' }
                  }
                >
                  <Icon
                    className={`w-3.5 h-3.5 ${isActive ? 'text-red-400' : 'text-stone-600'
                      }`}
                  />
                  <span>{item.label}</span>
                </button>
              );
            })}

            {/* Investigator View Link if investigator mode */}
            {userRole === 'investigator' && (
              <button
                onClick={() => handleNavClick('professional-analysis')}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-bold transition-all cursor-pointer ${activeTab === 'professional-analysis' || activeTab === 'forensic-viewer'
                  ? 'text-white shadow-xs'
                  : 'text-stone-800 hover:text-stone-950'
                  }`}
                style={
                  activeTab === 'professional-analysis' || activeTab === 'forensic-viewer'
                    ? { background: 'rgba(20, 18, 15, 0.85)' }
                    : { background: 'rgba(196, 30, 30, 0.12)', border: '1px solid rgba(196, 30, 30, 0.30)' }
                }
              >
                <Shield className="w-3.5 h-3.5 text-red-600" />
                <span>Investigator Suite</span>
              </button>
            )}
          </nav>

          {/* Right Utilities */}
          <div className="flex items-center gap-2 sm:gap-2.5">
            {/* Quick Search Input */}
            <div className="hidden md:flex items-center relative w-64 lg:w-72">
              <Search className="w-3.5 h-3.5 absolute left-3 text-stone-500 pointer-events-none" />
              <input
                type="text"
                placeholder="Quick search..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-8 pr-3 py-1 text-xs rounded-xl focus:outline-none text-stone-800 placeholder-stone-400"
                style={{
                  background: 'rgba(248, 242, 230, 0.45)',
                  border: '1px solid rgba(210, 196, 170, 0.50)',
                }}
              />
            </div>

            {/* Role Switcher Badge */}
            <button
              onClick={() => setUserRole(userRole === 'normal' ? 'investigator' : 'normal')}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl text-xs font-bold transition-all cursor-pointer shrink-0"
              style={
                userRole === 'investigator'
                  ? {
                    background: 'rgba(20, 18, 15, 0.85)',
                    border: '1px solid rgba(80, 60, 40, 0.60)',
                    color: '#fef3c7',
                  }
                  : {
                    background: 'rgba(237, 225, 200, 0.45)',
                    border: '1px solid rgba(200, 180, 140, 0.50)',
                    color: '#292524',
                  }
              }
              title="Click to toggle user role mode"
            >
              {userRole === 'investigator' ? (
                <>
                  <Shield className="w-3.5 h-3.5 text-red-400" />
                  <span className="hidden sm:inline">Investigator</span>
                </>
              ) : (
                <>
                  <UserCheck className="w-3.5 h-3.5 text-stone-600" />
                  <span className="hidden sm:inline">Standard User</span>
                </>
              )}
            </button>

            {/* Notifications Button */}
            <button
              className="p-2 text-stone-700 hover:text-stone-950 rounded-xl transition-colors cursor-pointer relative"
              style={{ background: 'rgba(230, 220, 200, 0.35)', border: '1px solid rgba(200, 185, 155, 0.30)' }}
              title="Notifications"
            >
              <Bell className="w-4 h-4" />
              <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-red-600" />
            </button>

            {/* Account Settings Button */}
            <button
              onClick={() => handleNavClick('settings')}
              className={`p-2 text-stone-700 hover:text-stone-950 rounded-xl transition-colors cursor-pointer ${activeTab === 'settings' ? 'text-stone-950 font-bold' : ''
                }`}
              style={{
                background: activeTab === 'settings' ? 'rgba(20, 18, 15, 0.85)' : 'rgba(230, 220, 200, 0.35)',
                color: activeTab === 'settings' ? '#ffffff' : undefined,
                border: '1px solid rgba(200, 185, 155, 0.30)',
              }}
              title="Account Settings"
            >
              <Settings className="w-4 h-4" />
            </button>

            {/* Trust Info Disclaimer Button */}
            <button
              onClick={() => setShowTrustModal(true)}
              className="p-2 text-stone-700 hover:text-stone-950 rounded-xl transition-colors cursor-pointer"
              style={{ background: 'rgba(230, 220, 200, 0.35)', border: '1px solid rgba(200, 185, 155, 0.30)' }}
              title="Trust & Forensic Principles"
            >
              <HelpCircle className="w-4 h-4" />
            </button>

            {/* Mobile Navigation Toggle Button */}
            <button
              onClick={() => setIsMobileOpen(!isMobileOpen)}
              className="p-2 rounded-xl text-stone-800 lg:hidden cursor-pointer"
              style={{ background: 'rgba(230, 220, 200, 0.45)', border: '1px solid rgba(200, 185, 155, 0.35)' }}
              aria-label="Toggle Mobile Menu"
            >
              {isMobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>

        {/* Mobile Compact Glass Dropdown */}
        {isMobileOpen && (
          <div
            className="lg:hidden mt-2 rounded-2xl p-4 shadow-xl space-y-2 transition-all"
            style={{
              background: 'rgba(255, 252, 244, 0.92)',
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
                  className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-xs font-bold transition-all cursor-pointer ${isActive ? 'text-white' : 'text-stone-800 hover:bg-[#ede4d4]/60'
                    }`}
                  style={
                    isActive ? { background: 'rgba(20, 18, 15, 0.85)' } : {}
                  }
                >
                  <Icon
                    className={`w-4 h-4 ${isActive ? 'text-red-400' : 'text-stone-600'
                      }`}
                  />
                  <span>{item.label}</span>
                </button>
              );
            })}

            {/* Investigator link in mobile menu */}
            <button
              onClick={() => {
                if (userRole === 'normal') setUserRole('investigator');
                handleNavClick('professional-analysis');
              }}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-xs font-bold transition-all cursor-pointer ${activeTab === 'professional-analysis' || activeTab === 'forensic-viewer'
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

            {/* Reports link */}
            <button
              onClick={() => handleNavClick('report')}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-xs font-bold transition-all cursor-pointer ${activeTab === 'report' ? 'text-white' : 'text-stone-800 hover:bg-[#ede4d4]/60'
                }`}
              style={activeTab === 'report' ? { background: 'rgba(20, 18, 15, 0.85)' } : {}}
            >
              <FileText className="w-4 h-4 text-amber-600" />
              <span>Forensic Reports</span>
            </button>
          </div>
        )}
      </header>

      {/* Trust & Transparency Principles Modal */}
      {showTrustModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 glass-modal-bg">
          <div
            className="rounded-2xl max-w-lg w-full p-6 shadow-2xl space-y-4"
            style={{
              background: 'rgba(250, 245, 232, 0.85)',
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
                probabilistic signals. ForensIQ does not issue absolute 100% guarantees; we present
                mathematical likelihoods and multi-model consensus.
              </p>
              <div className="space-y-2">
                <div className="flex gap-2">
                  <span className="font-bold text-stone-900 min-w-[80px]">Protection:</span>
                  <span>
                    EOT-based adversarial perturbation adds robust ownership signals against unexpected
                    web transformations.
                  </span>
                </div>
                <div className="flex gap-2">
                  <span className="font-bold text-stone-900 min-w-[80px]">Discovery:</span>
                  <span>
                    Misuse detection searches indexed public web sources and registered hash
                    registries.
                  </span>
                </div>
                <div className="flex gap-2">
                  <span className="font-bold text-stone-900 min-w-[80px]">Roles:</span>
                  <span>
                    Standard users receive intuitive authenticity ratings; verified investigators
                    access raw ELA maps, spectral metrics, and evidence binders.
                  </span>
                </div>
              </div>
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
