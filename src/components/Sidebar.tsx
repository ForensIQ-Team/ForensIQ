import React from 'react';
import {
  ShieldCheck,
  Search,
  Lock,
  History,
  FileSearch,
  FileText,
  Settings,
  Shield,
  Layers,
  Sparkles,
  Code2,
} from 'lucide-react';
import { NavItem, UserRole } from '../types';

interface SidebarProps {
  activeTab: NavItem;
  setActiveTab: (tab: NavItem) => void;
  userRole: UserRole;
  setUserRole: (role: UserRole) => void;
  isMobileOpen: boolean;
  setIsMobileOpen: (open: boolean) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  setActiveTab,
  userRole,
  setUserRole,
  isMobileOpen,
  setIsMobileOpen,
}) => {
  const handleNavClick = (tab: NavItem) => {
    setActiveTab(tab);
    setIsMobileOpen(false);
  };

  const navBtn = (tab: NavItem, label: string, Icon: any, accent?: boolean) => {
    const isActive = activeTab === tab || (tab === 'professional-analysis' && activeTab === 'forensic-viewer');
    return (
      <button
        onClick={() => handleNavClick(tab)}
        className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
          isActive
            ? 'nav-item-active'
            : 'text-stone-800 nav-item-hover hover:text-stone-950'
        }`}
      >
        <Icon className={`w-4 h-4 flex-shrink-0 ${isActive ? 'text-red-400' : accent ? 'text-red-700' : 'text-stone-600'}`} />
        <span>{label}</span>
      </button>
    );
  };

  return (
    <>
      {/* Mobile Backdrop */}
      {isMobileOpen && (
        <div
          className="fixed inset-0 z-40 lg:hidden"
          style={{ background: 'rgba(15, 12, 8, 0.45)', backdropFilter: 'blur(4px)' }}
          onClick={() => setIsMobileOpen(false)}
        />
      )}

      {/* Sidebar Drawer — liquid glass, shader visible through it */}
      <aside
        className={`fixed top-0 bottom-0 left-0 z-50 w-64 glass-sidebar flex flex-col justify-between transition-transform duration-200 ease-in-out lg:translate-x-0 ${
          isMobileOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="flex flex-col flex-1 overflow-y-auto">
          {/* Logo Brand Header */}
          <div
            className="p-5 flex items-center justify-between cursor-pointer group"
            style={{ borderBottom: '1px solid rgba(200, 185, 155, 0.25)' }}
            onClick={() => handleNavClick('home')}
          >
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg flex items-center justify-center text-amber-200 shadow-sm transition-colors"
                style={{ background: 'rgba(20, 18, 15, 0.80)' }}>
                <ShieldCheck className="w-5 h-5 text-red-500" />
              </div>
              <div>
                <div className="flex items-center gap-1.5">
                  <span className="font-serif font-black text-lg text-stone-950 tracking-tight">ForensIQ</span>
                  <span className="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded uppercase"
                    style={{ background: 'rgba(220, 200, 155, 0.40)', border: '1px solid rgba(180, 155, 100, 0.35)', color: '#4a3a20' }}>
                    v2.4
                  </span>
                </div>
                <p className="text-[11px] text-stone-600 font-medium leading-none mt-0.5">
                  Media Trust &amp; Forensics
                </p>
              </div>
            </div>
          </div>

          {/* Main Navigation */}
          <nav className="px-3 py-3 space-y-1">
            <div className="px-3 py-1 text-[10px] font-mono font-bold text-stone-500 uppercase tracking-wider">
              Navigation
            </div>
            {navBtn('home', 'Platform Overview', Layers)}
            {navBtn('check-media', 'Check Media', FileSearch)}
            {navBtn('protect-image', 'Protect Image', Lock)}
            {navBtn('find-misuse', 'Find Misuse', Search)}
            {navBtn('history', 'History', History)}
          </nav>

          {/* Investigator Suite */}
          <div className="px-3 pt-3 pb-1 mt-2 space-y-1"
            style={{ borderTop: '1px solid rgba(200, 185, 155, 0.22)' }}>
            <div className="px-3 py-1 text-[10px] font-mono font-bold text-stone-500 uppercase tracking-wider">
              Investigator Suite
            </div>

            {navBtn('investigator-workspace', 'Investigator Hub', Shield, true)}

            <button
              onClick={() => {
                if (userRole === 'normal') setUserRole('investigator');
                handleNavClick('professional-analysis');
              }}
              className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                activeTab === 'professional-analysis' || activeTab === 'forensic-viewer'
                  ? 'nav-item-active'
                  : 'text-stone-800 nav-item-hover hover:text-stone-950'
              }`}
            >
              <div className="flex items-center gap-3">
                <Shield className={`w-4 h-4 ${activeTab === 'professional-analysis' || activeTab === 'forensic-viewer' ? 'text-red-400' : 'text-red-700'}`} />
                <span>Professional Analysis</span>
              </div>
            </button>

            {navBtn('report', 'Forensic Reports', FileText)}
          </div>
        </div>

        {/* Footer: Account Settings */}
        <div className="p-3" style={{ borderTop: '1px solid rgba(200, 185, 155, 0.22)' }}>
          <button
            onClick={() => handleNavClick('settings')}
            className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
              activeTab === 'settings'
                ? 'nav-item-active'
                : 'text-stone-800 nav-item-hover hover:text-stone-950'
            }`}
          >
            <div className="flex items-center gap-2.5">
              <Settings className={`w-4 h-4 ${activeTab === 'settings' ? 'text-stone-300' : 'text-stone-600'}`} />
              <span>Account Settings</span>
            </div>
            <Sparkles className="w-3.5 h-3.5 text-stone-400" />
          </button>
        </div>
      </aside>
    </>
  );
};
