import React, { useState } from 'react';
import { NavItem, UserRole, ForensicAnalysis, DetectionResult, ForensicReportData } from './types';
import { MOCK_ANALYSES } from './data/mockData';
import { generateReportFromDetection } from './services/reportGenerator';
import { ShaderBackground } from './components/ShaderBackground';
import { Header } from './components/Header';
import { LandingPage } from './components/LandingPage';
import { HomeScreen } from './components/HomeScreen';
import { CheckMediaScreen } from './components/CheckMediaScreen';
import { ForensicViewer } from './components/ForensicViewer';
import { ProtectImageScreen } from './components/ProtectImageScreen';
import { FindMisuseScreen } from './components/FindMisuseScreen';
import { ProfessionalAnalysisScreen } from './components/ProfessionalAnalysisScreen';
import { ReportScreen } from './components/ReportScreen';
import { HistoryScreen } from './components/HistoryScreen';
import { AccountSettingsModal } from './components/AccountSettingsModal';
import { MagnifyingCursor } from './components/landing/MagnifyingCursor';

export default function App() {
  const [activeTab, setActiveTab] = useState<NavItem>('landing');
  const [userRole, setUserRole] = useState<UserRole>('normal');
  const [isMobileOpen, setIsMobileOpen] = useState<boolean>(false);
  const [activeAnalysis, setActiveAnalysis] = useState<ForensicAnalysis>(MOCK_ANALYSES['FQ-8091']);
  const [activeReportData, setActiveReportData] = useState<ForensicReportData | undefined>(undefined);

  // Navigate to Interactive Forensic Viewer with specific analysis item
  const handleOpenViewer = (analysis: ForensicAnalysis) => {
    setActiveAnalysis(analysis);
    setActiveTab('forensic-viewer');
  };

  // Open dynamic forensic report generated from analysis result
  const handleOpenReport = (analysisResult?: DetectionResult) => {
    if (analysisResult) {
      setActiveAnalysis(analysisResult);
      setActiveReportData(generateReportFromDetection(analysisResult));
    } else if (activeAnalysis) {
      setActiveReportData(generateReportFromDetection(activeAnalysis));
    }
    setActiveTab('report');
  };

  // Standalone Landing Page rendering (No ShaderGradient background on Landing Page)
  if (activeTab === 'landing') {
    return (
      <LandingPage
        setActiveTab={setActiveTab}
        setUserRole={setUserRole}
        userRole={userRole}
        onOpenViewer={handleOpenViewer}
      />
    );
  }

  return (
    /* Root wrapper: transparent so ShaderGradient is the true background */
    <div
      className="relative min-h-screen text-stone-950 font-sans flex flex-col antialiased selection:bg-red-900 selection:text-white"
      style={{ background: 'transparent' }}
       >
      <MagnifyingCursor />

      {/* Global ShaderGradient Background — fixed, lowest z-index, pointer-events none */}
      <ShaderBackground />

      {/* Main Application Content Layer — full-width, transparent bg, floating navbar */}
      <div className="relative z-10 flex flex-col flex-1 min-h-screen pt-2" style={{ background: 'transparent' }}>
        {/* Floating Top Glass Navbar */}
        <Header
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          userRole={userRole}
          setUserRole={setUserRole}
          isMobileOpen={isMobileOpen}
          setIsMobileOpen={setIsMobileOpen}
        />

        {/* Full-Width Viewport Container Stage — transparent, shader visible in gaps */}
        <main className="flex-1 w-full max-w-[1700px] mx-auto px-3 sm:px-6 lg:px-8 pb-12" style={{ background: 'transparent' }}>
          {activeTab === 'home' && (
            <HomeScreen
              setActiveTab={setActiveTab}
              userRole={userRole}
              onOpenReport={handleOpenReport}
            />
          )}

          {activeTab === 'check-media' && (
            <CheckMediaScreen
              userRole={userRole}
              setActiveTab={setActiveTab}
              onOpenViewer={handleOpenViewer}
              onOpenReport={handleOpenReport}
            />
          )}

          {activeTab === 'forensic-viewer' && (
            <ForensicViewer
              analysis={activeAnalysis}
              userRole={userRole}
              setActiveTab={setActiveTab}
              onBack={() => setActiveTab('check-media')}
            />
          )}

          {activeTab === 'protect-image' && (
            <ProtectImageScreen setActiveTab={setActiveTab} />
          )}

          {activeTab === 'find-misuse' && (
            <FindMisuseScreen
              userRole={userRole}
              setActiveTab={setActiveTab}
            />
          )}

          {activeTab === 'history' && (
            <HistoryScreen
              userRole={userRole}
              setActiveTab={setActiveTab}
              onOpenReport={handleOpenReport}
            />
          )}

          {activeTab === 'professional-analysis' && (
            <ProfessionalAnalysisScreen
              userRole={userRole}
              setActiveTab={setActiveTab}
              onOpenViewer={handleOpenViewer}
            />
          )}

          {activeTab === 'report' && (
            <ReportScreen
              report={activeReportData || generateReportFromDetection(activeAnalysis)}
              userRole={userRole}
              setActiveTab={setActiveTab}
            />
          )}

          {activeTab === 'settings' && (
            <AccountSettingsModal
              userRole={userRole}
              setUserRole={setUserRole}
            />
          )}
        </main>
      </div>
    </div>
  );
}
