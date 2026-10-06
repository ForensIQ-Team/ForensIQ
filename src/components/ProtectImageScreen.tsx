import React, { useState } from 'react';
import {
  Lock,
  Upload,
  CheckCircle2,
  ShieldCheck,
  Download,
  Info,
  RefreshCw,
  Copy,
  ExternalLink,
} from 'lucide-react';
import { ProtectionRecord, NavItem } from '../types';
import { MOCK_PROTECTIONS, SAMPLE_IMAGES } from '../data/mockData';

interface ProtectImageScreenProps {
  setActiveTab: (tab: NavItem) => void;
}

const glassPanelStyle: React.CSSProperties = {
  background: 'rgba(255, 252, 244, 0.78)',
  backdropFilter: 'blur(20px) saturate(130%)',
  WebkitBackdropFilter: 'blur(20px) saturate(130%)',
  border: '1px solid rgba(255, 255, 255, 0.7)',
  boxShadow: '0 8px 32px rgba(40, 30, 15, 0.08)',
};

const glassCardStyle: React.CSSProperties = {
  background: 'rgba(245, 238, 222, 0.45)',
  border: '1px solid rgba(226, 216, 195, 0.7)',
};

const glassInputStyle: React.CSSProperties = {
  background: 'rgba(255, 255, 255, 0.65)',
  border: '1px solid rgba(226, 216, 195, 0.8)',
};

export const ProtectImageScreen: React.FC<ProtectImageScreenProps> = ({ setActiveTab }) => {
  const [protectionLevel, setProtectionLevel] = useState<'EOT' | 'Hybrid' | 'Metadata'>('EOT');
  const [processState, setProcessState] = useState<'idle' | 'protecting' | 'complete'>('complete');
  const [activeRecord, setActiveRecord] = useState<ProtectionRecord>(MOCK_PROTECTIONS[0]);
  const [copiedId, setCopiedId] = useState<boolean>(false);

  const handleStartProtection = () => {
    setProcessState('protecting');
    setTimeout(() => {
      const newRecord: ProtectionRecord = {
        id: `PROT-${Math.floor(100 + Math.random() * 900)}`,
        registrationId: `REG-2026-${Math.floor(100000 + Math.random() * 900000)}`,
        originalName: 'My_Uploaded_Image_Protected.jpg',
        mediaUrl: SAMPLE_IMAGES.authenticPortrait,
        protectedDate: new Date().toISOString().replace('T', ' ').substring(0, 16) + ' UTC',
        protectionType:
          protectionLevel === 'EOT'
            ? 'EOT Adversarial'
            : protectionLevel === 'Hybrid'
            ? 'Robust Hybrid'
            : 'Metadata Provenance',
        watermarkSignal: 'Active',
        pHash: 'b3a7d2c8e1f9a0c4',
        clipEmbeddingId: 'CLIP-EMB-9904-81',
        status: 'Registered & Protected',
      };
      setActiveRecord(newRecord);
      setProcessState('complete');
    }, 1500);
  };

  const handleCopyRegistration = () => {
    navigator.clipboard.writeText(activeRecord.registrationId);
    setCopiedId(true);
    setTimeout(() => setCopiedId(false), 2000);
  };

  return (
    <div className="space-y-8 max-w-[1700px] mx-auto pb-12">
      {/* Title & Header */}
      <div>
        <h2 className="text-2xl font-bold text-stone-950 tracking-tight">Protect Your Image</h2>
        <p className="text-stone-600 text-xs sm:text-sm mt-1">
          Add invisible protection and a media fingerprint before you share your image publicly.
        </p>
      </div>

      {/* Main Protection Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Upload & Configuration Section */}
        <div className="lg:col-span-2 space-y-6">
          {/* Step 1: Upload / Selected Media */}
          <div className="rounded-2xl p-6 space-y-4" style={glassPanelStyle}>
            <h3 className="font-bold text-stone-950 text-sm flex items-center gap-2 border-b border-[#e2d8c3]/80 pb-2">
              <Upload className="w-4 h-4 text-red-700" />
              <span>1. Upload or Select Original Image</span>
            </h3>

            <div className="border-2 border-dashed border-[#d8ccb6] rounded-xl p-6 text-center flex flex-col items-center justify-center" style={glassCardStyle}>
              <img
                src={activeRecord.mediaUrl}
                alt="Original"
                className="w-32 h-32 object-cover rounded-lg border border-[#d8ccb6] shadow-xs mb-3"
              />
              <p className="text-xs font-bold text-stone-950">{activeRecord.originalName}</p>
              <p className="text-[11px] text-stone-500 mt-0.5">3.42 MB • 3840 x 2560 px</p>

              <label
                className="mt-3 inline-flex items-center gap-2 px-3 py-1.5 text-stone-800 text-xs font-semibold rounded-lg transition-colors cursor-pointer"
                style={glassInputStyle}
              >
                <span>Choose Different Image</span>
                <input
                  type="file"
                  accept="image/*"
                  onChange={() => handleStartProtection()}
                  className="hidden"
                />
              </label>
            </div>
          </div>

          {/* Step 2: Select Protection Intensity */}
          <div className="rounded-2xl p-6 space-y-4" style={glassPanelStyle}>
            <h3 className="font-bold text-stone-950 text-sm flex items-center gap-2 border-b border-[#e2d8c3]/80 pb-2">
              <Lock className="w-4 h-4 text-red-700" />
              <span>2. Select Protection Configuration</span>
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <button
                onClick={() => setProtectionLevel('EOT')}
                style={protectionLevel !== 'EOT' ? glassCardStyle : undefined}
                className={`p-4 rounded-xl text-left transition-all cursor-pointer ${
                  protectionLevel === 'EOT'
                    ? 'border border-stone-950 bg-[#1e1b18] text-white shadow-xs'
                    : 'text-stone-800 hover:opacity-90'
                }`}
              >
                <div className="font-bold text-xs">EOT Adversarial</div>
                <p
                  className={`text-[11px] mt-1 leading-snug ${
                    protectionLevel === 'EOT' ? 'text-amber-200' : 'text-stone-600'
                  }`}
                >
                  Applies expectation-over-transformation perturbation to confuse generative AI models.
                </p>
              </button>

              <button
                onClick={() => setProtectionLevel('Hybrid')}
                style={protectionLevel !== 'Hybrid' ? glassCardStyle : undefined}
                className={`p-4 rounded-xl text-left transition-all cursor-pointer ${
                  protectionLevel === 'Hybrid'
                    ? 'border border-stone-950 bg-[#1e1b18] text-white shadow-xs'
                    : 'text-stone-800 hover:opacity-90'
                }`}
              >
                <div className="font-bold text-xs">Robust Hybrid</div>
                <p
                  className={`text-[11px] mt-1 leading-snug ${
                    protectionLevel === 'Hybrid' ? 'text-amber-200' : 'text-stone-600'
                  }`}
                >
                  Combines invisible pHash frequency watermarking with CLIP vector embedding.
                </p>
              </button>

              <button
                onClick={() => setProtectionLevel('Metadata')}
                style={protectionLevel !== 'Metadata' ? glassCardStyle : undefined}
                className={`p-4 rounded-xl text-left transition-all cursor-pointer ${
                  protectionLevel === 'Metadata'
                    ? 'border border-stone-950 bg-[#1e1b18] text-white shadow-xs'
                    : 'text-stone-800 hover:opacity-90'
                }`}
              >
                <div className="font-bold text-xs">Provenance EXIF</div>
                <p
                  className={`text-[11px] mt-1 leading-snug ${
                    protectionLevel === 'Metadata' ? 'text-amber-200' : 'text-stone-600'
                  }`}
                >
                  Appends C2PA cryptographically signed ownership manifest to image headers.
                </p>
              </button>
            </div>

            <button
              onClick={handleStartProtection}
              disabled={processState === 'protecting'}
              className="w-full py-3 bg-[#1e1b18] hover:bg-stone-900 text-white text-xs font-semibold rounded-xl flex items-center justify-center gap-2 transition-colors cursor-pointer shadow-xs disabled:opacity-50"
            >
              {processState === 'protecting' ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin text-amber-300" />
                  <span>Embedding Ownership Signals &amp; Fingerprints...</span>
                </>
              ) : (
                <>
                  <ShieldCheck className="w-4 h-4 text-amber-300" />
                  <span>Apply Invisible Protection &amp; Register Media</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Right Info & Protection Checklist Panel */}
        <div className="space-y-6">
          {/* How Protection Works Card */}
          <div className="rounded-2xl p-5 space-y-4" style={glassPanelStyle}>
            <h4 className="font-bold text-stone-950 text-xs uppercase tracking-wider border-b border-[#e2d8c3]/80 pb-2">
              PROTECTION SUMMARY
            </h4>

            <p className="text-xs text-stone-600 leading-relaxed p-3 rounded-lg" style={glassInputStyle}>
              ForensIQ applies transformation-robust adversarial protection, adds an invisible ownership signal, and creates a media fingerprint.
            </p>

            <div className="space-y-2.5 text-xs text-stone-700">
              <div className="flex items-start gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
                <span>EOT-based adversarial perturbation</span>
              </div>
              <div className="flex items-start gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
                <span>Invisible ownership watermark</span>
              </div>
              <div className="flex items-start gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
                <span>pHash perceptual fingerprint</span>
              </div>
              <div className="flex items-start gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
                <span>CLIP neural media embedding</span>
              </div>
              <div className="flex items-start gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
                <span>Ownership registry registration</span>
              </div>
            </div>

            {/* Responsible Transparency Disclaimer */}
            <div className="p-3 bg-amber-50/80 border border-amber-200 rounded-xl text-[11px] text-amber-900 space-y-1">
              <div className="font-bold flex items-center gap-1.5">
                <Info className="w-3.5 h-3.5 text-amber-700" />
                <span>Responsible Protection Notice</span>
              </div>
              <p className="leading-snug">
                Designed to make malicious AI regeneration harder and help identify later copies or modified versions.
              </p>
            </div>
          </div>

          {/* Registration Record Box */}
          {processState === 'complete' && (
            <div className="rounded-2xl p-5 space-y-4" style={glassPanelStyle}>
              <div className="flex items-center justify-between border-b border-[#e2d8c3]/80 pb-2">
                <span className="text-[11px] font-bold text-stone-500 uppercase tracking-wider">
                  REGISTRATION RECORD
                </span>
                <span className="px-2 py-0.5 rounded bg-emerald-50/80 text-emerald-800 text-[10px] font-bold border border-emerald-200">
                  {activeRecord.status}
                </span>
              </div>

              <div className="space-y-2 text-xs">
                <div className="p-2.5 rounded-lg flex items-center justify-between" style={glassInputStyle}>
                  <div>
                    <p className="text-[10px] text-stone-500 font-bold uppercase">Registration ID</p>
                    <p className="font-mono font-bold text-stone-900">{activeRecord.registrationId}</p>
                  </div>
                  <button
                    onClick={handleCopyRegistration}
                    className="p-1.5 text-stone-500 hover:text-stone-900 hover:bg-[#e8decb]/60 rounded cursor-pointer"
                    title="Copy ID"
                  >
                    <Copy className="w-3.5 h-3.5" />
                  </button>
                </div>
                {copiedId && (
                  <p className="text-[10px] text-emerald-700 font-semibold text-right">Copied to clipboard!</p>
                )}

                <div className="py-1 flex justify-between text-stone-600">
                  <span>Type:</span>
                  <span className="font-semibold text-stone-900">{activeRecord.protectionType}</span>
                </div>
                <div className="py-1 flex justify-between text-stone-600">
                  <span>Registered Date:</span>
                  <span className="text-stone-800">{activeRecord.protectedDate}</span>
                </div>
                <div className="py-1 flex justify-between text-stone-600">
                  <span>pHash Signal:</span>
                  <span className="font-mono text-stone-800">{activeRecord.pHash}</span>
                </div>
              </div>

              <div className="pt-2 space-y-2">
                <button
                  onClick={() => alert('Downloading protected file...')}
                  className="w-full py-2.5 bg-[#1e1b18] hover:bg-stone-900 text-white text-xs font-semibold rounded-xl flex items-center justify-center gap-2 transition-colors cursor-pointer shadow-xs"
                >
                  <Download className="w-4 h-4 text-amber-300" />
                  <span>Download Protected Image</span>
                </button>

                <button
                  onClick={() => setActiveTab('find-misuse')}
                  className="w-full py-2 text-stone-800 text-xs font-semibold rounded-xl flex items-center justify-center gap-1.5 transition-colors cursor-pointer"
                  style={glassInputStyle}
                >
                  <span>Search Misuse for This Item</span>
                  <ExternalLink className="w-3.5 h-3.5 text-stone-500" />
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};