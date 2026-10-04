import React, { useState } from 'react';
import {
  Code2,
  Key,
  Copy,
  CheckCircle2,
  Terminal,
  Building2,
  GraduationCap,
  ArrowRight,
  ShieldCheck,
  FileCheck2,
} from 'lucide-react';

export const ApiIntegrationScreen: React.FC = () => {
  const [copiedKey, setCopiedKey] = useState<boolean>(false);
  const [activeLang, setActiveLang] = useState<'curl' | 'javascript' | 'python'>('javascript');

  const apiKey = 'fiq_live_9984102948102948019482';

  const handleCopyKey = () => {
    navigator.clipboard.writeText(apiKey);
    setCopiedKey(true);
    setTimeout(() => setCopiedKey(false), 2000);
  };

  const codeSnippets = {
    javascript: `import { ForensIQ } from '@forensiq/sdk';

const client = new ForensIQ({ apiKey: 'YOUR_API_KEY' });

// Verify image authenticity & synthetic probability
const result = await client.media.verify({
  fileUrl: 'https://university-portal.edu/submissions/doc_884.jpg',
  models: ['xception', 'vit', 'efficientnet']
});

console.log('Authenticity Score:', result.overallScore); // e.g. 94%
console.log('Is AI Generated:', result.isSynthetic);`,

    python: `from forensiq import ForensIQClient

client = ForensIQClient(api_key="YOUR_API_KEY")

# Verify document or student submission integrity
response = client.media.verify(
    file_path="./student_assignment_2026.pdf",
    include_ela=True
)

print(f"Authenticity Rating: {response.score}%")
print(f"Risk Assessment: {response.risk_level}")`,

    curl: `curl -X POST https://api.forensiq.app/v1/media/verify \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "media_url": "https://university.edu/uploads/receipt.jpg",
    "check_types": ["synthetic", "ela", "metadata"]
  }'`,
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-12">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-stone-950 tracking-tight">API &amp; College Integration Hub</h2>
        <p className="text-stone-950 text-xs sm:text-sm mt-1">
          Integrate ForensIQ media verification, watermarking, and document authentication directly into web platforms, university portals, or enterprise systems.
        </p>
      </div>

      {/* College / Educational Portal Flow Banner */}
      <div className="glass-panel rounded-xl p-6 shadow-xs space-y-4">
        <div className="flex items-center gap-2 border-b border-[#e2d8c3]/80 pb-3">
          <GraduationCap className="w-5 h-5 text-red-700" />
          <h3 className="font-bold text-stone-950 text-base">Educational Institution &amp; College Portal Integration</h3>
        </div>

        <p className="text-xs text-stone-950 leading-relaxed max-w-3xl">
          Universities and colleges embed ForensIQ REST APIs to safeguard academic integrity, prevent AI assignment generation fraud, register faculty research media ownership, and verify tuition payment receipts.
        </p>

        {/* Integration Flow Chart Diagram */}
        <div className="glass-card rounded-xl p-6 text-center">
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 text-xs font-bold">
            <div className="p-3 rounded-lg w-44" style={{ background: 'rgba(248, 242, 228, 0.42)', border: '1px solid rgba(200, 185, 155, 0.38)' }}>
              <Building2 className="w-5 h-5 text-stone-950 mx-auto mb-1" />
              <span>College Portal</span>
              <p className="text-xs text-stone-950 font-normal mt-0.5">Student Submissions</p>
            </div>

            <ArrowRight className="w-5 h-5 text-stone-950 rotate-90 sm:rotate-0" />

            <div className="p-3 rounded-lg border w-48" style={{ background: 'rgba(18, 16, 12, 0.78)', borderColor: 'rgba(80, 60, 40, 0.50)' }} >
              <ShieldCheck className="w-5 h-5 text-red-400 mx-auto mb-1" />
              <span className="text-white">ForensIQ REST API</span>
              <p className="text-xs text-amber-200 font-normal mt-0.5">Automated Analysis</p>
            </div>

            <ArrowRight className="w-5 h-5 text-stone-950 rotate-90 sm:rotate-0" />

            <div className="p-3 rounded-lg w-44" style={{ background: 'rgba(248, 242, 228, 0.42)', border: '1px solid rgba(200, 185, 155, 0.38)' }}>
              <FileCheck2 className="w-5 h-5 text-red-700 mx-auto mb-1" />
              <span>Verified Report</span>
              <p className="text-xs text-stone-950 font-normal mt-0.5">Pass / Flagged Result</p>
            </div>
          </div>
        </div>

        {/* Use Cases Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs pt-2">
          <div className="p-3 glass-card rounded-lg space-y-1">
            <h4 className="font-bold text-stone-950">1. Experiment Submissions</h4>
            <p className="text-stone-950">Detect duplicate image submissions or synthetic AI lab photo generation across student cohorts.</p>
          </div>
          <div className="p-3 glass-card rounded-lg space-y-1">
            <h4 className="font-bold text-stone-950">2. Research &amp; Notes</h4>
            <p className="text-stone-950">Register faculty lecture notes and research diagrams with pHash watermarks prior to public distribution.</p>
          </div>
          <div className="p-3 glass-card rounded-lg space-y-1">
            <h4 className="font-bold text-stone-950">3. Payment &amp; Receipts</h4>
            <p className="text-stone-950">Perform Error-Level Analysis (ELA) on uploaded wire transfers and fee receipts to flag text modifications.</p>
          </div>
        </div>
      </div>

      {/* Developer API Key & Code Snippet Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left API Credentials */}
        <div className="glass-panel rounded-xl p-5 shadow-xs space-y-4">
          <h4 className="font-bold text-stone-950 text-xs uppercase tracking-wider border-b border-[#e2d8c3]/80 pb-2 flex items-center gap-2">
            <Key className="w-4 h-4 text-red-700" />
            <span>API Credentials</span>
          </h4>

          <div className="space-y-2">
            <label className="text-xs font-semibold text-stone-950">Sandbox Key:</label>
            <div className="p-2.5 glass-input rounded-lg flex items-center justify-between text-xs">
              <span className="font-mono text-stone-950 text-xs truncate">{apiKey}</span>
              <button
                onClick={handleCopyKey}
                className="p-1 text-stone-950 hover:text-stone-900 rounded cursor-pointer"
                title="Copy API Key"
              >
                <Copy className="w-3.5 h-3.5" />
              </button>
            </div>
            {copiedKey && <p className="text-xs text-emerald-700 font-semibold text-right">Key copied!</p>}
          </div>

          <div className="p-3 bg-red-50/70 border border-red-200 rounded-lg text-xs text-red-950 space-y-1">
            <p className="font-bold">Developer SLA:</p>
            <p className="text-xs leading-relaxed">
              Unlimited verification requests, sub-300ms neural inference latency, 99.9% uptime.
            </p>
          </div>
        </div>

        {/* Right Code Snippets Console */}
        <div className="lg:col-span-2 bg-[#1e1b18]/95 text-stone-200 rounded-xl p-6 border border-stone-800 shadow-md space-y-4 backdrop-blur-md">
          <div className="flex items-center justify-between border-b border-stone-800 pb-3">
            <div className="flex items-center gap-2 font-mono text-xs text-amber-300 font-bold">
              <Terminal className="w-4 h-4 text-red-400" />
              <span>REST API ENDPOINT: POST /v1/media/verify</span>
            </div>

            {/* Language Selector */}
            <div className="flex items-center gap-1 bg-stone-900 p-0.5 rounded border border-stone-800 text-xs">
              <button
                onClick={() => setActiveLang('javascript')}
                className={`px-2.5 py-1 rounded font-mono font-semibold cursor-pointer ${
                  activeLang === 'javascript' ? 'bg-red-800 text-white' : 'text-stone-950 hover:text-white'
                }`}
              >
                Node.js
              </button>
              <button
                onClick={() => setActiveLang('python')}
                className={`px-2.5 py-1 rounded font-mono font-semibold cursor-pointer ${
                  activeLang === 'python' ? 'bg-red-800 text-white' : 'text-stone-950 hover:text-white'
                }`}
              >
                Python
              </button>
              <button
                onClick={() => setActiveLang('curl')}
                className={`px-2.5 py-1 rounded font-mono font-semibold cursor-pointer ${
                  activeLang === 'curl' ? 'bg-red-800 text-white' : 'text-stone-950 hover:text-white'
                }`}
              >
                cURL
              </button>
            </div>
          </div>

          <pre className="font-mono text-xs text-stone-300 overflow-x-auto p-4 bg-black/40 rounded-lg border border-stone-800/80 leading-relaxed">
            <code>{codeSnippets[activeLang]}</code>
          </pre>
        </div>
      </div>
    </div>
  );
};
