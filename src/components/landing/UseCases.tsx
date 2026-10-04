import React from 'react';
import { Newspaper, ShieldAlert, Landmark, GraduationCap } from 'lucide-react';

export const UseCases: React.FC = () => {
  const cases = [
    {
      id: 'uc-1',
      title: 'JOURNALISM',
      icon: Newspaper,
      description: 'Rapidly verify user-generated breaking news imagery before broadcasting.',
      tag: 'FACT-CHECKING',
    },
    {
      id: 'uc-2',
      title: 'LAW ENFORCEMENT',
      icon: ShieldAlert,
      description: 'Maintain strict chain of custody with cryptographically signed court binders.',
      tag: 'EVIDENCE AUDIT',
    },
    {
      id: 'uc-3',
      title: 'FINANCIAL FRAUD',
      icon: Landmark,
      description: 'Audit KYC identity documents and insurance claim photos for digital splicing.',
      tag: 'FRAUD PREVENTION',
    },
    {
      id: 'uc-4',
      title: 'DIGITAL FORENSICS',
      icon: GraduationCap,
      description: 'Academic research and laboratory image verification across published papers.',
      tag: 'RESEARCH & ACADEMIA',
    },
  ];

  return (
    <section id="use-cases" className="py-20 px-4 sm:px-6 lg:px-12 max-w-7xl mx-auto">
      <div className="text-center max-w-2xl mx-auto mb-14 space-y-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-red-100 border border-red-200 text-red-900 font-mono text-xs font-bold uppercase">
          <span>FORENSIC SECTORS</span>
        </div>
        <h2 className="text-3xl sm:text-4xl font-serif font-bold text-stone-900 tracking-tight">
          Investigative Use Cases
        </h2>
        <p className="text-stone-950 text-sm leading-relaxed">
          Engineered for institutions requiring absolute digital asset authenticity and provenance.
        </p>
      </div>

      {/* Horizontal Row of Pinned Case-File Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        {cases.map((c, idx) => {
          const Icon = c.icon;
          return (
            <div
              key={c.id}
              className="bg-[#faf8f5] border border-stone-300 p-6 rounded-sm shadow-md relative group hover:shadow-xl hover:-translate-y-1 transition-all duration-300"
            >
              {/* Top Red Pushpin */}
              <div className="absolute -top-3 left-1/2 -translate-x-1/2 z-20">
                <div className="w-4 h-4 rounded-full bg-red-700 border border-red-900 shadow-md" />
              </div>

              <div className="flex items-center justify-between border-b border-stone-200 pb-3 mb-4">
                <span className="font-mono text-xs font-bold text-red-700 bg-red-50 px-2 py-0.5 rounded border border-red-200 uppercase">
                  {c.tag}
                </span>
                <div className="w-8 h-8 rounded bg-stone-900 text-stone-100 flex items-center justify-center">
                  <Icon className="w-4 h-4 text-stone-200" />
                </div>
              </div>

              <h3 className="font-serif font-bold text-stone-900 text-lg mb-2">{c.title}</h3>
              <p className="text-xs text-stone-950 leading-relaxed font-sans">{c.description}</p>

              <div className="mt-4 pt-3 border-t border-stone-200 text-xs font-mono text-stone-950">
                FORENSIQ DEPLOYMENT READY
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
};
