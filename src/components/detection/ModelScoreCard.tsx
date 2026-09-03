import React from 'react';
import { DetailedModelScore } from '../../types';
import { ShieldCheck, AlertTriangle, Layers, Info, CheckCircle2 } from 'lucide-react';

interface ModelScoreCardProps {
  scores: DetailedModelScore[];
  modelAgreement: 'High Agreement' | 'Moderate Agreement' | 'Model Disagreement';
  modelAgreementDescription: string;
}

export const ModelScoreCard: React.FC<ModelScoreCardProps> = ({
  scores,
  modelAgreement,
  modelAgreementDescription,
}) => {
  const isAgreementHigh = modelAgreement === 'High Agreement';
  const isDisagreement = modelAgreement === 'Model Disagreement';

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-5">
      {/* Title Header */}
      <div className="flex items-center justify-between border-b border-slate-100 pb-3">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-indigo-600" />
          <h4 className="font-bold text-slate-900 text-xs uppercase tracking-wider">
            MODEL SCORE BREAKDOWN & FUSION
          </h4>
        </div>
        <span className="text-[10px] font-mono font-bold text-slate-500 bg-slate-100 px-2 py-0.5 rounded">
          Ensemble Architecture
        </span>
      </div>

      {/* Model Agreement Indicator Badge */}
      <div
        className={`p-3.5 rounded-lg border flex items-start gap-3 ${
          isAgreementHigh
            ? 'bg-emerald-50/80 border-emerald-200 text-emerald-900'
            : isDisagreement
            ? 'bg-amber-50/90 border-amber-200 text-amber-900'
            : 'bg-indigo-50/80 border-indigo-200 text-indigo-950'
        }`}
      >
        {isAgreementHigh ? (
          <CheckCircle2 className="w-5 h-5 text-emerald-600 shrink-0 mt-0.5" />
        ) : (
          <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
        )}
        <div className="text-xs space-y-0.5">
          <p className="font-bold">{modelAgreement}</p>
          <p className="text-[11px] leading-relaxed opacity-90">{modelAgreementDescription}</p>
        </div>
      </div>

      {/* Score List Cards */}
      <div className="space-y-3.5">
        {scores.map((m) => {
          const isFused = m.name.includes('Fused');

          return (
            <div
              key={m.name}
              className={`p-3.5 rounded-lg border text-xs space-y-2 transition-all ${
                isFused
                  ? 'bg-indigo-950 text-white border-indigo-900 shadow-2xs'
                  : 'bg-slate-50 border-slate-200 text-slate-800'
              }`}
            >
              <div className="flex items-center justify-between font-bold">
                <span className="flex items-center gap-1.5">
                  {isFused && <ShieldCheck className="w-4 h-4 text-indigo-300" />}
                  <span>{m.name}</span>
                </span>
                <span className={isFused ? 'text-indigo-200 font-mono' : 'text-slate-500 font-mono text-[11px]'}>
                  Confidence: {m.confidence}
                </span>
              </div>

              {/* Progress Bar Dual Indicators */}
              <div className="space-y-1.5 pt-1">
                <div className="flex items-center justify-between text-[11px]">
                  <span className={isFused ? 'text-emerald-300 font-semibold' : 'text-emerald-700 font-semibold'}>
                    Real: {m.realScore}%
                  </span>
                  <span className={isFused ? 'text-red-300 font-semibold' : 'text-red-600 font-semibold'}>
                    Fake: {m.fakeScore}%
                  </span>
                </div>

                <div className="w-full h-2.5 bg-slate-200/80 rounded-full overflow-hidden flex">
                  <div
                    className="bg-emerald-600 h-full transition-all duration-300"
                    style={{ width: `${m.realScore}%` }}
                    title={`Real ${m.realScore}%`}
                  />
                  <div
                    className="bg-red-600 h-full transition-all duration-300"
                    style={{ width: `${m.fakeScore}%` }}
                    title={`Fake ${m.fakeScore}%`}
                  />
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <p className="text-[11px] text-slate-500 leading-snug italic">
        Individual model logits are processed through Bayesian score fusion to compute the final authenticity score.
      </p>
    </div>
  );
};
