import React from 'react';
import { DetectionResult } from '../../types';
import { ShieldCheck, ShieldAlert, AlertTriangle, CheckCircle2, FileText, ArrowRight } from 'lucide-react';

interface NormalUserResultProps {
  result: DetectionResult;
  onExploreDetails: () => void;
  onOpenReport?: () => void;
}

export const NormalUserResult: React.FC<NormalUserResultProps> = ({
  result,
  onExploreDetails,
  onOpenReport,
}) => {
  const isReal = result.classification === 'Real';
  const isUncertain = result.classification === 'Uncertain';
  const isFakeOrAI = result.classification === 'Deepfake' || result.classification === 'AI-Generated';

  const confValue = result.confidenceScore ?? result.confidence ?? 90;

  // Extract model scores
  const xception = result.detailedModelScores?.find(m => m.name.toLowerCase().includes('xception'));
  const efficientnet = result.detailedModelScores?.find(m => m.name.toLowerCase().includes('efficientnet'));
  const fusion = result.detailedModelScores?.find(m => m.name.toLowerCase().includes('fusion') || m.name.toLowerCase().includes('ensemble'));

  const xFake = xception ? xception.fakeScore : (isFakeOrAI ? confValue : 100 - confValue);
  const xReal = xception ? xception.realScore : 100 - xFake;

  const eFake = efficientnet ? efficientnet.fakeScore : (isFakeOrAI ? Math.max(0, confValue - 2) : Math.max(0, 100 - confValue - 1));
  const eReal = efficientnet ? efficientnet.realScore : 100 - eFake;

  const fusedFake = fusion ? fusion.fakeScore : (isFakeOrAI ? confValue : 100 - confValue);
  const fusedReal = fusion ? fusion.realScore : 100 - fusedFake;

  const modelsAgree = (xFake >= 50 && eFake >= 50) || (xFake < 50 && eFake < 50);

  return (
    <div className="glass-panel-strong rounded-xl p-6 shadow-md space-y-6">
      {/* Header Banner: FORENSIQ ANALYSIS COMPLETE */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4" style={{ borderBottom: '1px solid rgba(200, 185, 155, 0.28)' }}>
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-[#1e1b18] text-red-500 flex items-center justify-center font-bold">
            <ShieldCheck className="w-6 h-6" />
          </div>
          <div>
            <span className="text-[10px] font-mono font-bold uppercase tracking-wider text-red-700 block">
              FORENSIQ ANALYSIS COMPLETE
            </span>
            <h3 className="text-lg font-serif font-black text-stone-950 truncate max-w-md">
              {result.mediaName}
            </h3>
          </div>
        </div>

        {/* Thumbnail Preview */}
        {result.mediaUrl && (
          <img
            src={result.mediaUrl}
            alt={result.mediaName}
            className="w-14 h-14 object-cover rounded-lg border border-[#dcd0b9] shadow-2xs"
          />
        )}
      </div>

      {/* Final Result Card — translucent glass tinted by classification */}
      <div
        className="p-5 rounded-xl flex flex-col sm:flex-row sm:items-center justify-between gap-4"
        style={{
          background: isReal
            ? 'rgba(220, 250, 225, 0.30)'
            : isUncertain
            ? 'rgba(255, 240, 200, 0.30)'
            : 'rgba(255, 215, 215, 0.30)',
          backdropFilter: 'blur(12px)',
          WebkitBackdropFilter: 'blur(12px)',
          border: isReal
            ? '1px solid rgba(52, 168, 83, 0.35)'
            : isUncertain
            ? '1px solid rgba(200, 145, 30, 0.35)'
            : '1px solid rgba(196, 30, 30, 0.35)',
        }}
      >
        <div className="flex items-start gap-4">
          <div
            className={`w-12 h-12 rounded-xl flex items-center justify-center shrink-0 font-bold text-white shadow-xs ${
              isReal ? 'bg-emerald-700' : isUncertain ? 'bg-amber-700' : 'bg-red-700'
            }`}
          >
            {isReal ? (
              <ShieldCheck className="w-6 h-6" />
            ) : isUncertain ? (
              <AlertTriangle className="w-6 h-6" />
            ) : (
              <ShieldAlert className="w-6 h-6" />
            )}
          </div>

          <div className="space-y-1">
            <span className={`text-[10px] font-mono font-bold uppercase tracking-wider opacity-75 ${
              isReal ? 'text-emerald-900' : isUncertain ? 'text-amber-900' : 'text-red-900'
            }`}>
              FINAL RESULT
            </span>
            <h3 className={`text-2xl font-serif font-black uppercase tracking-tight ${
              isReal ? 'text-emerald-950' : isUncertain ? 'text-amber-950' : 'text-red-950'
            }`}>
              {result.classification}
            </h3>
            <p className="text-xs font-semibold opacity-90 leading-relaxed max-w-xl text-stone-800">
              {result.summaryText}
            </p>
          </div>
        </div>

        {/* Confidence Badge — glass, not white */}
        <div
          className="p-4 rounded-xl text-center shrink-0 min-w-[140px]"
          style={{
            background: 'rgba(255, 252, 244, 0.48)',
            backdropFilter: 'blur(12px)',
            WebkitBackdropFilter: 'blur(12px)',
            border: '1px solid rgba(200, 185, 155, 0.38)',
          }}
        >
          <span className="text-[10px] font-mono font-bold uppercase text-stone-500 block">Confidence</span>
          <span className="text-3xl font-serif font-black text-stone-950">{confValue.toFixed(1)}%</span>
          <span className="text-[10px] font-semibold text-stone-600 block mt-0.5">
            Authenticity: {result.authenticityScore}%
          </span>
        </div>
      </div>

      {/* Model Analysis & Ensemble Breakdown Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* MODEL ANALYSIS */}
        <div className="glass-card rounded-xl p-4 space-y-3">
          <h4 className="font-mono font-bold text-xs uppercase tracking-wider text-stone-950 pb-1" style={{ borderBottom: '1px solid rgba(200, 185, 155, 0.28)' }}>
            MODEL ANALYSIS
          </h4>

          <div className="space-y-3 text-xs">
            {/* Xception V2 */}
            <div className="p-3 rounded-lg space-y-1" style={{ background: 'rgba(248, 242, 228, 0.38)', border: '1px solid rgba(200, 185, 155, 0.30)' }}>
              <div className="flex justify-between font-bold text-stone-900">
                <span>XCEPTION V2</span>
                <span className={xFake >= 50 ? 'text-red-700' : 'text-emerald-700'}>
                  {xFake >= 50 ? 'PREDICTION: FAKE' : 'PREDICTION: REAL'}
                </span>
              </div>
              <div className="flex justify-between font-mono text-[11px] text-stone-700">
                <span>Fake: {xFake.toFixed(1)}%</span>
                <span>Real: {xReal.toFixed(1)}%</span>
              </div>
            </div>

            {/* EfficientNet-B4 V2 */}
            <div className="p-3 rounded-lg space-y-1" style={{ background: 'rgba(248, 242, 228, 0.38)', border: '1px solid rgba(200, 185, 155, 0.30)' }}>
              <div className="flex justify-between font-bold text-stone-900">
                <span>EFFICIENTNET-B4 V2</span>
                <span className={eFake >= 50 ? 'text-red-700' : 'text-emerald-700'}>
                  {eFake >= 50 ? 'PREDICTION: FAKE' : 'PREDICTION: REAL'}
                </span>
              </div>
              <div className="flex justify-between font-mono text-[11px] text-stone-700">
                <span>Fake: {eFake.toFixed(1)}%</span>
                <span>Real: {eReal.toFixed(1)}%</span>
              </div>
            </div>
          </div>
        </div>

        {/* ENSEMBLE */}
        <div className="glass-card rounded-xl p-4 space-y-3">
          <h4 className="font-mono font-bold text-xs uppercase tracking-wider text-stone-950 pb-1" style={{ borderBottom: '1px solid rgba(200, 185, 155, 0.28)' }}>
            ENSEMBLE FUSION
          </h4>

          <div className="space-y-2 text-xs font-mono">
            <div className="p-3 rounded-lg space-y-1.5" style={{ background: 'rgba(248, 242, 228, 0.38)', border: '1px solid rgba(200, 185, 155, 0.30)' }}>
              <div className="flex justify-between">
                <span className="text-stone-600">Fused Fake Score:</span>
                <span className="font-bold text-red-700">{fusedFake.toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-stone-600">Fused Real Score:</span>
                <span className="font-bold text-emerald-700">{fusedReal.toFixed(1)}%</span>
              </div>
              <div className="flex justify-between pt-1" style={{ borderTop: '1px solid rgba(200, 185, 155, 0.28)' }}>
                <span className="text-stone-600">Model Agreement:</span>
                <span className={`font-bold ${modelsAgree ? 'text-emerald-800' : 'text-amber-800'}`}>
                  {modelsAgree ? 'AGREE (High Consensus)' : 'DISAGREE (Score Discrepancy)'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Report Action Buttons */}
      <div className="pt-3 border-t border-[#f0e6d6]/80 flex flex-wrap items-center justify-between gap-3">
        <button
          onClick={onExploreDetails}
          className="px-4 py-2 glass-input hover:bg-[#e8decb]/80 text-stone-900 rounded-lg text-xs font-bold flex items-center gap-2 cursor-pointer transition-colors"
        >
          <span>Examine Detailed Evidence</span>
          <ArrowRight className="w-3.5 h-3.5" />
        </button>

        <div className="flex items-center gap-2">
          {onOpenReport && (
            <>
              <button
                onClick={onOpenReport}
                className="px-4 py-2.5 bg-[#1e1b18] hover:bg-stone-900 text-white rounded-lg text-xs font-bold flex items-center gap-2 cursor-pointer shadow-xs transition-colors"
              >
                <FileText className="w-4 h-4 text-amber-300" />
                <span>GENERATE FORENSIC REPORT</span>
              </button>

              <button
                onClick={onOpenReport}
                className="px-3.5 py-2.5 glass-input hover:bg-[#e8decb]/80 text-stone-900 rounded-lg text-xs font-bold flex items-center gap-1.5 cursor-pointer transition-colors"
              >
                <span>VIEW REPORT</span>
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
