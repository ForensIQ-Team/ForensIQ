import React from 'react';
import { PipelineStage } from '../../types';
import {
  CheckCircle2,
  Clock,
  Loader2,
  AlertCircle,
  Cpu,
  Layers,
  Sparkles,
  ShieldAlert,
} from 'lucide-react';

interface DetectionPipelineProps {
  stages: PipelineStage[];
  currentStageIndex: number;
}

export const DetectionPipeline: React.FC<DetectionPipelineProps> = ({
  stages,
  currentStageIndex,
}) => {
  const completedCount = stages.filter((s) => s.status === 'completed').length;
  const progressPercent = Math.round((completedCount / stages.length) * 100);

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs space-y-6">
      {/* Header Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-indigo-950 text-indigo-300 flex items-center justify-center font-bold">
            <Cpu className="w-5 h-5 animate-pulse" />
          </div>
          <div>
            <h3 className="font-bold text-slate-900 text-base flex items-center gap-2">
              <span>ForensIQ AI Detection Pipeline</span>
              <span className="text-xs bg-indigo-100 text-indigo-800 font-mono font-bold px-2 py-0.5 rounded border border-indigo-200 uppercase">
                10-Stage Neural Engine
              </span>
            </h3>
            <p className="text-xs text-stone-950 mt-0.5">
              Evaluating multi-model spatial features, residual noise, and spatial attention vectors
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="text-right">
            <span className="text-xs text-stone-950 font-bold uppercase block">Analysis Progress</span>
            <span className="font-mono text-sm font-black text-indigo-950">{progressPercent}%</span>
          </div>
          <div className="w-12 h-12 rounded-full border-4 border-indigo-200 border-t-indigo-950 animate-spin flex items-center justify-center text-xs font-bold text-indigo-950">
            {completedCount}/10
          </div>
        </div>
      </div>

      {/* Global Progress Line Bar */}
      <div className="w-full bg-slate-100 h-2.5 rounded-full overflow-hidden">
        <div
          className="bg-indigo-600 h-full rounded-full transition-all duration-300 ease-out"
          style={{ width: `${progressPercent}%` }}
        />
      </div>

      {/* 10-Stage Grid View */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
        {stages.map((stage, idx) => {
          const isCurrent = idx === currentStageIndex && stage.status === 'processing';
          const isDone = stage.status === 'completed';
          const isError = stage.status === 'error';

          return (
            <div
              key={stage.id}
              className={`p-3 rounded-lg border text-xs transition-all flex flex-col justify-between ${
                isCurrent
                  ? 'border-indigo-600 bg-indigo-50/60 shadow-xs ring-2 ring-indigo-500/20'
                  : isDone
                  ? 'border-emerald-200 bg-emerald-50/40 text-stone-950'
                  : isError
                  ? 'border-red-200 bg-red-50 text-red-900'
                  : 'border-slate-200 bg-slate-50/50 text-stone-950 opacity-75'
              }`}
            >
              <div className="flex items-start justify-between gap-1 mb-1">
                <span className="font-mono text-xs font-bold text-stone-950">
                  {stage.id.toString().padStart(2, '0')}.
                </span>

                <div>
                  {isDone && <CheckCircle2 className="w-4 h-4 text-emerald-600" />}
                  {isCurrent && <Loader2 className="w-4 h-4 text-indigo-600 animate-spin" />}
                  {isError && <AlertCircle className="w-4 h-4 text-red-600" />}
                  {!isDone && !isCurrent && !isError && <Clock className="w-3.5 h-3.5 text-slate-300" />}
                </div>
              </div>

              <div>
                <p className={`font-bold text-xs leading-snug ${isDone ? 'text-slate-900' : isCurrent ? 'text-indigo-950' : 'text-stone-950'}`}>
                  {stage.name}
                </p>
                <p className="text-xs text-stone-950 line-clamp-2 mt-0.5 leading-tight">
                  {stage.description}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
