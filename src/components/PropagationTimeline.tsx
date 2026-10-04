import React from 'react';
import { GitCommit, Clock, ShieldCheck, AlertTriangle, ArrowRight } from 'lucide-react';
import { PropagationNode, UserRole } from '../types';
import { MOCK_PROPAGATION_NODES } from '../data/mockData';

interface PropagationTimelineProps {
  userRole: UserRole;
  nodes?: PropagationNode[];
}

export const PropagationTimeline: React.FC<PropagationTimelineProps> = ({
  userRole,
  nodes = MOCK_PROPAGATION_NODES,
}) => {
  return (
    <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs space-y-6">
      <div className="flex items-center justify-between border-b border-slate-100 pb-4">
        <div>
          <h3 className="font-bold text-slate-900 text-base flex items-center gap-2">
            <GitCommit className="w-5 h-5 text-indigo-600" />
            <span>Propagation Timeline & Lineage</span>
          </h3>
          <p className="text-xs text-stone-950 mt-0.5">
            Chronological dissemination path of registered media across public networks.
          </p>
        </div>

        {userRole === 'investigator' && (
          <span className="px-2.5 py-1 bg-indigo-950 text-indigo-200 text-xs font-mono font-bold rounded border border-indigo-900">
            Source Lineage Graph Active
          </span>
        )}
      </div>

      {/* Timeline Nodes */}
      <div className="relative pl-6 space-y-6 before:absolute before:left-2.5 before:top-3 before:bottom-3 before:w-0.5 before:bg-slate-200">
        {nodes.map((node, index) => (
          <div key={node.id} className="relative flex items-start gap-4 group">
            {/* Dot Indicator */}
            <div
              className={`absolute -left-6 top-1 w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold shadow-xs ${
                node.type === 'original'
                  ? 'bg-indigo-600 text-white ring-4 ring-indigo-50'
                  : node.type === 'manipulated'
                  ? 'bg-red-600 text-white ring-4 ring-red-50'
                  : 'bg-indigo-950 text-white ring-4 ring-slate-100'
              }`}
            >
              {index + 1}
            </div>

            {/* Card Content */}
            <div className="flex-1 bg-slate-50 border border-slate-200 rounded-xl p-4 hover:border-slate-300 transition-colors">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-200/60 pb-2">
                <div className="flex items-center gap-2">
                  <span className="font-bold text-slate-900 text-xs">{node.title}</span>
                  <span
                    className={`px-2 py-0.5 rounded text-xs font-extrabold uppercase border ${
                      node.classification === 'CONFIRMED MATCH'
                        ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                        : node.classification === 'AI MANIPULATED'
                        ? 'bg-red-50 text-red-800 border-red-200'
                        : 'bg-amber-50 text-amber-800 border-amber-200'
                    }`}
                  >
                    {node.classification}
                  </span>
                </div>

                <div className="flex items-center gap-1.5 text-xs text-stone-950 font-mono">
                  <Clock className="w-3.5 h-3.5 text-stone-950" />
                  <span>{node.timestamp}</span>
                </div>
              </div>

              <div className="mt-2 text-xs space-y-1">
                <p className="text-stone-950 font-semibold">Source Host: {node.source}</p>
                <p className="text-stone-950">{node.notes}</p>
                <div className="text-xs text-indigo-900 font-mono font-bold pt-1">
                  Vector Similarity: {node.similarity}%
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
