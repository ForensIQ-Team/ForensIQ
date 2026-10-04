import React, { useState } from 'react';
import { VideoAnalysisData } from '../../types';
import { Film, Play, AlertTriangle, CheckCircle2, Clock, Eye } from 'lucide-react';

interface VideoAnalysisTimelineProps {
  videoData: VideoAnalysisData;
  mediaUrl: string;
  mediaName: string;
}

export const VideoAnalysisTimeline: React.FC<VideoAnalysisTimelineProps> = ({
  videoData,
  mediaUrl,
  mediaName,
}) => {
  const [selectedFrame, setSelectedFrame] = useState<number>(videoData.frameResults[4]?.frameNumber || 1);

  const activeFrameData =
    videoData.frameResults.find((f) => f.frameNumber === selectedFrame) || videoData.frameResults[0];

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs space-y-6">
      {/* Title & Metrics Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-indigo-950 text-indigo-300 flex items-center justify-center font-bold">
            <Film className="w-5 h-5" />
          </div>
          <div>
            <h4 className="font-bold text-slate-900 text-base">Video Deepfake Frame Inspector</h4>
            <p className="text-xs text-stone-950 mt-0.5">
              Inspecting temporal facial alignment & inter-frame continuity across sampled video frames
            </p>
          </div>
        </div>

        {/* Top Badges */}
        <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
          <div className="px-2.5 py-1 rounded bg-slate-100 border border-slate-200 font-bold text-stone-950">
            {videoData.sampledFrames} Sampled Frames
          </div>
          <div className="px-2.5 py-1 rounded bg-indigo-50 text-indigo-950 border border-indigo-200 font-bold">
            {videoData.facesDetected} Face Tracked
          </div>
          <div className="px-2.5 py-1 rounded bg-red-50 text-red-800 border border-red-200 font-bold">
            {videoData.suspiciousFrameCount} Suspicious Frames Flagged
          </div>
        </div>
      </div>

      {/* Frame Timeline Selector */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs font-semibold text-stone-950">
          <span>Frame Timeline ({videoData.durationSeconds}s Duration)</span>
          <span className="font-mono text-stone-950">
            Selected Frame #{activeFrameData.frameNumber} ({activeFrameData.timestamp})
          </span>
        </div>

        {/* Frame Bar Sequence */}
        <div className="grid grid-cols-12 sm:grid-cols-24 gap-1 p-2 bg-slate-950 rounded-xl border border-slate-800 overflow-x-auto">
          {videoData.frameResults.map((frame) => {
            const isSelected = frame.frameNumber === selectedFrame;

            return (
              <button
                key={frame.frameNumber}
                onClick={() => setSelectedFrame(frame.frameNumber)}
                title={`Frame #${frame.frameNumber} (${frame.timestamp}) - Fake Score: ${frame.fakeScore}%`}
                className={`h-12 rounded transition-all cursor-pointer flex flex-col justify-between p-1 text-[11px] font-mono font-bold ${
                  frame.isSuspicious
                    ? 'bg-red-600 text-white hover:bg-red-500'
                    : 'bg-emerald-600/80 text-white hover:bg-emerald-500'
                } ${isSelected ? 'ring-2 ring-white scale-105 z-10' : 'opacity-80 hover:opacity-100'}`}
              >
                <span>{frame.frameNumber}</span>
                <span className="text-[11px]">{frame.fakeScore}%</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Frame Inspection Card */}
      <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
        {/* Left Player Preview */}
        <div className="relative bg-slate-950 rounded-lg overflow-hidden border border-slate-800 h-44 flex items-center justify-center">
          <video
            src={mediaUrl}
            controls
            className="w-full h-full object-cover"
          />
        </div>

        {/* Middle Frame Score Details */}
        <div className="space-y-2 md:col-span-2 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between border-b border-slate-200 pb-2">
              <span className="font-bold text-slate-900 text-sm">
                Frame #{activeFrameData.frameNumber} Analysis Detail
              </span>
              <span
                className={`px-2 py-0.5 rounded font-mono font-bold text-xs ${
                  activeFrameData.isSuspicious
                    ? 'bg-red-100 text-red-800 border border-red-200'
                    : 'bg-emerald-100 text-emerald-800 border border-emerald-200'
                }`}
              >
                {activeFrameData.isSuspicious ? 'FLAGGED ANOMALY' : 'PASS (AUTHENTIC)'}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-4 my-3 text-xs">
              <div>
                <span className="text-stone-950 block">Timestamp:</span>
                <span className="font-mono font-bold text-slate-900">{activeFrameData.timestamp}</span>
              </div>
              <div>
                <span className="text-stone-950 block">Manipulation Probability:</span>
                <span className="font-mono font-bold text-red-600">{activeFrameData.fakeScore}%</span>
              </div>
            </div>

            {activeFrameData.anomalyType && (
              <div className="p-2.5 bg-red-50 border border-red-200 text-red-900 rounded-lg text-xs flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-red-600 shrink-0" />
                <span>Anomalous Signal: {activeFrameData.anomalyType}</span>
              </div>
            )}
          </div>

          <p className="text-xs text-stone-950">
            Temporal inter-frame inconsistencies indicate neural autoencoder frame splicing during speech activity.
          </p>
        </div>
      </div>
    </div>
  );
};
