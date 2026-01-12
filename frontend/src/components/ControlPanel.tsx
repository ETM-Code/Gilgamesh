import type { ClientMessage } from '../lib/protocol';

interface ControlPanelProps {
  paused: boolean;
  currentStep: number;
  totalSteps: number;
  sampleIndex: number;
  totalSamples: number;
  send: (msg: ClientMessage) => void;
}

export function ControlPanel({
  paused,
  currentStep,
  totalSteps,
  sampleIndex,
  totalSamples,
  send,
}: ControlPanelProps) {
  return (
    <div className="flex items-center gap-4 p-4 bg-slate-800 rounded-lg">
      {/* Navigation */}
      <button
        onClick={() => send({ type: 'PrevSample' })}
        className="px-3 py-1 bg-slate-700 hover:bg-slate-600 rounded text-slate-200 transition-colors"
      >
        ← Prev
      </button>

      <span className="text-slate-400 text-sm min-w-[120px] text-center">
        Sample {sampleIndex + 1} / {totalSamples}
      </span>

      <button
        onClick={() => send({ type: 'NextSample' })}
        className="px-3 py-1 bg-slate-700 hover:bg-slate-600 rounded text-slate-200 transition-colors"
      >
        Next →
      </button>

      <div className="w-px h-6 bg-slate-600" />

      {/* Playback */}
      <button
        onClick={() => send(paused ? { type: 'Resume' } : { type: 'Pause' })}
        className="px-4 py-1 bg-blue-600 hover:bg-blue-500 rounded text-white font-medium transition-colors"
      >
        {paused ? '▶ Play' : '⏸ Pause'}
      </button>

      <button
        onClick={() => send({ type: 'RestartSample' })}
        className="px-3 py-1 bg-slate-700 hover:bg-slate-600 rounded text-slate-200 transition-colors"
      >
        ↻ Restart
      </button>

      <button
        onClick={() => send({ type: 'RandomSample' })}
        className="px-3 py-1 bg-slate-700 hover:bg-slate-600 rounded text-slate-200 transition-colors"
      >
        🎲 Random
      </button>

      <div className="w-px h-6 bg-slate-600" />

      {/* Progress */}
      <div className="flex-1 flex items-center gap-2">
        <span className="text-slate-500 text-xs">Step {currentStep}/{totalSteps}</span>
        <div className="flex-1 h-2 bg-slate-700 rounded overflow-hidden">
          <div
            className="h-full bg-blue-500 transition-all duration-100"
            style={{ width: `${(currentStep / totalSteps) * 100}%` }}
          />
        </div>
      </div>
    </div>
  );
}
