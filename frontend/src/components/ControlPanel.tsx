import type { ClientMessage } from '../lib/protocol';

interface ControlPanelProps {
  paused: boolean;
  currentStep: number;
  totalSteps: number;
  sampleIndex: number;
  totalSamples: number;
  send: (msg: ClientMessage) => void;
}

const secondaryBtn =
  'px-3 py-1.5 bg-white/5 hover:bg-white/10 rounded text-white/70 hover:text-white transition-colors text-sm';

export function ControlPanel({
  paused,
  currentStep,
  totalSteps,
  sampleIndex,
  totalSamples,
  send,
}: ControlPanelProps) {
  return (
    <div className="flex items-center gap-4 p-3 bg-white/5 rounded-lg border border-white/5">
      {/* Navigation */}
      <button
        onClick={() => send({ type: 'PrevSample' })}
        className={secondaryBtn}
      >
        ← Prev
      </button>

      <span className="text-white/40 text-xs min-w-[100px] text-center font-mono">
        {sampleIndex + 1} / {totalSamples}
      </span>

      <button
        onClick={() => send({ type: 'NextSample' })}
        className={secondaryBtn}
      >
        Next →
      </button>

      <div className="w-px h-5 bg-white/10" />

      {/* Playback */}
      <button
        onClick={() => send(paused ? { type: 'Resume' } : { type: 'Pause' })}
        className={`px-4 py-1.5 rounded text-sm font-medium transition-colors ${
          paused
            ? 'bg-blue-500/80 hover:bg-blue-500 text-white'
            : 'bg-orange-500/80 hover:bg-orange-500 text-white'
        }`}
      >
        {paused ? '▶ Play' : '⏸ Pause'}
      </button>

      <button
        onClick={() => send({ type: 'RestartSample' })}
        className={secondaryBtn}
      >
        ↻
      </button>

      <button
        onClick={() => send({ type: 'RandomSample' })}
        className={secondaryBtn}
      >
        🎲
      </button>

      <div className="w-px h-5 bg-white/10" />

      {/* Progress */}
      <div className="flex-1 flex items-center gap-3">
        <span className="text-white/30 text-xs font-mono w-12">
          {currentStep}/{totalSteps}
        </span>
        <div className="flex-1 h-1 bg-white/10 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-150"
            style={{ width: `${(currentStep / totalSteps) * 100}%` }}
          />
        </div>
      </div>
    </div>
  );
}
