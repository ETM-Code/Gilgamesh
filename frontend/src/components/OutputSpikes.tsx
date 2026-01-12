interface OutputSpikesProps {
  spikes: number[];
  prediction: number;
}

export function OutputSpikes({ spikes, prediction }: OutputSpikesProps) {
  const maxSpikes = Math.max(...spikes, 1);

  return (
    <div className="flex flex-col gap-1">
      <h3 className="text-slate-400 text-xs font-medium mb-1">Output Spikes</h3>
      {spikes.map((count, idx) => (
        <div key={idx} className="flex items-center gap-2">
          <span
            className={`w-4 text-xs font-mono ${
              idx === prediction ? 'text-green-400 font-bold' : 'text-slate-500'
            }`}
          >
            {idx}
          </span>
          <div className="flex-1 h-3 bg-slate-700 rounded overflow-hidden">
            <div
              className={`h-full transition-all duration-100 ${
                idx === prediction ? 'bg-green-500' : 'bg-blue-500'
              }`}
              style={{ width: `${(count / maxSpikes) * 100}%` }}
            />
          </div>
          <span className="w-6 text-xs text-slate-500 text-right font-mono">
            {count}
          </span>
        </div>
      ))}
    </div>
  );
}
