import { useEffect, useRef } from 'react';

interface MnistDisplayProps {
  pixels: number[];
  size: number;
  label: number;
  prediction: number;
  correct: boolean;
}

export function MnistDisplay({ pixels, size, label, prediction, correct }: MnistDisplayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || pixels.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const scale = canvas.width / size;

    // Draw pixels
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const value = pixels[y * size + x];
        const intensity = Math.floor(value * 255);
        ctx.fillStyle = `rgb(${intensity}, ${intensity}, ${intensity})`;
        ctx.fillRect(x * scale, y * scale, scale, scale);
      }
    }
  }, [pixels, size]);

  const displaySize = size === 7 ? 140 : 112;

  return (
    <div className="flex flex-col items-center gap-2">
      <canvas
        ref={canvasRef}
        width={displaySize}
        height={displaySize}
        className="rounded border border-slate-600"
        style={{ imageRendering: 'pixelated' }}
      />
      <div className="flex gap-4 text-sm">
        <span className="text-slate-400">
          Label: <span className="text-white font-mono">{label}</span>
        </span>
        <span className={correct ? 'text-green-400' : 'text-red-400'}>
          Pred: <span className="font-mono">{prediction}</span>
          {correct ? ' ✓' : ' ✗'}
        </span>
      </div>
    </div>
  );
}
