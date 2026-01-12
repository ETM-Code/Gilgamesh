import { useEffect, useState } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import { NetworkCanvas } from './components/NetworkCanvas';
import { MnistDisplay } from './components/MnistDisplay';
import { ControlPanel } from './components/ControlPanel';
import { OutputSpikes } from './components/OutputSpikes';

function App() {
  const wsUrl = `ws://${window.location.hostname}:3000/ws`;
  const { connected, frame, status, topology, send } = useWebSocket(wsUrl);

  const [dimensions, setDimensions] = useState({ width: 800, height: 500 });

  useEffect(() => {
    const updateDimensions = () => {
      const container = document.getElementById('canvas-container');
      if (container) {
        setDimensions({
          width: container.clientWidth,
          height: Math.max(400, window.innerHeight - 300),
        });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;

      switch (e.key) {
        case 'ArrowLeft':
        case 'a':
          send({ type: 'PrevSample' });
          break;
        case 'ArrowRight':
        case 'd':
          send({ type: 'NextSample' });
          break;
        case ' ':
          e.preventDefault();
          send(frame?.paused ? { type: 'Resume' } : { type: 'Pause' });
          break;
        case 'r':
          send({ type: 'RestartSample' });
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [send, frame?.paused]);

  return (
    <div className="min-h-screen bg-slate-900 text-white p-6">
      {/* Header */}
      <header className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">gilgamesh</h1>
          <p className="text-sm text-slate-400">Spiking Neural Network Visualizer</p>
        </div>
        <div className="flex items-center gap-4">
          <span className={`flex items-center gap-2 text-sm ${connected ? 'text-green-400' : 'text-red-400'}`}>
            <span className={`w-2 h-2 rounded-full ${connected ? 'bg-green-400' : 'bg-red-400'}`} />
            {connected ? 'Connected' : 'Disconnected'}
          </span>
          {status?.checkpoint_loaded && (
            <span className="text-xs text-slate-500">
              Model: {status.checkpoint_loaded.split('/').pop()}
            </span>
          )}
        </div>
      </header>

      {/* Main content */}
      <div className="flex gap-6">
        {/* Left sidebar - MNIST and output */}
        <div className="flex flex-col gap-4 w-48">
          {frame && (
            <>
              <MnistDisplay
                pixels={frame.image_pixels}
                size={frame.image_size}
                label={frame.label}
                prediction={frame.prediction}
                correct={frame.correct}
              />
              <OutputSpikes
                spikes={frame.output_spikes}
                prediction={frame.prediction}
              />
            </>
          )}
          {!frame && (
            <div className="text-slate-500 text-sm text-center py-8">
              {connected ? 'Waiting for data...' : 'Connecting...'}
            </div>
          )}
        </div>

        {/* Main visualization area */}
        <div className="flex-1 flex flex-col gap-4">
          <div
            id="canvas-container"
            className="bg-slate-800 rounded-lg overflow-hidden"
          >
            <NetworkCanvas
              neurons={frame?.neurons ?? []}
              topology={topology}
              width={dimensions.width}
              height={dimensions.height}
            />
          </div>

          {/* Controls */}
          <ControlPanel
            paused={frame?.paused ?? true}
            currentStep={frame?.step ?? 0}
            totalSteps={frame?.total_steps ?? 25}
            sampleIndex={frame?.sample_index ?? 0}
            totalSamples={status?.total_samples ?? 10000}
            send={send}
          />

          {/* Keyboard hints */}
          <div className="text-xs text-slate-500 text-center">
            <span className="mr-4">← / → or A/D: Navigate samples</span>
            <span className="mr-4">Space: Pause/Resume</span>
            <span>R: Restart sample</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
