import { useEffect, useState } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import { NetworkCanvas } from './components/NetworkCanvas';
import { MnistDisplay } from './components/MnistDisplay';
import { ControlPanel } from './components/ControlPanel';
import { OutputSpikes } from './components/OutputSpikes';

function App() {
  const wsUrl = `ws://${window.location.hostname}:3000/ws`;
  const { connected, frame, status, topology, send } = useWebSocket(wsUrl);

  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [speed, setSpeed] = useState(0.5); // Start slower

  useEffect(() => {
    const updateDimensions = () => {
      const container = document.getElementById('canvas-container');
      if (container) {
        setDimensions({
          width: container.clientWidth,
          height: Math.max(500, window.innerHeight - 280),
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

  // Send speed updates
  const handleSpeedChange = (newSpeed: number) => {
    setSpeed(newSpeed);
    send({ type: 'SetSpeed', speed: newSpeed });
  };

  // Calculate stats
  const neuronCount = topology?.total_neurons ?? 0;
  const synapseCount = topology?.synapses.length ?? 0;
  const activeNeurons = frame?.neurons.filter(n => n.spiking).length ?? 0;

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white flex flex-col">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-white/5">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-light tracking-wider text-white/90">gilgamesh</h1>
          <span className="text-xs text-white/30 tracking-widest">NEURAL SIMULATION</span>
        </div>

        {/* Stats */}
        <div className="flex items-center gap-8">
          <div className="text-center">
            <div className="text-2xl font-light text-blue-400">{neuronCount}</div>
            <div className="text-[10px] text-white/40 tracking-widest">NEURONS</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-light text-purple-400">{synapseCount}</div>
            <div className="text-[10px] text-white/40 tracking-widest">SYNAPSES</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-light text-orange-400">{activeNeurons}</div>
            <div className="text-[10px] text-white/40 tracking-widest">ACTIVE</div>
          </div>
          <div className="text-center">
            <div className={`text-2xl font-light ${connected ? 'text-green-400' : 'text-red-400'}`}>
              {connected ? '60' : '0'}
            </div>
            <div className="text-[10px] text-white/40 tracking-widest">FPS</div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <div className="flex-1 flex">
        {/* Left sidebar */}
        <div className="w-56 p-4 border-r border-white/5 flex flex-col gap-6">
          {/* MNIST Display */}
          {frame && (
            <div>
              <h3 className="text-[10px] text-white/40 tracking-widest mb-3">INPUT IMAGE</h3>
              <MnistDisplay
                pixels={frame.image_pixels}
                size={frame.image_size}
                label={frame.label}
                prediction={frame.prediction}
                correct={frame.correct}
              />
            </div>
          )}

          {/* Output Spikes */}
          {frame && (
            <div>
              <h3 className="text-[10px] text-white/40 tracking-widest mb-3">OUTPUT ACTIVITY</h3>
              <OutputSpikes
                spikes={frame.output_spikes}
                prediction={frame.prediction}
              />
            </div>
          )}

          {/* Speed control */}
          <div>
            <h3 className="text-[10px] text-white/40 tracking-widest mb-3">SIMULATION SPEED</h3>
            <input
              type="range"
              min="0.1"
              max="2"
              step="0.1"
              value={speed}
              onChange={(e) => handleSpeedChange(parseFloat(e.target.value))}
              className="w-full h-1 bg-white/10 rounded-full appearance-none tooling-pointer
                [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3
                [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-blue-400
                [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:tooling-pointer"
            />
            <div className="text-center text-white/50 text-xs mt-1">{speed.toFixed(1)}x</div>
          </div>

          {/* Connection status */}
          <div className="mt-auto">
            <div className={`flex items-center gap-2 text-xs ${connected ? 'text-green-400/70' : 'text-red-400/70'}`}>
              <span className={`w-1.5 h-1.5 rounded-full ${connected ? 'bg-green-400' : 'bg-red-400'}`} />
              {connected ? 'Connected' : 'Disconnected'}
            </div>
            {status?.checkpoint_loaded && (
              <div className="text-[10px] text-white/30 mt-1 truncate">
                {status.checkpoint_loaded.split('/').pop()}
              </div>
            )}
          </div>
        </div>

        {/* Main visualization */}
        <div className="flex-1 flex flex-col">
          <div
            id="canvas-container"
            className="flex-1 m-4"
          >
            <NetworkCanvas
              neurons={frame?.neurons ?? []}
              topology={topology}
              width={dimensions.width}
              height={dimensions.height}
            />
          </div>

          {/* Bottom controls */}
          <div className="px-4 pb-4">
            <ControlPanel
              paused={frame?.paused ?? true}
              currentStep={frame?.step ?? 0}
              totalSteps={frame?.total_steps ?? 25}
              sampleIndex={frame?.sample_index ?? 0}
              totalSamples={status?.total_samples ?? 10000}
              send={send}
            />

            {/* Keyboard hints */}
            <div className="text-[10px] text-white/30 text-center mt-2 tracking-wide">
              <span className="mr-6">← → Navigate</span>
              <span className="mr-6">SPACE Pause</span>
              <span>R Restart</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
