import { useEffect, useState } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import { NetworkCanvas } from './components/NetworkCanvas';
import { MnistDisplay } from './components/MnistDisplay';
import { ControlPanel } from './components/ControlPanel';
import { OutputSpikes } from './components/OutputSpikes';

type EndOfSampleBehavior = 'auto-advance' | 'stop' | 'loop';
type PulseStyle = 'ball' | 'electricity';

function App() {
  const wsUrl = `ws://${window.location.host}/ws`;
  const { connected, frame, status, topology, send } = useWebSocket(wsUrl);

  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [speed, setSpeed] = useState(0.5); // Start slower
  const [pulseStyle, setPulseStyle] = useState<PulseStyle>('ball');
  const [showInputCurrent, setShowInputCurrent] = useState(true);
  const [endOfSampleBehavior, setEndOfSampleBehavior] = useState<EndOfSampleBehavior>('auto-advance');

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
      // Allow keyboard shortcuts even when focused on form elements (except text inputs)
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT' && (target as HTMLInputElement).type === 'text') return;

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

  // Send end-of-sample behavior updates
  const handleBehaviorChange = (behavior: EndOfSampleBehavior) => {
    console.log('Setting behavior to:', behavior);
    setEndOfSampleBehavior(behavior);
    const msg = { type: 'SetEndOfSampleBehavior', behavior };
    console.log('Sending message:', JSON.stringify(msg));
    send(msg as any);
  };

  // Calculate stats
  const neuronCount = topology?.total_neurons ?? 0;
  const synapseCount = topology?.synapses.length ?? 0;
  const activeNeurons = frame?.neurons.filter(n => n.spiking).length ?? 0;

  return (
    <div className="h-screen bg-[#0a0a0f] text-white flex flex-col overflow-hidden">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-white/5 flex-shrink-0">
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
      <div className="flex-1 flex overflow-hidden">
        {/* Left sidebar */}
        <div className="w-56 p-4 border-r border-white/5 flex flex-col gap-4 overflow-y-auto flex-shrink-0">
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

          {/* Visualization Options */}
          <div>
            <h3 className="text-[10px] text-white/40 tracking-widest mb-3">VISUALIZATION</h3>

            {/* Pulse Style Toggle */}
            <div className="flex items-center justify-between mb-2">
              <span className="text-white/60 text-xs">Pulse Style</span>
              <div className="flex bg-white/5 rounded overflow-hidden">
                <button
                  onClick={() => setPulseStyle('ball')}
                  className={`px-2 py-1 text-xs transition-colors ${
                    pulseStyle === 'ball'
                      ? 'bg-blue-500/50 text-white'
                      : 'text-white/50 hover:text-white/70'
                  }`}
                >
                  Ball
                </button>
                <button
                  onClick={() => setPulseStyle('electricity')}
                  className={`px-2 py-1 text-xs transition-colors ${
                    pulseStyle === 'electricity'
                      ? 'bg-blue-500/50 text-white'
                      : 'text-white/50 hover:text-white/70'
                  }`}
                >
                  Electric
                </button>
              </div>
            </div>

            {/* Input Current Toggle */}
            <label className="flex items-center gap-2 tooling-pointer">
              <input
                type="checkbox"
                checked={showInputCurrent}
                onChange={(e) => setShowInputCurrent(e.target.checked)}
                className="w-3.5 h-3.5 rounded bg-white/10 border-white/20 text-blue-500
                  focus:ring-blue-500/30 focus:ring-offset-0"
              />
              <span className="text-white/60 text-xs">Show input current</span>
            </label>
          </div>

          {/* End of Sample Behavior */}
          <div>
            <h3 className="text-[10px] text-white/40 tracking-widest mb-3">AT END OF SAMPLE</h3>
            <div className="space-y-1.5">
              {[
                { value: 'auto-advance', label: 'Auto advance' },
                { value: 'stop', label: 'Stop' },
                { value: 'loop', label: 'Loop current' },
              ].map(option => (
                <label key={option.value} className="flex items-center gap-2 tooling-pointer">
                  <input
                    type="radio"
                    name="endBehavior"
                    value={option.value}
                    checked={endOfSampleBehavior === option.value}
                    onChange={() => handleBehaviorChange(option.value as EndOfSampleBehavior)}
                    className="w-3 h-3 text-blue-500 bg-white/10 border-white/20
                      focus:ring-blue-500/30 focus:ring-offset-0"
                  />
                  <span className="text-white/60 text-xs">{option.label}</span>
                </label>
              ))}
            </div>
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
        <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
          <div
            id="canvas-container"
            className="flex-1 m-4 min-h-0"
          >
            <NetworkCanvas
              neurons={frame?.neurons ?? []}
              topology={topology}
              width={dimensions.width}
              height={dimensions.height}
              speed={speed}
              pulseStyle={pulseStyle}
              showInputCurrent={showInputCurrent}
            />
          </div>

          {/* Bottom controls */}
          <div className="px-4 pb-3 flex-shrink-0">
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
