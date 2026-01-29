# Web-Based Architecture Proposal for Gilgamesh

This document outlines two approaches for moving gilgamesh to a web-based UI:
1. **Hybrid** - Rust backend + Web frontend via WebSocket
2. **Full Web** - WASM-compiled Rust + Web UI

---

## Current State

```
┌─────────────────────────────────────────────────────────────┐
│                    THREE SEPARATE UIs                        │
├─────────────────┬──────────────────┬────────────────────────┤
│  nannou (645L)  │   egui (374L)    │    Rerun.io (326L)     │
│  Animation      │   Dashboard      │    Visualization       │
│  - Neurons      │   - Loss plots   │    - Weight heatmaps   │
│  - Synapses     │   - Accuracy     │    - Spike rasters     │
│  - MNIST img    │   - LR schedule  │    - Membrane traces   │
│  - Controls     │   - Progress     │    - Epoch metrics     │
└─────────────────┴──────────────────┴────────────────────────┘
          │                  │                    │
          └──────────────────┴────────────────────┘
                             │
                    Arc<Mutex<State>>
                             │
                    ┌────────────────┐
                    │  Rust Core     │
                    │  Network       │
                    │  Trainer       │
                    └────────────────┘
```

**Problems:**
- 3 feature flags, 3 dependencies, ~1,350 lines of UI code
- Fragmented experience - must run separate commands
- No remote access
- Limited styling options

---

## Option A: Hybrid Architecture (Recommended)

Keep simulation in native Rust, serve web UI via embedded server.

```
┌─────────────────────────────────────────────────────────────┐
│                    UNIFIED WEB UI                            │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                   Browser (React/Svelte)                ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐ ││
│  │  │ Network  │ │ Training │ │ Metrics  │ │  Controls  │ ││
│  │  │ Canvas   │ │ Charts   │ │ Sidebar  │ │  Panel     │ ││
│  │  │ (WebGL)  │ │(Chart.js)│ │          │ │            │ ││
│  │  └──────────┘ └──────────┘ └──────────┘ └────────────┘ ││
│  └─────────────────────────────────────────────────────────┘│
│                            │                                 │
│                     WebSocket (JSON)                         │
│                            │                                 │
└────────────────────────────┼────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────┐
│                    RUST BACKEND                              │
│                            │                                 │
│  ┌─────────────────────────▼─────────────────────────────┐  │
│  │              WebSocket Server (axum/tokio)            │  │
│  │  - Broadcasts simulation state at 30-60fps            │  │
│  │  - Handles control commands (pause, next, config)     │  │
│  │  - Serves static files for frontend                   │  │
│  └───────────────────────────────────────────────────────┘  │
│                            │                                 │
│  ┌─────────────────────────▼─────────────────────────────┐  │
│  │                   Simulation Core                      │  │
│  │  Network, Trainer, Physics (unchanged)                 │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### New Rust Dependencies

```toml
# Web server
axum = "0.8"
tokio = { version = "1", features = ["full"] }
tokio-tungstenite = "0.24"
tower-http = { version = "0.6", features = ["fs", "cors"] }

# Better JSON serialization
serde_json = "1.0"  # already have this
```

### WebSocket Message Protocol

```rust
// Server -> Client messages
#[derive(Serialize)]
#[serde(tag = "type")]
enum ServerMessage {
    // Animation frame (30-60fps)
    AnimationFrame {
        neurons: Vec<NeuronState>,
        synapses: Vec<SynapseState>,
        viewer: ViewerState,
        timestamp: f64,
    },

    // Training update (per epoch)
    TrainingUpdate {
        epoch: usize,
        loss: f64,
        train_acc: f64,
        test_acc: f64,
        lr: f64,
    },

    // Weight matrix snapshot
    WeightMatrix {
        layer: String,
        data: Vec<f32>,
        rows: usize,
        cols: usize,
    },

    // System status
    Status {
        mode: String,  // "idle", "training", "inference"
        config: Config,
    },
}

// Compact neuron state for wire transfer
#[derive(Serialize)]
struct NeuronState {
    m: f32,      // membrane (0-1)
    s: bool,     // spiking
    sc: u32,     // spike_count (output only)
}

// Client -> Server messages
#[derive(Deserialize)]
#[serde(tag = "type")]
enum ClientMessage {
    // Navigation
    NextSample,
    PrevSample,
    JumpToSample { index: usize },

    // Playback
    Pause,
    Resume,
    SetSpeed { speed: f32 },

    // Training control
    StartTraining { config: Config },
    StopTraining,

    // Config
    LoadCheckpoint { path: String },
    UpdateConfig { config: Config },
}
```

### Server Architecture

```rust
// src/web/mod.rs
use axum::{
    extract::ws::{WebSocket, WebSocketUpgrade},
    routing::get,
    Router,
};
use std::sync::Arc;
use tokio::sync::broadcast;

pub struct WebServer {
    state: Arc<AppState>,
    tx: broadcast::Sender<ServerMessage>,
}

struct AppState {
    simulation: Mutex<Option<SimulationHandle>>,
    training: Mutex<Option<TrainingHandle>>,
    config: RwLock<Config>,
}

impl WebServer {
    pub async fn run(addr: &str) -> anyhow::Result<()> {
        let (tx, _) = broadcast::channel(100);
        let state = Arc::new(AppState::default());

        let app = Router::new()
            .route("/ws", get(ws_handler))
            .route("/api/config", get(get_config).post(set_config))
            .route("/api/checkpoints", get(list_checkpoints))
            .nest_service("/", ServeDir::new("frontend/dist"))
            .with_state(state);

        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app).await?;
        Ok(())
    }
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: Arc<AppState>) {
    // Subscribe to broadcast channel
    let mut rx = state.tx.subscribe();

    loop {
        tokio::select! {
            // Forward server messages to client
            Ok(msg) = rx.recv() => {
                let json = serde_json::to_string(&msg).unwrap();
                socket.send(Message::Text(json)).await.ok();
            }

            // Handle client messages
            Some(Ok(msg)) = socket.recv() => {
                if let Message::Text(text) = msg {
                    let cmd: ClientMessage = serde_json::from_str(&text)?;
                    handle_command(cmd, &state).await;
                }
            }
        }
    }
}
```

### Frontend Structure (React + TypeScript)

```
frontend/
├── package.json
├── vite.config.ts
├── tailwind.config.js
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── hooks/
│   │   ├── useWebSocket.ts      # WebSocket connection management
│   │   └── useSimulation.ts     # Simulation state
│   ├── components/
│   │   ├── NetworkCanvas.tsx    # WebGL neuron visualization
│   │   ├── TrainingCharts.tsx   # Loss/accuracy plots
│   │   ├── ControlPanel.tsx     # Playback controls
│   │   ├── ConfigEditor.tsx     # JSON config editing
│   │   ├── MnistDisplay.tsx     # Input image viewer
│   │   └── MetricsSidebar.tsx   # Real-time stats
│   ├── lib/
│   │   ├── protocol.ts          # Message types (generated from Rust)
│   │   └── renderer.ts          # WebGL neuron renderer
│   └── styles/
│       └── globals.css          # Tailwind + custom
```

### Key Frontend Components

```tsx
// NetworkCanvas.tsx - WebGL neuron visualization
import { useRef, useEffect } from 'react';
import { useSimulation } from '../hooks/useSimulation';

export function NetworkCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { neurons, synapses, layerSizes } = useSimulation();

  useEffect(() => {
    const gl = canvasRef.current?.getContext('webgl2');
    if (!gl) return;

    // Render neurons as instanced circles
    // Color by membrane potential, glow on spike
    // Draw synapses as lines with weight-based alpha
  }, [neurons, synapses]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full h-full bg-slate-900 rounded-lg"
    />
  );
}
```

```tsx
// ControlPanel.tsx - Playback and navigation
import { useSimulation } from '../hooks/useSimulation';

export function ControlPanel() {
  const {
    paused, setPaused,
    speed, setSpeed,
    currentSample, totalSamples,
    nextSample, prevSample,
    currentStep, totalSteps,
  } = useSimulation();

  return (
    <div className="flex items-center gap-4 p-4 bg-slate-800 rounded-lg">
      {/* Navigation */}
      <button onClick={prevSample} className="btn">
        <ChevronLeft />
      </button>
      <span className="text-sm text-slate-300">
        Sample {currentSample + 1} / {totalSamples}
      </span>
      <button onClick={nextSample} className="btn">
        <ChevronRight />
      </button>

      {/* Playback */}
      <button onClick={() => setPaused(!paused)} className="btn-primary">
        {paused ? <Play /> : <Pause />}
      </button>

      {/* Speed */}
      <input
        type="range"
        min="0.1" max="5" step="0.1"
        value={speed}
        onChange={(e) => setSpeed(parseFloat(e.target.value))}
        className="w-32"
      />

      {/* Progress */}
      <div className="flex-1 h-2 bg-slate-700 rounded">
        <div
          className="h-full bg-blue-500 rounded transition-all"
          style={{ width: `${(currentStep / totalSteps) * 100}%` }}
        />
      </div>
    </div>
  );
}
```

### CLI Integration

```rust
// src/cli/commands/web.rs
use clap::Args;

#[derive(Args)]
pub struct WebArgs {
    /// Port to run web server on
    #[arg(short, long, default_value = "3000")]
    port: u16,

    /// Open browser automatically
    #[arg(long, default_value = "true")]
    open: bool,

    /// Checkpoint to load initially
    #[arg(short, long)]
    checkpoint: Option<PathBuf>,
}

pub async fn run(args: WebArgs) -> anyhow::Result<()> {
    let addr = format!("0.0.0.0:{}", args.port);
    println!("Starting gilgamesh web interface at http://localhost:{}", args.port);

    if args.open {
        opener::open(&format!("http://localhost:{}", args.port))?;
    }

    WebServer::new(args.checkpoint)
        .run(&addr)
        .await
}
```

Usage:
```bash
cargo run --features web -- web --port 3000
cargo run --features web -- web --checkpoint model.json
```

### Bandwidth Optimization

For 60fps updates with 100 neurons:

```rust
// Compact binary protocol option (if JSON too slow)
// ~400 bytes per frame vs ~2KB for JSON

#[derive(Serialize)]
struct CompactFrame {
    // Pack membrane (0-255) + flags into single byte per neuron
    neurons: Vec<u8>,  // high 7 bits = membrane, low bit = spiking
    // Only send synapses with active propagation
    active_synapses: Vec<(u16, u8)>,  // (index, progress)
    step: u8,
}
```

Or use MessagePack instead of JSON:
```toml
rmp-serde = "1.3"  # ~40% smaller than JSON
```

---

## Option B: Full WASM Architecture

Compile entire Rust core to WebAssembly, run simulation in browser.

```
┌─────────────────────────────────────────────────────────────┐
│                       BROWSER                                │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                    Web UI (React)                       ││
│  └─────────────────────────────────────────────────────────┘│
│                            │                                 │
│                     JS <-> WASM FFI                          │
│                            │                                 │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              gilgamesh-core.wasm (~2MB)                 ││
│  │  Network, Trainer, Physics (compiled to WASM)           ││
│  └─────────────────────────────────────────────────────────┘│
│                            │                                 │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                    Web Workers                          ││
│  │  Simulation runs in worker, doesn't block UI            ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### WASM Setup

```toml
# Cargo.toml additions
[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
wasm-bindgen = "0.2"
js-sys = "0.3"
web-sys = { version = "0.3", features = ["console"] }
console_error_panic_hook = "0.1"

[target.'cfg(target_arch = "wasm32")'.dependencies]
getrandom = { version = "0.2", features = ["js"] }
```

```rust
// src/wasm.rs
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmNetwork {
    network: Network,
    config: Config,
}

#[wasm_bindgen]
impl WasmNetwork {
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str) -> Result<WasmNetwork, JsValue> {
        console_error_panic_hook::set_once();
        let config: Config = serde_json::from_str(config_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let network = Network::new(&config);
        Ok(Self { network, config })
    }

    pub fn load_checkpoint(&mut self, json: &str) -> Result<(), JsValue> {
        // Load weights from JSON
    }

    pub fn forward_step(&mut self, input: &[f32]) -> Box<[f32]> {
        let output = self.network.forward_step(input);
        output.into_boxed_slice()
    }

    pub fn get_neuron_states(&self) -> Box<[f32]> {
        // Return flat array of membrane potentials
    }

    pub fn get_spike_mask(&self) -> Box<[u8]> {
        // Return bitmask of which neurons spiked
    }
}
```

### Build Process

```bash
# Install wasm-pack
cargo install wasm-pack

# Build WASM module
wasm-pack build --target web --out-dir frontend/src/wasm

# Output:
# frontend/src/wasm/
#   gilgamesh.js       # JS glue code
#   gilgamesh_bg.wasm  # WASM binary (~2MB)
#   gilgamesh.d.ts     # TypeScript types
```

### Frontend Integration

```tsx
// hooks/useWasmSimulation.ts
import init, { WasmNetwork } from '../wasm/gilgamesh';

export function useWasmSimulation() {
  const [network, setNetwork] = useState<WasmNetwork | null>(null);
  const workerRef = useRef<Worker | null>(null);

  useEffect(() => {
    // Initialize WASM in web worker
    workerRef.current = new Worker(
      new URL('../workers/simulation.ts', import.meta.url)
    );

    workerRef.current.onmessage = (e) => {
      // Handle simulation updates
    };

    return () => workerRef.current?.terminate();
  }, []);

  const loadCheckpoint = async (checkpoint: string) => {
    workerRef.current?.postMessage({ type: 'load', checkpoint });
  };

  return { network, loadCheckpoint, ... };
}
```

```ts
// workers/simulation.ts
import init, { WasmNetwork } from '../wasm/gilgamesh';

let network: WasmNetwork | null = null;

self.onmessage = async (e) => {
  const { type, ...data } = e.data;

  switch (type) {
    case 'init':
      await init();
      network = new WasmNetwork(data.config);
      break;

    case 'step':
      if (!network) return;
      const states = network.forward_step(data.input);
      const spikes = network.get_spike_mask();
      self.postMessage({ type: 'frame', states, spikes });
      break;
  }
};
```

---

## Comparison

| Aspect | Hybrid (A) | Full WASM (B) |
|--------|-----------|---------------|
| **Performance** | Native Rust speed | ~50-70% of native |
| **Latency** | WebSocket overhead (1-5ms) | Direct memory access |
| **Remote Access** | Yes - connect from anywhere | No - browser only |
| **Deployment** | Server + frontend | Static files only |
| **MNIST Data** | Server loads files | Must bundle or fetch |
| **Large Networks** | Handles easily | Browser memory limits |
| **Development** | Two processes | Single build |
| **Offline** | No (needs server) | Yes |

### Recommendation: **Hybrid (Option A)**

Reasons:
1. Your physics simulation benefits from native speed
2. MNIST dataset (50MB+) stays on server
3. Remote access is valuable - run on workstation, view on laptop
4. Easier to integrate with existing checkpoints and data pipelines
5. Can still deploy frontend to CDN if server has public IP

---

## Implementation Roadmap

### Phase 1: WebSocket Server (2-3 files)
1. Add `axum`, `tokio`, `tokio-tungstenite` dependencies
2. Create `src/web/mod.rs` with basic WebSocket server
3. Define message protocol types
4. Add `web` command to CLI

### Phase 2: Simulation Bridge (1-2 files)
1. Create simulation runner that broadcasts to WebSocket
2. Handle control messages (pause, next, speed)
3. Add training mode with epoch broadcasts

### Phase 3: Frontend Core (React)
1. Set up Vite + React + TypeScript + Tailwind
2. WebSocket hook with reconnection
3. Basic layout with panels

### Phase 4: Visualization Components
1. NetworkCanvas with WebGL neuron rendering
2. TrainingCharts with Chart.js/Recharts
3. ControlPanel with playback controls
4. ConfigEditor with JSON editing

### Phase 5: Polish
1. Responsive design
2. Dark/light themes
3. Keyboard shortcuts
4. Export capabilities (screenshots, data)

---

## File Structure After Migration

```
gilgamesh/
├── Cargo.toml                  # Add web feature
├── src/
│   ├── main.rs
│   ├── lib.rs
│   ├── cli/
│   │   └── commands/
│   │       └── web.rs          # NEW: web command
│   ├── web/                    # NEW: web server module
│   │   ├── mod.rs
│   │   ├── server.rs
│   │   ├── protocol.rs
│   │   └── handlers.rs
│   ├── network/                # unchanged
│   ├── train/                  # unchanged
│   └── ...
├── frontend/                   # NEW: web frontend
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── components/
│       ├── hooks/
│       └── styles/
└── docs/
    └── web-architecture-proposal.md  # this file
```

### Features After Migration

```toml
[features]
default = []
web = ["dep:axum", "dep:tokio", "dep:tokio-tungstenite", "dep:tower-http"]
# Keep old features for backwards compatibility during transition
visualization = ["dep:rerun"]
dashboard = ["dep:eframe", "dep:egui", "dep:egui_plot"]
animation = ["dep:nannou"]
```

Eventually remove `visualization`, `dashboard`, `animation` features once web UI is complete.
