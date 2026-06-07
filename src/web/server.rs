//! WebSocket server for web UI
//!
//! Handles client connections and broadcasts simulation state.

use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    extract::{
        ws::{Message, WebSocket},
        State, WebSocketUpgrade,
    },
    response::IntoResponse,
    routing::get,
    Router,
};
use futures::{SinkExt, StreamExt};
use tokio::sync::{broadcast, RwLock};
use tower_http::{
    cors::{Any, CorsLayer},
    services::ServeDir,
};

use super::protocol::{ClientMessage, ServerMessage, SimulationMode};
use super::simulation::SimulationState;

/// Shared application state
pub struct AppState {
    /// Current simulation state
    pub simulation: RwLock<SimulationState>,
    /// Broadcast channel for server messages
    pub tx: broadcast::Sender<ServerMessage>,
    /// Data directory for MNIST
    pub data_dir: PathBuf,
}

impl AppState {
    pub fn new(data_dir: PathBuf) -> Self {
        let (tx, _) = broadcast::channel(256);
        Self {
            simulation: RwLock::new(SimulationState::new()),
            tx,
            data_dir,
        }
    }

    /// Broadcast a message to all connected clients
    pub fn broadcast(&self, msg: ServerMessage) {
        let _ = self.tx.send(msg);
    }

    /// Broadcast an `Ack` for the named command.
    fn ack(&self, command: impl Into<String>) {
        self.broadcast(ServerMessage::Ack {
            command: command.into(),
        });
    }

    /// Take the simulation write lock, apply `mutate`, then broadcast an `Ack`.
    ///
    /// Collapses the "lock, mutate, ack" trio shared by the simple control
    /// commands (pause/resume/next/prev/...).
    async fn with_sim_mut(
        &self,
        command: impl Into<String>,
        mutate: impl FnOnce(&mut SimulationState),
    ) {
        {
            let mut sim = self.simulation.write().await;
            mutate(&mut sim);
        }
        self.ack(command);
    }
}

/// Run the web server
pub async fn run_server(
    port: u16,
    data_dir: PathBuf,
    checkpoint: Option<PathBuf>,
    frontend_dir: Option<PathBuf>,
) -> anyhow::Result<()> {
    let state = Arc::new(AppState::new(data_dir.clone()));

    // Load checkpoint if provided
    if let Some(checkpoint_path) = checkpoint {
        let mut sim = state.simulation.write().await;
        sim.load_checkpoint(&checkpoint_path, &data_dir)?;
        println!("Loaded checkpoint: {}", checkpoint_path.display());
    }

    // Build router
    let app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/api/status", get(status_handler))
        .with_state(state.clone());

    // Serve frontend static files if directory exists
    let app = if let Some(frontend) = frontend_dir {
        if frontend.exists() {
            app.fallback_service(ServeDir::new(frontend))
        } else {
            println!(
                "Warning: Frontend directory not found: {}",
                frontend.display()
            );
            app
        }
    } else {
        // Try default location
        let default_frontend = PathBuf::from("frontend/dist");
        if default_frontend.exists() {
            app.fallback_service(ServeDir::new(default_frontend))
        } else {
            app
        }
    };

    // Add CORS for development
    let app = app.layer(
        CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any),
    );

    let addr = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("gilgamesh web UI running at http://localhost:{}", port);

    // Start simulation loop in background
    let state_clone = state.clone();
    tokio::spawn(async move {
        super::simulation::run_simulation_loop(state_clone).await;
    });

    axum::serve(listener, app).await?;
    Ok(())
}

/// Handle WebSocket upgrade
async fn ws_handler(ws: WebSocketUpgrade, State(state): State<Arc<AppState>>) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

/// Handle individual WebSocket connection
async fn handle_socket(socket: WebSocket, state: Arc<AppState>) {
    let (mut sender, mut receiver) = socket.split();

    // Subscribe to broadcast channel
    let mut rx = state.tx.subscribe();

    // Send initial status
    {
        let sim = state.simulation.read().await;
        let status = ServerMessage::Status {
            mode: sim.mode,
            config: sim.config.clone(),
            total_samples: sim.total_samples,
            checkpoint_loaded: sim.checkpoint_path.clone(),
        };
        let json = serde_json::to_string(&status).unwrap();
        let _ = sender.send(Message::Text(json.into())).await;

        // Send network topology if loaded
        if let Some(topology) = sim.get_topology() {
            let json = serde_json::to_string(&topology).unwrap();
            let _ = sender.send(Message::Text(json.into())).await;
        }
    }

    // Spawn task to forward broadcasts to this client
    let mut send_task = tokio::spawn(async move {
        while let Ok(msg) = rx.recv().await {
            let json = serde_json::to_string(&msg).unwrap();
            if sender.send(Message::Text(json.into())).await.is_err() {
                break;
            }
        }
    });

    // Handle incoming messages from client
    let state_clone = state.clone();
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            if let Message::Text(text) = msg {
                match serde_json::from_str::<ClientMessage>(&text) {
                    Ok(cmd) => {
                        handle_command(cmd, &state_clone).await;
                    }
                    Err(e) => {
                        eprintln!("Failed to parse client message: {} - Input: {}", e, text);
                    }
                }
            }
        }
    });

    // Wait for either task to finish
    tokio::select! {
        _ = &mut send_task => recv_task.abort(),
        _ = &mut recv_task => send_task.abort(),
    }
}

/// Handle a client command
async fn handle_command(cmd: ClientMessage, state: &Arc<AppState>) {
    match cmd {
        ClientMessage::GetStatus => {
            let sim = state.simulation.read().await;
            state.broadcast(ServerMessage::Status {
                mode: sim.mode,
                config: sim.config.clone(),
                total_samples: sim.total_samples,
                checkpoint_loaded: sim.checkpoint_path.clone(),
            });
        }

        ClientMessage::NextSample => {
            state
                .with_sim_mut("next_sample", |sim| sim.next_sample())
                .await;
        }

        ClientMessage::PrevSample => {
            state
                .with_sim_mut("prev_sample", |sim| sim.prev_sample())
                .await;
        }

        ClientMessage::JumpToSample { index } => {
            state
                .with_sim_mut(format!("jump_to_sample:{}", index), |sim| {
                    sim.jump_to_sample(index)
                })
                .await;
        }

        ClientMessage::RandomSample => {
            state
                .with_sim_mut("random_sample", |sim| sim.random_sample())
                .await;
        }

        ClientMessage::Pause => {
            state
                .with_sim_mut("pause", |sim| sim.paused = true)
                .await;
        }

        ClientMessage::Resume => {
            state
                .with_sim_mut("resume", |sim| sim.paused = false)
                .await;
        }

        ClientMessage::SetSpeed { speed } => {
            state
                .with_sim_mut(format!("set_speed:{}", speed), |sim| {
                    sim.speed = speed.clamp(0.1, 10.0)
                })
                .await;
        }

        ClientMessage::SetEndOfSampleBehavior { behavior } => {
            println!("Setting end-of-sample behavior to: {:?}", behavior);
            state
                .with_sim_mut(format!("set_end_of_sample_behavior:{:?}", behavior), |sim| {
                    sim.set_end_of_sample_behavior(behavior)
                })
                .await;
        }

        ClientMessage::RestartSample => {
            state
                .with_sim_mut("restart_sample", |sim| sim.restart_sample())
                .await;
        }

        ClientMessage::StartTraining { config } => {
            state
                .with_sim_mut("start_training", |sim| {
                    sim.config = config;
                    sim.mode = SimulationMode::Training;
                    // Training loop will pick this up
                })
                .await;
        }

        ClientMessage::StopTraining => {
            state
                .with_sim_mut("stop_training", |sim| sim.mode = SimulationMode::Idle)
                .await;
        }

        ClientMessage::LoadCheckpoint { path } => {
            let path = PathBuf::from(&path);
            let data_dir = state.data_dir.clone();
            let mut sim = state.simulation.write().await;
            match sim.load_checkpoint(&path, &data_dir) {
                Ok(()) => {
                    state.broadcast(ServerMessage::Status {
                        mode: sim.mode,
                        config: sim.config.clone(),
                        total_samples: sim.total_samples,
                        checkpoint_loaded: sim.checkpoint_path.clone(),
                    });
                }
                Err(e) => {
                    state.broadcast(ServerMessage::Error {
                        message: format!("Failed to load checkpoint: {}", e),
                    });
                }
            }
        }

        ClientMessage::UpdateConfig { config } => {
            state
                .with_sim_mut("update_config", |sim| sim.config = config)
                .await;
        }

        ClientMessage::GetWeights => {
            let sim = state.simulation.read().await;
            if let Some(weights) = sim.get_weight_matrices() {
                for (name, data, rows, cols) in weights {
                    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    state.broadcast(ServerMessage::WeightMatrix {
                        layer: name,
                        data,
                        rows,
                        cols,
                        min,
                        max,
                    });
                }
            }
        }
    }
}

/// Simple status endpoint for health checks
async fn status_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let sim = state.simulation.read().await;
    let status = serde_json::json!({
        "status": "ok",
        "mode": sim.mode,
        "total_samples": sim.total_samples,
        "checkpoint_loaded": sim.checkpoint_path,
    });
    axum::Json(status)
}
