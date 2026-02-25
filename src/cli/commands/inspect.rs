use anyhow::Result;

#[cfg(feature = "dashboard")]
use anyhow::Context;

#[cfg(feature = "dashboard")]
use crate::cli::utils::find_latest_checkpoint;

#[cfg(feature = "dashboard")]
pub(crate) fn run_inspector(
    checkpoint: Option<String>,
    data_dir: &str,
    num_steps: usize,
    seed: u64,
) -> Result<()> {
    use gilgamesh::inspector::InspectorApp;

    let checkpoint_path = match checkpoint {
        Some(path) => path,
        None => match find_latest_checkpoint() {
            Some(path) => {
                println!("Using most recent checkpoint: {}", path);
                path
            }
            None => {
                anyhow::bail!(
                    "No checkpoint specified and no checkpoint files found.\n\
                        Train a model first: gilgamesh train --save-checkpoint model.json"
                );
            }
        },
    };

    println!("=== gilgamesh Inspector ===");
    println!("Checkpoint: {}", checkpoint_path);
    println!("Data dir:   {}", data_dir);
    println!("Timesteps:  {}", num_steps);
    println!();

    let app = InspectorApp::from_checkpoint(&checkpoint_path, data_dir, num_steps, seed)
        .context("Failed to initialize inspector")?;

    app.run()
        .map_err(|e| anyhow::anyhow!("Inspector error: {}", e))
}

#[cfg(not(feature = "dashboard"))]
pub(crate) fn run_inspector(
    _checkpoint: Option<String>,
    _data_dir: &str,
    _num_steps: usize,
    _seed: u64,
) -> Result<()> {
    println!("Inspector feature not enabled. Rebuild with --features dashboard");
    Ok(())
}
