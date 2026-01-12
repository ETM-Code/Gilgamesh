use anyhow::Result;

#[cfg(feature = "web")]
use std::path::PathBuf;

#[cfg(feature = "web")]
use crate::cli::utils::find_latest_checkpoint;

#[cfg(feature = "web")]
pub(crate) fn run_web_server(
    port: u16,
    checkpoint: Option<String>,
    data_dir: &str,
    open_browser: bool,
) -> Result<()> {
    use gilgamesh::web::run_server;

    println!("=== gilgamesh Web UI ===");
    println!();

    let checkpoint_path = match checkpoint {
        Some(path) => Some(PathBuf::from(path)),
        None => find_latest_checkpoint().map(PathBuf::from),
    };

    if let Some(ref path) = checkpoint_path {
        println!("Will load checkpoint: {}", path.display());
    } else {
        println!("No checkpoint specified - starting in idle mode");
        println!("Load a checkpoint via the web UI or restart with --checkpoint");
    }

    let data_path = PathBuf::from(data_dir);

    // Open browser if requested
    if open_browser {
        let url = format!("http://localhost:{}", port);
        println!("Opening browser at {}", url);
        if let Err(e) = open::that(&url) {
            println!("Warning: Could not open browser: {}", e);
        }
    }

    // Run the server (blocking)
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        run_server(port, data_path, checkpoint_path, None).await
    })?;

    Ok(())
}

#[cfg(not(feature = "web"))]
pub(crate) fn run_web_server(
    _port: u16,
    _checkpoint: Option<String>,
    _data_dir: &str,
    _open_browser: bool,
) -> Result<()> {
    println!("Web feature not enabled. Rebuild with --features web");
    println!();
    println!("  cargo run --features web -- web");
    Ok(())
}
