use anyhow::Result;

#[cfg(feature = "dashboard")]
use anyhow::Context;

#[cfg(feature = "dashboard")]
use gilgamesh::config::Config;

#[cfg(feature = "dashboard")]
use gilgamesh::network::Network;

#[cfg(feature = "dashboard")]
use gilgamesh::surrogate::SurrogateGradient;

#[cfg(feature = "dashboard")]
pub(crate) fn run_dashboard(config: Option<String>, epochs: usize, data_dir: &str) -> Result<()> {
    use gilgamesh::dashboard::{create_shared_metrics, DashboardApp};
    use std::thread;

    let mut cfg = if let Some(ref path) = config {
        Config::load(path).with_context(|| format!("Failed to load config from {}", path))?
    } else {
        Config::default()
    };
    cfg.training.epochs = epochs;

    println!("Loading MNIST dataset...");
    let dataset = gilgamesh::data::MnistDataset::load(data_dir).context("Failed to load MNIST dataset")?;

    let input_size = dataset.feature_dim();
    let hidden_size = cfg.network.hidden_size;
    let output_size = cfg.network.output_size;

    let architecture = format!("{} → {} → {}", input_size, hidden_size, output_size);
    let metrics = create_shared_metrics(epochs, architecture);
    let metrics_clone = metrics.clone();

    {
        let mut metrics_guard = metrics.lock().unwrap();
        metrics_guard.is_training = true;
    }

    let thread_config = cfg.clone();
    let thread_data_dir = data_dir.to_string();
    thread::spawn(move || {
        let result = run_training_for_dashboard(&thread_config, &thread_data_dir, metrics_clone);
        if let Err(e) = result {
            eprintln!("Training error: {}", e);
        }
    });

    DashboardApp::run(metrics).map_err(|e| anyhow::anyhow!("Dashboard error: {}", e))
}

#[cfg(feature = "dashboard")]
fn run_training_for_dashboard(
    cfg: &Config,
    data_dir: &str,
    metrics: gilgamesh::dashboard::SharedMetrics,
) -> Result<()> {
    use gilgamesh::training::{Trainer, TrainingConfig};

    let dataset = gilgamesh::data::MnistDataset::load(data_dir)?;

    let input_size = dataset.feature_dim();
    let hidden_size = cfg.network.hidden_size;
    let output_size = cfg.network.output_size;
    let beta = cfg.neuron.beta;
    let seed = cfg.training.seed;

    let mut network = Network::new(input_size, hidden_size, output_size, beta, seed);
    network.lif1.spike_grad = SurrogateGradient::fast_sigmoid(cfg.neuron.slope);
    network.lif2.spike_grad = SurrogateGradient::fast_sigmoid(cfg.neuron.slope);

    let train_config = TrainingConfig {
        lr: cfg.training.lr,
        epochs: cfg.training.epochs,
        batch_size: cfg.training.batch_size,
        num_steps: cfg.training.num_steps,
        seed: cfg.training.seed,
        num_workers: cfg.training.num_workers,
    };

    let mut trainer = Trainer::new(network, train_config);
    trainer.max_grad_norm = Some(1.0);
    trainer.optimizer.set_weight_decay(0.01);

    if cfg.training.epochs > 1 {
        trainer.lr_scheduler = Some(gilgamesh::training::LRScheduler::new(
            cfg.training.lr,
            cfg.training.epochs,
        ));
    }

    for epoch in 1..=cfg.training.epochs {
        let lr = trainer.update_lr_for_epoch(epoch).unwrap_or(cfg.training.lr);
        let (train_loss, train_acc) = trainer.train_epoch(&dataset);
        let test_acc = trainer.evaluate(&dataset);

        {
            let mut metrics_guard = metrics.lock().unwrap();
            metrics_guard.record_epoch(train_loss as f64, train_acc as f64, test_acc as f64, lr as f64);
        }

        println!(
            "Epoch {:3} | Loss: {:.4} | Train: {:.2}% | Test: {:.2}%",
            epoch, train_loss, train_acc, test_acc
        );
    }

    {
        let mut metrics_guard = metrics.lock().unwrap();
        metrics_guard.is_training = false;
        metrics_guard.is_complete = true;
    }

    Ok(())
}

#[cfg(not(feature = "dashboard"))]
pub(crate) fn run_dashboard(_config: Option<String>, _epochs: usize, _data_dir: &str) -> Result<()> {
    println!("Dashboard feature not enabled. Rebuild with --features dashboard");
    Ok(())
}
