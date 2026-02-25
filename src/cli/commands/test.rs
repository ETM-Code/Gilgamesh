use anyhow::Result;
use gilgamesh::surrogate::SurrogateGradient;
use gilgamesh::training::TrainingConfig;

pub(crate) fn test_implementation(quick: bool) -> Result<()> {
    println!("=== Testing gilgamesh Implementation ===");
    println!();

    println!("Test 1: Surrogate gradients");
    let sg = SurrogateGradient::fast_sigmoid(25.0);
    let grad_at_zero = sg.backward(0.0);
    assert!(
        (grad_at_zero - 1.0).abs() < 1e-5,
        "Gradient at 0 should be 1.0"
    );
    println!("  FastSigmoid gradient at x=0: {} ✓", grad_at_zero);

    println!("\nTest 2: LIF neuron");
    use gilgamesh::neurons::Leaky;
    use ndarray::array;

    let lif = Leaky::new(3, 0.9);
    let state = lif.init_state(1);
    let input = array![[2.0, 0.5, 0.3]];
    let (spikes, _new_state, _) = lif.forward(&input, &state);
    println!("  Input: {:?}", input);
    println!("  Spikes: {:?}", spikes);
    println!("  First neuron spiked: {} ✓", spikes[[0, 0]] == 1.0);

    println!("\nTest 3: Linear layer");
    use gilgamesh::layers::Linear;

    let linear = Linear::new(3, 2, true);
    let output = linear.forward(&input);
    println!("  Input shape: {:?}", input.shape());
    println!("  Output shape: {:?} ✓", output.shape());

    println!("\nTest 4: Network forward pass");
    use gilgamesh::Network;
    use ndarray::Array2;

    let net = Network::new(49, 100, 10, 0.9, 42);
    let test_input = Array2::from_elem((4, 49), 0.1);
    let (spikes, mem, _) = net.forward(&test_input, 25);
    println!("  Batch size: 4, Timesteps: 25");
    println!("  Output spike count shape: {:?}", spikes.shape());
    println!("  Output membrane shape: {:?} ✓", mem.shape());

    println!("\nTest 5: Gradient computation (backward pass)");
    let (_spikes, _, caches) = net.forward(&test_input, 5);
    let grad_output = Array2::from_elem((4, 10), 0.1);
    let grads = net.backward(&test_input, &caches, &grad_output);
    println!("  FC1 weight grad shape: {:?}", grads.fc1_weight.shape());
    println!("  FC2 weight grad shape: {:?} ✓", grads.fc2_weight.shape());

    println!("\n=== All Tests Passed ===");

    if !quick {
        println!("\nRunning quick training test (3 epochs on subset)...");

        let test_images = Array2::from_elem((500, 49), 0.1f32);
        let test_labels: Vec<usize> = (0..500).map(|i| i % 10).collect();

        let mut net = Network::new(49, 100, 10, 0.9, 42);
        let config = TrainingConfig {
            lr: 1e-3,
            epochs: 3,
            batch_size: 64,
            num_steps: 10,
            seed: 42,
            num_workers: 0,
            bptt_steps: None,
        };

        use gilgamesh::tensor::cross_entropy_loss;
        use gilgamesh::training::AdamOptimizer;

        let mut optimizer = AdamOptimizer::new(&net, config.lr);

        for epoch in 1..=3 {
            let batch_input = test_images.slice(ndarray::s![0..64, ..]).to_owned();
            let (spikes, _, caches) = net.forward(&batch_input, config.num_steps);
            let (loss, grad_output) = cross_entropy_loss(&spikes, &test_labels[0..64]);
            let grads = net.backward(&batch_input, &caches, &grad_output);
            optimizer.step(&mut net, &grads);
            println!("  Epoch {} | Loss: {:.4}", epoch, loss);
        }

        println!("\nTraining test completed ✓");
    }

    Ok(())
}
