use std::time::Instant;
use gilgamesh::neurons::{Leaky, NeuronMode};
use ndarray::Array2;

fn main() {
    let tau_m = 0.0012f32;
    let dt = 1e-6f32;
    let duration = 0.05f32;  // 50ms
    let num_steps = (duration / dt) as usize;
    
    println!("Simulating {} timesteps (50ms at 1µs dt)\n", num_steps);
    
    for &batch_size in &[1, 100, 1000, 5000, 10000] {
        let mode = NeuronMode::physics_with_hardware_timing(
            tau_m, dt, 0.0005, 2.6, 50e-9, 0.00015
        );
        let neuron = Leaky::new(batch_size, (-dt / tau_m).exp())
            .with_mode(mode)
            .with_threshold(0.8);
        
        let mut state = neuron.init_state_with_hardware_timing(batch_size);
        let input = Array2::from_elem((1, batch_size), 0.001f32);
        
        let start = Instant::now();
        for _ in 0..num_steps {
            let (_, new_state, _) = neuron.forward_with_hardware_timing(&input, &state, dt);
            state = new_state;
        }
        let elapsed = start.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0;
        
        println!("{:>6} neurons: {:>7.1}ms  ({:.1}M neuron-steps/sec)", 
            batch_size, ms,
            (batch_size as f64 * num_steps as f64) / elapsed.as_secs_f64() / 1e6
        );
    }
}
