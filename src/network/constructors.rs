//! Network constructor variants and shared builder helpers.

use crate::layers::Linear;
use crate::neurons::Leaky;
use crate::surrogate::SurrogateGradient;

use super::network::Network;

const DEFAULT_SPIKE_SCALE: f32 = 1.0;
const DEFAULT_SPIKE_GRAD_SLOPE: f32 = 25.0;

impl Network {
    fn default_spike_grad() -> SurrogateGradient {
        SurrogateGradient::fast_sigmoid(DEFAULT_SPIKE_GRAD_SLOPE)
    }

    fn build_with_lif_layers(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        seed: u64,
        lif1: Leaky,
        lif2: Leaky,
    ) -> Self {
        let spike_grad = Self::default_spike_grad();
        Self {
            fc1: Linear::with_seed(input_size, hidden_size, false, seed),
            lif1: lif1.with_spike_grad(spike_grad.clone()),
            fc2: Linear::with_seed(hidden_size, output_size, false, seed.wrapping_add(1)),
            lif2: lif2.with_spike_grad(spike_grad),
            spiking_input: false,
            spike_scale: DEFAULT_SPIKE_SCALE,
            dac_max: f32::INFINITY,
        }
    }

    /// Create a new network with specified architecture (defaults to Physics mode)
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        beta: f32,
        seed: u64,
    ) -> Self {
        Self::build_with_lif_layers(
            input_size,
            hidden_size,
            output_size,
            seed,
            Leaky::new(hidden_size, beta),
            Leaky::new(output_size, beta),
        )
    }

    /// Create a new network in Physics mode with RC dynamics
    pub fn new_physics(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        tau_m: f32,
        dt: f32,
        seed: u64,
    ) -> Self {
        Self::build_with_lif_layers(
            input_size,
            hidden_size,
            output_size,
            seed,
            Leaky::new_physics(hidden_size, tau_m, dt),
            Leaky::new_physics(output_size, tau_m, dt),
        )
    }

    /// Create a new network in Physics mode with pulse stretching
    pub fn new_physics_with_pulse(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        tau_m: f32,
        dt: f32,
        tau_pulse: f32,
        v_peak: f32,
        seed: u64,
    ) -> Self {
        Self::build_with_lif_layers(
            input_size,
            hidden_size,
            output_size,
            seed,
            Leaky::new_physics_with_pulse(hidden_size, tau_m, dt, tau_pulse, v_peak),
            Leaky::new_physics_with_pulse(output_size, tau_m, dt, tau_pulse, v_peak),
        )
    }

    /// Create a new network in Physics mode with threshold adaptation
    pub fn new_physics_with_adaptation(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        tau_m: f32,
        dt: f32,
        tau_theta: f32,
        theta_low: f32,
        theta_high: f32,
        seed: u64,
    ) -> Self {
        Self::build_with_lif_layers(
            input_size,
            hidden_size,
            output_size,
            seed,
            Leaky::new_physics_with_threshold_adaptation(
                hidden_size,
                tau_m,
                dt,
                tau_theta,
                theta_low,
                theta_high,
            ),
            Leaky::new_physics_with_threshold_adaptation(
                output_size,
                tau_m,
                dt,
                tau_theta,
                theta_low,
                theta_high,
            ),
        )
    }
}
