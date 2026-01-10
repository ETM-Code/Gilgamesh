use ndarray::Array2;

use super::cache::LeakyCache;
use super::leaky::Leaky;

impl Leaky {
    /// Backward pass for a single timestep
    pub fn backward(
        &self,
        grad_spikes: &Array2<f32>,
        grad_mem_next: &Array2<f32>,
        cache: &LeakyCache,
    ) -> (Array2<f32>, Array2<f32>) {
        let surrogate_grad = cache.mem_shifted.mapv(|x| self.spike_grad.backward(x));
        let grad_mem_from_spikes = grad_spikes * &surrogate_grad;
        let grad_mem = &grad_mem_from_spikes + grad_mem_next;
        let grad_input = grad_mem.clone();
        let grad_mem_prev = &grad_mem * self.beta;
        (grad_input, grad_mem_prev)
    }
}

