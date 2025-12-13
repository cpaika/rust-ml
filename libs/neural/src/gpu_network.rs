//! GPU-accelerated neural network using WebGPU compute shaders

use crate::gpu::GpuContext;
use crate::{Activation, Network};

/// GPU-accelerated neural network
///
/// This wraps the CPU Network struct and provides GPU-accelerated
/// forward and backward passes using WebGPU compute shaders.
pub struct GpuNetwork {
    /// The underlying network structure (weights, biases)
    network: Network,
    /// GPU context for compute operations
    gpu: GpuContext,

    // Cached values for backpropagation
    layer_inputs: Vec<Vec<f32>>,
    layer_outputs: Vec<Vec<f32>>,
    pre_activations: Vec<Vec<f32>>,
}

impl GpuNetwork {
    /// Create a new GPU-accelerated network (async version for WASM)
    pub async fn new_async(layer_sizes: &[usize]) -> Result<Self, String> {
        let network = Network::new(layer_sizes);
        let gpu = GpuContext::new_async().await?;

        let num_layers = network.layers.len();
        let layer_inputs = vec![Vec::new(); num_layers];
        let layer_outputs = vec![Vec::new(); num_layers];
        let pre_activations = vec![Vec::new(); num_layers];

        Ok(GpuNetwork {
            network,
            gpu,
            layer_inputs,
            layer_outputs,
            pre_activations,
        })
    }

    /// Create GPU network from existing CPU network (async version for WASM)
    pub async fn from_network_async(network: Network) -> Result<Self, String> {
        let gpu = GpuContext::new_async().await?;

        let num_layers = network.layers.len();
        let layer_inputs = vec![Vec::new(); num_layers];
        let layer_outputs = vec![Vec::new(); num_layers];
        let pre_activations = vec![Vec::new(); num_layers];

        Ok(GpuNetwork {
            network,
            gpu,
            layer_inputs,
            layer_outputs,
            pre_activations,
        })
    }

    /// Create MNIST network with GPU acceleration (async version for WASM)
    pub async fn mnist_default_async() -> Result<Self, String> {
        Self::new_async(&[784, 128, 10]).await
    }

    /// Create a new GPU-accelerated network (blocking version for native)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(layer_sizes: &[usize]) -> Result<Self, String> {
        let network = Network::new(layer_sizes);
        let gpu = GpuContext::new()?;

        let num_layers = network.layers.len();
        let layer_inputs = vec![Vec::new(); num_layers];
        let layer_outputs = vec![Vec::new(); num_layers];
        let pre_activations = vec![Vec::new(); num_layers];

        Ok(GpuNetwork {
            network,
            gpu,
            layer_inputs,
            layer_outputs,
            pre_activations,
        })
    }

    /// Create GPU network from existing CPU network (blocking version for native)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_network(network: Network) -> Result<Self, String> {
        let gpu = GpuContext::new()?;

        let num_layers = network.layers.len();
        let layer_inputs = vec![Vec::new(); num_layers];
        let layer_outputs = vec![Vec::new(); num_layers];
        let pre_activations = vec![Vec::new(); num_layers];

        Ok(GpuNetwork {
            network,
            gpu,
            layer_inputs,
            layer_outputs,
            pre_activations,
        })
    }

    /// Create MNIST network with GPU acceleration (blocking version for native)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn mnist_default() -> Result<Self, String> {
        Self::new(&[784, 128, 10])
    }

    /// GPU-accelerated forward pass
    pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.network.layers[0].input_size);

        let mut current = input.to_vec();

        for (layer_idx, layer) in self.network.layers.iter().enumerate() {
            // Store input for backprop
            self.layer_inputs[layer_idx] = current.clone();

            // GPU matmul: z = input * W (1 x input_size) * (input_size x output_size)
            let mut z = self.gpu.matmul(&current, &layer.weights, 1, layer.input_size, layer.output_size);

            // GPU add bias
            self.gpu.add_bias(&mut z, &layer.biases);

            // Store pre-activation values
            self.pre_activations[layer_idx] = z.clone();

            // Apply activation
            current = match self.network.activations[layer_idx] {
                Activation::ReLU => {
                    self.gpu.relu(&mut z);
                    z
                }
                Activation::Softmax => {
                    self.gpu.softmax(&mut z);
                    z
                }
                Activation::None => z,
            };

            // Store output
            self.layer_outputs[layer_idx] = current.clone();
        }

        current
    }

    /// Train on a single sample using GPU acceleration
    pub fn train_sample(&mut self, input: &[f32], label: u8, learning_rate: f32) -> f32 {
        // Forward pass
        let output = self.forward(input);

        // Compute loss (cross-entropy)
        let prob = output[label as usize].max(1e-7);
        let loss = -prob.ln();

        // Backward pass
        self.backward(label, learning_rate);

        loss
    }

    /// GPU-accelerated backward pass
    fn backward(&mut self, label: u8, learning_rate: f32) {
        let num_layers = self.network.layers.len();

        // Output layer gradient (softmax + cross-entropy)
        // For softmax with cross-entropy: dL/dz = output - one_hot(label)
        let output = &self.layer_outputs[num_layers - 1];
        let mut delta: Vec<f32> = output.clone();
        delta[label as usize] -= 1.0;

        // Backpropagate through layers (reverse order)
        for layer_idx in (0..num_layers).rev() {
            // Get layer dimensions and weights for gradient computation
            let input_size = self.network.layers[layer_idx].input_size;
            let output_size = self.network.layers[layer_idx].output_size;
            let input = self.layer_inputs[layer_idx].clone();

            // Compute weight gradients: dW = input^T * delta
            let weight_grads = self.gpu.matmul_at_b(&input, &delta, input_size, 1, output_size);

            // Compute delta for previous layer BEFORE modifying weights
            let new_delta = if layer_idx > 0 {
                // delta_prev = delta * W^T (need old weights)
                let weights = &self.network.layers[layer_idx].weights;
                let mut nd = self.gpu.matmul_a_bt(&delta, weights, 1, output_size, input_size);

                // Apply activation derivative (ReLU backward)
                if self.network.activations[layer_idx - 1] == Activation::ReLU {
                    let prev_z = &self.pre_activations[layer_idx - 1];
                    self.gpu.relu_backward(&mut nd, prev_z);
                }
                Some(nd)
            } else {
                None
            };

            // Now apply weight updates: W = W - lr * dW
            let layer = &mut self.network.layers[layer_idx];
            self.gpu.saxpy(&weight_grads, &mut layer.weights, -learning_rate);

            // Apply bias updates: b = b - lr * delta
            self.gpu.saxpy(&delta, &mut layer.biases, -learning_rate);

            // Move to next delta
            if let Some(nd) = new_delta {
                delta = nd;
            }
        }
    }

    /// Train on a mini-batch using GPU acceleration
    pub fn train_batch(&mut self, inputs: &[Vec<f32>], labels: &[u8], learning_rate: f32) -> f32 {
        assert_eq!(inputs.len(), labels.len());

        let batch_size = inputs.len() as f32;
        let mut total_loss = 0.0;

        // Accumulate gradients
        let mut weight_grads: Vec<Vec<f32>> = self.network.layers
            .iter()
            .map(|l| vec![0.0; l.weights.len()])
            .collect();
        let mut bias_grads: Vec<Vec<f32>> = self.network.layers
            .iter()
            .map(|l| vec![0.0; l.biases.len()])
            .collect();

        for (input, &label) in inputs.iter().zip(labels.iter()) {
            // Forward pass
            let output = self.forward(input);

            // Compute loss
            let prob = output[label as usize].max(1e-7);
            total_loss -= prob.ln();

            // Compute gradients
            let (w_grads, b_grads) = self.compute_gradients(label);

            // Accumulate
            for (layer_idx, (wg, bg)) in w_grads.iter().zip(b_grads.iter()).enumerate() {
                for (i, &g) in wg.iter().enumerate() {
                    weight_grads[layer_idx][i] += g;
                }
                for (i, &g) in bg.iter().enumerate() {
                    bias_grads[layer_idx][i] += g;
                }
            }
        }

        // Apply averaged gradients using GPU
        let lr_scaled = -learning_rate / batch_size;
        for (layer_idx, layer) in self.network.layers.iter_mut().enumerate() {
            self.gpu.saxpy(&weight_grads[layer_idx], &mut layer.weights, lr_scaled);
            self.gpu.saxpy(&bias_grads[layer_idx], &mut layer.biases, lr_scaled);
        }

        total_loss / batch_size
    }

    /// Compute gradients without applying them (for batch training)
    fn compute_gradients(&self, label: u8) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let num_layers = self.network.layers.len();
        let mut weight_grads: Vec<Vec<f32>> = Vec::with_capacity(num_layers);
        let mut bias_grads: Vec<Vec<f32>> = Vec::with_capacity(num_layers);

        // Initialize gradient storage
        for layer in &self.network.layers {
            weight_grads.push(vec![0.0; layer.weights.len()]);
            bias_grads.push(vec![0.0; layer.biases.len()]);
        }

        // Output layer gradient
        let output = &self.layer_outputs[num_layers - 1];
        let mut delta: Vec<f32> = output.clone();
        delta[label as usize] -= 1.0;

        // Backpropagate through layers
        for layer_idx in (0..num_layers).rev() {
            let layer = &self.network.layers[layer_idx];
            let input = &self.layer_inputs[layer_idx];

            // Weight gradients: dW = input^T * delta
            weight_grads[layer_idx] = self.gpu.matmul_at_b(input, &delta, layer.input_size, 1, layer.output_size);

            // Bias gradients: db = delta
            bias_grads[layer_idx] = delta.clone();

            // Delta for previous layer
            if layer_idx > 0 {
                let mut new_delta = self.gpu.matmul_a_bt(&delta, &layer.weights, 1, layer.output_size, layer.input_size);

                if self.network.activations[layer_idx - 1] == Activation::ReLU {
                    let prev_z = &self.pre_activations[layer_idx - 1];
                    self.gpu.relu_backward(&mut new_delta, prev_z);
                }

                delta = new_delta;
            }
        }

        (weight_grads, bias_grads)
    }

    /// Compute loss and accuracy for evaluation
    pub fn evaluate(&mut self, inputs: &[Vec<f32>], labels: &[u8]) -> (f32, f32) {
        let mut total_loss = 0.0;
        let mut correct = 0;

        for (input, &label) in inputs.iter().zip(labels.iter()) {
            let output = self.forward(input);

            let prob = output[label as usize].max(1e-7);
            total_loss -= prob.ln();

            let predicted = output
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();

            if predicted == label as usize {
                correct += 1;
            }
        }

        let avg_loss = total_loss / inputs.len() as f32;
        let accuracy = correct as f32 / inputs.len() as f32;

        (avg_loss, accuracy)
    }

    /// Predict class for input
    pub fn predict(&mut self, input: &[f32]) -> usize {
        let output = self.forward(input);
        output
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Get underlying network (for visualization or serialization)
    pub fn network(&self) -> &Network {
        &self.network
    }

    /// Get mutable reference to underlying network
    pub fn network_mut(&mut self) -> &mut Network {
        &mut self.network
    }

    /// Convert back to CPU network
    pub fn into_network(self) -> Network {
        self.network
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mnist::MnistLoader;

    #[test]
    fn test_gpu_network_creation() {
        let network = GpuNetwork::new(&[784, 128, 10]);
        assert!(network.is_ok(), "Failed to create GPU network: {:?}", network.err());
    }

    #[test]
    fn test_gpu_forward_pass() {
        let mut gpu_net = GpuNetwork::new(&[4, 3, 2]).expect("Failed to create GPU network");
        let input = vec![1.0, 0.5, -0.5, 0.0];
        let output = gpu_net.forward(&input);

        assert_eq!(output.len(), 2);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Softmax sum should be 1, got {}", sum);
    }

    #[test]
    fn test_gpu_vs_cpu_forward() {
        // Create identical networks
        let cpu_net = Network::new(&[4, 3, 2]);
        let mut gpu_net = GpuNetwork::from_network(cpu_net.clone()).expect("Failed to create GPU network");
        let mut cpu_net = cpu_net;

        let input = vec![0.5, 0.3, -0.2, 0.1];

        let cpu_output = cpu_net.forward(&input);
        let gpu_output = gpu_net.forward(&input);

        // Results should match within floating point tolerance
        for (cpu, gpu) in cpu_output.iter().zip(gpu_output.iter()) {
            assert!((cpu - gpu).abs() < 1e-4,
                "CPU and GPU outputs differ: CPU {:?} vs GPU {:?}", cpu_output, gpu_output);
        }
    }

    #[test]
    fn test_gpu_training_step() {
        let mut network = GpuNetwork::new(&[4, 3, 2]).expect("Failed to create GPU network");
        let input = vec![1.0, 0.5, -0.5, 0.0];

        let loss1 = network.train_sample(&input, 0, 0.1);
        let loss2 = network.train_sample(&input, 0, 0.1);

        // Loss should generally decrease with training
        println!("GPU training - Loss 1: {}, Loss 2: {}", loss1, loss2);
        // Allow some variance but expect it's not increasing dramatically
        assert!(loss2 < loss1 * 2.0, "Loss increased too much: {} -> {}", loss1, loss2);
    }

    #[test]
    fn test_gpu_mnist_training() {
        let mut network = GpuNetwork::mnist_default().expect("Failed to create GPU network");

        let loader = MnistLoader::from_file(
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../perceptron/digit-recognizer/train.csv"),
            32
        ).unwrap();

        let samples = loader.samples();
        let inputs: Vec<Vec<f32>> = samples[0..32]
            .iter()
            .map(|s| s.normalized_pixels().iter().map(|&x| x as f32).collect())
            .collect();
        let labels: Vec<u8> = samples[0..32].iter().map(|s| s.label()).collect();

        let loss1 = network.train_batch(&inputs, &labels, 0.1);
        let loss2 = network.train_batch(&inputs, &labels, 0.1);

        println!("GPU MNIST batch training - Loss 1: {}, Loss 2: {}", loss1, loss2);
        // Training should generally reduce loss
        assert!(loss2 < loss1 * 1.5, "Loss increased too much");
    }

    #[test]
    fn test_gpu_vs_cpu_training() {
        // Create identical networks
        let cpu_net = Network::new(&[4, 3, 2]);
        let mut gpu_net = GpuNetwork::from_network(cpu_net.clone()).expect("Failed to create GPU network");
        let mut cpu_net = cpu_net;

        let input = vec![0.5, 0.3, -0.2, 0.1];

        // Train both with same input
        let cpu_loss = cpu_net.train_sample(&input, 0, 0.1);
        let gpu_loss = gpu_net.train_sample(&input, 0, 0.1);

        // Losses should be similar
        assert!((cpu_loss - gpu_loss).abs() < 0.1,
            "CPU and GPU losses differ significantly: CPU {} vs GPU {}", cpu_loss, gpu_loss);

        // Weights should be similar after training
        for (cpu_layer, gpu_layer) in cpu_net.layers.iter().zip(gpu_net.network().layers.iter()) {
            for (cpu_w, gpu_w) in cpu_layer.weights.iter().zip(gpu_layer.weights.iter()) {
                assert!((cpu_w - gpu_w).abs() < 0.01,
                    "Weights differ after training: CPU {} vs GPU {}", cpu_w, gpu_w);
            }
        }
    }

    // ==================== ASYNC TESTS ====================

    #[test]
    fn test_gpu_network_async_creation() {
        // Test async creation using pollster to block
        let network = pollster::block_on(GpuNetwork::new_async(&[784, 128, 10]));
        assert!(network.is_ok(), "Failed to create GPU network async: {:?}", network.err());
    }

    #[test]
    fn test_gpu_network_async_from_network() {
        let cpu_net = Network::new(&[4, 3, 2]);
        let gpu_net = pollster::block_on(GpuNetwork::from_network_async(cpu_net));
        assert!(gpu_net.is_ok(), "Failed to create GPU network from CPU network async: {:?}", gpu_net.err());
    }

    #[test]
    fn test_gpu_network_async_mnist_default() {
        let network = pollster::block_on(GpuNetwork::mnist_default_async());
        assert!(network.is_ok(), "Failed to create MNIST GPU network async: {:?}", network.err());
    }

    #[test]
    fn test_async_forward_pass() {
        let mut gpu_net = pollster::block_on(GpuNetwork::new_async(&[4, 3, 2]))
            .expect("Failed to create GPU network async");
        let input = vec![1.0, 0.5, -0.5, 0.0];
        let output = gpu_net.forward(&input);

        assert_eq!(output.len(), 2);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Softmax sum should be 1, got {}", sum);
    }

    #[test]
    fn test_async_training() {
        let cpu_net = Network::new(&[4, 3, 2]);
        let mut gpu_net = pollster::block_on(GpuNetwork::from_network_async(cpu_net.clone()))
            .expect("Failed to create GPU network async");
        let mut cpu_net = cpu_net;

        let input = vec![0.5, 0.3, -0.2, 0.1];

        let cpu_output = cpu_net.forward(&input);
        let gpu_output = gpu_net.forward(&input);

        // Results should match
        for (cpu, gpu) in cpu_output.iter().zip(gpu_output.iter()) {
            assert!((cpu - gpu).abs() < 1e-4,
                "Async GPU output differs from CPU: CPU {:?} vs GPU {:?}", cpu_output, gpu_output);
        }

        // Train both
        let cpu_loss = cpu_net.train_sample(&input, 0, 0.1);
        let gpu_loss = gpu_net.train_sample(&input, 0, 0.1);

        assert!((cpu_loss - gpu_loss).abs() < 0.1,
            "Async GPU loss differs from CPU: CPU {} vs GPU {}", cpu_loss, gpu_loss);
    }
}
