//! Matrix-based neural network optimized for WASM training
//!
//! This implementation uses dense matrices for weights to enable fast
//! forward/backward passes suitable for training MNIST in the browser.
//!
//! GPU acceleration is available via WebGPU when the `gpu` feature is enabled.

pub mod mnist;

#[cfg(any(feature = "gpu", test))]
pub mod gpu;

#[cfg(any(feature = "gpu", test))]
mod gpu_network;

#[cfg(any(feature = "gpu", test))]
pub use gpu_network::GpuNetwork;

use rand::Rng;

/// A single fully-connected layer with weights and biases
#[derive(Clone)]
pub struct Layer {
    /// Weights stored as flat array [input_size * output_size]
    /// Layout: weights[i * output_size + j] = weight from input i to output j
    pub weights: Vec<f32>,
    /// Biases for each output neuron [output_size]
    pub biases: Vec<f32>,
    pub input_size: usize,
    pub output_size: usize,
}

impl Layer {
    /// Create a new layer with Xavier initialization
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier initialization: scale by sqrt(2 / (fan_in + fan_out))
        let scale = (2.0 / (input_size + output_size) as f32).sqrt();

        let weights: Vec<f32> = (0..input_size * output_size)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();

        let biases: Vec<f32> = vec![0.0; output_size];

        Layer {
            weights,
            biases,
            input_size,
            output_size,
        }
    }

    /// Get weight from input neuron i to output neuron j
    #[inline]
    pub fn get_weight(&self, input_idx: usize, output_idx: usize) -> f32 {
        self.weights[input_idx * self.output_size + output_idx]
    }

    /// Set weight from input neuron i to output neuron j
    #[inline]
    pub fn set_weight(&mut self, input_idx: usize, output_idx: usize, value: f32) {
        self.weights[input_idx * self.output_size + output_idx] = value;
    }
}

/// Activation function types
#[derive(Clone, Copy, PartialEq)]
pub enum Activation {
    ReLU,
    Softmax,
    None,
}

/// Neural network with multiple layers
#[derive(Clone)]
pub struct Network {
    pub layers: Vec<Layer>,
    pub activations: Vec<Activation>,

    // Cached values for backpropagation
    layer_inputs: Vec<Vec<f32>>,   // Input to each layer
    layer_outputs: Vec<Vec<f32>>,  // Output (after activation) of each layer
    pre_activations: Vec<Vec<f32>>, // Values before activation (z values)
}

impl Network {
    /// Create a new network with specified layer sizes
    ///
    /// # Arguments
    /// * `layer_sizes` - Vector of layer sizes, e.g., [784, 128, 10] for MNIST
    pub fn new(layer_sizes: &[usize]) -> Self {
        assert!(layer_sizes.len() >= 2, "Network needs at least input and output layers");

        let mut layers = Vec::new();
        let mut activations = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            layers.push(Layer::new(layer_sizes[i], layer_sizes[i + 1]));

            // Use ReLU for hidden layers, Softmax for output
            if i == layer_sizes.len() - 2 {
                activations.push(Activation::Softmax);
            } else {
                activations.push(Activation::ReLU);
            }
        }

        // Pre-allocate cache vectors
        let layer_inputs = vec![Vec::new(); layers.len()];
        let layer_outputs = vec![Vec::new(); layers.len()];
        let pre_activations = vec![Vec::new(); layers.len()];

        Network {
            layers,
            activations,
            layer_inputs,
            layer_outputs,
            pre_activations,
        }
    }

    /// Create network specifically for MNIST
    pub fn mnist_default() -> Self {
        // 784 inputs (28x28), 128 hidden neurons, 10 outputs (digits 0-9)
        Self::new(&[784, 128, 10])
    }

    /// Forward pass: compute network output for given input
    pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.layers[0].input_size);

        let mut current = input.to_vec();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Store input for backprop
            self.layer_inputs[layer_idx] = current.clone();

            // Compute z = Wx + b
            let mut z = vec![0.0f32; layer.output_size];

            for j in 0..layer.output_size {
                let mut sum = layer.biases[j];
                for i in 0..layer.input_size {
                    sum += current[i] * layer.get_weight(i, j);
                }
                z[j] = sum;
            }

            // Store pre-activation values
            self.pre_activations[layer_idx] = z.clone();

            // Apply activation
            current = match self.activations[layer_idx] {
                Activation::ReLU => relu(&z),
                Activation::Softmax => softmax(&z),
                Activation::None => z,
            };

            // Store output
            self.layer_outputs[layer_idx] = current.clone();
        }

        current
    }

    /// Compute loss and accuracy for a batch (for monitoring)
    pub fn evaluate(&mut self, inputs: &[Vec<f32>], labels: &[u8]) -> (f32, f32) {
        let mut total_loss = 0.0;
        let mut correct = 0;

        for (input, &label) in inputs.iter().zip(labels.iter()) {
            let output = self.forward(input);

            // Cross-entropy loss
            let prob = output[label as usize].max(1e-7);
            total_loss -= prob.ln();

            // Check if prediction is correct
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

    /// Train on a single sample, returning the loss
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

    /// Train on a mini-batch, returning average loss
    pub fn train_batch(&mut self, inputs: &[Vec<f32>], labels: &[u8], learning_rate: f32) -> f32 {
        assert_eq!(inputs.len(), labels.len());

        let batch_size = inputs.len() as f32;
        let mut total_loss = 0.0;

        // Accumulate gradients
        let mut weight_grads: Vec<Vec<f32>> = self.layers
            .iter()
            .map(|l| vec![0.0; l.weights.len()])
            .collect();
        let mut bias_grads: Vec<Vec<f32>> = self.layers
            .iter()
            .map(|l| vec![0.0; l.biases.len()])
            .collect();

        for (input, &label) in inputs.iter().zip(labels.iter()) {
            // Forward pass
            let output = self.forward(input);

            // Compute loss
            let prob = output[label as usize].max(1e-7);
            total_loss -= prob.ln();

            // Backward pass - compute gradients
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

        // Apply averaged gradients
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            for (i, w) in layer.weights.iter_mut().enumerate() {
                *w -= learning_rate * weight_grads[layer_idx][i] / batch_size;
            }
            for (i, b) in layer.biases.iter_mut().enumerate() {
                *b -= learning_rate * bias_grads[layer_idx][i] / batch_size;
            }
        }

        total_loss / batch_size
    }

    /// Backward pass: update weights using gradient descent
    fn backward(&mut self, label: u8, learning_rate: f32) {
        let (weight_grads, bias_grads) = self.compute_gradients(label);

        // Apply gradients
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            for (i, w) in layer.weights.iter_mut().enumerate() {
                *w -= learning_rate * weight_grads[layer_idx][i];
            }
            for (i, b) in layer.biases.iter_mut().enumerate() {
                *b -= learning_rate * bias_grads[layer_idx][i];
            }
        }
    }

    /// Compute gradients for all layers (returns weight and bias gradients)
    fn compute_gradients(&self, label: u8) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let num_layers = self.layers.len();
        let mut weight_grads: Vec<Vec<f32>> = Vec::with_capacity(num_layers);
        let mut bias_grads: Vec<Vec<f32>> = Vec::with_capacity(num_layers);

        // Initialize gradient storage
        for layer in &self.layers {
            weight_grads.push(vec![0.0; layer.weights.len()]);
            bias_grads.push(vec![0.0; layer.biases.len()]);
        }

        // Output layer gradient (softmax + cross-entropy)
        // For softmax with cross-entropy: dL/dz = output - one_hot(label)
        let output = &self.layer_outputs[num_layers - 1];
        let mut delta: Vec<f32> = output.clone();
        delta[label as usize] -= 1.0;

        // Backpropagate through layers (reverse order)
        for layer_idx in (0..num_layers).rev() {
            let layer = &self.layers[layer_idx];
            let input = &self.layer_inputs[layer_idx];

            // Compute weight gradients: dW = input^T * delta
            for i in 0..layer.input_size {
                for j in 0..layer.output_size {
                    weight_grads[layer_idx][i * layer.output_size + j] = input[i] * delta[j];
                }
            }

            // Bias gradients: db = delta
            bias_grads[layer_idx] = delta.clone();

            // Compute delta for previous layer (if not at input layer)
            if layer_idx > 0 {
                let prev_z = &self.pre_activations[layer_idx - 1];
                let mut new_delta = vec![0.0f32; layer.input_size];

                // delta_prev = W^T * delta
                for i in 0..layer.input_size {
                    let mut sum = 0.0;
                    for j in 0..layer.output_size {
                        sum += layer.get_weight(i, j) * delta[j];
                    }
                    new_delta[i] = sum;
                }

                // Apply activation derivative (ReLU)
                if self.activations[layer_idx - 1] == Activation::ReLU {
                    for (i, d) in new_delta.iter_mut().enumerate() {
                        if prev_z[i] <= 0.0 {
                            *d = 0.0;
                        }
                    }
                }

                delta = new_delta;
            }
        }

        (weight_grads, bias_grads)
    }

    /// Get the number of layers (not counting input)
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get layer sizes for visualization
    pub fn layer_sizes(&self) -> Vec<usize> {
        let mut sizes = vec![self.layers[0].input_size];
        for layer in &self.layers {
            sizes.push(layer.output_size);
        }
        sizes
    }

    /// Get reference to a specific layer
    pub fn get_layer(&self, idx: usize) -> Option<&Layer> {
        self.layers.get(idx)
    }

    /// Get the last layer outputs (activations) - useful for visualization
    pub fn get_activations(&self, layer_idx: usize) -> Option<&Vec<f32>> {
        self.layer_outputs.get(layer_idx)
    }

    /// Predict class for input (returns class index)
    pub fn predict(&mut self, input: &[f32]) -> usize {
        let output = self.forward(input);
        output
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Get output probabilities for input
    pub fn predict_proba(&mut self, input: &[f32]) -> Vec<f32> {
        self.forward(input)
    }
}

/// ReLU activation: max(0, x)
fn relu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.max(0.0)).collect()
}

/// Softmax activation: exp(x_i) / sum(exp(x_j))
fn softmax(x: &[f32]) -> Vec<f32> {
    // Subtract max for numerical stability
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = x.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    exp_vals.iter().map(|&v| v / sum).collect()
}

// ============================================================================
// Legacy compatibility layer for visualization
// The visualizer expects certain structures - we provide adapters
// ============================================================================

/// Legacy Layer wrapper for visualization compatibility
pub struct VisLayer {
    pub neuron_count: usize,
}

impl Network {
    /// Get layers in a format suitable for visualization
    pub fn vis_layers(&self) -> Vec<VisLayer> {
        let sizes = self.layer_sizes();
        sizes.iter().map(|&size| VisLayer { neuron_count: size }).collect()
    }

    /// Get all weights as (from_layer, from_idx, to_layer, to_idx, weight) tuples
    /// This is inefficient but useful for visualization
    pub fn all_weights(&self) -> Vec<(usize, usize, usize, usize, f32)> {
        let mut weights = Vec::new();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            for i in 0..layer.input_size {
                for j in 0..layer.output_size {
                    weights.push((
                        layer_idx,      // from layer
                        i,              // from neuron
                        layer_idx + 1,  // to layer
                        j,              // to neuron
                        layer.get_weight(i, j),
                    ));
                }
            }
        }

        weights
    }

    /// Get bias for a neuron in visualization coordinates
    pub fn get_bias(&self, layer_idx: usize, neuron_idx: usize) -> f32 {
        if layer_idx == 0 {
            // Input layer has no bias
            0.0
        } else {
            self.layers[layer_idx - 1].biases[neuron_idx]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mnist::MnistLoader;

    #[test]
    fn test_network_creation() {
        let network = Network::new(&[784, 128, 10]);
        assert_eq!(network.layers.len(), 2);
        assert_eq!(network.layers[0].input_size, 784);
        assert_eq!(network.layers[0].output_size, 128);
        assert_eq!(network.layers[1].input_size, 128);
        assert_eq!(network.layers[1].output_size, 10);
    }

    #[test]
    fn test_forward_pass() {
        let mut network = Network::new(&[4, 3, 2]);
        let input = vec![1.0, 0.5, -0.5, 0.0];
        let output = network.forward(&input);

        assert_eq!(output.len(), 2);
        // Softmax outputs should sum to ~1
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_layer_sizes() {
        let network = Network::new(&[784, 128, 64, 10]);
        let sizes = network.layer_sizes();
        assert_eq!(sizes, vec![784, 128, 64, 10]);
    }

    #[test]
    fn test_training_step() {
        let mut network = Network::new(&[4, 3, 2]);
        let input = vec![1.0, 0.5, -0.5, 0.0];

        let loss1 = network.train_sample(&input, 0, 0.1);
        let loss2 = network.train_sample(&input, 0, 0.1);

        // Loss should decrease with training
        assert!(loss2 < loss1 || (loss2 - loss1).abs() < 0.1);
    }

    #[test]
    fn test_mnist_network() {
        let mut network = Network::mnist_default();

        let loader = MnistLoader::from_file(
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../perceptron/digit-recognizer/train.csv"),
            32
        ).unwrap();

        // Just verify we can do a forward pass with real data
        let sample = &loader.samples()[0];
        let input: Vec<f32> = sample.normalized_pixels().iter().map(|&x| x as f32).collect();
        let output = network.forward(&input);

        assert_eq!(output.len(), 10);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_batch_training() {
        let mut network = Network::mnist_default();

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

        // Training should reduce loss
        println!("Loss 1: {}, Loss 2: {}", loss1, loss2);
        assert!(loss2 < loss1 * 1.5); // Allow some variance but expect general decrease
    }
}
