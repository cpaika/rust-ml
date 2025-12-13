//! End-to-end tests for GPU training integration
//! These tests exercise the same code paths that run in the browser

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use common::parse_digits_from_bytes;
    use neural::mnist::MnistSample;
    use neural::{GpuNetwork, Network};

    const TRAIN_CSV: &[u8] = include_bytes!("../../perceptron/digit-recognizer/train.csv");
    const LEARNING_RATE: f32 = 0.1;
    const BATCH_SIZE: usize = 32;

    /// Helper to create training samples from embedded CSV
    fn load_samples() -> Vec<MnistSample> {
        let digits = parse_digits_from_bytes(TRAIN_CSV).expect("Failed to parse CSV");
        digits.iter().map(|d| MnistSample::new(d.clone())).collect()
    }

    /// Test basic GPU network creation from CPU network
    #[test]
    fn test_gpu_network_from_cpu_network() {
        let cpu_net = Network::mnist_default();
        let gpu_net = GpuNetwork::from_network(cpu_net.clone());
        assert!(gpu_net.is_ok(), "Failed to create GPU network: {:?}", gpu_net.err());

        let gpu_net = gpu_net.unwrap();
        // Verify network structure matches
        assert_eq!(gpu_net.network().layer_sizes(), cpu_net.layer_sizes());
    }

    /// Test GPU network creation with async API (simulates WASM path)
    #[test]
    fn test_gpu_network_from_cpu_network_async() {
        let cpu_net = Network::mnist_default();
        let gpu_net = pollster::block_on(GpuNetwork::from_network_async(cpu_net.clone()));
        assert!(gpu_net.is_ok(), "Failed to create GPU network async: {:?}", gpu_net.err());

        let gpu_net = gpu_net.unwrap();
        assert_eq!(gpu_net.network().layer_sizes(), cpu_net.layer_sizes());
    }

    /// Test single batch training on GPU
    #[test]
    fn test_gpu_single_batch_training() {
        let samples = load_samples();
        let mut gpu_net = GpuNetwork::mnist_default().expect("Failed to create GPU network");

        let inputs: Vec<Vec<f32>> = samples[0..BATCH_SIZE]
            .iter()
            .map(|s| s.normalized_pixels_f32())
            .collect();
        let labels: Vec<u8> = samples[0..BATCH_SIZE].iter().map(|s| s.label()).collect();

        let loss = gpu_net.train_batch(&inputs, &labels, LEARNING_RATE);
        assert!(loss.is_finite(), "Loss should be finite, got: {}", loss);
        assert!(loss > 0.0, "Loss should be positive for untrained network");
    }

    /// Test multiple batch training (simulates training loop)
    #[test]
    fn test_gpu_multiple_batch_training() {
        let samples = load_samples();
        let mut gpu_net = GpuNetwork::mnist_default().expect("Failed to create GPU network");

        let num_batches = 10;
        let mut losses = Vec::new();

        for batch_idx in 0..num_batches {
            let start = batch_idx * BATCH_SIZE;
            let end = start + BATCH_SIZE;

            let inputs: Vec<Vec<f32>> = samples[start..end]
                .iter()
                .map(|s| s.normalized_pixels_f32())
                .collect();
            let labels: Vec<u8> = samples[start..end].iter().map(|s| s.label()).collect();

            let loss = gpu_net.train_batch(&inputs, &labels, LEARNING_RATE);
            assert!(loss.is_finite(), "Loss should be finite at batch {}", batch_idx);
            losses.push(loss);
        }

        // Loss should generally decrease over training
        let first_avg = losses[..3].iter().sum::<f32>() / 3.0;
        let last_avg = losses[losses.len()-3..].iter().sum::<f32>() / 3.0;
        println!("First 3 avg loss: {}, Last 3 avg loss: {}", first_avg, last_avg);
    }

    /// Test weight synchronization from GPU to CPU (critical for visualization)
    #[test]
    fn test_weight_sync_gpu_to_cpu() {
        let cpu_net = Network::mnist_default();
        let original_weights: Vec<f32> = cpu_net.layers[0].weights.clone();

        let mut gpu_net = GpuNetwork::from_network(cpu_net).expect("Failed to create GPU network");

        // Train on GPU
        let samples = load_samples();
        let inputs: Vec<Vec<f32>> = samples[0..BATCH_SIZE]
            .iter()
            .map(|s| s.normalized_pixels_f32())
            .collect();
        let labels: Vec<u8> = samples[0..BATCH_SIZE].iter().map(|s| s.label()).collect();

        gpu_net.train_batch(&inputs, &labels, LEARNING_RATE);

        // Weights should have changed
        let new_weights = &gpu_net.network().layers[0].weights;
        let mut changed = false;
        for (old, new) in original_weights.iter().zip(new_weights.iter()) {
            if (old - new).abs() > 1e-6 {
                changed = true;
                break;
            }
        }
        assert!(changed, "GPU training should modify weights");
    }

    /// Test the exact flow from train_batches_gpu - simulate full epoch
    #[test]
    fn test_full_epoch_gpu_training() {
        let samples = load_samples();
        let total_samples = samples.len().min(1000); // Use subset for speed
        let num_batches = total_samples / BATCH_SIZE;

        let cpu_net = Network::mnist_default();
        let mut gpu_net = GpuNetwork::from_network(cpu_net.clone()).expect("Failed to create GPU network");
        let mut cpu_net_for_eval = cpu_net;

        println!("Training {} batches (batch size: {})", num_batches, BATCH_SIZE);

        for batch_idx in 0..num_batches {
            let start = batch_idx * BATCH_SIZE;
            let end = (start + BATCH_SIZE).min(total_samples);

            let inputs: Vec<Vec<f32>> = samples[start..end]
                .iter()
                .map(|s| s.normalized_pixels_f32())
                .collect();
            let labels: Vec<u8> = samples[start..end].iter().map(|s| s.label()).collect();

            let loss = gpu_net.train_batch(&inputs, &labels, LEARNING_RATE);
            assert!(loss.is_finite(), "Loss not finite at batch {}: {}", batch_idx, loss);

            // Every 10 batches, sync weights back to CPU (like train_batches_gpu does)
            if batch_idx % 10 == 0 {
                for (cpu_layer, gpu_layer) in cpu_net_for_eval.layers.iter_mut()
                    .zip(gpu_net.network().layers.iter())
                {
                    cpu_layer.weights.copy_from_slice(&gpu_layer.weights);
                    cpu_layer.biases.copy_from_slice(&gpu_layer.biases);
                }

                // Evaluate on CPU
                let val_size = 100;
                let eval_inputs: Vec<Vec<f32>> = samples[..val_size]
                    .iter()
                    .map(|s| s.normalized_pixels_f32())
                    .collect();
                let eval_labels: Vec<u8> = samples[..val_size].iter().map(|s| s.label()).collect();

                let (_, acc) = cpu_net_for_eval.evaluate(&eval_inputs, &eval_labels);
                println!("Batch {}: loss={:.4}, accuracy={:.1}%", batch_idx, loss, acc * 100.0);
            }
        }
    }

    /// Test GPU training matches CPU training behavior
    #[test]
    fn test_gpu_cpu_training_consistency() {
        let samples = load_samples();

        // Create identical networks
        let base_net = Network::mnist_default();
        let mut cpu_net = base_net.clone();
        let mut gpu_net = GpuNetwork::from_network(base_net).expect("Failed to create GPU network");

        // Train both with same data
        let inputs: Vec<Vec<f32>> = samples[0..BATCH_SIZE]
            .iter()
            .map(|s| s.normalized_pixels_f32())
            .collect();
        let labels: Vec<u8> = samples[0..BATCH_SIZE].iter().map(|s| s.label()).collect();

        let cpu_loss = cpu_net.train_batch(&inputs, &labels, LEARNING_RATE);
        let gpu_loss = gpu_net.train_batch(&inputs, &labels, LEARNING_RATE);

        println!("CPU loss: {}, GPU loss: {}", cpu_loss, gpu_loss);

        // Losses should be similar (allowing for floating point differences)
        assert!((cpu_loss - gpu_loss).abs() < 0.5,
            "CPU and GPU losses differ too much: CPU {} vs GPU {}", cpu_loss, gpu_loss);
    }

    /// Test network reset scenario (simulates clicking Reset button)
    #[test]
    fn test_gpu_network_reset_and_reinitialize() {
        let samples = load_samples();

        // First training session
        let cpu_net = Network::mnist_default();
        let mut gpu_net = GpuNetwork::from_network(cpu_net).expect("Failed to create GPU network");

        let inputs: Vec<Vec<f32>> = samples[0..BATCH_SIZE]
            .iter()
            .map(|s| s.normalized_pixels_f32())
            .collect();
        let labels: Vec<u8> = samples[0..BATCH_SIZE].iter().map(|s| s.label()).collect();

        gpu_net.train_batch(&inputs, &labels, LEARNING_RATE);

        // Reset - create new CPU network and new GPU network from it
        let new_cpu_net = Network::mnist_default();
        let mut new_gpu_net = GpuNetwork::from_network(new_cpu_net).expect("Failed to recreate GPU network");

        // Should be able to train again
        let loss = new_gpu_net.train_batch(&inputs, &labels, LEARNING_RATE);
        assert!(loss.is_finite(), "Should be able to train after reset");
    }

    /// Test training with edge case inputs
    #[test]
    fn test_gpu_training_edge_cases() {
        let mut gpu_net = GpuNetwork::mnist_default().expect("Failed to create GPU network");

        // All zeros input
        let zero_input = vec![0.0f32; 784];
        let output = gpu_net.forward(&zero_input);
        assert_eq!(output.len(), 10);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Softmax output should sum to 1");

        // All ones input
        let ones_input = vec![1.0f32; 784];
        let output = gpu_net.forward(&ones_input);
        assert_eq!(output.len(), 10);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Softmax output should sum to 1");

        // Train with extreme values
        let loss = gpu_net.train_sample(&ones_input, 5, LEARNING_RATE);
        assert!(loss.is_finite(), "Loss should be finite with all-ones input");
    }

    /// Test that GPU forward pass is deterministic
    #[test]
    fn test_gpu_forward_deterministic() {
        let gpu_net = GpuNetwork::mnist_default().expect("Failed to create GPU network");
        let mut gpu_net = gpu_net;

        let input = vec![0.5f32; 784];

        let output1 = gpu_net.forward(&input);
        let output2 = gpu_net.forward(&input);

        for (a, b) in output1.iter().zip(output2.iter()) {
            assert!((a - b).abs() < 1e-6, "Forward pass should be deterministic");
        }
    }

    /// Test simulating the exact browser training flow with batches per frame
    #[test]
    fn test_browser_training_flow_simulation() {
        let samples = load_samples();
        let total_samples = samples.len().min(500);
        let batches_per_frame = 5;  // Same as BATCHES_PER_FRAME in viz.rs

        let cpu_net = Network::mnist_default();
        let mut gpu_net = GpuNetwork::from_network(cpu_net.clone()).expect("Failed to create GPU network");
        let mut cpu_net_mirror = cpu_net;

        let mut current_batch_idx = 0;
        let num_total_batches = total_samples / BATCH_SIZE;
        let mut epoch = 0;

        // Simulate multiple "frames" of training
        for frame in 0..10 {
            for _ in 0..batches_per_frame {
                let start = current_batch_idx * BATCH_SIZE;
                let end = (start + BATCH_SIZE).min(total_samples);

                if start >= total_samples {
                    // Epoch complete
                    epoch += 1;
                    current_batch_idx = 0;
                    println!("Frame {}: Epoch {} complete", frame, epoch);
                    continue;
                }

                let inputs: Vec<Vec<f32>> = samples[start..end]
                    .iter()
                    .map(|s| s.normalized_pixels_f32())
                    .collect();
                let labels: Vec<u8> = samples[start..end].iter().map(|s| s.label()).collect();

                let loss = gpu_net.train_batch(&inputs, &labels, LEARNING_RATE);
                assert!(loss.is_finite(), "Frame {}, batch {}: loss not finite", frame, current_batch_idx);

                // Sync weights every 10 batches
                if current_batch_idx % 10 == 0 {
                    for (cpu_layer, gpu_layer) in cpu_net_mirror.layers.iter_mut()
                        .zip(gpu_net.network().layers.iter())
                    {
                        cpu_layer.weights.copy_from_slice(&gpu_layer.weights);
                        cpu_layer.biases.copy_from_slice(&gpu_layer.biases);
                    }
                }

                current_batch_idx += 1;
            }
        }

        println!("Completed {} frames, current batch: {}, epoch: {}", 10, current_batch_idx, epoch);
    }

    /// Test GPU evaluation matches CPU evaluation
    #[test]
    fn test_gpu_evaluation() {
        let samples = load_samples();

        let cpu_net = Network::mnist_default();
        let mut gpu_net = GpuNetwork::from_network(cpu_net.clone()).expect("Failed to create GPU network");
        let mut cpu_net = cpu_net;

        let eval_size = 100;
        let eval_inputs: Vec<Vec<f32>> = samples[..eval_size]
            .iter()
            .map(|s| s.normalized_pixels_f32())
            .collect();
        let eval_labels: Vec<u8> = samples[..eval_size].iter().map(|s| s.label()).collect();

        let (cpu_loss, cpu_acc) = cpu_net.evaluate(&eval_inputs, &eval_labels);
        let (gpu_loss, gpu_acc) = gpu_net.evaluate(&eval_inputs, &eval_labels);

        println!("CPU: loss={:.4}, acc={:.1}%", cpu_loss, cpu_acc * 100.0);
        println!("GPU: loss={:.4}, acc={:.1}%", gpu_loss, gpu_acc * 100.0);

        // Should be very close
        assert!((cpu_loss - gpu_loss).abs() < 0.01, "Loss differs");
        assert!((cpu_acc - gpu_acc).abs() < 0.001, "Accuracy differs");
    }

    /// Stress test - many rapid training calls
    #[test]
    fn test_gpu_rapid_training_stress() {
        let samples = load_samples();
        let mut gpu_net = GpuNetwork::mnist_default().expect("Failed to create GPU network");

        let inputs: Vec<Vec<f32>> = samples[0..BATCH_SIZE]
            .iter()
            .map(|s| s.normalized_pixels_f32())
            .collect();
        let labels: Vec<u8> = samples[0..BATCH_SIZE].iter().map(|s| s.label()).collect();

        // Rapid fire training
        for i in 0..100 {
            let loss = gpu_net.train_batch(&inputs, &labels, LEARNING_RATE);
            assert!(loss.is_finite(), "Stress test failed at iteration {}: loss={}", i, loss);
        }
    }
}
