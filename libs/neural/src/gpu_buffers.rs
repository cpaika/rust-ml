//! Persistent GPU buffer management for efficient neural network operations
//!
//! This module provides GPU buffer types that stay resident on the GPU,
//! eliminating the need for constant CPU<->GPU transfers during training.
//!
//! Key features:
//! - Persistent buffers that stay on GPU between operations
//! - Async buffer reading that works in WASM (no blocking)
//! - Staging buffer management for efficient readback

#[cfg(feature = "gpu")]
use wgpu;
#[cfg(feature = "gpu")]
use bytemuck;
#[cfg(feature = "gpu")]
use std::sync::Arc;

/// A persistent GPU buffer with known size
#[cfg(feature = "gpu")]
pub struct GpuBuffer {
    buffer: wgpu::Buffer,
    size: usize,  // Number of f32 elements
    label: String,
}

#[cfg(feature = "gpu")]
impl GpuBuffer {
    /// Create a new GPU buffer with the given size (in f32 elements)
    pub fn new(device: &wgpu::Device, size: usize, label: &str) -> Self {
        let byte_size = (size * std::mem::size_of::<f32>()) as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        GpuBuffer {
            buffer,
            size,
            label: label.to_string(),
        }
    }

    /// Create a GPU buffer initialized with data
    pub fn from_data(device: &wgpu::Device, _queue: &wgpu::Queue, data: &[f32], label: &str) -> Self {
        use wgpu::util::DeviceExt;

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        GpuBuffer {
            buffer,
            size: data.len(),
            label: label.to_string(),
        }
    }

    /// Upload data from CPU to this buffer
    pub fn write(&self, queue: &wgpu::Queue, data: &[f32]) {
        assert_eq!(data.len(), self.size, "Data size mismatch: expected {}, got {}", self.size, data.len());
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(data));
    }

    /// Get the underlying wgpu buffer
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Get the size in f32 elements
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the size in bytes
    pub fn byte_size(&self) -> u64 {
        (self.size * std::mem::size_of::<f32>()) as u64
    }
}

/// GPU buffers for a single layer
#[cfg(feature = "gpu")]
pub struct GpuLayerBuffers {
    pub weights: GpuBuffer,      // input_size × output_size
    pub biases: GpuBuffer,       // output_size
    pub activations: GpuBuffer,  // output_size - layer output after activation
    pub pre_activations: GpuBuffer, // output_size - z = Wx + b before activation
    pub weight_grads: GpuBuffer, // input_size × output_size - accumulated gradients
    pub bias_grads: GpuBuffer,   // output_size - accumulated gradients
    pub delta: GpuBuffer,        // output_size - error signal for backprop
    pub temp_weight_grads: GpuBuffer, // input_size × output_size - temp for single sample
    pub input_size: usize,
    pub output_size: usize,
}

#[cfg(feature = "gpu")]
impl GpuLayerBuffers {
    /// Create new layer buffers from CPU layer data
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        weights: &[f32],
        biases: &[f32],
        input_size: usize,
        output_size: usize,
        layer_idx: usize,
    ) -> Self {
        assert_eq!(weights.len(), input_size * output_size);
        assert_eq!(biases.len(), output_size);

        let prefix = format!("Layer{}", layer_idx);

        GpuLayerBuffers {
            weights: GpuBuffer::from_data(device, queue, weights, &format!("{}_weights", prefix)),
            biases: GpuBuffer::from_data(device, queue, biases, &format!("{}_biases", prefix)),
            activations: GpuBuffer::new(device, output_size, &format!("{}_activations", prefix)),
            pre_activations: GpuBuffer::new(device, output_size, &format!("{}_pre_activations", prefix)),
            weight_grads: GpuBuffer::new(device, input_size * output_size, &format!("{}_weight_grads", prefix)),
            bias_grads: GpuBuffer::new(device, output_size, &format!("{}_bias_grads", prefix)),
            delta: GpuBuffer::new(device, output_size, &format!("{}_delta", prefix)),
            temp_weight_grads: GpuBuffer::new(device, input_size * output_size, &format!("{}_temp_weight_grads", prefix)),
            input_size,
            output_size,
        }
    }

    /// Upload new weights from CPU
    pub fn upload_weights(&self, queue: &wgpu::Queue, weights: &[f32], biases: &[f32]) {
        self.weights.write(queue, weights);
        self.biases.write(queue, biases);
    }
}

/// Number of input buffers in the pool for batched submissions
/// Higher = fewer GPU submissions but more memory usage
#[cfg(feature = "gpu")]
pub const INPUT_BUFFER_POOL_SIZE: usize = 32;

/// All GPU buffers for a complete network
#[cfg(feature = "gpu")]
pub struct GpuNetworkBuffers {
    pub input: GpuBuffer,        // Single input buffer (legacy, for non-batched use)
    pub input_pool: Vec<GpuBuffer>,  // Pool of input buffers for batched submissions
    pub layers: Vec<GpuLayerBuffers>,
    pub staging_buffer: wgpu::Buffer,  // For async readback
    staging_size: usize,
}

#[cfg(feature = "gpu")]
impl GpuNetworkBuffers {
    /// Create network buffers from a CPU network
    pub fn from_network(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        network: &crate::Network,
    ) -> Self {
        let input_size = network.layers[0].input_size;

        // Find the largest buffer size we might need to read back
        let mut max_size = input_size;
        for layer in &network.layers {
            max_size = max_size.max(layer.output_size);
            max_size = max_size.max(layer.weights.len());
        }

        let input = GpuBuffer::new(device, input_size, "network_input");

        // Create input buffer pool for batched submissions
        let input_pool: Vec<GpuBuffer> = (0..INPUT_BUFFER_POOL_SIZE)
            .map(|i| GpuBuffer::new(device, input_size, &format!("input_pool_{}", i)))
            .collect();

        let layers: Vec<GpuLayerBuffers> = network.layers
            .iter()
            .enumerate()
            .map(|(i, layer)| {
                GpuLayerBuffers::new(
                    device,
                    queue,
                    &layer.weights,
                    &layer.biases,
                    layer.input_size,
                    layer.output_size,
                    i,
                )
            })
            .collect();

        // Create staging buffer for async readback
        let staging_byte_size = (max_size * std::mem::size_of::<f32>()) as u64;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_buffer"),
            size: staging_byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        GpuNetworkBuffers {
            input,
            input_pool,
            layers,
            staging_buffer,
            staging_size: max_size,
        }
    }

    /// Upload input data to a specific buffer in the pool
    pub fn upload_input_pooled(&self, queue: &wgpu::Queue, pool_index: usize, data: &[f32]) {
        self.input_pool[pool_index].write(queue, data);
    }

    /// Get the number of input buffers in the pool
    pub fn pool_size(&self) -> usize {
        self.input_pool.len()
    }

    /// Upload input data to GPU
    pub fn upload_input(&self, queue: &wgpu::Queue, data: &[f32]) {
        self.input.write(queue, data);
    }

    /// Sync weights from CPU network to GPU buffers
    pub fn sync_weights_from_cpu(&self, queue: &wgpu::Queue, network: &crate::Network) {
        for (gpu_layer, cpu_layer) in self.layers.iter().zip(network.layers.iter()) {
            gpu_layer.upload_weights(queue, &cpu_layer.weights, &cpu_layer.biases);
        }
    }

    /// Get the staging buffer for async reads
    pub fn staging_buffer(&self) -> &wgpu::Buffer {
        &self.staging_buffer
    }

    /// Get the maximum staging buffer size
    pub fn staging_size(&self) -> usize {
        self.staging_size
    }
}

// ==================== ASYNC BUFFER READING ====================
// These utilities allow non-blocking buffer reads that work in WASM.

/// State of a pending async buffer read
#[cfg(feature = "gpu")]
pub enum AsyncReadState {
    /// No read is pending
    Idle,
    /// Read is in progress, waiting for GPU
    Pending,
    /// Read completed, data available
    Ready(Vec<f32>),
    /// Read failed with an error
    Error(String),
}

/// Manages async buffer reads from GPU
///
/// This is designed to work in WASM where we can't block on GPU operations.
/// Instead, we initiate a read and poll for completion.
#[cfg(feature = "gpu")]
pub struct AsyncBufferReader {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    staging_buffer: wgpu::Buffer,
    staging_size: usize,
    // Use Arc<AtomicBool> for thread-safe callback signaling (works in both WASM and native)
    map_complete: Arc<std::sync::atomic::AtomicBool>,
    map_error: Arc<std::sync::atomic::AtomicBool>,
    pending_size: std::cell::Cell<usize>,
}

#[cfg(feature = "gpu")]
impl AsyncBufferReader {
    /// Create a new async buffer reader
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, max_size: usize) -> Self {
        let staging_byte_size = (max_size * std::mem::size_of::<f32>()) as u64;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("async_staging_buffer"),
            size: staging_byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        AsyncBufferReader {
            device,
            queue,
            staging_buffer,
            staging_size: max_size,
            map_complete: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            map_error: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            pending_size: std::cell::Cell::new(0),
        }
    }

    /// Initiate an async read from a source buffer
    ///
    /// This copies the source buffer to the staging buffer and initiates
    /// the map operation. Call `poll()` or `try_get_result()` to check for completion.
    pub fn request_read(&self, src_buffer: &wgpu::Buffer, size: usize) {
        assert!(size <= self.staging_size, "Read size {} exceeds staging buffer size {}", size, self.staging_size);

        // Reset state
        self.map_complete.store(false, std::sync::atomic::Ordering::SeqCst);
        self.map_error.store(false, std::sync::atomic::Ordering::SeqCst);
        self.pending_size.set(size);

        // Copy source to staging
        let byte_size = (size * std::mem::size_of::<f32>()) as u64;
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Async Read Encoder"),
        });
        encoder.copy_buffer_to_buffer(src_buffer, 0, &self.staging_buffer, 0, byte_size);
        self.queue.submit(Some(encoder.finish()));

        // Start the map operation with Arc callback
        let map_complete = self.map_complete.clone();
        let map_error = self.map_error.clone();

        self.staging_buffer.slice(..byte_size).map_async(wgpu::MapMode::Read, move |result| {
            match result {
                Ok(()) => {
                    map_complete.store(true, std::sync::atomic::Ordering::SeqCst);
                }
                Err(_e) => {
                    #[cfg(target_arch = "wasm32")]
                    web_sys::console::error_1(&"map_async callback: ERROR".into());
                    map_error.store(true, std::sync::atomic::Ordering::SeqCst);
                }
            }
        });
    }

    /// Poll the device for completion (call this in your render/update loop)
    pub fn poll(&self) {
        self.device.poll(wgpu::Maintain::Poll);
    }

    /// Check if a read result is available and return it if so
    ///
    /// Returns `Some(data)` if the read completed, `None` if still pending.
    /// If an error occurred, returns `Some` with an empty vec (check logs).
    pub fn try_get_result(&self) -> Option<Vec<f32>> {
        // First poll to check for completion
        self.poll();

        let map_complete = self.map_complete.load(std::sync::atomic::Ordering::SeqCst);
        let map_error = self.map_error.load(std::sync::atomic::Ordering::SeqCst);

        if map_complete {
            // Read the actual data from the mapped buffer
            let size = self.pending_size.get();
            let byte_size = (size * std::mem::size_of::<f32>()) as u64;
            let slice = self.staging_buffer.slice(..byte_size);
            let mapped = slice.get_mapped_range();
            let result: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
            drop(mapped);
            self.staging_buffer.unmap();

            self.map_complete.store(false, std::sync::atomic::Ordering::SeqCst);
            self.pending_size.set(0);
            Some(result)
        } else if map_error {
            // Error occurred
            #[cfg(target_arch = "wasm32")]
            web_sys::console::error_1(&"AsyncBufferReader: map error".into());

            self.staging_buffer.unmap();
            self.map_error.store(false, std::sync::atomic::Ordering::SeqCst);
            self.pending_size.set(0);
            Some(Vec::new())
        } else {
            None
        }
    }

    /// Check if a read is currently pending
    pub fn is_pending(&self) -> bool {
        self.pending_size.get() > 0 &&
        !self.map_complete.load(std::sync::atomic::Ordering::SeqCst) &&
        !self.map_error.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Blocking read for native platforms (will spin in WASM but eventually complete)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_blocking(&self, src_buffer: &wgpu::Buffer, size: usize) -> Vec<f32> {
        self.request_read(src_buffer, size);

        // Block until ready
        loop {
            self.device.poll(wgpu::Maintain::Wait);
            if let Some(result) = self.try_get_result() {
                return result;
            }
        }
    }

    /// Polling read for WASM - polls aggressively until data is ready
    /// Returns None if not ready after max_polls iterations
    #[cfg(target_arch = "wasm32")]
    pub fn read_with_polling(&self, src_buffer: &wgpu::Buffer, size: usize, max_polls: usize) -> Option<Vec<f32>> {
        // If there's already a pending read, we can't start a new one
        // The buffer would still be mapped
        if self.is_pending() {
            // Try to complete the pending read first
            for _ in 0..max_polls {
                self.device.poll(wgpu::Maintain::Poll);
                if !self.is_pending() {
                    break;
                }
            }
            // If still pending, we need to wait
            if self.is_pending() {
                return None;
            }
            // Discard the pending result by getting it
            let _ = self.try_get_result();
        }

        self.request_read(src_buffer, size);

        // Poll aggressively
        for _ in 0..max_polls {
            self.device.poll(wgpu::Maintain::Poll);
            if let Some(result) = self.try_get_result() {
                return Some(result);
            }
        }

        // Not ready yet - caller should try again later
        None
    }

    /// Safe single buffer read for WASM - only attempts if no pending read
    /// Returns None if a read is already pending or if polling doesn't complete in time
    #[cfg(target_arch = "wasm32")]
    pub fn try_single_read(&self, src_buffer: &wgpu::Buffer, size: usize, max_polls: usize) -> Option<Vec<f32>> {
        // If there's already a pending read, return None
        if self.is_pending() {
            return None;
        }

        // Check if previous read completed - consume the result
        if self.map_complete.load(std::sync::atomic::Ordering::SeqCst) ||
           self.map_error.load(std::sync::atomic::Ordering::SeqCst) {
            // There's a completed read we need to consume first
            let _ = self.try_get_result();
        }

        self.request_read(src_buffer, size);

        // Poll aggressively
        for _ in 0..max_polls {
            self.device.poll(wgpu::Maintain::Poll);
            if let Some(result) = self.try_get_result() {
                return Some(result);
            }
        }

        None
    }
}

/// Helper to copy buffer to staging and read asynchronously
#[cfg(feature = "gpu")]
pub fn copy_buffer_to_staging(
    encoder: &mut wgpu::CommandEncoder,
    src_buffer: &wgpu::Buffer,
    staging_buffer: &wgpu::Buffer,
    size: usize,
) {
    let byte_size = (size * std::mem::size_of::<f32>()) as u64;
    encoder.copy_buffer_to_buffer(src_buffer, 0, staging_buffer, 0, byte_size);
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use crate::Network;

    fn create_test_device() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })).expect("Failed to find adapter");

        pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Test Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )).expect("Failed to create device")
    }

    #[test]
    fn test_gpu_buffer_creation() {
        let (device, _queue) = create_test_device();

        let buffer = GpuBuffer::new(&device, 100, "test_buffer");
        assert_eq!(buffer.size(), 100);
        assert_eq!(buffer.byte_size(), 400); // 100 * 4 bytes
    }

    #[test]
    fn test_gpu_buffer_from_data() {
        let (device, queue) = create_test_device();

        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let buffer = GpuBuffer::from_data(&device, &queue, &data, "test_buffer");
        assert_eq!(buffer.size(), 4);
    }

    #[test]
    fn test_gpu_layer_buffers_creation() {
        let (device, queue) = create_test_device();

        let input_size = 4;
        let output_size = 3;
        let weights = vec![0.1f32; input_size * output_size];
        let biases = vec![0.0f32; output_size];

        let layer = GpuLayerBuffers::new(
            &device, &queue, &weights, &biases, input_size, output_size, 0
        );

        assert_eq!(layer.weights.size(), 12); // 4 * 3
        assert_eq!(layer.biases.size(), 3);
        assert_eq!(layer.activations.size(), 3);
        assert_eq!(layer.delta.size(), 3);
    }

    #[test]
    fn test_gpu_network_buffers_creation() {
        let (device, queue) = create_test_device();

        let network = Network::new(&[4, 3, 2]);
        let gpu_buffers = GpuNetworkBuffers::from_network(&device, &queue, &network);

        assert_eq!(gpu_buffers.input.size(), 4);
        assert_eq!(gpu_buffers.layers.len(), 2);
        assert_eq!(gpu_buffers.layers[0].output_size, 3);
        assert_eq!(gpu_buffers.layers[1].output_size, 2);
    }

    #[test]
    fn test_gpu_network_buffers_from_mnist_network() {
        let (device, queue) = create_test_device();

        let network = Network::mnist_default(); // 784 -> 128 -> 10
        let gpu_buffers = GpuNetworkBuffers::from_network(&device, &queue, &network);

        assert_eq!(gpu_buffers.input.size(), 784);
        assert_eq!(gpu_buffers.layers.len(), 2);
        assert_eq!(gpu_buffers.layers[0].input_size, 784);
        assert_eq!(gpu_buffers.layers[0].output_size, 128);
        assert_eq!(gpu_buffers.layers[1].input_size, 128);
        assert_eq!(gpu_buffers.layers[1].output_size, 10);
    }
}
