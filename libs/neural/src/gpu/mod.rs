//! GPU-accelerated neural network operations using wgpu/WebGPU
//!
//! This module provides compute shader implementations for neural network
//! operations that run on the GPU via WebGPU (in browser) or native backends
//! (Vulkan, Metal, DirectX 12).

use std::sync::Arc;

#[cfg(feature = "gpu")]
use wgpu;
#[cfg(feature = "gpu")]
use bytemuck::{Pod, Zeroable};

/// Helper to wait for buffer mapping in a cross-platform way
/// Takes the staging buffer and returns the mapped data as Vec<f32>
#[cfg(feature = "gpu")]
fn read_buffer_sync(device: &wgpu::Device, staging_buffer: &wgpu::Buffer, _size: u64) -> Vec<f32> {
    let buffer_slice = staging_buffer.slice(..);

    #[cfg(target_arch = "wasm32")]
    {
        // In WASM, use a Cell since we're single-threaded
        use std::cell::Cell;
        use std::rc::Rc;

        let mapped = Rc::new(Cell::new(false));
        let mapped_clone = mapped.clone();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            result.expect("Buffer mapping failed");
            mapped_clone.set(true);
        });

        // Poll until the buffer is mapped
        // In WebGPU, the callback is invoked during poll when work completes
        while !mapped.get() {
            device.poll(wgpu::Maintain::Poll);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        // On native, use channel-based blocking
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().unwrap();
    }

    let data = buffer_slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();
    result
}

/// Helper to read buffer back into a mutable slice in a cross-platform way
#[cfg(feature = "gpu")]
fn read_buffer_into_sync(device: &wgpu::Device, staging_buffer: &wgpu::Buffer, dest: &mut [f32]) {
    let buffer_slice = staging_buffer.slice(..);

    #[cfg(target_arch = "wasm32")]
    {
        use std::cell::Cell;
        use std::rc::Rc;

        let mapped = Rc::new(Cell::new(false));
        let mapped_clone = mapped.clone();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            result.expect("Buffer mapping failed");
            mapped_clone.set(true);
        });

        while !mapped.get() {
            device.poll(wgpu::Maintain::Poll);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().unwrap();
    }

    let mapped_data = buffer_slice.get_mapped_range();
    let result: &[f32] = bytemuck::cast_slice(&mapped_data);
    dest.copy_from_slice(result);
    drop(mapped_data);
    staging_buffer.unmap();
}

/// GPU context holding the device and queue
pub struct GpuContext {
    #[cfg(feature = "gpu")]
    device: Arc<wgpu::Device>,
    #[cfg(feature = "gpu")]
    queue: Arc<wgpu::Queue>,
    #[cfg(feature = "gpu")]
    matmul_pipeline: wgpu::ComputePipeline,
    #[cfg(feature = "gpu")]
    matmul_at_b_pipeline: wgpu::ComputePipeline,  // A^T * B for weight gradients
    #[cfg(feature = "gpu")]
    matmul_a_bt_pipeline: wgpu::ComputePipeline,  // A * B^T for delta backprop
    #[cfg(feature = "gpu")]
    relu_pipeline: wgpu::ComputePipeline,
    #[cfg(feature = "gpu")]
    relu_backward_pipeline: wgpu::ComputePipeline,
    #[cfg(feature = "gpu")]
    add_bias_pipeline: wgpu::ComputePipeline,
    #[cfg(feature = "gpu")]
    hadamard_pipeline: wgpu::ComputePipeline,
    #[cfg(feature = "gpu")]
    saxpy_pipeline: wgpu::ComputePipeline,  // y = alpha * x + y (for SGD updates)
}

/// Matrix dimensions for compute shaders
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "gpu", derive(Pod, Zeroable))]
pub struct MatrixDims {
    pub m: u32,  // rows of A, rows of C
    pub k: u32,  // cols of A, rows of B
    pub n: u32,  // cols of B, cols of C
    pub _pad: u32,
}

/// Vector dimensions for compute shaders
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "gpu", derive(Pod, Zeroable))]
pub struct VectorDims {
    pub size: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}

/// Scalar + vector dimensions for SAXPY (y = alpha * x + y)
#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "gpu", derive(Pod, Zeroable))]
pub struct SaxpyParams {
    pub size: u32,
    pub _pad1: u32,
    pub alpha: f32,
    pub _pad2: u32,
}

#[cfg(feature = "gpu")]
impl GpuContext {
    /// Create a new GPU context (async version for WASM compatibility)
    pub async fn new_async() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or("Failed to find suitable GPU adapter")?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Neural Network GPU"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| format!("Failed to create device: {}", e))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Create compute pipelines
        let matmul_pipeline = Self::create_matmul_pipeline(&device);
        let matmul_at_b_pipeline = Self::create_matmul_at_b_pipeline(&device);
        let matmul_a_bt_pipeline = Self::create_matmul_a_bt_pipeline(&device);
        let relu_pipeline = Self::create_relu_pipeline(&device);
        let relu_backward_pipeline = Self::create_relu_backward_pipeline(&device);
        let add_bias_pipeline = Self::create_add_bias_pipeline(&device);
        let hadamard_pipeline = Self::create_hadamard_pipeline(&device);
        let saxpy_pipeline = Self::create_saxpy_pipeline(&device);

        Ok(Self {
            device,
            queue,
            matmul_pipeline,
            matmul_at_b_pipeline,
            matmul_a_bt_pipeline,
            relu_pipeline,
            relu_backward_pipeline,
            add_bias_pipeline,
            hadamard_pipeline,
            saxpy_pipeline,
        })
    }

    /// Create a new GPU context (blocking version for native)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Result<Self, String> {
        pollster::block_on(Self::new_async())
    }

    fn create_matmul_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Matrix Multiply Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/matmul.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MatMul Bind Group Layout"),
            entries: &[
                // Dimensions uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Matrix A (input)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Matrix B (weights)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Matrix C (output)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MatMul Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MatMul Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    fn create_relu_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ReLU Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/relu.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ReLU Bind Group Layout"),
            entries: &[
                // Dimensions
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Input/Output buffer (in-place)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ReLU Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ReLU Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    fn create_add_bias_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Add Bias Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/add_bias.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Add Bias Bind Group Layout"),
            entries: &[
                // Dimensions
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Data buffer (in-place)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Bias buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Add Bias Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Add Bias Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    fn create_matmul_at_b_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MatMul A^T * B Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/matmul_at_b.wgsl").into()),
        });

        // Same layout as regular matmul
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MatMul A^T*B Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MatMul A^T*B Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MatMul A^T*B Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    fn create_matmul_a_bt_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MatMul A * B^T Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/matmul_a_bt.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MatMul A*B^T Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MatMul A*B^T Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MatMul A*B^T Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    fn create_relu_backward_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ReLU Backward Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/relu_backward.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ReLU Backward Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ReLU Backward Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ReLU Backward Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    fn create_hadamard_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Hadamard Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/hadamard.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Hadamard Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Hadamard Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Hadamard Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    fn create_saxpy_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SAXPY Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/saxpy.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SAXPY Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SAXPY Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SAXPY Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    /// Matrix multiply: C = A * B
    /// A is (m x k), B is (k x n), C is (m x n)
    pub fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        use wgpu::util::DeviceExt;

        let dims = MatrixDims {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            _pad: 0,
        };

        // Create buffers
        let dims_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dims Buffer"),
            contents: bytemuck::bytes_of(&dims),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let a_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matrix A"),
            contents: bytemuck::cast_slice(a),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let b_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matrix B"),
            contents: bytemuck::cast_slice(b),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_size = (m * n * std::mem::size_of::<f32>()) as u64;
        let c_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matrix C"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group
        let bind_group_layout = self.matmul_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MatMul Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dims_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: c_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch compute shader
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("MatMul Encoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MatMul Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.matmul_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            // Dispatch workgroups: ceil(m/8) x ceil(n/8)
            let workgroups_x = (m as u32 + 7) / 8;
            let workgroups_y = (n as u32 + 7) / 8;
            cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        encoder.copy_buffer_to_buffer(&c_buffer, 0, &staging_buffer, 0, output_size);
        self.queue.submit(Some(encoder.finish()));

        // Read back results using cross-platform helper
        read_buffer_sync(&self.device, &staging_buffer, output_size)
    }

    /// Apply ReLU activation in-place
    pub fn relu(&self, data: &mut [f32]) {
        use wgpu::util::DeviceExt;

        let dims = VectorDims {
            size: data.len() as u32,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };

        let dims_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dims Buffer"),
            contents: bytemuck::bytes_of(&dims),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let data_size = (data.len() * std::mem::size_of::<f32>()) as u64;
        let data_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Data Buffer"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: data_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = self.relu_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ReLU Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dims_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: data_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ReLU Encoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ReLU Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.relu_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (data.len() as u32 + 63) / 64;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&data_buffer, 0, &staging_buffer, 0, data_size);
        self.queue.submit(Some(encoder.finish()));

        // Read back results using cross-platform helper
        read_buffer_into_sync(&self.device, &staging_buffer, data);
    }

    /// Add bias to data: data[i] += bias[i]
    pub fn add_bias(&self, data: &mut [f32], bias: &[f32]) {
        use wgpu::util::DeviceExt;

        let dims = VectorDims {
            size: data.len() as u32,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };

        let dims_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dims Buffer"),
            contents: bytemuck::bytes_of(&dims),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let data_size = (data.len() * std::mem::size_of::<f32>()) as u64;
        let data_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Data Buffer"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let bias_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bias Buffer"),
            contents: bytemuck::cast_slice(bias),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: data_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = self.add_bias_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Add Bias Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dims_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bias_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Add Bias Encoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Add Bias Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.add_bias_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (data.len() as u32 + 63) / 64;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&data_buffer, 0, &staging_buffer, 0, data_size);
        self.queue.submit(Some(encoder.finish()));

        // Read back results using cross-platform helper
        read_buffer_into_sync(&self.device, &staging_buffer, data);
    }

    /// Softmax activation (CPU fallback for now due to reduction complexity)
    pub fn softmax(&self, data: &mut [f32]) {
        // Softmax requires a reduction (sum), which is complex in compute shaders
        // For now, use CPU implementation
        let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for x in data.iter_mut() {
            *x = (*x - max_val).exp();
            sum += *x;
        }
        for x in data.iter_mut() {
            *x /= sum;
        }
    }

    // ==================== BACKWARD PASS OPERATIONS ====================

    /// Matrix multiply with A transposed: C = A^T * B
    /// A is (k x m), B is (k x n), C is (m x n)
    /// Used for weight gradients: dW = input^T * delta
    pub fn matmul_at_b(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        use wgpu::util::DeviceExt;

        let dims = MatrixDims {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            _pad: 0,
        };

        let dims_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dims Buffer"),
            contents: bytemuck::bytes_of(&dims),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let a_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matrix A"),
            contents: bytemuck::cast_slice(a),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let b_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matrix B"),
            contents: bytemuck::cast_slice(b),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_size = (m * n * std::mem::size_of::<f32>()) as u64;
        let c_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matrix C"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = self.matmul_at_b_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MatMul A^T*B Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dims_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: c_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("MatMul A^T*B Encoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MatMul A^T*B Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.matmul_at_b_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups_x = (m as u32 + 7) / 8;
            let workgroups_y = (n as u32 + 7) / 8;
            cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        encoder.copy_buffer_to_buffer(&c_buffer, 0, &staging_buffer, 0, output_size);
        self.queue.submit(Some(encoder.finish()));

        // Read back results using cross-platform helper
        read_buffer_sync(&self.device, &staging_buffer, output_size)
    }

    /// Matrix multiply with B transposed: C = A * B^T
    /// A is (m x k), B is (n x k), C is (m x n)
    /// Used for delta backprop: delta_prev = delta * W^T
    pub fn matmul_a_bt(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        use wgpu::util::DeviceExt;

        let dims = MatrixDims {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            _pad: 0,
        };

        let dims_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dims Buffer"),
            contents: bytemuck::bytes_of(&dims),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let a_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matrix A"),
            contents: bytemuck::cast_slice(a),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let b_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matrix B"),
            contents: bytemuck::cast_slice(b),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_size = (m * n * std::mem::size_of::<f32>()) as u64;
        let c_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matrix C"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = self.matmul_a_bt_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MatMul A*B^T Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dims_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: c_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("MatMul A*B^T Encoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MatMul A*B^T Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.matmul_a_bt_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups_x = (m as u32 + 7) / 8;
            let workgroups_y = (n as u32 + 7) / 8;
            cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        encoder.copy_buffer_to_buffer(&c_buffer, 0, &staging_buffer, 0, output_size);
        self.queue.submit(Some(encoder.finish()));

        // Read back results using cross-platform helper
        read_buffer_sync(&self.device, &staging_buffer, output_size)
    }

    /// SAXPY: y = alpha * x + y
    /// Used for SGD weight updates: weights = -learning_rate * gradients + weights
    pub fn saxpy(&self, x: &[f32], y: &mut [f32], alpha: f32) {
        use wgpu::util::DeviceExt;

        let params = SaxpyParams {
            size: x.len() as u32,
            _pad1: 0,
            alpha,
            _pad2: 0,
        };

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let x_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("X Buffer"),
            contents: bytemuck::cast_slice(x),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let data_size = (y.len() * std::mem::size_of::<f32>()) as u64;
        let y_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Y Buffer"),
            contents: bytemuck::cast_slice(y),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: data_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = self.saxpy_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SAXPY Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: y_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("SAXPY Encoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SAXPY Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.saxpy_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (x.len() as u32 + 63) / 64;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&y_buffer, 0, &staging_buffer, 0, data_size);
        self.queue.submit(Some(encoder.finish()));

        // Read back results using cross-platform helper
        read_buffer_into_sync(&self.device, &staging_buffer, y);
    }

    /// ReLU backward: grad = grad * (pre_activation > 0 ? 1 : 0)
    /// Modifies grad in-place based on whether the pre-activation value was positive
    pub fn relu_backward(&self, grad: &mut [f32], pre_activation: &[f32]) {
        use wgpu::util::DeviceExt;

        let dims = VectorDims {
            size: grad.len() as u32,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };

        let dims_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dims Buffer"),
            contents: bytemuck::bytes_of(&dims),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let data_size = (grad.len() * std::mem::size_of::<f32>()) as u64;
        let grad_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grad Buffer"),
            contents: bytemuck::cast_slice(grad),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let pre_act_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Pre-activation Buffer"),
            contents: bytemuck::cast_slice(pre_activation),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: data_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = self.relu_backward_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ReLU Backward Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dims_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grad_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pre_act_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ReLU Backward Encoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ReLU Backward Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.relu_backward_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (grad.len() as u32 + 63) / 64;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&grad_buffer, 0, &staging_buffer, 0, data_size);
        self.queue.submit(Some(encoder.finish()));

        // Read back results using cross-platform helper
        read_buffer_into_sync(&self.device, &staging_buffer, grad);
    }

    /// Hadamard (element-wise) multiply: a = a * b
    pub fn hadamard(&self, a: &mut [f32], b: &[f32]) {
        use wgpu::util::DeviceExt;

        let dims = VectorDims {
            size: a.len() as u32,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };

        let dims_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dims Buffer"),
            contents: bytemuck::bytes_of(&dims),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let data_size = (a.len() * std::mem::size_of::<f32>()) as u64;
        let a_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("A Buffer"),
            contents: bytemuck::cast_slice(a),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let b_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("B Buffer"),
            contents: bytemuck::cast_slice(b),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: data_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = self.hadamard_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Hadamard Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dims_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Hadamard Encoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Hadamard Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.hadamard_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (a.len() as u32 + 63) / 64;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&a_buffer, 0, &staging_buffer, 0, data_size);
        self.queue.submit(Some(encoder.finish()));

        // Read back results using cross-platform helper
        read_buffer_into_sync(&self.device, &staging_buffer, a);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    fn vec_approx_eq(a: &[f32], b: &[f32], epsilon: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| approx_eq(*x, *y, epsilon))
    }

    #[test]
    fn test_gpu_context_creation() {
        let ctx = GpuContext::new();
        assert!(ctx.is_ok(), "Failed to create GPU context: {:?}", ctx.err());
    }

    #[test]
    fn test_matmul_identity() {
        let ctx = GpuContext::new().expect("Failed to create GPU context");

        // Multiply by identity matrix
        // A = [1, 2, 3, 4] (2x2), I = [1, 0, 0, 1] (2x2)
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let identity = vec![1.0, 0.0, 0.0, 1.0];

        let result = ctx.matmul(&a, &identity, 2, 2, 2);

        assert!(vec_approx_eq(&result, &a, 1e-5),
            "Identity multiplication failed: {:?} != {:?}", result, a);
    }

    #[test]
    fn test_matmul_simple() {
        let ctx = GpuContext::new().expect("Failed to create GPU context");

        // A = [[1, 2], [3, 4]] (2x2)
        // B = [[5, 6], [7, 8]] (2x2)
        // C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //   = [[19, 22], [43, 50]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let expected = vec![19.0, 22.0, 43.0, 50.0];

        let result = ctx.matmul(&a, &b, 2, 2, 2);

        assert!(vec_approx_eq(&result, &expected, 1e-5),
            "Matrix multiplication failed: {:?} != {:?}", result, expected);
    }

    #[test]
    fn test_matmul_vector() {
        let ctx = GpuContext::new().expect("Failed to create GPU context");

        // Matrix-vector multiplication (common in neural networks)
        // A = [[1, 2, 3], [4, 5, 6]] (2x3)
        // x = [1, 2, 3] (3x1)
        // result = [1*1+2*2+3*3, 4*1+5*2+6*3] = [14, 32]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 2.0, 3.0];
        let expected = vec![14.0, 32.0];

        let result = ctx.matmul(&a, &x, 2, 3, 1);

        assert!(vec_approx_eq(&result, &expected, 1e-5),
            "Matrix-vector multiplication failed: {:?} != {:?}", result, expected);
    }

    #[test]
    fn test_matmul_neural_layer() {
        let ctx = GpuContext::new().expect("Failed to create GPU context");

        // Simulate a neural network layer: output = input * weights
        // input = [0.5, 0.3, 0.2] (1x3)
        // weights = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]] (3x2)
        // output = [0.5*0.1+0.3*0.3+0.2*0.5, 0.5*0.2+0.3*0.4+0.2*0.6]
        //        = [0.05+0.09+0.1, 0.1+0.12+0.12] = [0.24, 0.34]
        let input = vec![0.5, 0.3, 0.2];
        let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let expected = vec![0.24, 0.34];

        let result = ctx.matmul(&input, &weights, 1, 3, 2);

        assert!(vec_approx_eq(&result, &expected, 1e-5),
            "Neural layer matmul failed: {:?} != {:?}", result, expected);
    }

    #[test]
    fn test_relu_positive() {
        let ctx = GpuContext::new().expect("Failed to create GPU context");

        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let expected = vec![1.0, 2.0, 3.0, 4.0];

        ctx.relu(&mut data);

        assert!(vec_approx_eq(&data, &expected, 1e-5),
            "ReLU positive failed: {:?} != {:?}", data, expected);
    }

    #[test]
    fn test_relu_negative() {
        let ctx = GpuContext::new().expect("Failed to create GPU context");

        let mut data = vec![-1.0, -2.0, -3.0, -4.0];
        let expected = vec![0.0, 0.0, 0.0, 0.0];

        ctx.relu(&mut data);

        assert!(vec_approx_eq(&data, &expected, 1e-5),
            "ReLU negative failed: {:?} != {:?}", data, expected);
    }

    #[test]
    fn test_relu_mixed() {
        let ctx = GpuContext::new().expect("Failed to create GPU context");

        let mut data = vec![-2.0, 1.0, -0.5, 3.0, 0.0, -1.0];
        let expected = vec![0.0, 1.0, 0.0, 3.0, 0.0, 0.0];

        ctx.relu(&mut data);

        assert!(vec_approx_eq(&data, &expected, 1e-5),
            "ReLU mixed failed: {:?} != {:?}", data, expected);
    }

    #[test]
    fn test_add_bias() {
        let ctx = GpuContext::new().expect("Failed to create GPU context");

        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let bias = vec![0.1, 0.2, 0.3, 0.4];
        let expected = vec![1.1, 2.2, 3.3, 4.4];

        ctx.add_bias(&mut data, &bias);

        assert!(vec_approx_eq(&data, &expected, 1e-5),
            "Add bias failed: {:?} != {:?}", data, expected);
    }

    #[test]
    fn test_softmax() {
        let ctx = GpuContext::new().expect("Failed to create GPU context");

        let mut data = vec![1.0, 2.0, 3.0];
        ctx.softmax(&mut data);

        // Check that softmax outputs sum to 1
        let sum: f32 = data.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-5), "Softmax sum != 1: {}", sum);

        // Check that values are in (0, 1)
        assert!(data.iter().all(|&x| x > 0.0 && x < 1.0),
            "Softmax values not in (0,1): {:?}", data);

        // Check ordering preserved (larger input -> larger output)
        assert!(data[2] > data[1] && data[1] > data[0],
            "Softmax ordering not preserved: {:?}", data);
    }

    #[test]
    fn test_forward_pass_layer() {
        let ctx = GpuContext::new().expect("Failed to create GPU context");

        // Simulate one layer: output = ReLU(input * weights + bias)
        let input = vec![1.0, 0.5];
        let weights = vec![0.5, -0.5, 0.3, 0.7]; // 2x2 (input_size x output_size)
        let bias = vec![0.1, -0.1];

        // Step 1: matmul
        // [1.0, 0.5] * [[0.5, -0.5], [0.3, 0.7]]
        // = [1.0*0.5 + 0.5*0.3, 1.0*(-0.5) + 0.5*0.7]
        // = [0.65, -0.15]
        let mut output = ctx.matmul(&input, &weights, 1, 2, 2);

        // Step 2: add bias
        // [0.65, -0.15] + [0.1, -0.1] = [0.75, -0.25]
        ctx.add_bias(&mut output, &bias);

        // Step 3: ReLU
        // ReLU([0.75, -0.25]) = [0.75, 0.0]
        ctx.relu(&mut output);

        let expected = vec![0.75, 0.0];
        assert!(vec_approx_eq(&output, &expected, 1e-5),
            "Forward pass layer failed: {:?} != {:?}", output, expected);
    }

    #[test]
    fn test_large_matmul() {
        let ctx = GpuContext::new().expect("Failed to create GPU context");

        // Test with larger matrices similar to MNIST layer sizes
        let m = 1;    // batch size
        let k = 784;  // input size (28x28)
        let n = 128;  // hidden layer size

        // Random-ish input and weights
        let input: Vec<f32> = (0..k).map(|i| (i as f32 % 10.0) / 10.0).collect();
        let weights: Vec<f32> = (0..k*n).map(|i| ((i % 17) as f32 - 8.0) / 100.0).collect();

        // GPU result
        let gpu_result = ctx.matmul(&input, &weights, m, k, n);

        // CPU reference
        let mut cpu_result = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += input[i * k + p] * weights[p * n + j];
                }
                cpu_result[i * n + j] = sum;
            }
        }

        assert!(vec_approx_eq(&gpu_result, &cpu_result, 1e-3),
            "Large matmul mismatch (first 10): GPU {:?} vs CPU {:?}",
            &gpu_result[..10], &cpu_result[..10]);
    }

    // ==================== BACKWARD PASS TESTS ====================

    #[test]
    fn test_matmul_at_b_simple() {
        let ctx = GpuContext::new().expect("Failed to create GPU context");

        // A^T * B where A is (k x m), B is (k x n), result is (m x n)
        // A = [[1, 2], [3, 4], [5, 6]] (3x2), so A^T = [[1, 3, 5], [2, 4, 6]] (2x3)
        // B = [[1, 0], [0, 1], [1, 1]] (3x2)
        // A^T * B = [[1*1+3*0+5*1, 1*0+3*1+5*1], [2*1+4*0+6*1, 2*0+4*1+6*1]]
        //         = [[6, 8], [8, 10]]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // (3x2) = (k x m)
        let b = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // (3x2) = (k x n)
        let expected = vec![6.0, 8.0, 8.0, 10.0];

        let result = ctx.matmul_at_b(&a, &b, 2, 3, 2); // m=2, k=3, n=2

        assert!(vec_approx_eq(&result, &expected, 1e-5),
            "matmul_at_b failed: {:?} != {:?}", result, expected);
    }

    #[test]
    fn test_matmul_at_b_weight_gradient() {
        let ctx = GpuContext::new().expect("Failed to create GPU context");

        // Simulating dW = input^T * delta
        // input = [[0.5, 0.3, 0.2]] (1x3) => input^T is (3x1)
        // delta = [[0.1, 0.2]] (1x2)
        // dW = input^T * delta = (3x1) * (1x2) = (3x2)
        //    = [[0.5*0.1, 0.5*0.2], [0.3*0.1, 0.3*0.2], [0.2*0.1, 0.2*0.2]]
        //    = [[0.05, 0.1], [0.03, 0.06], [0.02, 0.04]]
        let input = vec![0.5, 0.3, 0.2]; // (1x3) treated as (k x m) = (1 x 3)
        let delta = vec![0.1, 0.2];      // (1x2) treated as (k x n) = (1 x 2)
        let expected = vec![0.05, 0.1, 0.03, 0.06, 0.02, 0.04];

        let result = ctx.matmul_at_b(&input, &delta, 3, 1, 2); // m=3, k=1, n=2

        assert!(vec_approx_eq(&result, &expected, 1e-5),
            "matmul_at_b weight gradient failed: {:?} != {:?}", result, expected);
    }

    #[test]
    fn test_matmul_a_bt_simple() {
        let ctx = GpuContext::new().expect("Failed to create GPU context");

        // A * B^T where A is (m x k), B is (n x k), result is (m x n)
        // A = [[1, 2, 3], [4, 5, 6]] (2x3)
        // B = [[1, 0, 1], [0, 1, 1]] (2x3), so B^T = [[1, 0], [0, 1], [1, 1]] (3x2)
        // A * B^T = [[1*1+2*0+3*1, 1*0+2*1+3*1], [4*1+5*0+6*1, 4*0+5*1+6*1]]
        //         = [[4, 5], [10, 11]]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // (2x3) = (m x k)
        let b = vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0]; // (2x3) = (n x k)
        let expected = vec![4.0, 5.0, 10.0, 11.0];

        let result = ctx.matmul_a_bt(&a, &b, 2, 3, 2); // m=2, k=3, n=2

        assert!(vec_approx_eq(&result, &expected, 1e-5),
            "matmul_a_bt failed: {:?} != {:?}", result, expected);
    }

    #[test]
    fn test_matmul_a_bt_delta_backprop() {
        let ctx = GpuContext::new().expect("Failed to create GPU context");

        // Simulating delta_prev = delta * W^T
        // delta = [[0.1, 0.2]] (1x2) = (m x k)
        // W = [[0.5, 0.3], [0.4, 0.2], [0.1, 0.6]] (3x2) = (n x k)
        // delta_prev = delta * W^T = (1x2) * (2x3) = (1x3)
        //            = [[0.1*0.5+0.2*0.3, 0.1*0.4+0.2*0.2, 0.1*0.1+0.2*0.6]]
        //            = [[0.11, 0.08, 0.13]]
        let delta = vec![0.1, 0.2];
        let weights = vec![0.5, 0.3, 0.4, 0.2, 0.1, 0.6]; // (3x2)
        let expected = vec![0.11, 0.08, 0.13];

        let result = ctx.matmul_a_bt(&delta, &weights, 1, 2, 3); // m=1, k=2, n=3

        assert!(vec_approx_eq(&result, &expected, 1e-5),
            "matmul_a_bt delta backprop failed: {:?} != {:?}", result, expected);
    }

    #[test]
    fn test_saxpy_simple() {
        let ctx = GpuContext::new().expect("Failed to create GPU context");

        // y = alpha * x + y
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![10.0, 20.0, 30.0, 40.0];
        let alpha = 2.0;
        let expected = vec![12.0, 24.0, 36.0, 48.0]; // 2*1+10, 2*2+20, 2*3+30, 2*4+40

        ctx.saxpy(&x, &mut y, alpha);

        assert!(vec_approx_eq(&y, &expected, 1e-5),
            "saxpy failed: {:?} != {:?}", y, expected);
    }

    #[test]
    fn test_saxpy_sgd_update() {
        let ctx = GpuContext::new().expect("Failed to create GPU context");

        // SGD: weights = -learning_rate * gradients + weights
        let learning_rate = 0.01;
        let gradients = vec![0.5, -0.3, 0.2, -0.1]; // dW
        let mut weights = vec![1.0, 2.0, 3.0, 4.0];
        // expected = -0.01 * [0.5, -0.3, 0.2, -0.1] + [1.0, 2.0, 3.0, 4.0]
        //          = [-0.005, 0.003, -0.002, 0.001] + [1.0, 2.0, 3.0, 4.0]
        //          = [0.995, 2.003, 2.998, 4.001]
        let expected = vec![0.995, 2.003, 2.998, 4.001];

        ctx.saxpy(&gradients, &mut weights, -learning_rate);

        assert!(vec_approx_eq(&weights, &expected, 1e-5),
            "saxpy SGD update failed: {:?} != {:?}", weights, expected);
    }

    #[test]
    fn test_saxpy_negative_alpha() {
        let ctx = GpuContext::new().expect("Failed to create GPU context");

        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![5.0, 5.0, 5.0];
        let alpha = -1.0;
        let expected = vec![4.0, 3.0, 2.0]; // -1*1+5, -1*2+5, -1*3+5

        ctx.saxpy(&x, &mut y, alpha);

        assert!(vec_approx_eq(&y, &expected, 1e-5),
            "saxpy negative alpha failed: {:?} != {:?}", y, expected);
    }

    #[test]
    fn test_relu_backward_positive() {
        let ctx = GpuContext::new().expect("Failed to create GPU context");

        // ReLU'(z) = 1 if z > 0, 0 otherwise
        let mut grad = vec![1.0, 2.0, 3.0, 4.0];
        let pre_activation = vec![0.5, 1.0, 2.0, 0.1]; // all positive
        let expected = vec![1.0, 2.0, 3.0, 4.0]; // grad unchanged

        ctx.relu_backward(&mut grad, &pre_activation);

        assert!(vec_approx_eq(&grad, &expected, 1e-5),
            "relu_backward positive failed: {:?} != {:?}", grad, expected);
    }

    #[test]
    fn test_relu_backward_negative() {
        let ctx = GpuContext::new().expect("Failed to create GPU context");

        let mut grad = vec![1.0, 2.0, 3.0, 4.0];
        let pre_activation = vec![-0.5, -1.0, -2.0, -0.1]; // all negative
        let expected = vec![0.0, 0.0, 0.0, 0.0]; // grad zeroed

        ctx.relu_backward(&mut grad, &pre_activation);

        assert!(vec_approx_eq(&grad, &expected, 1e-5),
            "relu_backward negative failed: {:?} != {:?}", grad, expected);
    }

    #[test]
    fn test_relu_backward_mixed() {
        let ctx = GpuContext::new().expect("Failed to create GPU context");

        let mut grad = vec![1.0, 2.0, 3.0, 4.0];
        let pre_activation = vec![0.5, -1.0, 2.0, -0.1]; // mixed
        let expected = vec![1.0, 0.0, 3.0, 0.0]; // grad * ReLU'

        ctx.relu_backward(&mut grad, &pre_activation);

        assert!(vec_approx_eq(&grad, &expected, 1e-5),
            "relu_backward mixed failed: {:?} != {:?}", grad, expected);
    }

    #[test]
    fn test_hadamard_simple() {
        let ctx = GpuContext::new().expect("Failed to create GPU context");

        let mut a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let expected = vec![2.0, 6.0, 12.0, 20.0]; // element-wise multiply

        ctx.hadamard(&mut a, &b);

        assert!(vec_approx_eq(&a, &expected, 1e-5),
            "hadamard failed: {:?} != {:?}", a, expected);
    }

    #[test]
    fn test_hadamard_with_zeros() {
        let ctx = GpuContext::new().expect("Failed to create GPU context");

        let mut a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![0.0, 1.0, 0.0, 1.0];
        let expected = vec![0.0, 2.0, 0.0, 4.0];

        ctx.hadamard(&mut a, &b);

        assert!(vec_approx_eq(&a, &expected, 1e-5),
            "hadamard with zeros failed: {:?} != {:?}", a, expected);
    }

    #[test]
    fn test_backward_pass_integration() {
        let ctx = GpuContext::new().expect("Failed to create GPU context");

        // Full backward pass through one layer:
        // Forward: output = ReLU(input * W + b)
        // Backward: given delta (gradient from next layer)
        //   1. delta_relu = delta * ReLU'(z)  (where z = input * W + b before ReLU)
        //   2. dW = input^T * delta_relu
        //   3. delta_prev = delta_relu * W^T
        //   4. W = W - lr * dW

        // Setup: input (1x3), weights (3x2), output (1x2)
        let input = vec![0.5, 0.3, 0.2];
        let mut weights = vec![0.5, -0.3, 0.4, 0.2, -0.1, 0.6]; // (3x2) row-major
        let bias = vec![0.1, -0.1];

        // Forward pass (compute z before ReLU for backward)
        let mut z = ctx.matmul(&input, &weights, 1, 3, 2);
        ctx.add_bias(&mut z, &bias);
        // z = [0.5*0.5+0.3*0.4+0.2*(-0.1)+0.1, 0.5*(-0.3)+0.3*0.2+0.2*0.6+(-0.1)]
        //   = [0.25+0.12-0.02+0.1, -0.15+0.06+0.12-0.1] = [0.45, -0.07]

        let mut output = z.clone();
        ctx.relu(&mut output);
        // output = [0.45, 0.0] (ReLU zeros out the negative value)

        // Backward pass
        let delta = vec![1.0, 1.0]; // Assume gradient from loss

        // Step 1: ReLU backward - manually compute since we haven't impl relu_backward yet
        // ReLU'(z) = 1 if z > 0, else 0
        // delta_relu = delta * ReLU'(z) = [1.0*1, 1.0*0] = [1.0, 0.0]
        let delta_relu = vec![
            if z[0] > 0.0 { delta[0] } else { 0.0 },
            if z[1] > 0.0 { delta[1] } else { 0.0 },
        ]; // [1.0, 0.0]

        // Step 2: Weight gradients dW = input^T * delta_relu
        // input (1x3)^T = (3x1), delta_relu (1x2)
        // dW = (3x1) * (1x2) = (3x2)
        let dw = ctx.matmul_at_b(&input, &delta_relu, 3, 1, 2);
        // dW = [[0.5*1.0, 0.5*0.0], [0.3*1.0, 0.3*0.0], [0.2*1.0, 0.2*0.0]]
        //    = [[0.5, 0.0], [0.3, 0.0], [0.2, 0.0]]
        let expected_dw = vec![0.5, 0.0, 0.3, 0.0, 0.2, 0.0];
        assert!(vec_approx_eq(&dw, &expected_dw, 1e-5),
            "Weight gradients incorrect: {:?} != {:?}", dw, expected_dw);

        // Step 3: Delta for previous layer = delta_relu * W^T
        // delta_relu (1x2), W (3x2)^T = (2x3)
        // delta_prev = (1x2) * (2x3) = (1x3)
        let delta_prev = ctx.matmul_a_bt(&delta_relu, &weights, 1, 2, 3);
        // = [1.0*0.5+0.0*(-0.3), 1.0*0.4+0.0*0.2, 1.0*(-0.1)+0.0*0.6]
        // = [0.5, 0.4, -0.1]
        let expected_delta_prev = vec![0.5, 0.4, -0.1];
        assert!(vec_approx_eq(&delta_prev, &expected_delta_prev, 1e-5),
            "Delta prev incorrect: {:?} != {:?}", delta_prev, expected_delta_prev);

        // Step 4: SGD update W = W - lr * dW
        let learning_rate = 0.1;
        ctx.saxpy(&dw, &mut weights, -learning_rate);
        // weights = -0.1 * dW + weights
        // = [0.5-0.05, -0.3-0.0, 0.4-0.03, 0.2-0.0, -0.1-0.02, 0.6-0.0]
        // = [0.45, -0.3, 0.37, 0.2, -0.12, 0.6]
        let expected_weights = vec![0.45, -0.3, 0.37, 0.2, -0.12, 0.6];
        assert!(vec_approx_eq(&weights, &expected_weights, 1e-5),
            "Updated weights incorrect: {:?} != {:?}", weights, expected_weights);
    }
}
