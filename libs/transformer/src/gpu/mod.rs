//! GPU acceleration for transformer operations using WebGPU
//!
//! This module provides GPU compute shaders for key transformer operations:
//! - Attention computation (QK^T, softmax, attention @ V)
//! - Layer normalization
//! - GELU activation
//! - Matrix multiplication
//! - Embedding lookup

pub mod ops;
pub mod forward;

use std::sync::Arc;
use wgpu;

pub use ops::GpuOps;
pub use forward::GpuTransformer;

/// GPU context holding device, queue, and compute pipelines
pub struct GpuContext {
    /// WebGPU device
    pub device: Arc<wgpu::Device>,
    /// Command queue
    pub queue: Arc<wgpu::Queue>,

    // Compute pipelines
    attention_qk_pipeline: wgpu::ComputePipeline,
    attention_v_pipeline: wgpu::ComputePipeline,
    softmax_pipeline: wgpu::ComputePipeline,
    layer_norm_pipeline: wgpu::ComputePipeline,
    gelu_pipeline: wgpu::ComputePipeline,
    matmul_pipeline: wgpu::ComputePipeline,
    add_pipeline: wgpu::ComputePipeline,
    embedding_pipeline: wgpu::ComputePipeline,

    // Bind group layouts
    attention_qk_layout: wgpu::BindGroupLayout,
    attention_v_layout: wgpu::BindGroupLayout,
    softmax_layout: wgpu::BindGroupLayout,
    layer_norm_layout: wgpu::BindGroupLayout,
    gelu_layout: wgpu::BindGroupLayout,
    matmul_layout: wgpu::BindGroupLayout,
    add_layout: wgpu::BindGroupLayout,
    embedding_layout: wgpu::BindGroupLayout,
}

impl GpuContext {
    /// Create a new GPU context asynchronously
    pub async fn new_async() -> Result<Self, String> {
        // Request adapter
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
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
            .ok_or("Failed to find GPU adapter")?;

        // Request device
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Transformer GPU"),
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

        // Create pipelines
        Self::create_pipelines(device, queue)
    }

    /// Create a new GPU context (blocking)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Result<Self, String> {
        pollster::block_on(Self::new_async())
    }

    fn create_pipelines(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    ) -> Result<Self, String> {
        // Attention QK pipeline
        let attention_qk_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Attention QK Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/attention_qk.wgsl").into()),
        });

        let attention_qk_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Attention QK Layout"),
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

        let attention_qk_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Attention QK Pipeline Layout"),
            bind_group_layouts: &[&attention_qk_layout],
            push_constant_ranges: &[],
        });

        let attention_qk_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Attention QK Pipeline"),
            layout: Some(&attention_qk_pipeline_layout),
            module: &attention_qk_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Attention V pipeline
        let attention_v_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Attention V Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/attention_v.wgsl").into()),
        });

        let attention_v_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Attention V Layout"),
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

        let attention_v_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Attention V Pipeline Layout"),
            bind_group_layouts: &[&attention_v_layout],
            push_constant_ranges: &[],
        });

        let attention_v_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Attention V Pipeline"),
            layout: Some(&attention_v_pipeline_layout),
            module: &attention_v_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Softmax pipeline
        let softmax_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Softmax Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/softmax_rows.wgsl").into()),
        });

        let softmax_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Softmax Layout"),
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
            ],
        });

        let softmax_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Softmax Pipeline Layout"),
            bind_group_layouts: &[&softmax_layout],
            push_constant_ranges: &[],
        });

        let softmax_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Softmax Pipeline"),
            layout: Some(&softmax_pipeline_layout),
            module: &softmax_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Layer norm pipeline
        let layer_norm_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Layer Norm Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/layer_norm.wgsl").into()),
        });

        let layer_norm_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Layer Norm Layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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

        let layer_norm_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Layer Norm Pipeline Layout"),
            bind_group_layouts: &[&layer_norm_layout],
            push_constant_ranges: &[],
        });

        let layer_norm_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Layer Norm Pipeline"),
            layout: Some(&layer_norm_pipeline_layout),
            module: &layer_norm_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // GELU pipeline
        let gelu_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GELU Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gelu.wgsl").into()),
        });

        let gelu_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GELU Layout"),
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
            ],
        });

        let gelu_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GELU Pipeline Layout"),
            bind_group_layouts: &[&gelu_layout],
            push_constant_ranges: &[],
        });

        let gelu_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GELU Pipeline"),
            layout: Some(&gelu_pipeline_layout),
            module: &gelu_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Matmul pipeline
        let matmul_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Matmul Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/matmul.wgsl").into()),
        });

        let matmul_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Matmul Layout"),
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

        let matmul_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Matmul Pipeline Layout"),
            bind_group_layouts: &[&matmul_layout],
            push_constant_ranges: &[],
        });

        let matmul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Matmul Pipeline"),
            layout: Some(&matmul_pipeline_layout),
            module: &matmul_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Add pipeline
        let add_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Add Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/add.wgsl").into()),
        });

        let add_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Add Layout"),
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

        let add_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Add Pipeline Layout"),
            bind_group_layouts: &[&add_layout],
            push_constant_ranges: &[],
        });

        let add_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Add Pipeline"),
            layout: Some(&add_pipeline_layout),
            module: &add_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Embedding pipeline
        let embedding_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Embedding Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/embedding.wgsl").into()),
        });

        let embedding_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Embedding Layout"),
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

        let embedding_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Embedding Pipeline Layout"),
            bind_group_layouts: &[&embedding_layout],
            push_constant_ranges: &[],
        });

        let embedding_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Embedding Pipeline"),
            layout: Some(&embedding_pipeline_layout),
            module: &embedding_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            attention_qk_pipeline,
            attention_v_pipeline,
            softmax_pipeline,
            layer_norm_pipeline,
            gelu_pipeline,
            matmul_pipeline,
            add_pipeline,
            embedding_pipeline,
            attention_qk_layout,
            attention_v_layout,
            softmax_layout,
            layer_norm_layout,
            gelu_layout,
            matmul_layout,
            add_layout,
            embedding_layout,
        })
    }

    /// Get bind group layout for attention QK computation
    pub fn attention_qk_layout(&self) -> &wgpu::BindGroupLayout {
        &self.attention_qk_layout
    }

    /// Get bind group layout for attention V computation
    pub fn attention_v_layout(&self) -> &wgpu::BindGroupLayout {
        &self.attention_v_layout
    }

    /// Get bind group layout for softmax
    pub fn softmax_layout(&self) -> &wgpu::BindGroupLayout {
        &self.softmax_layout
    }

    /// Get bind group layout for layer normalization
    pub fn layer_norm_layout(&self) -> &wgpu::BindGroupLayout {
        &self.layer_norm_layout
    }

    /// Get bind group layout for GELU activation
    pub fn gelu_layout(&self) -> &wgpu::BindGroupLayout {
        &self.gelu_layout
    }

    /// Get bind group layout for matrix multiplication
    pub fn matmul_layout(&self) -> &wgpu::BindGroupLayout {
        &self.matmul_layout
    }

    /// Get bind group layout for element-wise addition
    pub fn add_layout(&self) -> &wgpu::BindGroupLayout {
        &self.add_layout
    }

    /// Get bind group layout for embedding lookup
    pub fn embedding_layout(&self) -> &wgpu::BindGroupLayout {
        &self.embedding_layout
    }

    /// Get attention QK pipeline
    pub fn attention_qk_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.attention_qk_pipeline
    }

    /// Get attention V pipeline
    pub fn attention_v_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.attention_v_pipeline
    }

    /// Get softmax pipeline
    pub fn softmax_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.softmax_pipeline
    }

    /// Get layer norm pipeline
    pub fn layer_norm_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.layer_norm_pipeline
    }

    /// Get GELU pipeline
    pub fn gelu_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.gelu_pipeline
    }

    /// Get matmul pipeline
    pub fn matmul_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.matmul_pipeline
    }

    /// Get add pipeline
    pub fn add_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.add_pipeline
    }

    /// Get embedding pipeline
    pub fn embedding_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.embedding_pipeline
    }
}

/// Check if GPU is available
pub async fn is_gpu_available() -> bool {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_gpu_context_creation() {
        // This test requires GPU hardware
        let result = GpuContext::new();
        // GPU might not be available in CI
        if result.is_ok() {
            let _ctx = result.unwrap();
            // Context created successfully
        }
    }
}
