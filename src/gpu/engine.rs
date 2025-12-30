// gpu/engine.rs
// Almost-Production-ready GPU engine

use crate::core::architecture::*;
use wgpu::util::DeviceExt;
use std::sync::Arc;

pub struct GPUEngine {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipelines: GPUPipelines,
    precision: PrecisionMode,
}

pub struct GPUPipelines {
    pub gram_schmidt: wgpu::ComputePipeline,
    pub matrix_mult: wgpu::ComputePipeline,
    pub lll_reduction: wgpu::ComputePipeline,
    pub svp_enumeration: wgpu::ComputePipeline,
}

impl GPUEngine {
    pub async fn new(precision: PrecisionMode) -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| LatticeCoreError::GpuError("No GPU adapter found".to_string()))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("lattice_gpu_engine"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                    memory_hints: wgpu::MemoryHints::default(),
                    trace: wgpu::Trace::Off,
                },
                None,
            )
            .await
            .map_err(|e| LatticeCoreError::GpuError(format!("Device creation failed: {}", e)))?;

        let pipelines = Self::create_pipelines(&device).await?;

        Ok(GPUEngine {
            device: Arc::new(device),
            queue: Arc::new(queue),
            pipelines,
            precision,
        })
    }

    async fn create_pipelines(device: &wgpu::Device) -> Result<GPUPipelines> {
        let gram_schmidt = Self::create_compute_pipeline(
            device,
            include_str!("shaders/gram_schmidt.wgsl"),
            "gram_schmidt",
        )?;

        let matrix_mult = Self::create_compute_pipeline(
            device,
            include_str!("shaders/matrix_mult.wgsl"), 
            "matrix_mult",
        )?;

        let lll_reduction = Self::create_compute_pipeline(
            device,
            include_str!("shaders/lll.wgsl"),
            "lll_reduce",
        )?;

        let svp_enumeration = Self::create_compute_pipeline(
            device,
            include_str!("shaders/svp.wgsl"),
            "svp_enum",
        )?;

        Ok(GPUPipelines {
            gram_schmidt,
            matrix_mult,
            lll_reduction,
            svp_enumeration,
        })
    }

    fn create_compute_pipeline(
        device: &wgpu::Device,
        source: &str,
        entry_point: &str,
    ) -> Result<wgpu::ComputePipeline> {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compute_shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compute_pipeline"),
            layout: None,
            module: &shader_module,
            entry_point: Some(entry_point),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(pipeline)
    }

    pub async fn gram_schmidt(&self, basis: &GPUBasis) -> Result<GPUBasis> {
        match basis {
            GPUBasis::F32(data) => self.gram_schmidt_f32(data).await,
        }
    }

    async fn gram_schmidt_f32(&self, matrix: &[Vec<f32>]) -> Result<GPUBasis> {
        let rows = matrix.len();
        let cols = matrix[0].len();

        // Flatten matrix data
        let flat_data: Vec<f32> = matrix.iter().flat_map(|row| row.iter().copied()).collect();

        // Create buffers
        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("input_matrix"),
            contents: bytemuck::cast_slice(&flat_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output_matrix"),
            size: (flat_data.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let dimensions_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dimensions"),
            contents: bytemuck::cast_slice(&[rows as u32, cols as u32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group
        let bind_group_layout = self.pipelines.gram_schmidt.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gram_schmidt_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dimensions_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch compute
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gram_schmidt_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gram_schmidt_compute"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipelines.gram_schmidt);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(
                ((rows as u32 + 31) / 32) as u32,
                ((cols as u32 + 31) / 32) as u32,
                1,
            );
        }

        // Copy to staging buffer for readback
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_buffer"),
            size: output_buffer.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer.size());
        self.queue.submit(Some(encoder.finish()));

        // Read back results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_channel::oneshot::channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        self.device.poll(wgpu::PollType::WaitIndefinitely);
        
        receiver.await
            .map_err(|_| LatticeCoreError::GpuError("Channel failed".to_string()))?
            .map_err(|_| LatticeCoreError::GpuError("Buffer mapping failed".to_string()))?;

        let mapped_data = buffer_slice.get_mapped_range();
        let result_data: &[f32] = bytemuck::cast_slice(&mapped_data);

        // Reconstruct matrix
        let mut result_matrix = Vec::with_capacity(rows);
        for i in 0..rows {
            let start = i * cols;
            let end = start + cols;
            result_matrix.push(result_data[start..end].to_vec());
        }

        drop(mapped_data);
        staging_buffer.unmap();

        Ok(GPUBasis::F32(result_matrix))
    }
}

#[async_trait]
impl LatticeEngine for GPUEngine {
    async fn reduce_lll(&self, basis: &LatticeBasis) -> Result<LatticeBasis> {
        let gpu_basis = basis.to_gpu_format().await?;
        let result = self.gram_schmidt(&gpu_basis).await?;
        
        // Convert back to original precision
        match result {
            GPUBasis::F32(data) => {
                let f64_data: Vec<Vec<f64>> = data
                    .iter()
                    .map(|row| row.iter().map(|&x| x as f64).collect())
                    .collect();
                Ok(LatticeBasis::new_f64(f64_data))
            }
        }
    }

    async fn reduce_bkz(&self, _basis: &LatticeBasis, _beta: usize) -> Result<LatticeBasis> {
        // Implementation for BKZ on GPU
        todo!("BKZ GPU implementation")
    }

    async fn solve_svp(&self, _basis: &LatticeBasis) -> Result<SVPResult> {
        // Implementation for SVP on GPU  
        todo!("SVP GPU implementation")
    }

    async fn solve_cvp(&self, _basis: &LatticeBasis, _target: &[f64]) -> Result<CVPResult> {
        // Implementation for CVP on GPU
        todo!("CVP GPU implementation")
    }

    fn precision(&self) -> PrecisionMode {
        self.precision
    }

    fn supports_gpu(&self) -> bool {
        true
    }
}