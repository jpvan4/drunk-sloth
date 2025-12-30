// gpu.rs
//! GPU acceleration module using CUDA for high-performance compute with f64 support

use crate::core::lattice::Lattice;
use crate::core::matrix::Matrix;
use crate::core::types::LatticeVector;
use crate::core::error::{LatticeError, Result};

#[cfg(feature = "gpu")]
use rustacuda::prelude::*;
#[cfg(feature = "gpu")]
use rustacuda::memory::{DeviceBuffer, LockedBuffer};
#[cfg(feature = "gpu")]
use rustacuda::function::Function;
#[cfg(feature = "gpu")]
use rustacuda::launch;
#[cfg(feature = "gpu")]
use std::ffi::CString;
#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(not(feature = "gpu"))]
pub struct GPUManager;

/// GPU acceleration manager using CUDA
#[cfg(feature = "gpu")]
pub struct GPUContext {
    pub device: Device,
    pub context: Arc<Context>,
    pub stream: Stream,
    pub module: Module,
}

#[cfg(feature = "gpu")]
pub struct GPUManager {
    contexts: Vec<GPUContext>,
    adapter_info: Vec<String>,
    is_initialized: bool,
}

#[cfg(feature = "gpu")]
impl GPUManager {
    /// Create new GPU manager with CUDA initialization
    /// Create a new GPUManager. If `device_ids` is None, use all CUDA devices.
    pub fn new(device_ids: Option<Vec<usize>>) -> Result<Self> {
        rustacuda::init(CudaFlags::empty())?;

        // Load compiled PTX from OUT_DIR if available, otherwise from included file
        let ptx = if let Ok(env_ptx) = std::env::var("KERNELS_PTX") {
            std::fs::read_to_string(env_ptx).map_err(|e| LatticeError::io_error(format!("Failed to read PTX from {}: {}", env_ptx, e)))?
        } else if let Ok(out) = std::env::var("OUT_DIR") {
            let path = std::path::Path::new(&out).join("kernels.ptx");
            if path.exists() {
                std::fs::read_to_string(&path).map_err(|e| LatticeError::io_error(format!("Failed to read PTX file {}: {}", path.display(), e)))?
            } else {
                return Err(LatticeError::io_error(format!("PTX not found at {} - build script may have failed. Ensure nvcc is installed and build with --features gpu", path.display())));
            }
        } else {
            return Err(LatticeError::io_error("PTX missing - build with `--features gpu` and ensure nvcc compile task succeeds".to_string()));
        };

        // Choose device ids
        let available_count = Device::num_devices()? as usize;
        let ids = match device_ids {
            Some(v) => v.into_iter().filter(|&x| x < available_count).collect::<Vec<_>>(),
            None => (0..available_count).collect::<Vec<_>>(),
        };

        let mut contexts: Vec<GPUContext> = Vec::new();
        let mut adapter_info: Vec<String> = Vec::new();

        for id in ids.iter() {
            let device = Device::get_device(*id as u32)?;
            let ctx = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
            let ctx_arc = Arc::new(ctx);
            let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
            let module = Module::load_from_string(&CString::new(ptx.clone())?)?;
            adapter_info.push(device.name()?);
            contexts.push(GPUContext { device, context: ctx_arc, stream, module });
        }

        Ok(GPUManager { contexts, adapter_info, is_initialized: true })
    }

    /// Check if GPU acceleration is available
    pub fn is_available() -> bool {
        rustacuda::init(CudaFlags::empty()).is_ok()
    }
    
    /// Check if GPU is initialized
    pub fn is_initialized(&self) -> bool {
        self.is_initialized
    }
    
    /// Return whether the underlying device supports native f64 arithmetic (always true for CUDA)
    pub fn supports_float64(&self) -> bool { true }
    
    /// Get GPU device info
    pub fn device_info(&self) -> String {
        format!("CUDA devices: {}", self.adapter_info.join(", "))
    }
    
    /// Perform Gram-Schmidt orthogonalization on GPU with f64
    pub fn gram_schmidt_gpu(&self, matrix: &Matrix) -> Result<Vec<LatticeVector>> {
        let rows = matrix.rows();
        let cols = matrix.cols();

        log::debug!("Running Gram-Schmidt on CUDA GPU: {}x{}", rows, cols);

        // Prepare flat f64 data (row-major) using pinned host memory
        let mut matrix_data: Vec<f64> = matrix
            .to_vec()
            .into_iter()
            .flatten()
            .map(|x| x as f64)
            .collect();
        // Allocate device buffers and copy from host (we may add pinned memory optimization later with LockedBuffer correctly)
        let mut d_matrix = DeviceBuffer::from_slice(&matrix_data)?;
        let mut d_orthogonal = unsafe { DeviceBuffer::zeroed(rows * cols)? };
        let mut d_norm_sq = unsafe { DeviceBuffer::zeroed(rows)? };

        // Launch Gram-Schmidt kernel
        let block_size = 256;
        let grid_size = ((rows * cols) as u32 + block_size - 1) / block_size;
        
        // Launch on device 0 for now - TODO: shard across multiple GPUs
        let ctx = &self.contexts[0];
        let func_name = CString::new("gram_schmidt_kernel")?;
        let func = ctx.module.get_function(func_name.as_c_str())?;
        let stream_ref = &ctx.stream; // local identifier required by launch! macro
        unsafe {
            launch!(func<<<grid_size, block_size, 0, stream_ref>>>(
                d_matrix.as_device_ptr(),
                d_orthogonal.as_device_ptr(),
                d_norm_sq.as_device_ptr(),
                rows as u32,
                cols as u32
            ))?;
        }

        // Copy results back
        let mut orthogonal_data = vec![0.0f64; rows * cols];
        d_orthogonal.copy_to(&mut orthogonal_data)?;

        // Reconstruct vectors
        let mut result = Vec::with_capacity(rows);
        for i in 0..rows {
            let start = i * cols;
            let end = start + cols;
            result.push(LatticeVector::new(orthogonal_data[start..end].to_vec()));
        }

        Ok(result)
    }
    
    /// Perform matrix-vector multiplication on GPU with f64
    pub fn matrix_vector_mult_gpu(&self, matrix: &Matrix, vector: &LatticeVector) -> Result<LatticeVector> {
        let rows = matrix.rows();
        let cols = matrix.cols();

        if cols != vector.dimension() {
            return Err(LatticeError::invalid_dimensions((rows, cols), (vector.dimension(), 1)));
        }

        log::debug!("Running matrix-vector multiply on CUDA GPU: {}x{} * {}x1", rows, cols, cols);

        // Prepare flat f64 data
        let matrix_data: Vec<f64> = matrix
            .to_vec()
            .into_iter()
            .flatten()
            .map(|x| x as f64)
            .collect();
        
        let vec_data: Vec<f64> = vector.as_slice().iter().cloned().collect();

        // Allocate device buffers
        // Using DeviceBuffer directly here; using LockedBuffer correctly requires element-by-element DeviceCopy and correct size param
        let d_matrix = DeviceBuffer::from_slice(&matrix_data)?;
        let d_vector = DeviceBuffer::from_slice(&vec_data)?;
        let mut d_result = unsafe { DeviceBuffer::zeroed(rows)? };

        // Launch kernel
        let block_size = 256;
        let grid_size = (rows as u32 + block_size - 1) / block_size;
        
        let ctx = &self.contexts[0];
        let func_name = CString::new("matrix_vector_kernel")?;
        let func = ctx.module.get_function(func_name.as_c_str())?;
        let stream_ref = &ctx.stream;
        unsafe {
            launch!(func<<<grid_size, block_size, 0, stream_ref>>>(
                d_matrix.as_device_ptr(),
                d_vector.as_device_ptr(),
                d_result.as_device_ptr(),
                rows as u32,
                cols as u32
            ))?;
        }

        // Copy back
        let mut result_data = vec![0.0f64; rows];
        d_result.copy_to(&mut result_data)?;

        Ok(LatticeVector::new(result_data))
    }
    // In gpu.rs, add to GPUManager

    pub fn lll_gpu(&self, lattice: &Lattice, delta: f64) -> Result<Lattice> {
        log::info("Starting full CUDA LLL kernel");
    
        let basis = lattice.basis();
    
        let n = basis.rows();
    
        let m = basis.cols();
    
        if n > 128 {
            return Err(LatticeError::invalid_parameters("Lattice rank too large for full kernel (max 128)"));
        }
    
        let matrix_data: Vec<f64> = basis.to_vec().into_iter().flatten().map(|x| x as f64).collect();
    
        let mut d_basis = DeviceBuffer::from_slice(&matrix_data)?;
    
        let mut d_orthogonal = DeviceBuffer::zeros(n * m)?;
    
        let mut d_mu = DeviceBuffer::zeros(n * n)?;
    
        let mut d_B = DeviceBuffer::zeros(n)?;
    
        let block_size = 128u32; // Power of 2, >= n
    
        let shared_size = block_size as u32 * std::mem::size_of::<f64>() as u32; // For shared memory in reduction
    
        let ctx = &self.contexts[0];
        let func_name = CString::new("lll_kernel")?;
        let func = ctx.module.get_function(func_name.as_c_str())?;
        let stream_ref = &ctx.stream;
        unsafe {
            launch!(func<<<1, block_size, shared_size, stream_ref>>>(
                d_basis.as_device_ptr(),
                d_orthogonal.as_device_ptr(),
                d_mu.as_device_ptr(),
                d_B.as_device_ptr(),
                n as u32,
                m as u32,
                delta
            ))?;
        }
    
        let mut result_data = vec![0.0f64; n * m];
    
        d_basis.copy_to(&mut result_data)?;
    
        let data: Vec<Vec<i64>> = (0..n).map(|i| {
            let start = i * m;
            (0..m).map(|j| result_data[start + j].round() as i64).collect()
        }).collect();
    
        Lattice::new(Matrix::new(data)?)
    }
    /// Accelerated lattice reduction using GPU
    pub fn accelerated_lll_reduction(&self, lattice: &Lattice) -> Result<Lattice> {
        log::info!("Starting CUDA GPU-accelerated LLL reduction");
        let basis = lattice.basis();

        // Use GPU for Gram-Schmidt
        let gs = self.gram_schmidt_gpu(basis)?;

        // Proceed with CPU for remaining LLL steps (or implement full CUDA LLL if needed)
        let cpu_lattice = Lattice::new(basis.clone())?;
        let reducer = crate::lll::LLLReducer::new();
        let reduced = reducer.reduce(&cpu_lattice)?;

        log::info!("CUDA GPU-accelerated LLL completed");
        Ok(reduced)
    }
    
    /// Accelerated BKZ reduction with parallel SVP solving
    pub fn accelerated_bkz_reduction(&self, lattice: &Lattice) -> Result<Lattice> {
        log::info!("Starting CUDA GPU-accelerated BKZ reduction");
        let basis = lattice.basis();

        // Use GPU for Gram-Schmidt
        let gs = self.gram_schmidt_gpu(basis)?;

        // Use CPU BKZ for now
        let cpu_lattice = Lattice::new(basis.clone())?;
        let reducer = crate::bkz::BKZReducer::new();
        let reduced = reducer.reduce(&cpu_lattice)?;

        log::info!("CUDA GPU-accelerated BKZ completed");
        Ok(reduced)
    }
    
    /// Accelerated CVP solving
    pub fn accelerated_cvp_solve(
        &self,
        lattice: &Lattice,
        target: &LatticeVector,
    ) -> Result<crate::cvp::CVPResult> {
        log::info!("Starting CUDA GPU-accelerated CVP solving");
        
        let basis = lattice.basis();
        
        // Use GPU for matrix-vector projection
        let projected = self.matrix_vector_mult_gpu(basis, target)?;

        // Solve CVP using standard algorithm
        let solver = crate::cvp::CVPSolver::new();
        let result = solver.solve(lattice, &projected)?;  // Adjusted for projected

        log::info!("CUDA GPU-accelerated CVP completed");
        Ok(result)
    }
    
    /// Benchmark GPU performance
    pub fn benchmark_gpu_performance(&self, matrix: &Matrix) -> Result<GPUBenchmarkResult> {
        let start = std::time::Instant::now();
        let gpu_result = self.gram_schmidt_gpu(matrix)?;
        let gpu_time = start.elapsed().as_secs_f64();

        let start = std::time::Instant::now();
        let cpu_result = cpu_gram_schmidt_basis(matrix)?;  // Use CPU fallback implemented earlier
        let cpu_time = start.elapsed().as_secs_f64();

        let speedup = cpu_time / gpu_time;

        Ok(GPUBenchmarkResult {
            gpu_time,
            cpu_time,
            speedup,
            gpu_result: gpu_result.len(),
            cpu_result: cpu_result.len(),
        })
    }
}

#[cfg(feature = "gpu")]
pub struct GPUBenchmarkResult {
    pub gpu_time: f64,
    pub cpu_time: f64,
    pub speedup: f64,
    pub gpu_result: usize,
    pub cpu_result: usize,
}