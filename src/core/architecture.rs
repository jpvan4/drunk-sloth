// core/architecture.rs
// Complete architectural foundation

use std::sync::Arc;
use async_trait::async_trait;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LatticeCoreError {
    #[error("GPU acceleration failed: {0}")]
    GpuError(String),
    #[error("Precision error: {0}")]
    PrecisionError(String),
    #[error("Numerical instability: {0}")]
    NumericalError(String),
    #[error("Invalid parameters: {0}")]
    ParameterError(String),
    #[error("Dimension mismatch: expected {expected:?}, got {actual:?}")]
    DimensionError { expected: (usize, usize), actual: (usize, usize) },
    #[error("Algorithm failed to converge")]
    ConvergenceError,
}

pub type Result<T> = std::result::Result<T, LatticeCoreError>;

/// Unified precision system
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PrecisionMode {
    F64,
    F32, 
    BigInt(usize), // bits
    BigFloat(usize), // bits
}

impl PrecisionMode {
    pub fn bits(&self) -> usize {
        match self {
            PrecisionMode::F64 => 64,
            PrecisionMode::F32 => 32,
            PrecisionMode::BigInt(bits) => *bits,
            PrecisionMode::BigFloat(bits) => *bits,
        }
    }
    
    pub fn is_big(&self) -> bool {
        matches!(self, PrecisionMode::BigInt(_) | PrecisionMode::BigFloat(_))
    }
}

/// Core lattice trait that works across all backends
#[async_trait]
pub trait LatticeEngine: Send + Sync {
    async fn reduce_lll(&self, basis: &LatticeBasis) -> Result<LatticeBasis>;
    async fn reduce_bkz(&self, basis: &LatticeBasis, beta: usize) -> Result<LatticeBasis>;
    async fn solve_svp(&self, basis: &LatticeBasis) -> Result<SVPResult>;
    async fn solve_cvp(&self, basis: &LatticeBasis, target: &[f64]) -> Result<CVPResult>;
    
    fn precision(&self) -> PrecisionMode;
    fn supports_gpu(&self) -> bool;
}

/// Unified basis representation
#[derive(Debug, Clone)]
pub struct LatticeBasis {
    data: MatrixData,
    precision: PrecisionMode,
}

#[derive(Debug, Clone)]
pub enum MatrixData {
    F64(Vec<Vec<f64>>),
    F32(Vec<Vec<f32>>),
    BigInt(Vec<Vec<rug::Integer>>),
    BigFloat(Vec<Vec<rug::Float>>),
}

impl LatticeBasis {
    pub fn new_f64(data: Vec<Vec<f64>>) -> Self {
        Self {
            data: MatrixData::F64(data),
            precision: PrecisionMode::F64,
        }
    }
    
    pub fn new_bigint(data: Vec<Vec<rug::Integer>>, bits: usize) -> Self {
        Self {
            data: MatrixData::BigInt(data),
            precision: PrecisionMode::BigInt(bits),
        }
    }
    
    pub async fn to_gpu_format(&self) -> Result<GPUBasis> {
        match &self.data {
            MatrixData::F64(data) => {
                // Convert to f32 for GPU (wgpu doesn't handle f64 well)
                let f32_data: Vec<Vec<f32>> = data
                    .iter()
                    .map(|row| row.iter().map(|&x| x as f32).collect())
                    .collect();
                Ok(GPUBasis::F32(f32_data))
            }
            MatrixData::BigInt(data) => {
                // Convert big int to f64 approximation for GPU
                let f64_data: Vec<Vec<f64>> = data
                    .iter()
                    .map(|row| row.iter().map(|x| x.to_f64()).collect())
                    .collect();
                let f32_data: Vec<Vec<f32>> = f64_data
                    .iter()
                    .map(|row| row.iter().map(|&x| x as f32).collect())
                    .collect();
                Ok(GPUBasis::F32(f32_data))
            }
            _ => Err(LatticeCoreError::PrecisionError(
                "Unsupported precision for GPU conversion".to_string()
            )),
        }
    }
}

/// GPU-compatible basis
pub enum GPUBasis {
    F32(Vec<Vec<f32>>),
    // Add other formats as needed
}

/// Results that work across all backends
#[derive(Debug, Clone)]
pub struct SVPResult {
    pub vector: LatticeBasis,
    pub norm: f64,
    pub iterations: u64,
    pub precision: PrecisionMode,
}

#[derive(Debug, Clone)]
pub struct CVPResult {
    pub closest_vector: LatticeBasis,
    pub distance: f64,
    pub target: LatticeBasis,
    pub iterations: u64,
}