//! lattice reduction crate with LLL, BKZ, SVP, and CVP solvers
//!
//! This crate provides lattice reduction algorithms with support for:
//! - LLL (Lenstra-Lenstra-Lovász) reduction
//! - BKZ (Block Korkine-Zolotarev) reduction  
//! - SVP (Shortest Vector Problem) solving
//! - CVP (Closest Vector Problem) solving
//! - GPU acceleration (optional)
//! - High-precision arithmetic (optional)
//!
//! # Examples
//!
//! Basic LLL reduction:
//! ```rust
//! use lattice_solver::{Lattice, LLLReducer, LLLParams};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let lattice = Lattice::from_matrix(vec![
//!     vec![1, 2, 3],
//!     vec![4, 5, 6],
//!     vec![7, 8, 9],
//! ])?;
//!
//! let params = LLLParams::default();
//! let reducer = LLLReducer::with_params(params);
//! let reduced = reducer.reduce(&lattice)?;
//! println!("Reduced lattice dimension: {:?}", reduced.dimension());
//! # Ok(())
//! # }
//! ```
//!
//! BKZ reduction with custom block size:
//! ```rust
//! use lattice_solver::{BKZReducer, BKZParams, Lattice};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let lattice = Lattice::from_matrix(vec![
//!     vec![1, 0, 0],
//!     vec![0, 1, 0],
//!     vec![0, 0, 1],
//! ])?;
//! let params = BKZParams::new(20); // block size = 20
//! let reducer = BKZReducer::with_params(params);
//! let reduced = reducer.reduce(&lattice)?;
//! println!("BKZ reduced dimension: {:?}", reduced.dimension());
//! # Ok(())
//! # }
//! ```

//! lattice reduction crate

pub mod core;
pub mod lll;
pub mod bkz;
pub mod svp;
pub mod cvp;
pub mod gpu;
pub mod precision;
pub mod utils;

pub use core::*;
pub use lll::*;
pub use bkz::*;
pub use svp::*;
pub use cvp::*;
pub use precision::PrecisionManager;
#[cfg(feature = "gpu")]
pub use gpu::{GPUManager, GPUAcceleratedOperations};
#[cfg(not(feature = "gpu"))]
pub use gpu::GPUManager;

// Re-export commonly used types
pub use core::lattice::Lattice;
pub use core::matrix::Matrix;
pub use core::error::{LatticeError, Result};

/// Feature flag utilities
pub mod features {
    /// Check if GPU support is enabled and available
    pub fn gpu_available() -> bool {
        cfg!(feature = "gpu")
    }
    
    /// Check if high precision arithmetic is enabled
    pub fn high_precision_enabled() -> bool {
        cfg!(feature = "high-precision")
    }
    
    /// Check if parallel processing is enabled
    pub fn parallel_enabled() -> bool {
        cfg!(feature = "parallel")
    }
}

/// Validate that requested features are available
pub fn validate_features() -> Result<()> {
    log::info!("Feature status - GPU: {}, High-Precision: {}, Parallel: {}", 
               features::gpu_available(),
               features::high_precision_enabled(),
               features::parallel_enabled());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lattice_creation() {
        let matrix = vec![vec![1, 2], vec![3, 4]];
        let lattice = Lattice::from_matrix(matrix).unwrap();
        assert_eq!(lattice.dimension(), (2, 2));
    }
    
    #[test]
    fn test_feature_detection() {
        assert_eq!(features::gpu_available(), cfg!(feature = "gpu"));
        assert_eq!(features::high_precision_enabled(), cfg!(feature = "high-precision"));
    }

    #[test]
    fn test_lll_reduction_2d() {
        let matrix = vec![vec![1, 1], vec![1, 0]];
        let lattice = Lattice::from_matrix(matrix).unwrap();
        
        let params = LLLParams::default();
        let reducer = LLLReducer::with_params(params);
        let reduced = reducer.reduce(&lattice).unwrap();
        
        assert!(reduced.is_reduced().unwrap());
    }
}
