//! Core error types for lattice operations

/// Error types for lattice reduction operations
#[derive(Debug, Clone)]
pub enum LatticeError {
    /// Invalid matrix dimensions
    InvalidDimensions {
        expected: (usize, usize),
        found: (usize, usize),
    },
    
    /// Matrix is not full rank
    NotFullRank(usize),
    
    /// Numerical instability detected
    NumericalInstability(String),
    
    /// Invalid parameters
    InvalidParameters(String),
    
    /// Operation not supported
    NotSupported(String),
    
    /// Memory allocation failure
    MemoryAllocation,
    
    /// GPU computation error
    GpuError(String),
    
    /// High precision arithmetic error
    PrecisionError(String),
    
    /// File I/O error
    IoError(String),
    
    /// Custom error with message
    Custom(String),
}

impl std::fmt::Display for LatticeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LatticeError::InvalidDimensions { expected, found } => {
                write!(f, "Invalid dimensions: expected {:?}, found {:?}", expected, found)
            }
            LatticeError::NotFullRank(rank) => {
                write!(f, "Matrix is not full rank, rank = {}", rank)
            }
            LatticeError::NumericalInstability(msg) => {
                write!(f, "Numerical instability: {}", msg)
            }
            LatticeError::InvalidParameters(msg) => {
                write!(f, "Invalid parameters: {}", msg)
            }
            LatticeError::NotSupported(msg) => {
                write!(f, "Operation not supported: {}", msg)
            }
            LatticeError::MemoryAllocation => {
                write!(f, "Memory allocation failed")
            }
            LatticeError::GpuError(msg) => {
                write!(f, "GPU error: {}", msg)
            }
            LatticeError::PrecisionError(msg) => {
                write!(f, "Precision error: {}", msg)
            }
            LatticeError::IoError(msg) => {
                write!(f, "I/O error: {}", msg)
            }
            LatticeError::Custom(msg) => {
                write!(f, "Error: {}", msg)
            }
        }
    }
}

impl std::error::Error for LatticeError {}

// Convert from rustacuda::error::CudaError to LatticeError for `?` convenience
// impl From<rustacuda::error::CudaError> for LatticeError {
//     fn from(e: rustacuda::error::CudaError) -> Self {
//         LatticeError::gpu_error(format!("CUDA error: {}", e))
//     }
// }

impl From<std::ffi::NulError> for LatticeError {
    fn from(e: std::ffi::NulError) -> Self {
        LatticeError::io_error(format!("NulError: {}", e))
    }
}

impl From<std::io::Error> for LatticeError {
    fn from(e: std::io::Error) -> Self {
        LatticeError::io_error(format!("I/O Error: {}", e))
    }
}

/// Result type for lattice operations
pub type Result<T> = std::result::Result<T, LatticeError>;

impl LatticeError {
    /// Create a custom error with the given message
    pub fn custom(msg: impl Into<String>) -> Self {
        LatticeError::Custom(msg.into())
    }
    
    /// Create an invalid dimensions error
    pub fn invalid_dimensions(expected: (usize, usize), found: (usize, usize)) -> Self {
        LatticeError::InvalidDimensions { expected, found }
    }
    
    /// Create a not full rank error
    pub fn not_full_rank(rank: usize) -> Self {
        LatticeError::NotFullRank(rank)
    }
    
    /// Create a numerical instability error
    pub fn numerical_instability(msg: impl Into<String>) -> Self {
        LatticeError::NumericalInstability(msg.into())
    }
    
    /// Create an invalid parameters error
    pub fn invalid_parameters(msg: impl Into<String>) -> Self {
        LatticeError::InvalidParameters(msg.into())
    }
    
    /// Create a not supported error
    pub fn not_supported(msg: impl Into<String>) -> Self {
        LatticeError::NotSupported(msg.into())
    }
    
    /// Create a GPU error
    pub fn gpu_error(msg: impl Into<String>) -> Self {
        LatticeError::GpuError(msg.into())
    }
    
    /// Create a precision error
    pub fn precision_error(msg: impl Into<String>) -> Self {
        LatticeError::PrecisionError(msg.into())
    }
    
    /// Create an I/O error
    pub fn io_error(msg: impl Into<String>) -> Self {
        LatticeError::IoError(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_creation() {
        let err = LatticeError::custom("test error");
        assert_eq!(format!("{}", err), "Error: test error");
    }
    
    #[test]
    fn test_invalid_dimensions_error() {
        let err = LatticeError::invalid_dimensions((3, 3), (2, 2));
        assert_eq!(format!("{}", err), "Invalid dimensions: expected (3, 3), found (2, 2)");
    }
}