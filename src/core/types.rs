//! Core types for lattice operations

use crate::core::error::{LatticeError, Result};
use crate::core::matrix::Matrix;
use serde::{Deserialize, Serialize};

/// Lattice basis represented as a matrix
pub type LatticeBasis = Matrix;

/// Vector in lattice space
#[derive(Debug, Clone, PartialEq)]
pub struct LatticeVector {
    data: Vec<f64>,
    dimension: usize,
}

impl LatticeVector {
    /// Create a new lattice vector
    pub fn new(data: Vec<f64>) -> Self {
        let dimension = data.len();
        LatticeVector { data, dimension }
    }
    
    /// Create a zero vector of given dimension
    pub fn zeros(dimension: usize) -> Self {
        LatticeVector {
            data: vec![0.0; dimension],
            dimension,
        }
    }
    
    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    
    /// Get element at index
    pub fn get(&self, index: usize) -> Option<&f64> {
        self.data.get(index)
    }

    /// Get backing slice (read-only)
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Get mutable slice (internal use)
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }
    
    /// Set element at index
    pub fn set(&mut self, index: usize, value: f64) -> Result<()> {
        if index >= self.dimension {
            return Err(LatticeError::invalid_parameters(format!(
                "index {} out of bounds for dimension {}",
                index, self.dimension
            )));
        }
        self.data[index] = value;
        Ok(())
    }
    
    /// Euclidean norm
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
    
    /// Dot product with another vector
    pub fn dot(&self, other: &LatticeVector) -> Result<f64> {
        if self.dimension != other.dimension {
            return Err(LatticeError::invalid_dimensions(
                (self.dimension, 1),
                (other.dimension, 1),
            ));
        }
        
        Ok(self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).sum())
    }
    
    /// Addition
    pub fn add(&self, other: &LatticeVector) -> Result<Self> {
        if self.dimension != other.dimension {
            return Err(LatticeError::invalid_dimensions(
                (self.dimension, 1),
                (other.dimension, 1),
            ));
        }
        
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect();
        Ok(LatticeVector::new(data))
    }
    
    /// Subtraction
    pub fn sub(&self, other: &LatticeVector) -> Result<Self> {
        if self.dimension != other.dimension {
            return Err(LatticeError::invalid_dimensions(
                (self.dimension, 1),
                (other.dimension, 1),
            ));
        }
        
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a - b).collect();
        Ok(LatticeVector::new(data))
    }
    
    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: f64) -> Self {
        let data = self.data.iter().map(|x| x * scalar).collect();
        LatticeVector::new(data)
    }
    
    /// Convert to integer vector
    pub fn to_integer_vec(&self) -> Vec<i64> {
        self.data.iter().map(|x| x.round() as i64).collect()
    }
    
    /// Create from integer vector
    pub fn from_integer_vec(data: Vec<i64>) -> Self {
        let float_data: Vec<f64> = data.iter().map(|&x| x as f64).collect();
        LatticeVector::new(float_data)
    }
}

impl std::fmt::Display for LatticeVector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}]", self.data.iter().map(|x| format!("{:.3}", x)).collect::<Vec<_>>().join(", "))
    }
}

/// Gram-Schmidt coefficients and vectors
#[derive(Debug, Clone)]
pub struct GramSchmidt {
    /// Orthogonal vectors (b*_i)
    pub b_star: Vec<LatticeVector>,
    /// Coefficients mu[i][j] = <b_i, b*_j> / ||b*_j||^2
    pub mu: Vec<Vec<f64>>,
    /// Squared norms of orthogonal vectors
    pub norm_squared: Vec<f64>,
}

impl GramSchmidt {
    /// Create new Gram-Schmidt decomposition
    pub fn new(b_star: Vec<LatticeVector>, mu: Vec<Vec<f64>>, norm_squared: Vec<f64>) -> Self {
        GramSchmidt {
            b_star,
            mu,
            norm_squared,
        }
    }
    
    /// Create from lattice basis
    pub fn from_basis(basis: &Matrix) -> Result<Self> {
        let n = basis.rows();
        let mut b_star: Vec<LatticeVector> = Vec::with_capacity(n);
        let mut mu = vec![vec![0.0; n]; n];
        let mut norm_squared = Vec::with_capacity(n);
        
        // Convert basis to vector representation
        let mut b_vectors = Vec::with_capacity(n);
        for i in 0..n {
            let row = basis.get_row(i)?;
            b_vectors.push(LatticeVector::from_integer_vec(row));
        }
        
        for i in 0..n {
            // Compute mu[i][j] for j < i
            let mut b_i = b_vectors[i].clone();
            let dim = b_i.dimension();
            for j in 0..i {
                let mut dot_product = 0.0;
                let b_i_slice = b_vectors[i].as_slice();
                let b_star_slice = b_star[j].as_slice();
                for k in 0..dim {
                    dot_product += b_i_slice[k] * b_star_slice[k];
                }
                mu[i][j] = dot_product / norm_squared[j];
                
                // Subtract projection: b_i = b_i - mu[i][j] * b*_j
                let b_star_slice = b_star[j].as_slice();
                let b_i_mut = b_i.as_mut_slice();
                for k in 0..dim {
                    b_i_mut[k] -= mu[i][j] * b_star_slice[k];
                }
            }
            
            // b*_i is the remaining vector
            let norm = b_i.norm();
            norm_squared.push(norm * norm);
            b_star.push(b_i);
            
            // Set diagonal mu[i][i] = 1
            mu[i][i] = 1.0;
        }
        
        Ok(GramSchmidt {
            b_star,
            mu,
            norm_squared,
        })
    }
    /// Get mu coefficient
    pub fn get_mu(&self, i: usize, j: usize) -> Option<f64> {
        self.mu.get(i).and_then(|row| row.get(j)).copied()
    }

    /// Get mutable reference to mu coefficient
    pub fn get_mu_mut(&mut self, i: usize, j: usize) -> Option<&mut f64> {
        self.mu.get_mut(i).and_then(|row| row.get_mut(j))
    }

    /// Update after basis change
    pub fn update_after_swap(&mut self, i: usize, j: usize) -> Result<()> {
        if i >= self.mu.len() || j >= self.mu.len() {
            return Err(LatticeError::invalid_parameters(
                format!("swap indices {} and {} exceed basis size {}", i, j, self.mu.len()),
            ));
        }

        let mut basis_vectors = self.reconstruct_basis();
        basis_vectors.swap(i, j);
        let integer_basis: Vec<Vec<i64>> = basis_vectors
            .into_iter()
            .map(|row| row.into_iter().map(|val| val.round() as i64).collect())
            .collect();
        let matrix = Matrix::new(integer_basis)?;
        let updated = GramSchmidt::from_basis(&matrix)?;
        *self = updated;
        Ok(())
    }

    fn reconstruct_basis(&self) -> Vec<Vec<f64>> {
        let rows = self.b_star.len();
        if rows == 0 {
            return vec![];
        }
        let cols = self.b_star[0].dimension();
        let mut basis = vec![vec![0.0; cols]; rows];
        for i in 0..rows {
            let orth_slice = self.b_star[i].as_slice();
            for k in 0..cols {
                basis[i][k] = orth_slice[k];
            }
            for j in 0..i {
                let coeff = self.mu[i][j];
                let orth = self.b_star[j].as_slice();
                for k in 0..cols {
                    basis[i][k] += coeff * orth[k];
                }
            }
        }
        basis
    }

    /// Check Lovász condition for LLL
    pub fn check_lovasz_condition(&self, delta: f64, eta: f64) -> Result<bool> {
        let n = self.b_star.len();
        if n < 2 {
            return Ok(false);
        }
        
        for i in 0..n - 1 {
            let lhs = self.norm_squared[i + 1];
            
            if lhs < (delta - eta * eta) * self.norm_squared[i] {
                return Ok(false);
            }
            
            if lhs < (delta - self.mu[i + 1][i].powi(2)) * self.norm_squared[i] {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
}

/// Reduction status
#[derive(Debug, Clone, PartialEq)]
pub enum ReductionStatus {
    /// Reduction is complete
    Complete,
    /// Reduction is in progress
    InProgress {
        current_step: usize,
        total_steps: usize,
        progress: f64,
    },
    /// Reduction failed
    Failed(String),
}

/// Algorithm parameters
#[derive(Debug, Clone)]
pub struct AlgorithmParams {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Tolerance for numerical stability
    pub tolerance: f64,
    /// Enable verbose logging
    pub verbose: bool,
    /// Use GPU acceleration if available
    pub use_gpu: bool,
    /// Number of threads for parallel processing
    pub num_threads: Option<usize>,
}

impl Default for AlgorithmParams {
    fn default() -> Self {
        AlgorithmParams {
            max_iterations: 1000,
            tolerance: 1e-12,
            verbose: false,
            use_gpu: false,
            num_threads: None,
        }
    }
}

/// Benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Algorithm name
    pub algorithm: String,
    /// Input size (dimension)
    pub input_size: usize,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: usize,
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
    /// GPU usage
    pub gpu_used: bool,
    /// Precision used
    pub precision: PrecisionType,
}

impl BenchmarkResult {
    /// Create new benchmark result
    pub fn new(
        algorithm: String,
        input_size: usize,
        execution_time_ms: f64,
        peak_memory_bytes: usize,
        quality_metrics: QualityMetrics,
        gpu_used: bool,
        precision: PrecisionType,
    ) -> Self {
        BenchmarkResult {
            algorithm,
            input_size,
            execution_time_ms,
            peak_memory_bytes,
            quality_metrics,
            gpu_used,
            precision,
        }
    }
}

/// Quality metrics for reduction algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Length of shortest vector
    pub shortest_vector_length: f64,
    /// Hermite constant approximation
    pub hermite_constant: f64,
    /// Conditioning number
    pub conditioning_number: f64,
    /// Orthogonality defect
    pub orthogonality_defect: f64,
    /// Gap to optimal solution (if known)
    pub optimality_gap: Option<f64>,
}

impl QualityMetrics {
    /// Create default quality metrics
    pub fn new() -> Self {
        QualityMetrics {
            shortest_vector_length: 0.0,
            hermite_constant: 0.0,
            conditioning_number: 0.0,
            orthogonality_defect: 0.0,
            optimality_gap: None,
        }
    }
    
    /// Update with actual shortest vector length
    pub fn with_shortest_vector_length(mut self, length: f64) -> Self {
        self.shortest_vector_length = length;
        self
    }
    
    /// Update with Hermite constant
    pub fn with_hermite_constant(mut self, constant: f64) -> Self {
        self.hermite_constant = constant;
        self
    }
    
    /// Update with conditioning number
    pub fn with_conditioning_number(mut self, number: f64) -> Self {
        self.conditioning_number = number;
        self
    }
    
    /// Update with orthogonality defect
    pub fn with_orthogonality_defect(mut self, defect: f64) -> Self {
        self.orthogonality_defect = defect;
        self
    }
    
    /// Update with optimality gap
    pub fn with_optimality_gap(mut self, gap: f64) -> Self {
        self.optimality_gap = Some(gap);
        self
    }
}

/// Precision types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PrecisionType {
    /// Standard double precision (64-bit)
    Double,
    /// Extended precision (80-bit)
    Extended,
    /// Arbitrary precision (configurable bits)
    Arbitrary {
        bits: usize,
    },
}

impl PrecisionType {
    /// Get the number of bits for this precision
    pub fn bits(&self) -> usize {
        match self {
            PrecisionType::Double => 64,
            PrecisionType::Extended => 80,
            PrecisionType::Arbitrary { bits } => *bits,
        }
    }
    
    /// Check if high precision arithmetic is needed
    pub fn requires_high_precision(&self) -> bool {
        match self {
            PrecisionType::Double | PrecisionType::Extended => false,
            PrecisionType::Arbitrary { .. } => true,
        }
    }
}

impl Default for PrecisionType {
    fn default() -> Self {
        PrecisionType::Double
    }
}

/// Statistical utilities for lattice operations
pub mod stats {
    use super::*;
    
    /// Compute mean of a vector
    pub fn mean(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f64>() / data.len() as f64
    }
    
    /// Compute variance of a vector
    pub fn variance(data: &[f64]) -> f64 {
        if data.len() <= 1 {
            return 0.0;
        }
        let mean_val = mean(data);
        let mut sum_sq = 0.0;
        for &x in data {
            sum_sq += (x - mean_val).powi(2);
        }
        sum_sq / (data.len() - 1) as f64
    }
    
    /// Compute standard deviation
    pub fn std_dev(data: &[f64]) -> f64 {
        variance(data).sqrt()
    }
    
    /// Compute correlation between two vectors
    pub fn correlation(x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(LatticeError::invalid_dimensions(
                (x.len(), 1),
                (y.len(), 1),
            ));
        }
        
        if x.is_empty() {
            return Ok(0.0);
        }
        
        let mean_x = mean(x);
        let mean_y = mean(y);
        
        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;
        
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            numerator += dx * dy;
            sum_sq_x += dx * dx;
            sum_sq_y += dy * dy;
        }
        
        if sum_sq_x == 0.0 || sum_sq_y == 0.0 {
            return Ok(0.0);
        }
        
        Ok(numerator / (sum_sq_x * sum_sq_y).sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lattice_vector_creation() {
        let vec = LatticeVector::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(vec.dimension(), 3);
        assert_eq!(vec.get(0), Some(&1.0));
    }
    
    #[test]
    fn test_lattice_vector_operations() {
        let vec1 = LatticeVector::new(vec![1.0, 2.0]);
        let vec2 = LatticeVector::new(vec![3.0, 4.0]);
        
        let sum = vec1.add(&vec2).unwrap();
        assert_eq!(sum.get(0), Some(&4.0));
        
        let diff = vec2.sub(&vec1).unwrap();
        assert_eq!(diff.get(0), Some(&2.0));
        
        let scaled = vec1.scalar_mul(2.0);
        assert_eq!(scaled.get(0), Some(&2.0));
    }
    
    #[test]
    fn test_precision_types() {
        let double = PrecisionType::Double;
        assert_eq!(double.bits(), 64);
        assert!(!double.requires_high_precision());
        
        let arbitrary = PrecisionType::Arbitrary { bits: 256 };
        assert_eq!(arbitrary.bits(), 256);
        assert!(arbitrary.requires_high_precision());
    }
}
