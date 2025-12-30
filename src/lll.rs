//! LLL (Lenstra-Lenstra-Lovász) lattice reduction algorithm

use crate::core::lattice::Lattice;
use crate::core::matrix::Matrix;
use crate::core::types::{GramSchmidt, AlgorithmParams, PrecisionType};
use crate::core::error::{LatticeError, Result};

/// Parameters for LLL reduction
#[derive(Debug, Clone)]
pub struct LLLParams {
    /// Reduction parameter (0.5 < delta < 1), typically 0.99
    pub delta: f64,
    /// Approximation parameter (0.5 < eta < sqrt(delta)), typically 0.51
    pub eta: f64,
    /// Precision type for computations
    pub precision: PrecisionType,
    /// Algorithm parameters
    pub algorithm_params: AlgorithmParams,
}

impl Default for LLLParams {
    fn default() -> Self {
        LLLParams {
            delta: 0.99,
            eta: 0.51,
            precision: PrecisionType::Double,
            algorithm_params: AlgorithmParams::default(),
        }
    }
}

impl LLLParams {
    /// Create new LLL parameters with custom delta and eta
    pub fn new(delta: f64, eta: f64) -> Self {
        LLLParams {
            delta,
            eta,
            precision: PrecisionType::Double,
            algorithm_params: AlgorithmParams::default(),
        }
    }
    
    /// Create with high precision arithmetic
    pub fn with_high_precision(bits: usize) -> Self {
        LLLParams {
            precision: PrecisionType::Arbitrary { bits },
            ..Default::default()
        }
    }
    
    /// Validate parameters
    pub fn validate(&self) -> Result<()> {
        if !(0.5 < self.delta && self.delta < 1.0) {
            return Err(LatticeError::invalid_parameters(
                format!("Delta must be in (0.5, 1.0), got {}", self.delta)
            ));
        }
        
        if !(0.5 < self.eta && self.eta < (self.delta.sqrt())) {
            return Err(LatticeError::invalid_parameters(
                format!("Eta must be in (0.5, sqrt(delta)), got {}", self.eta)
            ));
        }
        
        Ok(())
    }
}

/// Status of LLL reduction
#[derive(Debug, Clone)]
pub struct LLLStatus {
    /// Current size reduction parameter
    pub size_reduction_count: usize,
    /// Number of swaps performed
    pub swap_count: usize,
    /// Current iteration
    pub iteration: usize,
    /// Total iterations performed
    pub total_iterations: usize,
    /// Quality improvement
    pub quality_improvement: f64,
}

/// LLL reducer implementation
pub struct LLLReducer {
    params: LLLParams,
}

impl LLLReducer {
    /// Create new LLL reducer with default parameters
    pub fn new() -> Self {
        Self::with_params(LLLParams::default())
    }
    
    /// Create new LLL reducer with custom parameters
    pub fn with_params(params: LLLParams) -> Self {
        LLLReducer { params }
    }
    
    /// Reduce a lattice using LLL algorithm
    pub fn reduce(&self, lattice: &Lattice) -> Result<Lattice> {
        self.params.validate()?;
        
        if !lattice.is_full_rank() {
            return Err(LatticeError::not_full_rank(lattice.rank()));
        }
        
        let mut basis = lattice.basis().clone();
        let n = lattice.rank();
        
        let mut status = LLLStatus {
            size_reduction_count: 0,
            swap_count: 0,
            iteration: 0,
            total_iterations: 0,
            quality_improvement: 0.0,
        };
        
        // Compute initial Gram-Schmidt
        let mut gs = GramSchmidt::from_basis(&basis)?;
        
        let mut k = 1;
        
        while k < n {
            status.total_iterations += 1;
            
            // Size reduction for vector k against vectors 0..k-1
            self.size_reduce(&mut basis, &mut gs, k)?;
            
            // Check Lovász condition
            if self.check_lovasz_condition(&gs, k, self.params.delta)? {
                k += 1;
            } else {
                // Swap vectors k-1 and k
                basis.swap_rows(k - 1, k)?;
                status.swap_count += 1;
                
                // Recompute Gram-Schmidt after swap
                gs = GramSchmidt::from_basis(&basis)?;
                
                if k > 1 {
                    k -= 1;
                }
            }
            
            status.iteration += 1;
            
            // Limit iterations to prevent infinite loops
            if status.total_iterations > self.params.algorithm_params.max_iterations {
                log::warn!("LLL reduction reached maximum iterations");
                break;
            }
            
            if status.iteration % 100 == 0 && self.params.algorithm_params.verbose {
                log::info!("LLL iteration {}: swaps={}", 
                          status.iteration, status.swap_count);
            }
        }
        
        // Final size reduction
        for k in 1..n {
            self.size_reduce(&mut basis, &mut gs, k)?;
        }
        
        // Create new lattice with reduced basis
        Lattice::new(basis)
    }
    
    /// Proper size reduction implementation
    fn size_reduce(&self, basis: &mut Matrix, gs: &mut GramSchmidt, k: usize) -> Result<bool> {
        let mut changed = false;
        
        for j in (0..k).rev() {
            let mu_kj = gs.get_mu(k, j)
                .ok_or_else(|| LatticeError::numerical_instability("Invalid mu index"))?;
            
            let coefficient = mu_kj.round() as i64;
            
            if coefficient != 0 {
                // Perform size reduction: b_k = b_k - coefficient * b_j
                for col in 0..basis.cols() {
                    let current_val = *basis.get(k, col).ok_or_else(|| {
                        LatticeError::invalid_dimensions(
                            (basis.rows(), basis.cols()),
                            (k + 1, col + 1),
                        )
                    })?;
                    let subtract_val = *basis.get(j, col).ok_or_else(|| {
                        LatticeError::invalid_dimensions(
                            (basis.rows(), basis.cols()),
                            (j + 1, col + 1),
                        )
                    })?;
                    let new_val = current_val - coefficient * subtract_val;
                    basis.set(k, col, new_val)?;
                }
                
                // Update mu coefficients
                self.update_mu_coefficients(gs, k, j, coefficient as f64)?;
                changed = true;
            }
        }
        
        Ok(changed)
    }

    /// Update mu coefficients after size reduction
    fn update_mu_coefficients(&self, gs: &mut GramSchmidt, k: usize, j: usize, coefficient: f64) -> Result<()> {
        let row_j_snapshot = if let Some(row_j) = gs.mu.get(j) {
            row_j.clone()
        } else {
            return Ok(());
        };

        if let Some(row_k) = gs.mu.get_mut(k) {
            if let Some(mu_kj) = row_k.get_mut(j) {
                *mu_kj -= coefficient;
            }
            for i in 0..j {
                if let Some(mu_ki) = row_k.get_mut(i) {
                    if let Some(mu_ji) = row_j_snapshot.get(i) {
                        *mu_ki -= coefficient * *mu_ji;
                    }
                }
            }
        }

        Ok(())
    }

    /// Fix GramSchmidt access methods
    /// Check Lovász condition for LLL
    fn check_lovasz_condition(&self, gs: &GramSchmidt, k: usize, delta: f64) -> Result<bool> {
        if k == 0 || k >= gs.norm_squared.len() {
            return Ok(true);
        }
        
        let norm_k_sq = gs.norm_squared[k];
        let norm_km1_sq = gs.norm_squared[k - 1];
        
        let mu_k_km1 = gs.get_mu(k, k - 1)
            .ok_or_else(|| LatticeError::numerical_instability("Invalid mu index for Lovász condition"))?;
        
        // Lovász condition: ||b*_k||^2 >= (delta - mu[k][k-1]^2) * ||b*_{k-1}||^2
        let rhs = (delta - mu_k_km1.powi(2)) * norm_km1_sq;
        
        Ok(norm_k_sq >= rhs - 1e-10) // Small epsilon for floating point
    }
    
    /// Get reduction status
    pub fn reduction_status(&self, status: &LLLStatus) -> String {
        format!("Swaps: {}, Iterations: {}, Total: {}", 
                status.swap_count, status.iteration, status.total_iterations)
    }
}

/// Extended LLL with different variants
pub struct ExtendedLLLReducer {
    params: LLLParams,
    variant: LLLVariant,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LLLVariant {
    /// Standard LLL
    Standard,
    /// LLL with deep insertions
    Deep,
    /// LLL with block reduction
    Block,
}

impl Default for ExtendedLLLReducer {
    fn default() -> Self {
        ExtendedLLLReducer {
            params: LLLParams::default(),
            variant: LLLVariant::Standard,
        }
    }
}

impl ExtendedLLLReducer {
    /// Create new extended LLL reducer
    pub fn new(params: LLLParams, variant: LLLVariant) -> Self {
        ExtendedLLLReducer { params, variant }
    }
    
    /// Reduce with specific variant
    pub fn reduce_with_variant(&self, lattice: &Lattice) -> Result<Lattice> {
        match self.variant {
            LLLVariant::Standard => {
                let reducer = LLLReducer::with_params(self.params.clone());
                reducer.reduce(lattice)
            }
            LLLVariant::Deep => self.reduce_with_deep_insertions(lattice),
            LLLVariant::Block => self.reduce_with_block_reduction(lattice),
        }
    }
    
    /// LLL with deep insertions
    fn reduce_with_deep_insertions(&self, lattice: &Lattice) -> Result<Lattice> {
        self.params.validate()?;
        
        let mut basis = lattice.basis().clone();
        let n = lattice.rank();
        
        // Compute initial Gram-Schmidt
        let mut gs = GramSchmidt::from_basis(&basis)?;
        
        let mut k = 1;
        let mut total_iterations = 0;
        
        while k < n {
            total_iterations += 1;
            
            // Size reduction
            let reducer = LLLReducer::with_params(self.params.clone());
            reducer.size_reduce(&mut basis, &mut gs, k)?;
            
            // Deep insertion check
            let mut swapped = false;
            for l in (0..k-1).rev() {
                let mu_kl = gs.get_mu(k, l)
                    .ok_or_else(|| LatticeError::numerical_instability("Invalid mu index"))?;
                
                if mu_kl.abs() > self.params.eta {
                    // Perform deep insertion: move vector k to position l+1
                    self.deep_insert(&mut basis, k, l + 1)?;
                    swapped = true;
                    
                    // Recompute Gram-Schmidt
                    gs = GramSchmidt::from_basis(&basis)?;
                    
                    k = l + 1;
                    break;
                }
            }
            
            if !swapped {
                // Check standard Lovász condition
                if reducer.check_lovasz_condition(&gs, k, self.params.delta)? {
                    k += 1;
                } else {
                    // Standard swap
                    basis.swap_rows(k - 1, k)?;
                    
                    // Recompute Gram-Schmidt
                    gs = GramSchmidt::from_basis(&basis)?;
                    
                    if k > 1 {
                        k -= 1;
                    }
                }
            }
            
            if total_iterations > self.params.algorithm_params.max_iterations {
                log::warn!("Deep LLL reduction reached maximum iterations");
                break;
            }
        }
        
        Lattice::new(basis)
    }
    
    /// Perform deep insertion
    fn deep_insert(&self, basis: &mut Matrix, from: usize, to: usize) -> Result<()> {
        if from <= to || from >= basis.rows() || to >= basis.rows() {
            return Err(LatticeError::invalid_parameters("Invalid deep insertion indices"));
        }
        
        // Remove vector at 'from' and insert at 'to'
        let row = basis.remove_row(from)?;
        basis.insert_row(to, row)?;
        
        Ok(())
    }
    
    /// LLL with block reduction (simplified)
    fn reduce_with_block_reduction(&self, lattice: &Lattice) -> Result<Lattice> {
        let n = lattice.rank();
        
        if n <= 2 {
            // For small dimensions, use standard LLL
            let reducer = LLLReducer::with_params(self.params.clone());
            return reducer.reduce(lattice);
        }
        
        let block_size = (n / 2).max(2);
        let mut current_basis = lattice.basis().clone();
        
        // Iterative block reduction
        for _ in 0..self.params.algorithm_params.max_iterations {
            let mut changed = false;
            
            for start in (0..n).step_by(block_size) {
                let end = (start + block_size).min(n);
                if end - start < 2 {
                    continue;
                }
                
                // Extract block
                let block_data: Vec<Vec<i64>> = (start..end)
                    .map(|i| current_basis.get_row(i).unwrap().clone())
                    .collect();
                
                let block_lattice = Lattice::from_matrix(block_data)?;
                
                // Reduce block
                let reducer = LLLReducer::with_params(self.params.clone());
                let reduced_block = reducer.reduce(&block_lattice)?;
                
                // Update basis if block was reduced
                if reduced_block.basis() != block_lattice.basis() {
                    for (idx, i) in (start..end).enumerate() {
                        let reduced_row = reduced_block.basis().get_row(idx)?;
                        for j in 0..current_basis.cols() {
                            current_basis.set(i, j, reduced_row[j])?;
                        }
                    }
                    changed = true;
                }
            }
            
            if !changed {
                break;
            }
        }
        
        Lattice::new(current_basis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lll_params_validation() {
        let valid_params = LLLParams::new(0.99, 0.51);
        assert!(valid_params.validate().is_ok());
        
        let invalid_delta = LLLParams::new(0.4, 0.51);
        assert!(invalid_delta.validate().is_err());
        
        let invalid_eta = LLLParams::new(0.6, 0.9);
        assert!(invalid_eta.validate().is_err());
    }
    
    #[test]
    fn test_lll_reduction_2d() {
        // Create a non-reduced basis
        let data = vec![vec![2, 1], vec![1, 1]];
        let lattice = Lattice::from_matrix(data).unwrap();
        
        let reducer = LLLReducer::new();
        let reduced = reducer.reduce(&lattice).unwrap();
        
        // Check that the reduced basis has shorter vectors
        let orig_norms: Vec<f64> = (0..lattice.rank())
            .map(|i| {
                let row = lattice.basis().get_row(i).unwrap();
                row.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt()
            })
            .collect();
            
        let reduced_norms: Vec<f64> = (0..reduced.rank())
            .map(|i| {
                let row = reduced.basis().get_row(i).unwrap();
                row.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt()
            })
            .collect();
        
        assert!(reduced_norms[0] <= orig_norms[0]);
        assert_eq!(reduced.dimension(), lattice.dimension());
    }
    
    #[test]
    fn test_lll_reduction_3d() {
        let data = vec![
            vec![1, 1, 1],
            vec![1, 0, 0],
            vec![0, 1, 0],
        ];
        let lattice = Lattice::from_matrix(data).unwrap();
        
        let reducer = LLLReducer::new();
        let reduced = reducer.reduce(&lattice).unwrap();
        
        // Should preserve determinant
        let orig_det = lattice.determinant().unwrap();
        let reduced_det = reduced.determinant().unwrap();
        assert!((orig_det - reduced_det).abs() < 1e-10);
    }
    
    #[test]
    fn test_extended_lll_variants() {
        let data = vec![vec![2, 1], vec![1, 1]];
        let lattice = Lattice::from_matrix(data).unwrap();
        
        let variants = vec![
            LLLVariant::Standard,
            LLLVariant::Deep,
            LLLVariant::Block,
        ];
        
        for variant in variants {
            let reducer = ExtendedLLLReducer::new(LLLParams::default(), variant);
            let reduced = reducer.reduce_with_variant(&lattice).unwrap();
            assert_eq!(reduced.dimension(), lattice.dimension());
        }
    }
}
