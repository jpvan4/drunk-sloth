//! BKZ (Block Korkine-Zolotarev) lattice reduction algorithm

use crate::core::lattice::Lattice;
use crate::core::matrix::Matrix;
use crate::core::types::{GramSchmidt, LatticeVector, AlgorithmParams, ReductionStatus};
use crate::core::error::{LatticeError, Result};
use crate::lll::{LLLReducer, LLLParams};
use crate::svp::{SVPSolver, SVPSolverParams};

/// Parameters for BKZ reduction
#[derive(Debug, Clone)]
pub struct BKZParams {
    /// Block size (typically between 10 and 100)
    pub beta: usize,
    /// Maximum number of BKZ rounds
    pub max_rounds: usize,
    /// LLL parameters for preprocessing
    pub lll_params: LLLParams,
    /// SVP solver parameters
    pub svp_params: SVPSolverParams,
    /// Enable progressive BKZ (increase block size gradually)
    pub progressive: bool,
    /// Enable pruning in SVP enumeration
    pub enable_pruning: bool,
    /// Algorithm parameters
    pub algorithm_params: AlgorithmParams,
}

impl Default for BKZParams {
    fn default() -> Self {
        BKZParams {
            beta: 20,
            max_rounds: 10,
            lll_params: LLLParams::default(),
            svp_params: SVPSolverParams::default(),
            progressive: true,
            enable_pruning: true,
            algorithm_params: AlgorithmParams::default(),
        }
    }
}

impl BKZParams {
    /// Create new BKZ parameters with given block size
    pub fn new(beta: usize) -> Self {
        BKZParams {
            beta,
            ..Default::default()
        }
    }
    
    /// Validate BKZ parameters
    pub fn validate(&self) -> Result<()> {
        if self.beta < 2 {
            return Err(LatticeError::invalid_parameters(
                format!("Block size must be at least 2, got {}", self.beta)
            ));
        }
        
        if self.beta > 200 {
            log::warn!("Large block size {} may cause performance issues", self.beta);
        }
        
        self.lll_params.validate()?;
        self.svp_params.validate()?;
        
        Ok(())
    }
    
    /// Get effective block size (for progressive BKZ)
    pub fn effective_beta(&self, round: usize) -> usize {
        if self.progressive {
            (self.beta + round).min(self.beta * 2)
        } else {
            self.beta
        }
    }
}

/// Status of BKZ reduction
#[derive(Debug, Clone)]
pub struct BKZStatus {
    /// Current round
    pub round: usize,
    /// Total rounds performed
    pub total_rounds: usize,
    /// Current block being processed
    pub current_block: usize,
    /// Number of blocks processed
    pub blocks_processed: usize,
    /// SVP solutions found
    pub svp_solutions: usize,
    /// Quality improvement per round
    pub quality_improvement: Vec<f64>,
    /// Status of current operation
    pub status: ReductionStatus,
}

/// BKZ reducer implementation
pub struct BKZReducer {
    params: BKZParams,
    lll_reducer: LLLReducer,
    svp_solver: SVPSolver,
}

impl BKZReducer {
    /// Create new BKZ reducer with default parameters
    pub fn new() -> Self {
        Self::with_params(BKZParams::default())
    }
    
    /// Create new BKZ reducer with custom parameters
    pub fn with_params(params: BKZParams) -> Self {
        let lll_reducer = LLLReducer::with_params(params.lll_params.clone());
        let svp_solver = SVPSolver::with_params(params.svp_params.clone());
        
        BKZReducer {
            params,
            lll_reducer,
            svp_solver,
        }
    }
    
    /// Reduce a lattice using BKZ algorithm
    pub fn reduce(&self, lattice: &Lattice) -> Result<Lattice> {
        self.params.validate()?;
        
        if !lattice.is_full_rank() {
            return Err(LatticeError::not_full_rank(lattice.rank()));
        }
        
        let mut basis = lattice.basis().clone();
        let n = lattice.rank();
        
        let mut status = BKZStatus {
            round: 0,
            total_rounds: 0,
            current_block: 0,
            blocks_processed: 0,
            svp_solutions: 0,
            quality_improvement: Vec::new(),
            status: ReductionStatus::InProgress {
                current_step: 0,
                total_steps: n * self.params.max_rounds,
                progress: 0.0,
            },
        };
        
        // First, perform LLL reduction
        let lll_lattice = Lattice::new(basis.clone())?;
        let reduced_lattice = self.lll_reducer.reduce(&lll_lattice)?;
        basis = reduced_lattice.basis().clone();
        
        if self.params.algorithm_params.verbose {
            log::info!("Initial LLL reduction completed");
        }
        
        // Perform BKZ rounds
        for round in 0..self.params.max_rounds {
            let effective_beta = self.params.effective_beta(round);
            
            if self.params.algorithm_params.verbose {
                log::info!("BKZ round {}/{} with effective block size {}", 
                          round + 1, self.params.max_rounds, effective_beta);
            }
            
            let mut round_improved = false;
            
            // Skip rounds where the block size exceeds dimension
            if effective_beta > n {
                continue;
            }

            // Process each block
            for k in 0..=(n - effective_beta) {
                status.current_block = k;
                
                // Extract block
                let block_matrix = self.extract_block(&basis, k, effective_beta)?;
                let block_basis = Lattice::new(block_matrix)?;
                
                // Find shortest vector in block
                let svp_result = self.svp_solver.solve(&block_basis)?;
                
                if svp_result.found {
                    // Replace vector k with SVP solution
                    if self.update_with_svp_solution(&mut basis, k, &svp_result.solution)? {
                        status.svp_solutions += 1;
                        round_improved = true;
                    }
                }
                
                status.blocks_processed += 1;
                status.status = ReductionStatus::InProgress {
                    current_step: status.blocks_processed,
                    total_steps: n * self.params.max_rounds,
                    progress: status.blocks_processed as f64 / (n * self.params.max_rounds) as f64,
                };
                
                if self.params.algorithm_params.verbose && k % 10 == 0 {
                    log::debug!("Processed block {}/{}", k, n - effective_beta + 1);
                }
            }
            
            // Perform LLL reduction after each round
            let temp_lattice = Lattice::new(basis.clone())?;
            let new_reduced = self.lll_reducer.reduce(&temp_lattice)?;
            basis = new_reduced.basis().clone();
            
            // Check quality improvement
            if self.params.algorithm_params.verbose {
                let old_quality = self.compute_quality_metric(&lattice.basis())?;
                let new_quality = self.compute_quality_metric(&basis)?;
                let improvement = (old_quality - new_quality).abs();
                status.quality_improvement.push(improvement);
                
                log::info!("Round {} quality: {:.6}, improvement: {:.6}", 
                          round + 1, new_quality, improvement);
            }
            
            // Check if reduction is stable
            if !round_improved {
                if self.params.algorithm_params.verbose {
                    log::info!("BKZ reduction converged after {} rounds", round + 1);
                }
                break;
            }
            
            status.round += 1;
            status.total_rounds += 1;
        }
        
        status.status = ReductionStatus::Complete;
        
        // Final LLL reduction
        let final_lattice = Lattice::new(basis)?;
        let result = self.lll_reducer.reduce(&final_lattice)?;
        
        if self.params.algorithm_params.verbose {
            log::info!("BKZ reduction completed: {} rounds, {} SVP solutions found", 
                      status.total_rounds, status.svp_solutions);
        }
        
        Ok(result)
    }
    
    /// Extract a block from the basis
    fn extract_block(&self, basis: &Matrix, start: usize, size: usize) -> Result<Matrix> {
        let n = basis.rows();
        if start + size > n {
            return Err(LatticeError::invalid_parameters(
                format!("Block extends beyond basis: start={}, size={}, basis_size={}", 
                        start, size, n)
            ));
        }
        
        let mut block_data = Vec::with_capacity(size);
        for i in start..(start + size) {
            let row = basis.get_row(i)?;
            block_data.push(row);
        }
        
        Matrix::new(block_data)
    }
    
    /// Update basis with SVP solution
    fn update_with_svp_solution(&self, basis: &mut Matrix, position: usize, solution: &LatticeVector) -> Result<bool> {
        if solution.dimension() < basis.cols() {
            return Err(LatticeError::invalid_dimensions(
                (basis.cols(), 1),
                (solution.dimension(), 1),
            ));
        }
        let coordinates = solution.as_slice();
        
        // Convert SVP solution to integer coefficients
        let coefficients: Vec<i64> = coordinates.iter().map(|&val| val.round() as i64).collect();
        
        // Check if solution is non-trivial
        let norm = coordinates.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-10 {
            return Ok(false);
        }
        
        // Update the basis vector at position
        for j in 0..basis.cols() {
            let new_val = coefficients[j];
            basis.set(position, j, new_val)?;
        }
        
        Ok(true)
    }
    
    /// Compute quality metric for comparison
    fn compute_quality_metric(&self, basis: &Matrix) -> Result<f64> {
        let gs = GramSchmidt::from_basis(basis)?;
        
        // Use product of Gram-Schmidt norms as quality metric
        let mut product = 1.0f64;
        for &norm_sq in &gs.norm_squared {
            product *= norm_sq.sqrt();
        }
        
        Ok(product)
    }
    
    /// Get reduction status
    pub fn reduction_status(&self, status: &BKZStatus) -> String {
        format!("Round: {}/{}, Blocks: {}, SVP solutions: {}, Progress: {:.1}%", 
                status.round + 1, 
                status.total_rounds,
                status.blocks_processed,
                status.svp_solutions,
                match &status.status {
                    ReductionStatus::InProgress { progress, .. } => progress * 100.0,
                    ReductionStatus::Complete => 100.0,
                    ReductionStatus::Failed(_) => 0.0,
                })
    }
}

/// Progressive BKZ with adaptive block size
pub struct ProgressiveBKZReducer {
    params: BKZParams,
    lll_reducer: LLLReducer,
    svp_solver: SVPSolver,
    schedule: Vec<usize>,
}

impl ProgressiveBKZReducer {
    /// Create new progressive BKZ reducer
    pub fn new(params: BKZParams) -> Self {
        let lll_reducer = LLLReducer::with_params(params.lll_params.clone());
        let svp_solver = SVPSolver::with_params(params.svp_params.clone());
        let schedule = Self::generate_schedule(params.beta);
        
        ProgressiveBKZReducer {
            params,
            lll_reducer,
            svp_solver,
            schedule,
        }
    }
    
    /// Generate progressive schedule
    fn generate_schedule(target_beta: usize) -> Vec<usize> {
        let mut schedule = Vec::new();
        
        // Start with small blocks and increase gradually
        let mut beta = 2;
        while beta < target_beta {
            schedule.push(beta);
            beta = if beta < 10 {
                beta + 1
            } else if beta < 30 {
                beta + 2
            } else {
                beta + 5
            };
        }
        
        // Add target beta multiple times for convergence
        for _ in 0..5 {
            schedule.push(target_beta);
        }
        
        schedule
    }
    
    /// Reduce with progressive block sizes
    pub fn reduce(&self, lattice: &Lattice) -> Result<Lattice> {
        let mut basis = lattice.basis().clone();

        if self.params.algorithm_params.verbose {
            log::debug!(
                "Progressive schedule {:?}, SVP helper: {}",
                self.schedule,
                self.svp_solver.algorithm_info()
            );
        }
        
        // Initial LLL reduction
        let initial_lattice = Lattice::new(basis.clone())?;
        let reduced = self.lll_reducer.reduce(&initial_lattice)?;
        basis = reduced.basis().clone();
        
        for (round, &beta) in self.schedule.iter().enumerate() {
            if self.params.algorithm_params.verbose {
                log::info!("Progressive BKZ round {}: block size {}", round + 1, beta);
            }
            
            let bkz_params = BKZParams {
                beta,
                max_rounds: 1, // Only one round per step
                ..self.params.clone()
            };
            
            let reducer = BKZReducer::with_params(bkz_params);
            let temp_lattice = Lattice::new(basis.clone())?;
            let new_reduced = reducer.reduce(&temp_lattice)?;
            basis = new_reduced.basis().clone();
        }
        
        // Final LLL reduction
        let final_lattice = Lattice::new(basis)?;
        self.lll_reducer.reduce(&final_lattice)
    }
}

/// BKZ with preprocessing optimizations
pub struct OptimizedBKZReducer {
    params: BKZParams,
    lll_reducer: LLLReducer,
    svp_solver: SVPSolver,
}

impl OptimizedBKZReducer {
    /// Create new optimized BKZ reducer
    pub fn new(params: BKZParams) -> Self {
        let lll_reducer = LLLReducer::with_params(params.lll_params.clone());
        let svp_solver = SVPSolver::with_params(params.svp_params.clone());
        
        OptimizedBKZReducer {
            params,
            lll_reducer,
            svp_solver,
        }
    }
    
    /// Reduce with optimizations
    pub fn reduce(&self, lattice: &Lattice) -> Result<Lattice> {
        let mut basis = lattice.basis().clone();
        
        // Preprocessing: Sort vectors by length
        self.sort_vectors_by_length(&mut basis)?;
        let preprocessed = self.lll_reducer.reduce(&Lattice::new(basis.clone())?)?;
        basis = preprocessed.basis().clone();
        log::debug!(
            "Optimized BKZ preprocessing complete (SVP helper: {})",
            self.svp_solver.algorithm_info()
        );
        
        // Multiple rounds of BKZ with increasing block size
        let mut beta = self.params.beta.min(20); // Start smaller
        loop {
            let params = BKZParams {
                beta: 48,
                max_rounds: 1,
                ..self.params.clone()
            };
            
            let reducer = ProgressiveBKZReducer::new(params);
            let _temp_lattice = Lattice::new(basis.clone())?;
            let reduced = reducer.reduce(&lattice)?;
            basis = reduced.basis().clone();
            
            if beta >= self.params.beta {
                break;
            }
            beta = (beta + 2).min(self.params.beta);
        }
        
        Lattice::new(basis)
    }
    
    /// Sort vectors by their length
    fn sort_vectors_by_length(&self, basis: &mut Matrix) -> Result<()> {
        let n = basis.rows();
        let mut indices: Vec<usize> = (0..n).collect();
        
        // Sort indices by vector length
        indices.sort_by(|&i, &j| {
            let len_i = self.vector_length(basis, i).unwrap_or(f64::INFINITY);
            let len_j = self.vector_length(basis, j).unwrap_or(f64::INFINITY);
            len_i.partial_cmp(&len_j).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Reorder basis vectors
        let mut new_data = Vec::with_capacity(n);
        for &idx in &indices {
            new_data.push(basis.get_row(idx).unwrap());
        }
        
        *basis = Matrix::new(new_data)?;
        Ok(())
    }
    
    /// Compute length of a basis vector
    fn vector_length(&self, basis: &Matrix, index: usize) -> Result<f64> {
        let row = basis.get_row(index)?;
        let norm_sq: f64 = row.iter().map(|&x| (x as f64).powi(2)).sum();
        Ok(norm_sq.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bkz_params_validation() {
        let valid_params = BKZParams::new(20);
        assert!(valid_params.validate().is_ok());
        
        let invalid_params = BKZParams::new(1);
        assert!(invalid_params.validate().is_err());
    }
    
    #[test]
    fn test_bkz_reduction_2d() {
        let data = vec![vec![1, 1], vec![1, 0]];
        let lattice = Lattice::from_matrix(data).unwrap();
        
        let reducer = BKZReducer::new();
        let reduced = reducer.reduce(&lattice).unwrap();
        
        assert_eq!(reduced.dimension(), lattice.dimension());
    }
    
    #[test]
    fn test_bkz_reduction_3d() {
        let data = vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
            vec![7, 8, 10],
        ];
        let lattice = Lattice::from_matrix(data).unwrap();
        
        let params = BKZParams::new(3);
        let reducer = BKZReducer::with_params(params);
        let reduced = reducer.reduce(&lattice).unwrap();
        
        assert_eq!(reduced.dimension(), lattice.dimension());
    }
    
    #[test]
    fn test_progressive_bkz_schedule() {
        let schedule = ProgressiveBKZReducer::generate_schedule(10);
        
        assert!(!schedule.is_empty());
        assert_eq!(schedule.last(), Some(&10));
        assert!(schedule.iter().all(|&b| b >= 2));
    }
    
    #[test]
    fn test_optimized_bkz() {
        let data = vec![vec![1, 1], vec![1, 0]];
        let lattice = Lattice::from_matrix(data).unwrap();
        
        let reducer = OptimizedBKZReducer::new(BKZParams::default());
        let reduced = reducer.reduce(&lattice).unwrap();
        
        assert_eq!(reduced.dimension(), lattice.dimension());
    }
}
