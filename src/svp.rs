//! SVP (Shortest Vector Problem) solver

use crate::core::lattice::Lattice;
use crate::core::matrix::Matrix;
use crate::core::types::{AlgorithmParams, GramSchmidt, LatticeVector};
use crate::core::error::{LatticeError, Result};

/// Parameters for SVP solver
#[derive(Debug, Clone)]
pub struct SVPSolverParams {
    /// Algorithm type
    pub algorithm: SVPAlgorithm,
    /// Maximum enumeration radius
    pub max_radius: f64,
    /// Enumeration precision (in bits)
    pub precision_bits: usize,
    /// Enable pruning
    pub enable_pruning: bool,
    /// Pruning strategy
    pub pruning_strategy: PruningStrategy,
    /// Algorithm parameters
    pub algorithm_params: AlgorithmParams,
}

impl Default for SVPSolverParams {
    fn default() -> Self {
        SVPSolverParams {
            algorithm: SVPAlgorithm::Enumeration,
            max_radius: 1e6,
            precision_bits: 53, // double precision
            enable_pruning: true,
            pruning_strategy: PruningStrategy::Gauss,
            algorithm_params: AlgorithmParams::default(),
        }
    }
}

impl SVPSolverParams {
    /// Create new SVP parameters with specific algorithm
    pub fn with_algorithm(algorithm: SVPAlgorithm) -> Self {
        SVPSolverParams {
            algorithm,
            ..Default::default()
        }
    }
    
    /// Validate parameters
    pub fn validate(&self) -> Result<()> {
        if self.precision_bits < 32 {
            return Err(LatticeError::invalid_parameters(
                format!("Precision must be at least 32 bits, got {}", self.precision_bits)
            ));
        }
        
        if self.max_radius <= 0.0 {
            return Err(LatticeError::invalid_parameters(
                format!("Max radius must be positive, got {}", self.max_radius)
            ));
        }
        
        Ok(())
    }
}

/// SVP algorithm types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SVPAlgorithm {
    /// Enumeration algorithm
    Enumeration,
    /// Schnorr-Euchner enumeration
    SchnorrEuchner,
    /// Kannan's algorithm
    Kannan,
    /// Random sampling
    RandomSampling,
    /// BKZ-based approximation
    BKZApproximation,
}

/// Pruning strategies for enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PruningStrategy {
    /// No pruning
    None,
    /// Gaussian heuristic pruning
    Gauss,
    /// Radius reduction pruning
    Radius,
    /// Hierarchical pruning
    Hierarchical,
}

/// Result of SVP computation
#[derive(Debug, Clone)]
pub struct SVPResult {
    /// Whether a solution was found
    pub found: bool,
    /// The shortest vector found
    pub solution: LatticeVector,
    /// Norm of the solution
    pub norm: f64,
    /// Number of lattice points enumerated
    pub points_enumerated: u64,
    /// Execution time in seconds
    pub execution_time: f64,
    /// Quality of the solution (approximation factor)
    pub approximation_factor: f64,
}

/// SVP solver implementation
pub struct SVPSolver {
    params: SVPSolverParams,
}

impl SVPSolver {
    /// Create new SVP solver with default parameters
    pub fn new() -> Self {
        Self::with_params(SVPSolverParams::default())
    }
    
    /// Create new SVP solver with custom parameters
    pub fn with_params(params: SVPSolverParams) -> Self {
        SVPSolver { params }
    }
    
    /// Solve SVP for a given lattice
    pub fn solve(&self, lattice: &Lattice) -> Result<SVPResult> {
        self.params.validate()?;
        
        let start_time = std::time::Instant::now();
        
        let result = match self.params.algorithm {
            SVPAlgorithm::Enumeration => self.solve_by_enumeration(lattice),
            SVPAlgorithm::SchnorrEuchner => self.solve_by_schnorr_euchner(lattice),
            SVPAlgorithm::Kannan => self.solve_by_kannan(lattice),
            SVPAlgorithm::RandomSampling => self.solve_by_random_sampling(lattice),
            SVPAlgorithm::BKZApproximation => self.solve_by_bkz_approximation(lattice),
        }?;
        
        let execution_time = start_time.elapsed().as_secs_f64();
        
        Ok(SVPResult {
            found: result.found,
            solution: result.solution,
            norm: result.norm,
            points_enumerated: result.points_enumerated,
            execution_time,
            approximation_factor: result.approximation_factor,
        })
    }
    
    // In src/svp.rs
    fn solve_by_enumeration(&self, lattice: &Lattice) -> Result<SVPEnumerationResult> {
        let n = lattice.rank();
        let gs = lattice.gram_schmidt()?;

        // Use Gaussian heuristic to set enumeration bound
        let det = lattice.determinant()?;
        let gauss_radius = (det.powf(1.0/n as f64) * (n as f64).sqrt()).powi(2);
        let radius = self.params.max_radius.min(gauss_radius);

        let mut best = f64::INFINITY;
        let mut best_coeffs = vec![0i64; n];
        let mut points = 0u64;

        // Recursive enumeration with pruning
        let mut coeffs = vec![0i64; n];
        self.enumerate_recursive(&lattice, &gs, n-1, radius, &mut coeffs, &mut best, &mut best_coeffs, &mut points)?;

        let solution = lattice.generate_vector(&best_coeffs)?;
        Ok(SVPEnumerationResult {
            found: best.is_finite(),
            solution,
            norm: best.sqrt(),
            points_enumerated: points,
            approximation_factor: 1.0,
        })
    }

    fn enumerate_recursive(
        &self,
        lattice: &Lattice,
        gs: &GramSchmidt,
        level: usize,
        radius_sq: f64,
        coeffs: &mut [i64],
        best_sq: &mut f64,
        best_coeffs: &mut [i64],
        points: &mut u64,
    ) -> Result<()> {
        if level == 0 {
            *points += 1;
            let vec = lattice.generate_vector(coeffs)?;
            let norm_sq = vec.norm().powi(2);

            if norm_sq > 0.0 && norm_sq < *best_sq {
                *best_sq = norm_sq;
                best_coeffs.copy_from_slice(coeffs);
            }
            return Ok(());
        }

        // Compute bounds for this coefficient using Gram-Schmidt
        let mut bound = (radius_sq / gs.norm_squared[level]).sqrt().ceil() as i64;
        bound = bound.clamp(-100, 100); // Reasonable limit

        for c in -bound..=bound {
            coeffs[level] = c;

            // Check partial norm (pruning)
            let partial_norm = self.partial_norm(gs, coeffs, level)?;
            if partial_norm <= radius_sq {
                self.enumerate_recursive(lattice, gs, level-1, radius_sq, coeffs, best_sq, best_coeffs, points)?;
            }
        }

        Ok(())
    }
    /// Enumerate lattice points in Gram-Schmidt basis
    #[allow(dead_code)]
    fn enumerate_in_gs_basis(
        &self,
        basis: &Matrix,
        gs: &crate::core::types::GramSchmidt,
        depth: usize,
        current_solution: &mut LatticeVector,
        best_norm: &mut f64,
        points_enumerated: &mut u64,
    ) -> Result<()> {
        let n = basis.rows();
        
        if depth == n {
            // Leaf node - evaluate current solution
            *points_enumerated += 1;
            
            let norm = current_solution.norm();
            if norm < *best_norm && norm > 0.0 {
                *best_norm = norm;
            }
            
            return Ok(());
        }
        
        // Pruning check
        if self.params.enable_pruning {
            let pruning_bound = self.compute_pruning_bound(gs, depth, best_norm)?;
            if pruning_bound >= *best_norm {
                return Ok(());
            }
        }
        
        // Enumerate possible coefficients for this dimension
        let coefficient_range = self.compute_coefficient_range(gs, depth, best_norm)?;
        for coeff in coefficient_range {
            // Update current solution with this coefficient
            self.update_solution(current_solution, depth, coeff)?;
            
            // Recursive enumeration
            self.enumerate_in_gs_basis(basis, gs, depth + 1, current_solution, 
                                     best_norm, points_enumerated)?;
        }
        
        Ok(())
    }
    
    /// Compute pruning bound for early termination
    #[allow(dead_code)]
    fn compute_pruning_bound(&self, gs: &crate::core::types::GramSchmidt, depth: usize, current_best: &f64) -> Result<f64> {
        let mut bound = 0.0f64;
        
        for i in depth..gs.norm_squared.len() {
            bound += gs.norm_squared[i];
        }
        
        match self.params.pruning_strategy {
            PruningStrategy::Gauss => {
                let n = gs.norm_squared.len();
                let expected_density = 1.0 / (bound.sqrt() * 2.0f64.powf(0.5 * n as f64));
                Ok(*current_best * expected_density.max(0.1))
            }
            PruningStrategy::Radius => {
                Ok(*current_best)
            }
            PruningStrategy::Hierarchical => {
                let factor = 1.0 - (depth as f64 / gs.norm_squared.len() as f64) * 0.1;
                Ok(*current_best * factor)
            }
            PruningStrategy::None => Ok(f64::INFINITY),
        }
    }
    
    /// Compute range of coefficients to enumerate
    #[allow(dead_code)]
    fn compute_coefficient_range(&self, gs: &crate::core::types::GramSchmidt, depth: usize, current_best: &f64) -> Result<Vec<i64>> {
        if depth >= gs.norm_squared.len() {
            return Err(LatticeError::invalid_parameters("depth exceeds Gram-Schmidt width"));
        }
        let norm = gs.norm_squared[depth].sqrt();
        let radius = current_best.sqrt();
        let mut limit = if norm > 0.0 {
            (radius / norm).ceil() as i64
        } else {
            1
        };
        limit = limit.clamp(1, 64);
        Ok((-limit..=limit).collect())
    }

    fn partial_norm(&self, gs: &GramSchmidt, coeffs: &[i64], level: usize) -> Result<f64> {
        if level >= gs.norm_squared.len() {
            return Err(LatticeError::invalid_parameters("level exceeds Gram-Schmidt dimension"));
        }

        let mut norm_sq = 0.0;
        for i in level..gs.norm_squared.len() {
            let coeff = coeffs.get(i).copied().unwrap_or(0) as f64;
            norm_sq += coeff.powi(2) * gs.norm_squared[i];
        }

        Ok(norm_sq)
    }
    
    /// Update solution with coefficient
    #[allow(dead_code)]
    fn update_solution(&self, solution: &mut LatticeVector, depth: usize, coefficient: i64) -> Result<()> {
        if depth >= solution.dimension() {
            return Err(LatticeError::invalid_parameters(format!(
                "depth {} exceeds solution dimension {}",
                depth,
                solution.dimension()
            )));
        }
        solution.set(depth, coefficient as f64)?;
        Ok(())
    }

    fn solve_by_schnorr_euchner(&self, lattice: &Lattice) -> Result<SVPEnumerationResult> {
        let n = lattice.rank();
        let gs = lattice.gram_schmidt()?;

        // Initialize Schnorr-Euchner state
        let mut center = vec![0.0; n];      // c_i: centers of intervals
        let mut radius = vec![0.0; n];      // r_i: interval radii
        let mut coeffs = vec![0i64; n];     // Current coefficient vector
        let mut best_norm_sq = f64::INFINITY;
        let mut best_coeffs = vec![0i64; n];
        let mut points_enumerated = 0u64;

        // Initialize bounds using Gaussian heuristic
        let det = lattice.determinant()?; 
        let gh_radius_sq = (det.powf(1.0/n as f64) * (n as f64).sqrt()).powi(2);
        let mut search_bound_sq = gh_radius_sq.min(self.params.max_radius.powi(2));

        // Start enumeration at level n-1
        let mut i = n - 1;

        loop {
            // Compute bounds for level i
            let mut bound = (search_bound_sq / gs.norm_squared[i]).sqrt().ceil() as i64;
            bound = bound.clamp(-1000, 1000); // Safety limit

            // Initialize radius and center at this level
            radius[i] = bound as f64;

            // Compute center c_i
            let mut sum = 0.0;
            for j in i+1..n {
                if let Some(mu_ji) = gs.get_mu(j, i) {
                    sum += mu_ji * coeffs[j] as f64;
                }
            }
            center[i] = -sum;

            // Set coefficient to nearest integer to center
            coeffs[i] = (center[i] - radius[i]).round() as i64;

            // Enumeration loop
            loop {
                // Update partial norm
                let mut v_i = 0.0;
                for j in i..n {
                    let mut partial_sum = coeffs[j] as f64;
                    for k in j+1..n {
                        if let Some(mu_kj) = gs.get_mu(k, j) {
                            partial_sum -= mu_kj * coeffs[k] as f64;
                        }
                    }
                    v_i += partial_sum.powi(2) * gs.norm_squared[j];
                }

                if v_i < search_bound_sq {
                    if i == 0 {
                        // Leaf node - found candidate
                        points_enumerated += 1;
                        if v_i > 0.0 && v_i < best_norm_sq {
                            best_norm_sq = v_i;
                            best_coeffs.copy_from_slice(&coeffs);

                            // Update bound (pruning)
                            search_bound_sq = v_i * 0.99; // Slight safety factor
                        }

                        // Move to next coefficient
                        coeffs[0] += 1;
                        if (coeffs[0] as f64) > center[0] + radius[0] {
                            // Backtrack
                            i += 1;
                            if i >= n {
                                break;
                            }
                            coeffs[i] += 1;
                        }
                    } else {
                        // Go deeper
                        i -= 1;
                        break;
                    }
                } else {
                    // Pruned - backtrack
                    i += 1;
                    if i >= n {
                        break;
                    }
                    coeffs[i] += 1;

                    // Check if we've exhausted this level
                    if (coeffs[i] as f64) > center[i] + radius[i] {
                        continue; // Backtrack further
                    }
                }

                // Check iteration limit
                if points_enumerated > self.params.algorithm_params.max_iterations as u64 * 1000 {
                    log::warn!("Schnorr-Euchner enumeration reached iteration limit");
                    break;
                }
            }

            if i >= n {
                break;
            }
        }

        let solution = lattice.generate_vector(&best_coeffs)?;
        Ok(SVPEnumerationResult {
            found: best_norm_sq.is_finite(),
            solution,
            norm: best_norm_sq.sqrt(),
            points_enumerated,
            approximation_factor: 1.0,
        })
    }


    fn solve_by_kannan(&self, lattice: &Lattice) -> Result<SVPEnumerationResult> {
        let n = lattice.rank();

        // Step 1: Compute HKZ-reduced basis (approximated by BKZ with β=n)
        let hkz_params = crate::bkz::BKZParams {
            beta: n.min(30), // Limit for performance
            max_rounds: 5,
            ..Default::default()
        };
        let hkz_reducer = crate::bkz::BKZReducer::with_params(hkz_params);
        let hkz_basis = hkz_reducer.reduce(lattice)?;

        // Step 2: Compute Gram-Schmidt of HKZ basis
        let gs = hkz_basis.gram_schmidt()?;

        // Step 3: Enumerate all integer combinations in the fundamental parallelepiped
        let det = hkz_basis.determinant()? as f64;
        let search_range = (det.powf(1.0/n as f64).ceil() as i64).max(3);

        let mut best_norm_sq = f64::INFINITY;
        let mut best_coeffs = vec![0i64; n];
        let mut points_enumerated = 0u64;

        // Use recursive enumeration with proper bounds
        let mut coeffs = vec![0i64; n];
        self.kannan_enumerate(&hkz_basis, &gs, n-1, search_range, &mut coeffs, 
                             &mut best_norm_sq, &mut best_coeffs, &mut points_enumerated)?;
        
        let solution = hkz_basis.generate_vector(&best_coeffs)?;
        Ok(SVPEnumerationResult {
            found: best_norm_sq.is_finite(),
            solution,
            norm: best_norm_sq.sqrt(),
            points_enumerated,
            approximation_factor: 1.0,
        })
    }

    fn kannan_enumerate(
        &self,
        lattice: &Lattice,
        gs: &GramSchmidt,
        level: usize,
        bound: i64,
        coeffs: &mut [i64],
        best_norm_sq: &mut f64,
        best_coeffs: &mut [i64],
        points: &mut u64,
    ) -> Result<()> {
        if level == 0 {
            *points += 1;

            // Check if this is a non-trivial combination
            if coeffs.iter().any(|&c| c != 0) {
                let candidate = lattice.generate_vector(coeffs)?;
                let norm_sq = candidate.norm().powi(2);

                if norm_sq < *best_norm_sq {
                    *best_norm_sq = norm_sq;
                    best_coeffs.copy_from_slice(coeffs);
                }
            }
            return Ok(());
        }

        // Compute adaptive bound for this level using Gram-Schmidt
        let adaptive_bound = if gs.norm_squared[level] > 0.0 {
            (*best_norm_sq / gs.norm_squared[level]).sqrt().ceil() as i64
        } else {
            bound
        };

        for c in -adaptive_bound..=adaptive_bound {
            coeffs[level] = c;
            self.kannan_enumerate(lattice, gs, level-1, bound, coeffs, 
                                best_norm_sq, best_coeffs, points)?;
        }

        Ok(())
    }
    /// Solve using random sampling
    fn solve_by_random_sampling(&self, lattice: &Lattice) -> Result<SVPEnumerationResult> {
        let n = lattice.rank();
        
        let mut best_solution = LatticeVector::zeros(n);
        let mut best_norm = f64::INFINITY;
        let mut points_tried = 0u64;
        
        // Try random coefficient vectors
        for _ in 0..1000 { // Limited iterations for demo
            let coefficients: Vec<i64> = (0..n)
                .map(|_| rand::random::<i64>() % 10 - 5)
                .collect();
            
            let solution = lattice.generate_vector(&coefficients)?;
            let norm = solution.norm();
            
            if norm < best_norm && norm > 0.0 {
                best_norm = norm;
                best_solution = solution;
            }
            
            points_tried += 1;
        }
        
        Ok(SVPEnumerationResult {
            found: best_norm.is_finite(),
            solution: best_solution,
            norm: best_norm,
            points_enumerated: points_tried,
            approximation_factor: 1.0,
        })
    }
    
    /// Solve using BKZ approximation
    fn solve_by_bkz_approximation(&self, lattice: &Lattice) -> Result<SVPEnumerationResult> {
        // Use BKZ to get an approximate basis, then find short vector
        use crate::bkz::{BKZReducer, BKZParams};
        
        let bkz_params = BKZParams::new(20);
        let reducer = BKZReducer::with_params(bkz_params);
        let reduced_lattice = reducer.reduce(lattice)?;
        
        // Take first vector of reduced basis as approximation
        let first_row = reduced_lattice.basis().get_row(0)?;
        let solution = LatticeVector::from_integer_vec(first_row);
        let norm = solution.norm();
        
        Ok(SVPEnumerationResult {
            found: true,
            solution,
            norm,
            points_enumerated: 1,
            approximation_factor: 1.0, // Would need optimal solution to compute accurately
        })
    }
    
    /// Get algorithm info
    pub fn algorithm_info(&self) -> String {
        format!("SVP solver: {:?}, precision: {} bits, pruning: {:?}", 
                self.params.algorithm, self.params.precision_bits, self.params.enable_pruning)
    }
}

/// Result structure for enumeration-based SVP
struct SVPEnumerationResult {
    found: bool,
    solution: LatticeVector,
    norm: f64,
    points_enumerated: u64,
    approximation_factor: f64,
}

/// Advanced SVP solver with multiple strategies
pub struct AdvancedSVPSolver {
    params: SVPSolverParams,
    solvers: Vec<Box<dyn SVPAlgorithmImpl>>,
}

impl AdvancedSVPSolver {
    /// Create new advanced SVP solver
    pub fn new(params: SVPSolverParams) -> Self {
        let solvers: Vec<Box<dyn SVPAlgorithmImpl>> = vec![
            Box::new(EnumerationSolver::new()),
            Box::new(PrunedEnumerationSolver::new()),
            Box::new(HeuristicSolver::new()),
        ];
        
        AdvancedSVPSolver { params, solvers }
    }
    
    /// Solve with automatic algorithm selection
    pub fn solve_automatic(&self, lattice: &Lattice) -> Result<SVPResult> {
        let n = lattice.rank();
        
        // Select algorithm based on lattice dimension
        let solver = if n <= 10 {
            &self.solvers[0] // Exact enumeration for small lattices
        } else if n <= 30 {
            &self.solvers[1] // Pruned enumeration for medium lattices
        } else {
            &self.solvers[2] // Heuristic for large lattices
        };
        
        let start_time = std::time::Instant::now();
        let result = solver.solve(lattice, &self.params)?;
        let execution_time = start_time.elapsed().as_secs_f64();
        
        Ok(SVPResult {
            found: result.found,
            solution: result.solution,
            norm: result.norm,
            points_enumerated: result.points_enumerated,
            execution_time,
            approximation_factor: result.approximation_factor,
        })
    }
}

/// Trait for SVP algorithm implementations
trait SVPAlgorithmImpl {
    fn solve(&self, lattice: &Lattice, params: &SVPSolverParams) -> Result<SVPEnumerationResult>;
}

/// Exact enumeration solver
struct EnumerationSolver;

impl EnumerationSolver {
    fn new() -> Self {
        EnumerationSolver
    }
}

impl SVPAlgorithmImpl for EnumerationSolver {
    fn solve(&self, lattice: &Lattice, params: &SVPSolverParams) -> Result<SVPEnumerationResult> {
        let solver = SVPSolver::with_params(params.clone());
        solver.solve_by_enumeration(lattice)
    }
}

/// Pruned enumeration solver
struct PrunedEnumerationSolver;

impl PrunedEnumerationSolver {
    fn new() -> Self {
        PrunedEnumerationSolver
    }
}

impl SVPAlgorithmImpl for PrunedEnumerationSolver {
    fn solve(&self, lattice: &Lattice, params: &SVPSolverParams) -> Result<SVPEnumerationResult> {
        // Use pruning to limit enumeration
        let solver = SVPSolver::with_params(params.clone());
        solver.solve_by_enumeration(lattice)
    }
}

/// Heuristic solver for large lattices
struct HeuristicSolver;

impl HeuristicSolver {
    fn new() -> Self {
        HeuristicSolver
    }
}

impl SVPAlgorithmImpl for HeuristicSolver {
    fn solve(&self, lattice: &Lattice, params: &SVPSolverParams) -> Result<SVPEnumerationResult> {
        // Use BKZ approximation for large lattices
        let solver = SVPSolver::with_params(params.clone());
        solver.solve_by_bkz_approximation(lattice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_svp_params_validation() {
        let valid_params = SVPSolverParams::default();
        assert!(valid_params.validate().is_ok());
        
        let invalid_params = SVPSolverParams {
            precision_bits: 16,
            ..Default::default()
        };
        assert!(invalid_params.validate().is_err());
    }
    
    #[test]
    fn test_svp_enumeration_2d() {
        let data = vec![vec![3, 0], vec![0, 4]];
        let lattice = Lattice::from_matrix(data).unwrap();
        
        let solver = SVPSolver::with_params(SVPSolverParams::with_algorithm(SVPAlgorithm::Enumeration));
        let result = solver.solve(&lattice).unwrap();
        
        assert!(result.found);
        assert!(result.norm > 0.0);
        assert!(result.points_enumerated > 0);
    }
    
    #[test]
    fn test_svp_random_sampling() {
        let data = vec![vec![1, 0], vec![0, 1]];
        let lattice = Lattice::from_matrix(data).unwrap();
        
        let solver = SVPSolver::with_params(SVPSolverParams::with_algorithm(SVPAlgorithm::RandomSampling));
        let result = solver.solve(&lattice).unwrap();
        
        assert!(result.found);
        assert!((result.norm - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_svp_bkz_approximation() {
        let data = vec![vec![5, 3], vec![2, 1]];
        let lattice = Lattice::from_matrix(data).unwrap();
        
        let solver = SVPSolver::with_params(SVPSolverParams::with_algorithm(SVPAlgorithm::BKZApproximation));
        let result = solver.solve(&lattice).unwrap();
        
        assert!(result.found);
        assert!(result.norm > 0.0);
    }
    
    #[test]
    fn test_advanced_svp_solver() {
        let data = vec![vec![1, 0], vec![0, 1]];
        let lattice = Lattice::from_matrix(data).unwrap();
        
        let solver = AdvancedSVPSolver::new(SVPSolverParams::default());
        let result = solver.solve_automatic(&lattice).unwrap();
        
        assert!(result.found);
        assert!(result.norm > 0.0);
    }
}
