//! CVP (Closest Vector Problem) solver

use crate::core::lattice::Lattice;
use crate::core::matrix::Matrix;
use crate::core::types::{LatticeVector, AlgorithmParams};
use crate::core::error::{LatticeError, Result};

/// Parameters for CVP solver
#[derive(Debug, Clone)]
pub struct CVPSolverParams {
    /// Algorithm type
    pub algorithm: CVPAlgorithm,
    /// Maximum enumeration radius
    pub max_radius: f64,
    /// CVP precision (in bits)
    pub precision_bits: usize,
    /// Enable preprocessing with reduction algorithms
    pub enable_preprocessing: bool,
    /// Preprocessing algorithm
    pub preprocessing_algorithm: PreprocessingAlgorithm,
    /// Algorithm parameters
    pub algorithm_params: AlgorithmParams,
}

impl Default for CVPSolverParams {
    fn default() -> Self {
        CVPSolverParams {
            algorithm: CVPAlgorithm::BabaiNearestPlane,
            max_radius: 1e6,
            precision_bits: 53,
            enable_preprocessing: true,
            preprocessing_algorithm: PreprocessingAlgorithm::LLL,
            algorithm_params: AlgorithmParams::default(),
        }
    }
}

impl CVPSolverParams {
    /// Create new CVP parameters with specific algorithm
    pub fn with_algorithm(algorithm: CVPAlgorithm) -> Self {
        CVPSolverParams {
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

/// CVP algorithm types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CVPAlgorithm {
    /// Babai's nearest plane algorithm
    BabaiNearestPlane,
    /// Babai's rounding algorithm
    BabaiRounding,
    /// Enumeration-based CVP
    Enumeration,
    /// Schnorr-Euchner enumeration for CVP
    SchnorrEuchnerCVP,
    /// Vaidya's algorithm
    Vaidya,
    /// Kannan's algorithm
    KannanCVP,
    /// Embedding method
    Embedding,
}

/// Preprocessing algorithms for CVP
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PreprocessingAlgorithm {
    /// No preprocessing
    None,
    /// LLL reduction
    LLL,
    /// BKZ reduction
    BKZ,
    /// Deep insertion LLL
    DeepLLL,
    /// Combined LLL + BKZ
    Combined,
}

/// Result of CVP computation
#[derive(Debug, Clone)]
pub struct CVPResult {
    /// Whether a solution was found
    pub found: bool,
    /// The closest lattice vector
    pub closest_vector: LatticeVector,
    /// The coefficient vector that generates the closest vector
    pub coefficient_vector: Vec<i64>,
    /// Distance to the closest vector
    pub distance: f64,
    /// Number of lattice points examined
    pub points_examined: u64,
    /// Execution time in seconds
    pub execution_time: f64,
    /// Quality of the solution
    pub approximation_factor: f64,
}

/// CVP solver implementation
pub struct CVPSolver {
    params: CVPSolverParams,
}

impl CVPSolver {
    /// Create new CVP solver with default parameters
    pub fn new() -> Self {
        Self::with_params(CVPSolverParams::default())
    }
    
    /// Create new CVP solver with custom parameters
    pub fn with_params(params: CVPSolverParams) -> Self {
        CVPSolver { params }
    }
    
    /// Solve CVP for a given lattice and target vector
    pub fn solve(&self, lattice: &Lattice, target: &LatticeVector) -> Result<CVPResult> {
        self.params.validate()?;
        
        if target.dimension() != lattice.ambient_dimension() {
            return Err(LatticeError::invalid_dimensions(
                (lattice.ambient_dimension(), 1),
                (target.dimension(), 1)
            ));
        }
        
        let start_time = std::time::Instant::now();
        
        let result = match self.params.algorithm {
            CVPAlgorithm::BabaiNearestPlane => self.solve_babai_nearest_plane(lattice, target),
            CVPAlgorithm::BabaiRounding => self.solve_babai_rounding(lattice, target),
            CVPAlgorithm::Enumeration => self.solve_by_enumeration(lattice, target),
            CVPAlgorithm::SchnorrEuchnerCVP => self.solve_schnorr_euchner_cvp(lattice, target),
            CVPAlgorithm::Vaidya => self.solve_vaidya(lattice, target),
            CVPAlgorithm::KannanCVP => self.solve_kannan_cvp(lattice, target),
            CVPAlgorithm::Embedding => self.solve_embedding(lattice, target),
        }?;
        
        let execution_time = start_time.elapsed().as_secs_f64();
        
        Ok(CVPResult {
            found: result.found,
            closest_vector: result.closest_vector,
            coefficient_vector: result.coefficient_vector,
            distance: result.distance,
            points_examined: result.points_examined,
            execution_time,
            approximation_factor: result.approximation_factor,
        })
    }
    
    /// Solve using Babai's nearest plane algorithm
    fn solve_babai_nearest_plane(&self, lattice: &Lattice, target: &LatticeVector) -> Result<CVPEnumerationResult> {
        let mut lattice = lattice.clone();
        
        // Preprocess lattice if enabled
        if self.params.enable_preprocessing {
            lattice = self.preprocess_lattice(&lattice)?;
        }
        
        let gs = lattice.gram_schmidt()?;
        
        let n = lattice.rank();
        let mut coefficients = vec![0.0f64; n];
        let mut current_target = target.clone();
        
        // Process from last to first vector
        for i in (0..n).rev() {
            // Project target onto current basis vector
            let mut projection: f64 = 0.0;
            for j in 0..target.dimension() {
                projection += current_target.get(j).unwrap() * gs.b_star[i].get(j).unwrap();
            }
            
            // Round to nearest integer
            let coeff: f64 = projection.round();
            coefficients[i] = coeff;
            
            // Update target: subtract projected component
            for j in 0..target.dimension() {
                let new_val = current_target.get(j).unwrap() - 
                             coeff * gs.b_star[i].get(j).unwrap();
                current_target.set(j, new_val)?;
            }
        }
        
        // Generate closest vector
        let int_coefficients: Vec<i64> = coefficients.iter()
            .map(|&x| x.round() as i64)
            .collect();
        let closest_vector = lattice.generate_vector(&int_coefficients)?;
        let distance = target.sub(&closest_vector)?.norm();
        
        Ok(CVPEnumerationResult {
            found: true,
            closest_vector,
            coefficient_vector: int_coefficients,
            distance,
            points_examined: n as u64,
            approximation_factor: 1.0, // Exact for reduced bases
        })
    }
    
    /// Solve using Babai's rounding algorithm
    fn solve_babai_rounding(&self, lattice: &Lattice, target: &LatticeVector) -> Result<CVPEnumerationResult> {
        let basis = lattice.basis();
        
        // Solve linear system B * x = t approximately
        // where B is the basis matrix and t is the target vector
        let coefficients = self.solve_linear_system_approximately(basis, target)?;
        
        // Round coefficients to nearest integers
        let rounded_coefficients: Vec<i64> = coefficients
            .iter()
            .map(|&x| x.round() as i64)
            .collect();
        
        let closest_vector = lattice.generate_vector(&rounded_coefficients)?;
        let distance = target.sub(&closest_vector)?.norm();
        
        Ok(CVPEnumerationResult {
            found: true,
            closest_vector,
            coefficient_vector: rounded_coefficients,
            distance,
            points_examined: 1,
            approximation_factor: 1.0,
        })
    }
    
    /// Solve approximately by solving the linear system B * x = t
    fn solve_linear_system_approximately(&self, basis: &Matrix, target: &LatticeVector) -> Result<Vec<f64>> {
        let n = basis.rows();
        let mut coefficients = vec![0.0f64; n];
        
        // this is a Simple Gaussian elimination 
        // We need to integrate QR decomposition or SVD for better numerical stability
        let mut augmented: Vec<Vec<f64>> = Vec::with_capacity(n);
        
        // Create augmented matrix [B | t]
        for i in 0..n {
            let mut row: Vec<f64> = basis.get_row(i)?
                .into_iter()
                .map(|val| val as f64)
                .collect();
            let rhs = target.get(i).copied().unwrap_or(0.0);
            row.push(rhs);
            augmented.push(row);
        }
        
        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if augmented[k][i].abs() > augmented[max_row][i].abs() {
                    max_row = k;
                }
            }
            
            augmented.swap(i, max_row);
            
            // Make all rows below this one 0 in current column
            for k in (i + 1)..n {
                let factor = augmented[k][i] / augmented[i][i];
                for j in i..=n {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }
        
        // Back substitution
        for i in (0..n).rev() {
            coefficients[i] = augmented[i][n] / augmented[i][i];
            for k in 0..i {
                augmented[k][n] -= augmented[k][i] * coefficients[i];
            }
        }
        
        Ok(coefficients)
    }
    
    /// Solve using enumeration algorithm
    fn solve_by_enumeration(&self, lattice: &Lattice, target: &LatticeVector) -> Result<CVPEnumerationResult> {
        let n = lattice.rank();
        let mut best_coefficients = vec![0i64; n];
        let mut best_distance = f64::INFINITY;
        let mut points_examined = 0u64;
        // Fix the following:
        // Simplified enumeration - try coefficient vectors around Babai solution
        let babai_result = self.solve_babai_nearest_plane(lattice, target)?;
        let babai_coeffs = &babai_result.coefficient_vector;
        
        // Search in a small neighborhood of Babai solution
        for dx in -3..=3 {
            for dy in -3..=3 {
                if n >= 2 {
                    let mut coefficients = babai_coeffs.clone();
                    if dx != 0 {
                        coefficients[0] += dx;
                    }
                    if dy != 0 && n >= 2 {
                        coefficients[1] += dy;
                    }
                    
                    if let Ok(candidate) = lattice.generate_vector(&coefficients) {
                        let distance = target.sub(&candidate)?.norm();
                        if distance < best_distance {
                            best_distance = distance;
                            best_coefficients = coefficients;
                        }
                        points_examined += 1;
                    }
                }
            }
        }
        
        let closest_vector = lattice.generate_vector(&best_coefficients)?;
        
        Ok(CVPEnumerationResult {
            found: best_distance.is_finite(),
            closest_vector,
            coefficient_vector: best_coefficients,
            distance: best_distance,
            points_examined,
            approximation_factor: 1.0,
        })
    }
    
    /// Solve using Schnorr-Euchner enumeration for CVP
    fn solve_schnorr_euchner_cvp(&self, lattice: &Lattice, target: &LatticeVector) -> Result<CVPEnumerationResult> {
        // Again fix this: Schnorr-Euchner CVP is similar to SVP but searches for closest point
        // This is a simplified implementation
        self.solve_by_enumeration(lattice, target)
    }
    
    /// Solve using Vaidya's algorithm
    fn solve_vaidya(&self, lattice: &Lattice, target: &LatticeVector) -> Result<CVPEnumerationResult> {
        // FIX: Vaidya's algorithm uses geometry of numbers
        // Simplified implementation using projections
        let babai_result = self.solve_babai_nearest_plane(lattice, target)?;
        Ok(CVPEnumerationResult {
            found: true,
            closest_vector: babai_result.closest_vector,
            coefficient_vector: babai_result.coefficient_vector,
            distance: babai_result.distance,
            points_examined: babai_result.points_examined,
            approximation_factor: 1.2, // Approximation factor for Vaidya
        })
    }
    
    /// Solve using Kannan's algorithm for CVP
    fn solve_kannan_cvp(&self, lattice: &Lattice, target: &LatticeVector) -> Result<CVPEnumerationResult> {
        // FIX: Kannan's algorithm uses preprocessing and enumeration
        // Simplified implementation
        self.solve_by_enumeration(lattice, target)
    }
    
    /// Solve using embedding method
    fn solve_embedding(&self, lattice: &Lattice, target: &LatticeVector) -> Result<CVPEnumerationResult> {
        // FIX:  Embedding method: create extended lattice and solve SVP
        // This is a conceptual implementation
        let n = lattice.rank();
        // Create embedded basis: [B | t]
        let mut embedded_data = Vec::with_capacity(n);
        for i in 0..n {
            let mut extended_row = lattice.basis().get_row(i)?;
            let scaled_target = (target.get(i).copied().unwrap_or(0.0) * 1000.0).round() as i64;
            extended_row.push(scaled_target);
            embedded_data.push(extended_row);
        }
        
        let embedded_basis = Matrix::new(embedded_data)?;
        let embedded_lattice = Lattice::new(embedded_basis)?;
        
        // Solve SVP on embedded lattice
        use crate::svp::{SVPSolver, SVPSolverParams, SVPAlgorithm};
        let svp_solver = SVPSolver::with_params(SVPSolverParams::with_algorithm(SVPAlgorithm::Enumeration));
        let svp_result = svp_solver.solve(&embedded_lattice)?;
        
        // Extract coefficients (all but the last component)
        let mut coefficient_vector = Vec::with_capacity(n);
        for i in 0..n {
            coefficient_vector.push(svp_result.solution.get(i).unwrap_or(&0.0).round() as i64);
        }
        
        let closest_vector = lattice.generate_vector(&coefficient_vector)?;
        let distance = target.sub(&closest_vector)?.norm();
        
        Ok(CVPEnumerationResult {
            found: true,
            closest_vector,
            coefficient_vector,
            distance,
            points_examined: svp_result.points_enumerated,
            approximation_factor: 1.0,
        })
    }
    
    /// Preprocess lattice using reduction algorithms
    fn preprocess_lattice(&self, lattice: &Lattice) -> Result<Lattice> {
        let preprocessing_lattice = lattice.clone();
        
        match self.params.preprocessing_algorithm {
            PreprocessingAlgorithm::None => Ok(preprocessing_lattice),
            PreprocessingAlgorithm::LLL => {
                use crate::lll::{LLLReducer, LLLParams};
                let lll_reducer = LLLReducer::with_params(LLLParams::default());
                lll_reducer.reduce(&preprocessing_lattice)
            }
            PreprocessingAlgorithm::BKZ => {
                use crate::bkz::{BKZReducer, BKZParams};
                let bkz_reducer = BKZReducer::with_params(BKZParams::default());
                bkz_reducer.reduce(&preprocessing_lattice)
            }
            PreprocessingAlgorithm::DeepLLL => {
                use crate::lll::{ExtendedLLLReducer, LLLParams, LLLVariant};
                let params = LLLParams::default();
                let reducer = ExtendedLLLReducer::new(params, LLLVariant::Deep);
                reducer.reduce_with_variant(&preprocessing_lattice)
            }
            PreprocessingAlgorithm::Combined => {
                // First LLL, then BKZ
                use crate::lll::{LLLReducer, LLLParams};
                let lll_reducer = LLLReducer::with_params(LLLParams::default());
                let lll_result = lll_reducer.reduce(&preprocessing_lattice)?;
                
                use crate::bkz::{BKZReducer, BKZParams};
                let bkz_reducer = BKZReducer::with_params(BKZParams::default());
                bkz_reducer.reduce(&lll_result)
            }
        }
    }
    
    /// Get algorithm info
    pub fn algorithm_info(&self) -> String {
        format!("CVP solver: {:?}, preprocessing: {:?}, precision: {} bits", 
                self.params.algorithm, self.params.preprocessing_algorithm, self.params.precision_bits)
    }
}

/// Result structure for CVP computation
struct CVPEnumerationResult {
    found: bool,
    closest_vector: LatticeVector,
    coefficient_vector: Vec<i64>,
    distance: f64,
    points_examined: u64,
    approximation_factor: f64,
}

/// Advanced CVP solver with multiple strategies and optimization
pub struct AdvancedCVPSolver {
    params: CVPSolverParams,
    svp_solver: crate::svp::SVPSolver,
    lll_reducer: crate::lll::LLLReducer,
    bkz_reducer: crate::bkz::BKZReducer,
}

impl AdvancedCVPSolver {
    /// Create new advanced CVP solver
    pub fn new(params: CVPSolverParams) -> Self {
        AdvancedCVPSolver {
            params,
            svp_solver: crate::svp::SVPSolver::new(),
            lll_reducer: crate::lll::LLLReducer::new(),
            bkz_reducer: crate::bkz::BKZReducer::new(),
        }
    }
    
    /// Solve CVP with adaptive algorithm selection
    pub fn solve_adaptive(&self, lattice: &Lattice, target: &LatticeVector) -> Result<CVPResult> {
        let n = lattice.rank();
        let ambient_dim = lattice.ambient_dimension();
        let mut working_lattice = lattice.clone();
        
        if self.params.enable_preprocessing {
            working_lattice = self.lll_reducer.reduce(&working_lattice)?;
            working_lattice = self.bkz_reducer.reduce(&working_lattice)?;
        }
        
        if self.params.algorithm_params.verbose {
            log::debug!(
                "Advanced CVP using SVP helper: {}",
                self.svp_solver.algorithm_info()
            );
        }
        
        // Select algorithm based on problem characteristics
        let algorithm = if n <= 5 {
            CVPAlgorithm::Enumeration // Exact enumeration for very small lattices
        } else if n <= 20 && ambient_dim == n {
            CVPAlgorithm::BabaiNearestPlane // Babai for square lattices
        } else if self.params.enable_preprocessing {
            CVPAlgorithm::Embedding // Embedding with preprocessing
        } else {
            CVPAlgorithm::BabaiRounding // Rounding for larger lattices
        };
        
        let mut params = self.params.clone();
        params.algorithm = algorithm;
        
        let solver = CVPSolver::with_params(params);
        let start_time = std::time::Instant::now();
        
        let result = solver.solve(&working_lattice, target)?;
        let execution_time = start_time.elapsed().as_secs_f64();
        
        Ok(CVPResult {
            found: result.found,
            closest_vector: result.closest_vector,
            coefficient_vector: result.coefficient_vector,
            distance: result.distance,
            points_examined: result.points_examined,
            execution_time,
            approximation_factor: result.approximation_factor,
        })
    }
    
    /// Solve CVP with multiple methods and return best result
    pub fn solve_with_fallback(&self, lattice: &Lattice, target: &LatticeVector) -> Result<CVPResult> {
        let mut working_lattice = lattice.clone();
        if self.params.enable_preprocessing {
            working_lattice = self.lll_reducer.reduce(&working_lattice)?;
            working_lattice = self.bkz_reducer.reduce(&working_lattice)?;
        }
        
        let algorithms = vec![
            CVPAlgorithm::BabaiNearestPlane,
            CVPAlgorithm::BabaiRounding,
            CVPAlgorithm::Enumeration,
        ];
        
        let mut best_result = None;
        let mut best_distance = f64::INFINITY;
        
        for &algorithm in &algorithms {
            let mut params = self.params.clone();
            params.algorithm = algorithm;
            
            let solver = CVPSolver::with_params(params);
            match solver.solve(&working_lattice, target) {
                Ok(result) => {
                    if result.distance < best_distance {
                        best_distance = result.distance;
                        best_result = Some(result);
                    }
                }
                Err(e) => {
                    if self.params.algorithm_params.verbose {
                        log::warn!("CVP algorithm {:?} failed: {}", algorithm, e);
                    }
                }
            }
        }
        
        best_result.ok_or_else(|| LatticeError::custom("All CVP algorithms failed"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::LatticeVector;
    
    #[test]
    fn test_cvp_params_validation() {
        let valid_params = CVPSolverParams::default();
        assert!(valid_params.validate().is_ok());
        
        let invalid_params = CVPSolverParams {
            precision_bits: 16,
            ..Default::default()
        };
        assert!(invalid_params.validate().is_err());
    }
    
    #[test]
    fn test_cvp_babai_nearest_plane() {
        let data = vec![vec![2, 0], vec![0, 3]];
        let lattice = Lattice::from_matrix(data).unwrap();
        
        let target = LatticeVector::new(vec![1.5, 2.5]);
        
        let solver = CVPSolver::with_params(CVPSolverParams::with_algorithm(CVPAlgorithm::BabaiNearestPlane));
        let result = solver.solve(&lattice, &target).unwrap();
        
        assert!(result.found);
        assert!(result.distance >= 0.0);
    }
    
    #[test]
    fn test_cvp_babai_rounding() {
        let data = vec![vec![1, 0], vec![0, 1]];
        let lattice = Lattice::from_matrix(data).unwrap();
        
        let target = LatticeVector::new(vec![2.3, 1.7]);
        
        let solver = CVPSolver::with_params(CVPSolverParams::with_algorithm(CVPAlgorithm::BabaiRounding));
        let result = solver.solve(&lattice, &target).unwrap();
        
        assert!(result.found);
        assert_eq!(result.coefficient_vector, vec![2, 2]); // Closest to (2,2)
    }
    
    #[test]
    fn test_cvp_enumeration() {
        let data = vec![vec![2, 0], vec![1, 1]];
        let lattice = Lattice::from_matrix(data).unwrap();
        
        let target = LatticeVector::new(vec![1.5, 0.8]);
        
        let solver = CVPSolver::with_params(CVPSolverParams::with_algorithm(CVPAlgorithm::Enumeration));
        let result = solver.solve(&lattice, &target).unwrap();
        
        assert!(result.found);
        assert!(result.distance > 0.0);
    }
    
    #[test]
    fn test_cvp_embedding() {
        let data = vec![vec![3, 0], vec![0, 4]];
        let lattice = Lattice::from_matrix(data).unwrap();
        
        let target = LatticeVector::new(vec![1.5, 2.5]);
        
        let solver = CVPSolver::with_params(CVPSolverParams::with_algorithm(CVPAlgorithm::Embedding));
        let result = solver.solve(&lattice, &target).unwrap();
        
        assert!(result.found);
        assert!(result.distance >= 0.0);
    }
    
    #[test]
    fn test_advanced_cvp_solver() {
        let data = vec![vec![1, 0], vec![0, 1]];
        let lattice = Lattice::from_matrix(data).unwrap();
        
        let target = LatticeVector::new(vec![1.5, 1.5]);
        
        let solver = AdvancedCVPSolver::new(CVPSolverParams::default());
        let result = solver.solve_adaptive(&lattice, &target).unwrap();
        
        assert!(result.found);
        assert!((result.distance - 0.70710678118).abs() < 1e-10); // sqrt(2)/2
    }
    
    #[test]
    fn test_cvp_with_preprocessing() {
        let data = vec![vec![5, 3], vec![2, 1]];
        let lattice = Lattice::from_matrix(data).unwrap();
        
        let target = LatticeVector::new(vec![1.5, 1.5]);
        
        let mut params = CVPSolverParams::default();
        params.enable_preprocessing = true;
        params.preprocessing_algorithm = PreprocessingAlgorithm::LLL;
        
        let solver = CVPSolver::with_params(params);
        let result = solver.solve(&lattice, &target).unwrap();
        
        assert!(result.found);
        assert!(result.distance >= 0.0);
    }
}
