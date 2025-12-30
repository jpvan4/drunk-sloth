//! Utility functions and helpers for lattice operations

use crate::core::lattice::Lattice;
use crate::core::matrix::Matrix;
use crate::core::types::LatticeVector;
use crate::core::error::{LatticeError, Result};

/// Matrix utilities
pub mod matrix_utils {
    use super::*;
    
    /// Generate random lattice basis
    pub fn generate_random_lattice(n: usize, m: usize, seed: Option<u64>) -> Result<Lattice> {
        use rand::{rng, Rng, SeedableRng};
        use rand::rngs::StdRng;
        
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => {
                let mut entropy = rng();
                <StdRng as SeedableRng>::from_rng(&mut entropy)
            }
        };
        
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            let mut row = Vec::with_capacity(m);
            for _ in 0..m {
                // Generate random integers in a reasonable range
                row.push(rng.random_range(-100..=100));
            }
            data.push(row);
        }
        
        Lattice::from_matrix(data)
    }
    
    /// Create well-conditioned lattice basis
    pub fn create_well_conditioned_lattice(n: usize) -> Result<Lattice> {
        let mut data = Vec::with_capacity(n);
        
        for i in 0..n {
            let mut row = vec![0i64; n];
            row[i] = 1; // Identity matrix
            
            // Add small perturbations for realistic testing
            for j in (i + 1)..n {
                if i > 0 {
                    row[j] = (rand::random::<i64>() % 10) - 5;
                }
            }
            
            data.push(row);
        }
        
        Lattice::from_matrix(data)
    }
    
    /// Check if matrix is upper triangular
    pub fn is_upper_triangular(matrix: &Matrix) -> bool {
        for i in 0..matrix.rows() {
            for j in 0..i {
                if matrix.get(i, j).unwrap_or(&0) != &0 {
                    return false;
                }
            }
        }
        true
    }
    
    /// Check if matrix is lower triangular
    pub fn is_lower_triangular(matrix: &Matrix) -> bool {
        for i in 0..matrix.rows() {
            for j in (i + 1)..matrix.cols() {
                if matrix.get(i, j).unwrap_or(&0) != &0 {
                    return false;
                }
            }
        }
        true
    }
    
    /// Convert to row echelon form
    pub fn to_row_echelon_form(matrix: &Matrix) -> Result<Matrix> {
        let mut data = matrix.to_vec();
        let rows = data.len();
        let cols = data[0].len();
        let mut pivot_row = 0;
        
        for col in 0..cols {
            // Find pivot
            let mut pivot = pivot_row;
            for i in pivot_row..rows {
                if data[i][col].abs() > data[pivot][col].abs() {
                    pivot = i;
                }
            }
            
            if data[pivot][col] == 0 {
                continue;
            }
            
            // Swap rows
            data.swap(pivot_row, pivot);
            
            // Normalize pivot row
            let pivot_val = data[pivot_row][col];
            for j in col..cols {
                data[pivot_row][j] /= pivot_val;
            }
            
            // Eliminate below
            for i in (pivot_row + 1)..rows {
                let factor = data[i][col];
                for j in col..cols {
                    data[i][j] -= factor * data[pivot_row][j];
                }
            }
            
            pivot_row += 1;
            if pivot_row >= rows {
                break;
            }
        }
        
        Matrix::new(data)
    }
    
    /// Compute matrix rank using row reduction
    pub fn matrix_rank(matrix: &Matrix) -> Result<usize> {
        let echelon = to_row_echelon_form(matrix)?;
        
        let mut rank = 0;
        for i in 0..echelon.rows() {
            let row = echelon.get_row(i)?;
            if row.iter().any(|&x| x != 0) {
                rank += 1;
            }
        }
        
        Ok(rank)
    }
    
    /// Compute null space
    pub fn null_space(matrix: &Matrix) -> Result<Vec<Vec<f64>>> {
        let rows = matrix.rows();
        let cols = matrix.cols();
        if cols == 0 {
            return Ok(vec![]);
        }

        let mut working: Vec<Vec<f64>> = matrix
            .to_vec()
            .into_iter()
            .map(|row| row.into_iter().map(|val| val as f64).collect())
            .collect();

        let mut pivot_cols = Vec::new();
        let mut pivot_row = 0usize;
        for col in 0..cols {
            if pivot_row >= rows {
                break;
            }
            let mut pivot = None;
            for r in pivot_row..rows {
                if working[r][col].abs() > 1e-9 {
                    pivot = Some(r);
                    break;
                }
            }
            if let Some(row_idx) = pivot {
                working.swap(pivot_row, row_idx);
                let denom = working[pivot_row][col];
                for c in col..cols {
                    working[pivot_row][c] /= denom;
                }
                for r in 0..rows {
                    if r != pivot_row {
                        let factor = working[r][col];
                        if factor.abs() > 1e-9 {
                            for c in col..cols {
                                working[r][c] -= factor * working[pivot_row][c];
                            }
                        }
                    }
                }
                pivot_cols.push(col);
                pivot_row += 1;
            }
        }

        if pivot_cols.len() == cols {
            return Ok(vec![]);
        }

        let free_cols: Vec<usize> = (0..cols).filter(|c| !pivot_cols.contains(c)).collect();
        let mut basis = Vec::with_capacity(free_cols.len());

        for &free_col in &free_cols {
            let mut vector = vec![0.0; cols];
            vector[free_col] = 1.0;
            for (r, &pivot_col) in pivot_cols.iter().enumerate().rev() {
                let mut value = -working[r][free_col];
                for &other_free in &free_cols {
                    if other_free != free_col {
                        value -= working[r][other_free] * vector[other_free];
                    }
                }
                vector[pivot_col] = value;
            }
            basis.push(vector);
        }

        Ok(basis)
    }
}

/// Vector utilities
pub mod vector_utils {
    use super::*;
    
    /// Generate random lattice vector
    pub fn generate_random_vector(dimension: usize, coefficients: &[i64]) -> Result<LatticeVector> {
        if coefficients.is_empty() {
            return Err(LatticeError::invalid_parameters("Coefficients cannot be empty"));
        }
        
        let mut data = vec![0.0f64; dimension];
        
        for (i, &coeff) in coefficients.iter().enumerate() {
            if i < dimension {
                data[i] = coeff as f64;
            }
        }
        
        Ok(LatticeVector::new(data))
    }
    
    /// Normalize vector
    pub fn normalize(vector: &LatticeVector) -> LatticeVector {
        let norm = vector.norm();
        if norm == 0.0 {
            return vector.clone();
        }
        
        let data: Vec<f64> = vector.as_slice().iter().map(|x| x / norm).collect();
        LatticeVector::new(data)
    }
    
    /// Project vector onto another vector
    pub fn project(u: &LatticeVector, v: &LatticeVector) -> Result<LatticeVector> {
        if u.dimension() != v.dimension() {
            return Err(LatticeError::invalid_parameters("Vectors must have same dimension"));
        }
        
        let dot_product = u.dot(v)?;
        let v_norm_sq = v.norm().powi(2);
        
        if v_norm_sq == 0.0 {
            return Err(LatticeError::invalid_parameters("Cannot project onto zero vector"));
        }
        
        let scalar = dot_product / v_norm_sq;
        Ok(v.scalar_mul(scalar))
    }
    
    /// Compute angle between two vectors
    pub fn angle_between(u: &LatticeVector, v: &LatticeVector) -> Result<f64> {
        let dot_product = u.dot(v)?;
        let norms_product = u.norm() * v.norm();
        
        if norms_product == 0.0 {
            return Err(LatticeError::invalid_parameters("Cannot compute angle with zero vector"));
        }
        
        let cosine = dot_product / norms_product;
        let cosine = cosine.clamp(-1.0, 1.0); // Clamp to valid range
        
        Ok(cosine.acos())
    }
    
    /// Check if vectors are orthogonal
    pub fn is_orthogonal(u: &LatticeVector, v: &LatticeVector, tolerance: f64) -> Result<bool> {
        let dot_product = u.dot(v)?;
        Ok(dot_product.abs() < tolerance)
    }
    
    /// Gram-Schmidt process for a set of vectors
    pub fn gram_schmidt_process(vectors: &[LatticeVector]) -> Result<Vec<LatticeVector>> {
        if vectors.is_empty() {
            return Ok(vec![]);
        }
        
        let dimension = vectors[0].dimension();
        let mut orthogonal_vectors = Vec::new();
        
        for v in vectors {
            if v.dimension() != dimension {
                return Err(LatticeError::invalid_parameters("All vectors must have same dimension"));
            }
            
            let mut u = v.clone();
            
            // Subtract projections onto previous orthogonal vectors
            for orth_vector in &orthogonal_vectors {
                let projection = project(&u, orth_vector)?;
                u = u.sub(&projection)?;
            }
            
            let norm = u.norm();
            if norm > 1e-12 {
                orthogonal_vectors.push(normalize(&u));
            }
        }
        
        Ok(orthogonal_vectors)
    }
}

/// Lattice-specific utilities
pub mod lattice_utils {
    use super::*;
    
    /// Compute lattice packing density
    pub fn packing_density(lattice: &Lattice) -> Result<f64> {
        let determinant = lattice.determinant()?;
        let n = lattice.ambient_dimension() as f64;
        
        // Volume of unit ball in n dimensions
        let volume_unit_ball = std::f64::consts::PI.powf(n / 2.0) / gamma(n / 2.0 + 1.0);
        
        Ok(volume_unit_ball / determinant.powf(2.0 / n))
    }
    
    /// Gamma function approximation (Lanczos approximation)
    fn gamma(mut x: f64) -> f64 {
        let g = 7.0;
        let p = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ];
        
        if x < 0.5 {
            return std::f64::consts::PI / (x * std::f64::consts::PI).sin() * gamma(1.0 - x);
        }
        
        x -= 1.0;
        let mut a = p[0];
        for i in 1..p.len() {
            a += p[i] / (x + i as f64);
        }
        
        let t = x + g + 0.5;
        (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * a
    }
    
    /// Compute successive minima
    pub fn successive_minima(lattice: &Lattice, k: usize) -> Result<Vec<f64>> {
        if k > lattice.rank() {
            return Err(LatticeError::invalid_parameters(
                format!("k cannot exceed lattice rank: {} > {}", k, lattice.rank())
            ));
        }
        
        // Simplified computation of successive minima
        // In practice, this would require solving multiple SVP instances
        let mut minima = Vec::with_capacity(k);
        
        // First minimum is the shortest vector
        let sv = lattice.shortest_vector()?;
        minima.push(sv.norm());
        
        // For higher minima, we'd need more sophisticated algorithms
        for i in 1..k {
            // Simplified: use norm of basis vectors
            let basis = lattice.basis();
            let row = basis.get_row(i - 1)?;
            let norm: f64 = row.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
            minima.push(norm);
        }
        
        Ok(minima)
    }
    
    /// Check if lattice is ideal
    pub fn is_ideal_lattice(lattice: &Lattice) -> Result<bool> {
        // Check if the lattice can be represented as an ideal in a number field
        // This is a complex property requiring algebraic number theory
        // Simplified check: verify certain algebraic properties
        
        let rank = lattice.rank();
        let ambient_dim = lattice.ambient_dimension();
        
        // For simplicity, check if rank equals ambient dimension (full rank)
        // In practice, ideal lattices have special algebraic structure
        Ok(rank == ambient_dim && lattice.is_full_rank())
    }
    
    /// Compute Minkowski sum decomposition
    pub fn minkowski_decompose(lattice: &Lattice) -> Result<(Lattice, f64)> {
        // Decompose lattice into product of fundamental parallelepiped and scaling factor
        let determinant = lattice.determinant()?;
        let n = lattice.ambient_dimension() as f64;
        
        let scaling_factor = determinant.powf(1.0 / n);
        
        // Create scaled version
        let basis = lattice.basis();
        let scaled_data: Vec<Vec<i64>> = basis.to_vec()
            .into_iter()
            .map(|row| row.into_iter().map(|x| (x as f64 / scaling_factor).round() as i64).collect())
            .collect();
        
        let scaled_lattice = Lattice::from_matrix(scaled_data)?;
        Ok((scaled_lattice, scaling_factor))
    }
    
    /// Generate lattice points within a ball
    pub fn lattice_points_in_ball(lattice: &Lattice, center: &LatticeVector, radius: f64) -> Result<Vec<LatticeVector>> {
        let mut points = Vec::new();
        
        // Simplified enumeration within radius
        // In practice, you'd use more sophisticated enumeration
        
        for i in -radius.ceil() as i64..=radius.ceil() as i64 {
            if i == 0 { continue; } // Skip zero vector
            
            let coefficients = vec![i];
            if let Ok(point) = lattice.generate_vector(&coefficients) {
                let distance = point.sub(center)?.norm();
                if distance <= radius {
                    points.push(point);
                }
            }
        }
        
        Ok(points)
    }
}

/// Performance profiling utilities
pub mod profiling {
    use std::time::{Duration, Instant};
    
    /// Simple performance profiler
    pub struct Profiler {
        start_time: Instant,
        operations: Vec<(String, Duration)>,
    }
    
    impl Default for Profiler {
        fn default() -> Self {
            Profiler::new()
        }
    }
    
    impl Profiler {
        /// Create new profiler
        pub fn new() -> Self {
            Profiler {
                start_time: Instant::now(),
                operations: Vec::new(),
            }
        }
        
        /// Start timing an operation
        pub fn start_operation(&mut self, name: String) -> OperationTimer<'_> {
            OperationTimer {
                name,
                start_time: Instant::now(),
                profiler: self,
            }
        }
        
        /// Record an operation with known duration
        pub fn record_operation(&mut self, name: String, duration: Duration) {
            self.operations.push((name, duration));
        }
        
        /// Get total elapsed time
        pub fn total_time(&self) -> Duration {
            self.start_time.elapsed()
        }
        
        /// Print performance report
        pub fn print_report(&self) {
            println!("Performance Report:");
            println!("Total time: {:.2?}", self.total_time());
            
        for (name, duration) in &self.operations {
            let percentage = duration.as_secs_f64() / self.total_time().as_secs_f64() * 100.0;
            println!("  {}: {:.2?} ({:.1}%)", name, duration, percentage);
        }
        }
    }
    
    /// RAII timer for profiling operations
    pub struct OperationTimer<'a> {
        name: String,
        start_time: Instant,
        profiler: &'a mut Profiler,
    }
    
    impl<'a> Drop for OperationTimer<'a> {
        fn drop(&mut self) {
            let duration = self.start_time.elapsed();
            self.profiler.operations.push((self.name.clone(), duration));
        }
    }
    
    /// Profile a function execution
    pub fn profile_function<F, R>(name: &str, func: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = func();
        let duration = start.elapsed();
        
        log::debug!("Function '{}' took {:.2?}", name, duration);
        result
    }
}

/// Statistical utilities
pub mod statistics {
    use super::*;
    
    /// Compute statistics for a set of lattice vectors
    pub fn lattice_statistics(vectors: &[LatticeVector]) -> LatticeStatistics {
        if vectors.is_empty() {
            return LatticeStatistics::default();
        }
        
        let norms: Vec<f64> = vectors.iter().map(|v| v.norm()).collect();
        let mean_norm = mean(&norms);
        let variance = variance(&norms);
        let std_dev = variance.sqrt();
        
        let mut min_norm = f64::INFINITY;
        let mut max_norm: f64 = 0.0;
        for &norm in &norms {
            min_norm = min_norm.min(norm);
            max_norm = max_norm.max(norm);
        }
        
        LatticeStatistics {
            count: vectors.len(),
            mean_norm,
            variance,
            std_dev,
            min_norm,
            max_norm,
        }
    }
    
    fn mean(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f64>() / data.len() as f64
    }
    
    fn variance(data: &[f64]) -> f64 {
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
}

/// Statistics for lattice vectors
#[derive(Debug, Clone)]
pub struct LatticeStatistics {
    pub count: usize,
    pub mean_norm: f64,
    pub variance: f64,
    pub std_dev: f64,
    pub min_norm: f64,
    pub max_norm: f64,
}

impl Default for LatticeStatistics {
    fn default() -> Self {
        LatticeStatistics {
            count: 0,
            mean_norm: 0.0,
            variance: 0.0,
            std_dev: 0.0,
            min_norm: 0.0,
            max_norm: 0.0,
        }
    }
}

/// File I/O utilities
pub mod file_io {
    use super::*;
    
    /// Save lattice to file in custom format
    pub fn save_lattice_custom(lattice: &Lattice, filename: &str) -> Result<()> {
        let mut content = format!("# Custom lattice format\n");
        content.push_str(&lattice.to_fplll_format());
        
        std::fs::write(filename, content)
            .map_err(|e| LatticeError::io_error(e.to_string()))
    }
    
    /// Load lattice from custom format
    pub fn load_lattice_custom(filename: &str) -> Result<Lattice> {
        let content = std::fs::read_to_string(filename)
            .map_err(|e| LatticeError::io_error(e.to_string()))?;
        
        // Skip comments and parse lattice
        let lines: Vec<&str> = content.lines()
            .filter(|line| !line.starts_with('#'))
            .collect();
        
        if lines.is_empty() {
            return Err(LatticeError::invalid_parameters("No valid data in file"));
        }
        
        let first_line = lines[0];
        let dimensions: Vec<usize> = first_line.split_whitespace()
            .filter_map(|x| x.parse().ok())
            .collect();
        
        if dimensions.len() != 2 {
            return Err(LatticeError::invalid_parameters("First line must contain dimensions"));
        }
        
        let (rows, cols) = (dimensions[0], dimensions[1]);
        let mut data = Vec::with_capacity(rows);
        
        for line in &lines[1..] {
            let values: Vec<i64> = line.split_whitespace()
                .filter_map(|x| x.parse().ok())
                .collect();
            
            if values.len() == cols {
                data.push(values);
            }
        }
        
        if data.len() != rows {
            return Err(LatticeError::invalid_parameters(
                format!("Expected {} rows, found {}", rows, data.len())
            ));
        }
        
        Lattice::from_matrix(data)
    }
    
    /// Export lattice as CSV
    pub fn export_lattice_csv(lattice: &Lattice, filename: &str) -> Result<()> {
        let basis = lattice.basis();
        let mut csv_content = String::new();
        
        // Write header
        csv_content.push_str(&format!("rank,{},ambient_dimension,{}\n", lattice.rank(), lattice.ambient_dimension()));
        
        // Write basis matrix
        for i in 0..basis.rows() {
            let row = basis.get_row(i)?;
            csv_content.push_str(&row.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
            csv_content.push('\n');
        }
        
        std::fs::write(filename, csv_content)
            .map_err(|e| LatticeError::io_error(e.to_string()))
    }
    
    /// Generate test lattice files
    pub fn generate_test_lattices(output_dir: &str) -> Result<()> {
        use std::fs;
        
        fs::create_dir_all(output_dir)
            .map_err(|e| LatticeError::io_error(e.to_string()))?;
        
        // Generate various test lattices
        let test_cases = vec![
            ("identity_2d.mat", create_well_conditioned_lattice(2)?),
            ("identity_3d.mat", create_well_conditioned_lattice(3)?),
            ("random_4d.mat", generate_random_lattice(4, 4, Some(42))?),
            ("random_5d.mat", generate_random_lattice(5, 5, Some(123))?),
        ];
        
        for (filename, lattice) in test_cases {
            let filepath = format!("{}/{}", output_dir, filename);
            lattice.save_to_file(&filepath)?;
        }
        
        Ok(())
    }
}

// Re-export common utilities
pub use matrix_utils::*;
pub use vector_utils::*;
pub use lattice_utils::*;
pub use profiling::*;
pub use statistics::*;
pub use file_io::*;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_random_lattice_generation() {
        let lattice = generate_random_lattice(3, 3, Some(42)).unwrap();
        assert_eq!(lattice.dimension(), (3, 3));
        assert!(lattice.is_full_rank());
    }
    
    #[test]
    fn test_well_conditioned_lattice() {
        let lattice = create_well_conditioned_lattice(3).unwrap();
        assert_eq!(lattice.dimension(), (3, 3));
        assert!(lattice.is_full_rank());
        
        let basis = lattice.basis();
        let det = basis.determinant().unwrap();
        assert!(det.abs() > 0);
    }
    
    #[test]
    fn test_matrix_rank() {
        let data = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let matrix = Matrix::new(data).unwrap();
        let rank = matrix_rank(&matrix).unwrap();
        assert_eq!(rank, 2); // Matrix is rank-deficient
    }
    
    #[test]
    fn test_vector_operations() {
        let v1 = LatticeVector::new(vec![1.0, 2.0, 3.0]);
        let v2 = LatticeVector::new(vec![4.0, 5.0, 6.0]);
        
        let normalized = normalize(&v1);
        assert!((normalized.norm() - 1.0).abs() < 1e-10);
        
        let dot = v1.dot(&v2).unwrap();
        assert_eq!(dot, 32.0);
    }
    
    #[test]
    fn test_lattice_statistics() {
        let v1 = LatticeVector::new(vec![1.0, 0.0]);
        let v2 = LatticeVector::new(vec![0.0, 1.0]);
        let v3 = LatticeVector::new(vec![1.0, 1.0]);
        
        let stats = lattice_statistics(&[v1, v2, v3]);
        assert_eq!(stats.count, 3);
        assert!(stats.min_norm > 0.0);
        assert!(stats.max_norm > stats.min_norm);
    }
    
    #[test]
    fn test_profiler() {
        let mut profiler = Profiler::new();
        
        {
            let _timer = profiler.start_operation("test_op".to_string());
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        
        assert!(profiler.total_time().as_millis() >= 10);
    }
    
    #[test]
    fn test_custom_format_io() {
        let lattice = create_well_conditioned_lattice(2).unwrap();
        
        let filename = "test_lattice.custom";
        save_lattice_custom(&lattice, filename).unwrap();
        let loaded = load_lattice_custom(filename).unwrap();
        
        assert_eq!(lattice.dimension(), loaded.dimension());
        
        // Clean up
        let _ = std::fs::remove_file(filename);
    }
}
