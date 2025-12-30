//! Lattice representation and operations

use crate::core::matrix::Matrix;
use crate::core::error::{LatticeError, Result};
use crate::core::types::{LatticeVector, GramSchmidt};
use std::fs;
#[cfg(feature = "high-precision")]
use crate::core::bigint_lattice::BigIntLattice;
#[cfg(feature = "high-precision")]
use rug::Integer;

/// A lattice represented by its basis matrix
#[derive(Debug, Clone, PartialEq)]
pub struct Lattice {
    basis: Matrix,
    dimension: (usize, usize),
}

impl Lattice {
    /// Create a new lattice from a basis matrix
    pub fn new(basis: Matrix) -> Result<Self> {
        let rows = basis.rows();
        let cols = basis.cols();

        if rows == 0 || cols == 0 {
            return Err(LatticeError::invalid_parameters("Basis matrix cannot be empty"));
        }
        
        Ok(Lattice {
            basis,
            dimension: (rows, cols),
        })
    }
      pub fn cvp_babai(&self, target: &LatticeVector) -> Result<(LatticeVector, Vec<i64>)> {
        // sanity
        if target.dimension() != self.ambient_dimension() {
            return Err(LatticeError::invalid_dimensions(
                (self.ambient_dimension(), 1),
                (target.dimension(), 1),
            ));
        }

        let gs = self.gram_schmidt()?; // uses your existing decomposition

        let n = self.rank();
        let m = self.ambient_dimension();

        // Precompute alpha_i = <t, b*_i> / ||b*_i||^2
        let t = target.as_slice();
        if gs.b_star.len() != n || gs.norm_squared.len() != n {
            return Err(LatticeError::numerical_instability(
                "Gram-Schmidt shape mismatch",
            ));
        }

        let mut alpha = vec![0.0f64; n];
        for i in 0..n {
            let bs_i = &gs.b_star[i];
            if bs_i.dimension() != m {
                return Err(LatticeError::numerical_instability(
                    "b* length mismatch",
                ));
            }
            let nsq = gs.norm_squared[i];
            if nsq.abs() < 1e-18 {
                return Err(LatticeError::numerical_instability(
                    "near-zero GS norm (dependent basis?)",
                ));
            }
            alpha[i] = dot_f64(t, bs_i.as_slice()) / nsq;
        }

        // Back-substitute: z_i = round(alpha_i - sum_{j>i} mu_{j,i} * z_j)
        let mut z = vec![0.0f64; n];
        for i_rev in 0..n {
            let i = n - 1 - i_rev;
            let mut s = 0.0f64;
            for j in (i + 1)..n {
                if let Some(mu_ji) = gs.get_mu(j, i) {
                    s += mu_ji * z[j];
                }
            }
            z[i] = (alpha[i] - s).round();
        }

        // Build closest vector v = sum_i z_i * b_i
        let mut v = vec![0.0f64; m];
        for i in 0..n {
            let row = self.basis.get_row(i)?; // Vec<i64>
            let zi = z[i];
            if zi != 0.0 {
                for j in 0..m {
                    v[j] += zi * (row[j] as f64);
                }
            }
        }

        // Cast integer coefficients (guarding overflow)
        let mut coeffs = Vec::with_capacity(n);
        for zi in z {
            // Wide cast & clamp to i64 range if needed
            let r = zi.round();
            if r < i64::MIN as f64 || r > i64::MAX as f64 {
                return Err(LatticeError::numerical_instability(
                    "CVP produced coefficient outside i64 range",
                ));
            }
            coeffs.push(r as i64);
        }

        Ok((LatticeVector::new(v), coeffs))
    }

    /// Convenience: distance from target to its Babai projection
    pub fn cvp_babai_distance(&self, target: &LatticeVector) -> Result<f64> {
        let (v, _) = self.cvp_babai(target)?;
        let mut d2 = 0.0f64;
        let t = target.as_slice();
        let vv = v.as_slice();
        for k in 0..t.len() {
            let d = t[k] - vv[k];
            d2 += d * d;
        }
        Ok(d2.sqrt())
    }
    /// Create a lattice from a 2D vector representation
    pub fn from_matrix(data: Vec<Vec<i64>>) -> Result<Self> {
        let matrix = Matrix::new(data)?;
        Lattice::new(matrix)
    }
    
    /// Get the dimension of the lattice (rows, cols)
    pub fn dimension(&self) -> (usize, usize) {
        self.dimension
    }
    
    /// Get the rank of the lattice (number of basis vectors)
    pub fn rank(&self) -> usize {
        self.basis.rows()
    }
    
    /// Get the ambient dimension (length of basis vectors)
    pub fn ambient_dimension(&self) -> usize {
        self.basis.cols()
    }
    
    /// Get a reference to the basis matrix
    pub fn basis(&self) -> &Matrix {
        &self.basis
    }
    
    /// Check if the lattice is full rank
    pub fn is_full_rank(&self) -> bool {
        self.rank() == self.ambient_dimension()
    }
    
    /// Compute the Gram-Schmidt orthogonalization
    pub fn gram_schmidt(&self) -> Result<GramSchmidt> {
        GramSchmidt::from_basis(&self.basis)
    }
        /// Tiny-n (≤8) CVP refinement around Babai using bounded enumeration.
    /// radius_coeff controls the ±range around each Babai coeff to search.
    pub fn cvp_refine_small_n(
        &self,
        target: &LatticeVector,
        radius_coeff: i32,
    ) -> Result<(LatticeVector, Vec<i64>)> {
        let (babai_vec, z0) = self.cvp_babai(target)?;
        let n = self.rank();
        if n > 8 {
            // Keep it cheap by design
            return Ok((babai_vec, z0));
        }

        // Preload basis rows
        let m = self.ambient_dimension();
        let mut basis_rows: Vec<Vec<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            basis_rows.push(self.basis.get_row(i)?.into_iter().map(|x| x as f64).collect());
        }

        let t = target.as_slice().to_vec();

        // Depth-first bounded search around z0
        let best_dist2 = diff_norm2(&t, babai_vec.as_slice());
        let best_coeffs = z0.clone();
        let best_vec = babai_vec.as_slice().to_vec();

        let mut cur = z0.clone();
        let mut accum = vec![0.0f64; m]; // current lattice vector sum_i cur[i]*b_i
        // Initialize accum with z0
        for i in 0..n {
            if cur[i] != 0 {
                saxpy_inplace(&mut accum, cur[i] as f64, &basis_rows[i]);
            }
        }

        fn rec(
            i: usize,
            n: usize,
            radius: i32,
            cur: &mut [i64],
            accum: &mut [f64],
            basis_rows: &[Vec<f64>],
            target: &[f64],
            best: &mut (f64, Vec<i64>, Vec<f64>),
        ) {
            if i == n {
                let d2 = diff_norm2(target, accum);
                if d2 < best.0 {
                    best.0 = d2;
                    best.1.clone_from_slice(cur);
                    best.2.clone_from_slice(accum);
                }
                return;
            }

            // subtract baseline contribution (so we can add candidate cleanly)
            saxpy_inplace(accum, -(cur[i] as f64), &basis_rows[i]);

            let base = cur[i];
            for delta in -radius..=radius {
                let z = base + delta as i64;
                saxpy_inplace(accum, z as f64, &basis_rows[i]);
                cur[i] = z;
                rec(i + 1, n, radius, cur, accum, basis_rows, target, best);
                saxpy_inplace(accum, -(z as f64), &basis_rows[i]);
            }

            // restore baseline
            saxpy_inplace(accum, base as f64, &basis_rows[i]);
            cur[i] = base;
        }

        let mut best = (best_dist2, best_coeffs.clone(), best_vec.clone());
        rec(0, n, radius_coeff, &mut cur, &mut accum, &basis_rows, &t, &mut best);

        Ok((LatticeVector::new(best.2), best.1))
    }

    /// Check if the lattice is reduced (proper implementation)
    pub fn is_reduced(&self) -> Result<bool> {
        if !self.is_full_rank() {
            // For non-full rank, check linear independence
            let gs = self.gram_schmidt()?;
            for &norm_sq in &gs.norm_squared {
                if norm_sq < 1e-10 {
                    return Ok(false);
                }
            }
            return Ok(true);
        }

        // For full rank, check determinant and basic reduction properties
        let det = self.determinant()?;
        if det == 0.0 {
            return Ok(false);
        }

        // Check orthogonality defect as a proxy for reduction quality
        let od = self.orthogonality_defect()?;
        Ok(od < 2.0) // Reasonable threshold for "reduced"
    }

    /// Fix determinant calculation
    pub fn determinant(&self) -> Result<f64> {
        if self.is_full_rank() {
            // For full rank, use the basis determinant
            let det = self.basis.determinant()?;
            Ok(det.abs() as f64)
        } else {
            // For non-full rank, use Gram matrix determinant
            let gram = self.basis.gram()?;
            let det = gram.determinant()?;
            Ok((det.abs() as f64).sqrt())
        }
    }

    /// Fix shortest vector implementation
    pub fn shortest_vector(&self) -> Result<LatticeVector> {
        let n = self.rank();
        
        // Try basis vectors first
        let mut shortest_norm = f64::INFINITY;
        let mut shortest_vec = LatticeVector::zeros(self.ambient_dimension());
        
        for i in 0..n {
            let row = self.basis.get_row(i)?;
            let vec = LatticeVector::from_integer_vec(row);
            let norm = vec.norm();
            
            if norm > 0.0 && norm < shortest_norm {
                shortest_norm = norm;
                shortest_vec = vec;
            }
        }
        
        if shortest_norm.is_finite() {
            Ok(shortest_vec)
        } else {
            Err(LatticeError::custom("No non-zero vector found"))
        }
    }
    
    /// Generate a lattice vector from integer coefficients
    pub fn generate_vector(&self, coefficients: &[i64]) -> Result<LatticeVector> {
        if coefficients.len() != self.rank() {
            return Err(LatticeError::invalid_dimensions(
                (self.rank(), 1),
                (coefficients.len(), 1)
            ));
        }
        
        let mut result = vec![0.0; self.ambient_dimension()];
        
        for i in 0..self.rank() {
            let row = self.basis.get_row(i)?;
            for j in 0..self.ambient_dimension() {
                result[j] += (coefficients[i] as f64) * (row[j] as f64);
            }
        }
        
        Ok(LatticeVector::new(result))
    }
    
    /// Compute the covering radius (approximate)
    pub fn covering_radius(&self) -> Result<f64> {
        // This is a complex computation - using approximation
        let det = self.determinant()?;
        let gamma = std::f64::consts::PI.powf(self.ambient_dimension() as f64);
        Ok((det * gamma).powf(1.0 / self.ambient_dimension() as f64))
    }
    
    /// Compute Hermite constant approximation
    pub fn hermite_constant(&self) -> Result<f64> {
        let sv_length = self.shortest_vector()?.norm();
        let det = self.determinant()?;
        
        let n = self.ambient_dimension() as f64;
        Ok((sv_length.powi(2) / det.powf(2.0 / n)).sqrt())
    }
    
    /// Check if another lattice is equivalent (up to unimodular transformation)
    pub fn is_equivalent(&self, other: &Lattice) -> Result<bool> {
        if self.dimension() != other.dimension() {
            return Ok(false);
        }
        
        // This is a simplified check - full equivalence testing is complex
        let det1 = self.determinant()?;
        let det2 = other.determinant()?;
        
        Ok((det1 - det2).abs() < 1e-10)
    }
    
    /// Compute the conditioning number
    pub fn conditioning_number(&self) -> Result<f64> {
        let gs = self.gram_schmidt()?;
        // core/lattice.rs (continued)  
        let mut max_norm: f64 = 0.0;
        let mut min_norm = f64::INFINITY;
        
        for &norm_sq in &gs.norm_squared {
            let norm = norm_sq.sqrt();
            max_norm = max_norm.max(norm);
            min_norm = min_norm.min(norm);
        }
        
        if min_norm == 0.0 {
            return Err(LatticeError::numerical_instability("Zero norm in conditioning number computation"));
        }
        
        Ok(max_norm / min_norm)
    }
    
    /// Compute orthogonality defect
    pub fn orthogonality_defect(&self) -> Result<f64> {
        let gs = self.gram_schmidt()?;
        let lattice_det = self.determinant()?;
        
        let mut product_gs = 1.0f64;
        for &norm_sq in &gs.norm_squared {
            product_gs *= norm_sq.sqrt();
        }
        
        Ok(product_gs / lattice_det)
    }
    
    /// Apply unimodular transformation (for testing)
    pub fn transform(&self, unimodular_matrix: &Matrix) -> Result<Self> {
        // Check if transformation matrix is unimodular (determinant ±1)
        let det = unimodular_matrix.determinant()?;
        
        if det != 1 && det != -1 {
            return Err(LatticeError::invalid_parameters(
                "Transformation matrix must be unimodular (determinant ±1)"
            ));
        }
        
        // Check dimensions
        if unimodular_matrix.rows() != self.rank() || unimodular_matrix.cols() != self.rank() {
            return Err(LatticeError::invalid_dimensions(
                (self.rank(), self.rank()),
                (unimodular_matrix.rows(), unimodular_matrix.cols())
            ));
        }
        
        let new_basis = unimodular_matrix.mul(&self.basis)?;
        Lattice::new(new_basis)
    }
    
    /// Check if lattice contains a given vector
        /// Check if lattice contains a given vector.
    /// Uses CVP(Babai) and equality within a tight epsilon.
    pub fn contains(&self, vector: &LatticeVector) -> Result<bool> {
        if vector.dimension() != self.ambient_dimension() {
            return Ok(false);
        }
        let (closest, _coeffs) = self.cvp_babai(vector)?;
        let x = vector.as_slice();
        let y = closest.as_slice();
        let mut max_abs = 0.0f64;
        for k in 0..x.len() {
            max_abs = max_abs.max((x[k] - y[k]).abs());
        }
        // epsilon tuned for purely integer bases + f64 math
        Ok(max_abs < 1e-9)
    }

    
    /// Sample random vector from the lattice
    pub fn sample_random(&self, coefficients: &[i64]) -> Result<LatticeVector> {
        self.generate_vector(coefficients)
    }
    
    /// Compute quality metrics
    pub fn quality_metrics(&self) -> Result<crate::core::types::QualityMetrics> {
        use crate::core::types::QualityMetrics;
        
        let sv_length = self.shortest_vector()?.norm();
        let hermite_constant = self.hermite_constant()?;
        let conditioning_number = self.conditioning_number()?;
        let orthogonality_defect = self.orthogonality_defect()?;
        
        Ok(QualityMetrics::new()
            .with_shortest_vector_length(sv_length)
            .with_hermite_constant(hermite_constant)
            .with_conditioning_number(conditioning_number)
            .with_orthogonality_defect(orthogonality_defect))
    }
    
    /// Convert to string representation (fplll format)
    pub fn to_fplll_format(&self) -> String {
        let mut output = format!("{} {}\n", self.rank(), self.ambient_dimension());
        for i in 0..self.rank() {
            let row = self.basis.get_row(i).unwrap();
            output.push_str(&row.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "));
            output.push('\n');
        }
        output
    }
    
    /// Parse from fplll format
    pub fn from_fplll_format(input: &str) -> Result<Self> {
        let mut lines = input.lines();
        let header_line = loop {
            match lines.next() {
                Some(line) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() || trimmed.starts_with('#') {
                        continue;
                    } else {
                        break trimmed.to_string();
                    }
                }
                None => {
                    return Err(LatticeError::invalid_parameters(
                        "Empty input",
                    ))
                }
            }
        };
        
        let dimensions: Vec<usize> = header_line
            .split_whitespace()
            .filter_map(|x| x.parse().ok())
            .collect();
        
        if dimensions.len() != 2 {
            return Err(LatticeError::invalid_parameters(
                "First line must contain two integers: rows and cols",
            ));
        }
        
        let (rows, cols) = (dimensions[0], dimensions[1]);
        let mut data = Vec::with_capacity(rows);
        
        let mut row_idx = 0usize;
        while row_idx < rows {
            if let Some(line) = lines.next() {
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.starts_with('#') {
                    continue;
                }
                let values: Vec<i64> = trimmed
                    .split_whitespace()
                    .filter_map(|x| x.parse().ok())
                    .collect();
                
                if values.len() != cols {
                    return Err(LatticeError::invalid_parameters(format!(
                        "Row {} has {} entries, but the header specifies {} columns",
                        row_idx + 1,
                        values.len(),
                        cols
                    )));
                }
                
                data.push(values);
                row_idx += 1;
            } else {
                return Err(LatticeError::invalid_parameters(format!(
                    "File ended after {} rows, but the header specifies {} rows",
                    row_idx,
                    rows
                )));
            }
        }
        
        // Ensure there are no extra non-empty lines after reading the expected rows
        if lines.any(|line| {
            let trimmed = line.trim();
            !trimmed.is_empty() && !trimmed.starts_with('#')
        }) {
            return Err(LatticeError::invalid_parameters(format!(
                "File contains more than {} rows; remove extra data after the lattice definition",
                rows
            )));
        }
        
        Lattice::from_matrix(data)
    }

    #[cfg(feature = "high-precision")]
    pub fn from_fplll_format_bigint(input: &str) -> Result<BigIntLattice> {
        let mut lines = input.lines();
        let header_line = loop {
            match lines.next() {
                Some(line) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() || trimmed.starts_with('#') { continue; }
                    else { break trimmed.to_string(); }
                }
                None => { return Err(LatticeError::invalid_parameters("Empty input")); }
            }
        };
        let dimensions: Vec<usize> = header_line.split_whitespace().filter_map(|x| x.parse().ok()).collect();
        if dimensions.len() != 2 { return Err(LatticeError::invalid_parameters("First line must contain two integers: rows and cols")); }
        let (rows, cols) = (dimensions[0], dimensions[1]);
        let mut data: Vec<Vec<Integer>> = Vec::with_capacity(rows);
        let mut row_idx = 0usize;
        while row_idx < rows {
            if let Some(line) = lines.next() {
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.starts_with('#') { continue; }
                let tokens: Vec<&str> = trimmed.split_whitespace().collect();
                if tokens.len() != cols { return Err(LatticeError::invalid_parameters(format!("Row {} has {} entries, but the header specifies {} columns", row_idx + 1, tokens.len(), cols))); }
                let mut row = Vec::with_capacity(cols);
                for tok in tokens {
                    let int = Integer::from_str_radix(tok, 10).map_err(|e| LatticeError::invalid_parameters(format!("Failed to parse integer token {}: {}", tok, e)))?;
                    row.push(int);
                }
                data.push(row);
                row_idx += 1;
            } else { return Err(LatticeError::invalid_parameters(format!("File ended after {} rows, but the header specifies {} rows", row_idx, rows))); }
        }
        let big_mat = crate::core::bigint_matrix::BigIntMatrix::new(data)?;
        BigIntLattice::new(big_mat)
    }
    
    /// Save to file in fplll format
    pub fn save_to_file(&self, filename: &str) -> Result<()> {
        let content = self.to_fplll_format();
        std::fs::write(filename, content)
            .map_err(|e| LatticeError::io_error(e.to_string()))
    }
    /// Helper: parse a single matrix row from a line
    fn parse_row(line: &str, line_no: usize) -> Result<Vec<i64>> {
    // Trim whitespace and strip leading/trailing brackets
       let mut clean = line.trim();
        
       // Strip any leading '[' characters (handles "[[" too)
       while clean.starts_with('[') {
           clean = &clean[1..];
           clean = clean.trim_start();
       }

       // Strip any trailing ']' characters (handles "]]")
       while clean.ends_with(']') {
           clean = &clean[..clean.len() - 1];
           clean = clean.trim_end();
       }

       // Now split on whitespace and common delimiters
       let tokens: Vec<&str> = clean
           .split(|c: char| c.is_whitespace() || c == ',' || c == ';')
           .filter(|t| !t.is_empty())
           .collect();

       if tokens.is_empty() {
           return Err(LatticeError::invalid_parameters(format!(
               "Row {} has 0 entries",
               line_no
           )));
       }

       let mut row = Vec::with_capacity(tokens.len());
       for tok in tokens {
           let val = tok.parse::<i64>().map_err(|e| {
               LatticeError::invalid_parameters(format!(
                   "Failed to parse integer at row {}: '{}': {}",
                   line_no, tok, e
               ))
           })?;
           row.push(val);
       }
       Ok(row)
    }

    
    /// Load a lattice from a file.
    /// all of the .basis/matrix files adhere to fplll standard format. for example they all have double brackets in the beginning and end and single brackets throughtout the body
    /// [[0 0 1 0]
    ///  [0 0 0 1]
    ///  [0 1 0 0]]
    /// and none of them have the two integer header at top
    /// matrix files without [[bracket:]] format are not supported
    /// Supported formats:
    ///  1. JSON: [[..., ...], [..., ...], ...]
    ///  2. Text with optional header: "rows cols" on first line,
    ///     followed by rows of integers separated by spaces/commas/semicolons.
    pub fn load_from_file(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path).map_err(|e| {
            LatticeError::invalid_parameters(format!(
                "Failed to read lattice file {}: {}",
                path, e
            ))
        })?;

        let trimmed = content.trim();

        // 1) Try JSON / fplll-like `[[ ... ], ... ]` . only vectors with [] are accepted
        if trimmed.starts_with('[') {
            // try to parse as Vec<Vec<i64>>
            if let Ok(data) = serde_json::from_str::<Vec<Vec<i64>>>(trimmed) {
                return Self::from_matrix(data);
            }
            #[cfg(feature = "high-precision")]
            {
                // Try to parse JSON with arbitrary numeric values and build BigIntLattice
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed) {
                    if let serde_json::Value::Array(rows) = val {
                        // Try to detect large integers or format non-parseable by i64
                        let mut any_big = false;
                        for r in &rows {
                            if let serde_json::Value::Array(cols) = r {
                                for cval in cols {
                                    if let serde_json::Value::Number(num) = cval {
                                        if num.as_i64().is_none() {
                                            any_big = true;
                                            break;
                                        }
                                    } else if !cval.is_string() {
                                        any_big = true;
                                        break;
                                    }
                                }
                            }
                            if any_big { break; }
                        }

                        if any_big {
                            // Attempt to parse into BigIntLattice
                            return Self::load_from_file_bigint(path)
                                .and_then(|big_lat| {
                                    // Convert to f64 lattice for compatibility with current algorithms
                                    big_lat.to_f64_lattice()
                                });
                        }
                    }
                }
            }
            // fall through to text parser if JSON fails
        }

        // 2) Plain text matrix with optional header
        let mut lines = trimmed
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with('#'));

        let first_line = match lines.next() {
            Some(l) => l,
            None => {
                return Err(LatticeError::invalid_parameters(
                    "Lattice file is empty after stripping comments/whitespace",
                ))
            }
        };

        // Try "rows cols" header on first line -> if present, remove them.
        let first_tokens: Vec<&str> = first_line.split_whitespace().collect();
        let (mut rows, expected_rows, expected_cols, mut line_no) =
            if first_tokens.len() == 2
                && first_tokens[0].parse::<usize>().is_ok()
                && first_tokens[1].parse::<usize>().is_ok()
            {
                let r = first_tokens[0].parse::<usize>().unwrap();
                let c = first_tokens[1].parse::<usize>().unwrap();
                (Vec::with_capacity(r), Some(r), Some(c), 2usize)
            } else {
                // No header → thats great! continue and treat first line as row 1
                let row = Self::parse_row(first_line, 1)?;
                let c = row.len();
                let mut rows_vec = Vec::new();
                rows_vec.push(row);
                (rows_vec, None, Some(c), 2usize)
            };

        // Parse remaining lines
        while let Some(line) = lines.next() {
            let row = Self::parse_row(line, line_no)?;

            if let Some(cols) = expected_cols {
                if row.len() != cols {
                    return Err(LatticeError::invalid_parameters(format!(
                        "Row {} has {} entries, but expected {} columns",
                        line_no,
                        row.len(),
                        cols
                    )));
                }
            }

            rows.push(row);
            line_no += 1;
        }

        // If header specified #rows, remove header from file
        if let Some(r) = expected_rows {
            if rows.len() != r {
                return Err(LatticeError::invalid_parameters(format!(
                    "Header specifies {} rows, but file contains {} rows. Do the sensible thing and remove header row from file entirely.",
                    r,
                    rows.len()
                )));
            }
        }

        Self::from_matrix(rows)
    }

    #[cfg(feature = "high-precision")]
    pub fn load_from_file_bigint(path: &str) -> Result<BigIntLattice> {
        let content = fs::read_to_string(path).map_err(|e| {
            LatticeError::invalid_parameters(format!("Failed to read lattice file {}: {}", path, e))
        })?;

        let trimmed = content.trim();

        // 1) Try JSON
        if trimmed.starts_with('[') {
            // Try parse as JSON, but fall through to bracketed plain-text parse
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed) {
                if let serde_json::Value::Array(rows) = val {
                let mut data: Vec<Vec<Integer>> = Vec::with_capacity(rows.len());
                for r in rows {
                    if let serde_json::Value::Array(cols) = r {
                        let mut row: Vec<Integer> = Vec::with_capacity(cols.len());
                        for c in cols {
                            let s = match c {
                                serde_json::Value::Number(n) => n.to_string(),
                                serde_json::Value::String(s) => s,
                                _ => return Err(LatticeError::invalid_parameters("Unsupported JSON matrix element type (not number/string)")),
                            };
                            let int = Integer::from_str_radix(&s, 10).map_err(|e| LatticeError::precision_error(format!("Failed to parse big integer: {}", e)))?;
                            row.push(int);
                        }
                        data.push(row);
                    } else {
                        return Err(LatticeError::invalid_parameters("Invalid JSON matrix: expected array of rows"));
                    }
                }
                    return BigIntLattice::new(crate::core::bigint_matrix::BigIntMatrix::new(data)?);
                }
            }
        }

        // 2) Plain text - parse rows of numbers
        let mut lines = trimmed
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with('#'));

        let first_line = match lines.next() {
            Some(l) => l,
            None => {
                return Err(LatticeError::invalid_parameters("Lattice file is empty after stripping comments/whitespace"))
            }
        };

        // Try header "rows cols"
        let first_tokens: Vec<&str> = first_line.split_whitespace().collect();
        let (mut rows_data, expected_rows, expected_cols, mut line_no) = if first_tokens.len() == 2
            && first_tokens[0].parse::<usize>().is_ok()
            && first_tokens[1].parse::<usize>().is_ok()
        {
            let r = first_tokens[0].parse::<usize>().unwrap();
            let c = first_tokens[1].parse::<usize>().unwrap();
            (Vec::with_capacity(r), Some(r), Some(c), 2usize)
        } else {
            let row = Self::parse_row_bigint(first_line, 1)?;
            let c = row.len();
            let mut rows_vec = Vec::new();
            rows_vec.push(row);
            (rows_vec, None, Some(c), 2usize)
        };

        while let Some(line) = lines.next() {
            let row = Self::parse_row_bigint(line, line_no)?;
            if let Some(cols) = expected_cols {
                if row.len() != cols {
                    return Err(LatticeError::invalid_parameters(format!("Row {} has {} entries, but expected {} columns", line_no, row.len(), cols)));
                }
            }
            rows_data.push(row);
            line_no += 1;
        }

        if let Some(r) = expected_rows {
            if rows_data.len() != r {
                return Err(LatticeError::invalid_parameters(format!("Header specifies {} rows, but file contains {} rows. Remove header from file entirely seems like the sensible thing to do here.", r, rows_data.len())));
            }
        }

        let big_mat = crate::core::bigint_matrix::BigIntMatrix::new(rows_data)?;
        BigIntLattice::new(big_mat)
    }

    #[cfg(feature = "high-precision")]
    fn parse_row_bigint(line: &str, line_no: usize) -> Result<Vec<Integer>> {
        let mut out: Vec<Integer> = Vec::new();
        // Split by whitespace, comma, semicolon or bracket characters
        let cleaned = line.replace('[', " ").replace(']', " ").replace(',', " ").replace(';', " ");
        for (i, token) in cleaned.split_whitespace().enumerate() {
            let token = token.trim();
            if token.is_empty() { continue; }
            let int = Integer::from_str_radix(token, 10).map_err(|e| LatticeError::invalid_parameters(format!("Failed to parse integer on line {} token {}: {}", line_no, i+1, e)))?;
            out.push(int);
        }
        if out.is_empty() {
            return Err(LatticeError::invalid_parameters(format!("No numeric data found on line {}", line_no)));
        }
        Ok(out)
    }
}

impl std::fmt::Display for Lattice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Lattice of dimension {}x{}:", self.rank(), self.ambient_dimension())?;
        write!(f, "{}", self.basis)
    }
}
fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "Vectors must be the same length for dot product"
    );
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}
fn saxpy_inplace(y: &mut [f64], a: f64, x: &[f64]) {
    debug_assert_eq!(y.len(), x.len());
    for i in 0..y.len() {
        y[i] += a * x[i];
    }
}

fn diff_norm2(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}




#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lattice_creation() {
        let data = vec![vec![1, 0], vec![0, 1]];
        let lattice = Lattice::from_matrix(data).unwrap();
        assert_eq!(lattice.dimension(), (2, 2));
        assert!(lattice.is_full_rank());
    }
    
    #[test]
    fn test_lattice_determinant() {
        let data = vec![vec![2, 0], vec![0, 3]];
        let lattice = Lattice::from_matrix(data).unwrap();
        let det = lattice.determinant().unwrap();
        assert!((det - 6.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_lattice_shortest_vector() {
        let data = vec![vec![1, 0], vec![1, 1]];
        let lattice = Lattice::from_matrix(data).unwrap();
        let sv = lattice.shortest_vector().unwrap();
        assert!(sv.norm() > 0.0);
    }
    
    #[test]
    fn test_fplll_format() {
        let data = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let lattice = Lattice::from_matrix(data).unwrap();
        
        let format_string = lattice.to_fplll_format();
        let loaded_lattice = Lattice::from_fplll_format(&format_string).unwrap();
        
        assert_eq!(lattice.dimension(), loaded_lattice.dimension());
    }
    
    #[test]
    fn test_orthogonality_defect() {
        let data = vec![vec![1, 0], vec![0, 1]];
        let lattice = Lattice::from_matrix(data).unwrap();
        let od = lattice.orthogonality_defect().unwrap();
        assert!((od - 1.0).abs() < 1e-10); // Perfect orthogonal basis should have OD = 1
    }
}
