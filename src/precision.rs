//! High-precision arithmetic management with rug support

use crate::core::error::{LatticeError, Result};
use crate::core::types::PrecisionType;

#[cfg(feature = "high-precision")]
use rug::Float;

/// Precision manager for configurable arithmetic precision
pub struct PrecisionManager {
    precision_bits: usize,
    current_precision: PrecisionType,
    #[cfg(feature = "high-precision")]
    float_context: Option<Float>, // just a marker/scratch value
    is_initialized: bool,
}

impl Default for PrecisionManager {
    fn default() -> Self {
        PrecisionManager::new(64) // default to 64-bit
    }
}

impl PrecisionManager {
    /// Create new precision manager with specified precision
    pub fn new(bits: usize) -> Self {
        PrecisionManager {
            precision_bits: bits,
            current_precision: PrecisionType::Arbitrary { bits },
            #[cfg(feature = "high-precision")]
            float_context: None,
            is_initialized: false,
        }
    }

    /// Initialize precision manager
    pub fn initialize(&mut self) -> Result<()> {
        self.precision_bits = self.current_precision.bits();

        #[cfg(feature = "high-precision")]
        {
            if self.current_precision.requires_high_precision() {
                // Allocate a scratch Float just to validate the backend is usable.
                let f = Float::with_val(self.precision_bits as u32, 0);
                self.float_context = Some(f);
            } else {
                self.float_context = None;
            }
            self.is_initialized = true;
        }

        #[cfg(not(feature = "high-precision"))]
        {
            if self.current_precision.requires_high_precision() {
                return Err(LatticeError::precision_error(
                    "High precision requested but 'high-precision' feature is not enabled. \
                     Enable the 'high-precision' feature in Cargo.toml (rug backend) and rebuild, \
                     or choose a lower precision mode.",
                ));
            }
            // Pure double-precision mode is fine
            self.is_initialized = true;
        }

        Ok(())
    }

    /// Get current precision in bits
    pub fn precision_bits(&self) -> usize {
        self.precision_bits
    }

    /// Get current precision type
    pub fn precision_type(&self) -> PrecisionType {
        self.current_precision
    }

    /// Set precision to specific number of bits (Arbitrary)
    pub fn set_precision(&mut self, bits: usize) -> Result<()> {
        if bits < 32 {
            return Err(LatticeError::invalid_parameters(format!(
                "Precision must be at least 32 bits, got {}",
                bits
            )));
        }

        if bits > 16384 {
            log::warn!(
                "Very high precision ({} bits) may cause performance issues",
                bits
            );
        }

        self.precision_bits = bits;
        self.current_precision = PrecisionType::Arbitrary { bits };

        #[cfg(feature = "high-precision")]
        {
            if self.current_precision.requires_high_precision() {
                let f = Float::with_val(self.precision_bits as u32, 0);
                self.float_context = Some(f);
            } else {
                self.float_context = None;
            }
        }

        Ok(())
    }

    /// Set precision via a `PrecisionType`
    pub fn set_precision_type(&mut self, precision: PrecisionType) -> Result<()> {
        self.current_precision = precision;
        self.precision_bits = precision.bits();

        #[cfg(feature = "high-precision")]
        {
            if precision.requires_high_precision() {
                let f = Float::with_val(self.precision_bits as u32, 0);
                self.float_context = Some(f);
            } else {
                self.float_context = None;
            }
        }

        Ok(())
    }

    /// Check if manager is initialized
    pub fn is_ready(&self) -> bool {
        self.is_initialized
    }

    /// Old name kept for backward compatibility – now just calls `ensure_high_precision`.
    pub fn require_high_precision(&self) -> Result<()> {
        self.ensure_high_precision()
    }

    /// Ensure high precision is actually available and initialized.
    pub fn ensure_high_precision(&self) -> Result<()> {
        // If high precision is not required by the current mode, nothing to do.
        if !self.current_precision.requires_high_precision() {
        // We're in pure double/extended mode; nothing special needed.
        return Ok(());
    }

    #[cfg(not(feature = "high-precision"))]
    {
        return Err(LatticeError::precision_error(
            "High precision arithmetic requested but 'high-precision' feature \
             is not enabled. Enable it in Cargo.toml (rug backend) and rebuild.",
        ));
    }

    #[cfg(feature = "high-precision")]
    {
        if !self.is_initialized {
            return Err(LatticeError::precision_error(
                "High precision manager not initialized. Call initialize() first.",
            ));
        }

        if self.float_context.is_none() {
            return Err(LatticeError::precision_error(
                "High precision context is missing. This is a bug in PrecisionManager.",
            ));
        }

        Ok(())
    }
}
    /// Create a strongly-typed high-precision float (rug backend)
    pub fn create_hp_float(&self, value: f64) -> Result<HighPrecisionFloat> {
        self.ensure_high_precision()?; // enforce contract

        #[cfg(feature = "high-precision")]
        {
            let f = Float::with_val(self.precision_bits as u32, value);
            Ok(HighPrecisionFloat::Rug(f))
        }

        #[cfg(not(feature = "high-precision"))]
        {
            let _ = value;
            // Should never be reachable due to ensure_high_precision()
            unreachable!("High precision feature validation failed");
        }
    }

    /// High-precision dot product
    pub fn hp_dot_product(&self, a: &[f64], b: &[f64]) -> Result<f64> {
        self.ensure_high_precision()?;

        if a.len() != b.len() {
            return Err(LatticeError::invalid_parameters(format!(
                "Vectors must have same length: {} vs {}",
                a.len(),
                b.len()
            )));
        }

        #[cfg(feature = "high-precision")]
        {
            if self.current_precision.requires_high_precision() {
                return self.hp_dot_product_rug(a, b);
            }
        }

        // Fallback: standard double precision
        let mut sum = 0.0f64;
        for (&ai, &bi) in a.iter().zip(b.iter()) {
            sum += ai * bi;
        }
        Ok(sum)
    }

    #[cfg(feature = "high-precision")]
    fn hp_dot_product_rug(&self, a: &[f64], b: &[f64]) -> Result<f64> {
        let prec = self.precision_bits as u32;
        let mut acc = Float::with_val(prec, 0);

        for (&ai, &bi) in a.iter().zip(b.iter()) {
            let tmp = Float::with_val(prec, ai) * Float::with_val(prec, bi);
            acc += tmp;
        }

        Ok(acc.to_f64())
    }

    /// High-precision Euclidean norm
    pub fn hp_norm(&self, v: &[f64]) -> Result<f64> {
        self.ensure_high_precision()?;
        let dot = self.hp_dot_product(v, v)?;
        Ok(dot.sqrt())
    }

    /// High-precision matrix-vector product
    pub fn hp_matrix_vector_mult(
        &self,
        matrix: &[Vec<f64>],
        vector: &[f64],
    ) -> Result<Vec<f64>> {
        self.ensure_high_precision()?;
        if matrix.is_empty() {
            return Ok(vec![]);
        }

        let rows = matrix.len();
        let cols = matrix[0].len();

        if cols != vector.len() {
            return Err(LatticeError::invalid_dimensions(
                (rows, cols),
                (vector.len(), 1),
            ));
        }

        let mut result = vec![0.0f64; rows];
        for i in 0..rows {
            result[i] = self.hp_dot_product(&matrix[i], vector)?;
        }
        Ok(result)
    }

    /// High-precision matrix-matrix product (using hp_dot_product for columns)
    pub fn hp_matrix_mult(
        &self,
        a: &[Vec<f64>],
        b: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>> {
        self.ensure_high_precision()?;

        let rows_a = a.len();
        if rows_a == 0 {
            return Ok(vec![]);
        }
        let cols_a = a[0].len();
        let rows_b = b.len();
        if rows_b == 0 {
            return Err(LatticeError::invalid_parameters(
                "Right-hand matrix must have at least one row",
            ));
        }
        let cols_b = b[0].len();

        if cols_a != rows_b {
            return Err(LatticeError::invalid_dimensions(
                (rows_a, cols_a),
                (rows_b, cols_b),
            ));
        }

        // Precompute columns of B for nicer reuse
        let mut columns_b = Vec::with_capacity(cols_b);
        for j in 0..cols_b {
            let mut col = Vec::with_capacity(rows_b);
            for i in 0..rows_b {
                col.push(b[i][j]);
            }
            columns_b.push(col);
        }

        let mut result = vec![vec![0.0f64; cols_b]; rows_a];
        for i in 0..rows_a {
            for j in 0..cols_b {
                result[i][j] = self.hp_dot_product(&a[i], &columns_b[j])?;
            }
        }

        Ok(result)
    }

    /// High-precision eigenvalues using a simple power iteration + deflation
    pub fn hp_eigenvalues(&self, matrix: &[Vec<f64>]) -> Result<Vec<f64>> {
        self.ensure_high_precision()?;

        let n = matrix.len();
        if n == 0 {
            return Ok(vec![]);
        }
        if matrix.iter().any(|row| row.len() != n) {
            return Err(LatticeError::invalid_parameters(
                "Matrix must be square",
            ));
        }

        let mut working = matrix.to_vec();
        let mut eigenvalues = Vec::with_capacity(n);
        let tolerance = 1e-10;

        for idx in 0..n {
            let mut v = vec![0.0f64; n];
            v[idx % n] = 1.0;
            let mut prev = vec![0.0f64; n];

            for _ in 0..200 {
                let mv = self.hp_matrix_vector_mult(&working, &v)?;
                let norm = self.hp_norm(&mv)?;
                if norm <= tolerance {
                    break;
                }

                let next: Vec<f64> = mv.into_iter().map(|x| x / norm).collect();

                let delta = next
                    .iter()
                    .zip(prev.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);

                prev = v;
                v = next;

                if delta < tolerance {
                    break;
                }
            }

            let av = self.hp_matrix_vector_mult(&working, &v)?;
            let lambda = self.hp_dot_product(&av, &v)?;
            eigenvalues.push(lambda);

            // Deflation: A <- A - λ v v^T
            for i in 0..n {
                for j in 0..n {
                    working[i][j] -= lambda * v[i] * v[j];
                }
            }
        }

        Ok(eigenvalues)
    }

    /// High-precision condition number (spectral)
    pub fn hp_condition_number(&self, matrix: &[Vec<f64>]) -> Result<f64> {
        self.ensure_high_precision()?;

        if matrix.is_empty() || matrix.len() != matrix[0].len() {
            return Err(LatticeError::invalid_parameters(
                "Matrix must be non-empty and square",
            ));
        }

        let eigenvalues = self.hp_eigenvalues(matrix)?;
        let max_ev = eigenvalues
            .iter()
            .fold(0.0f64, |acc, &v| acc.max(v.abs()));
        let min_ev = eigenvalues
            .iter()
            .fold(f64::INFINITY, |acc, &v| acc.min(v.abs()));

        if min_ev == 0.0 || !min_ev.is_finite() {
            return Err(LatticeError::precision_error(
                "Matrix is singular or ill-conditioned (zero eigenvalue)",
            ));
        }

        Ok(max_ev / min_ev)
    }

    /// High-precision solve Ax = b via Gaussian elimination (still mostly f64,
    /// but guarded by high-precision controls for consistency).
    pub fn hp_solve_linear_system(
        &self,
        a: &[Vec<f64>],
        b: &[f64],
    ) -> Result<Vec<f64>> {
        self.ensure_high_precision()?;
        let n = a.len();

        if n == 0
            || a[0].len() != n
            || b.len() != n
        {
            return Err(LatticeError::invalid_parameters(
                "Invalid matrix or vector dimensions",
            ));
        }

        let mut aug = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = a[i].clone();
            row.push(b[i]);
            aug.push(row);
        }

        // Forward elimination
        for i in 0..n {
            // pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if aug[k][i].abs() > aug[max_row][i].abs() {
                    max_row = k;
                }
            }
            aug.swap(i, max_row);

            // zero out below
            for k in (i + 1)..n {
                let factor = aug[k][i] / aug[i][i];
                for j in i..=n {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }

        // Back substitution
        let mut x = vec![0.0f64; n];
        for i in (0..n).rev() {
            x[i] = aug[i][n];
            for j in (i + 1)..n {
                x[i] -= aug[i][j] * x[j];
            }
            x[i] /= aug[i][i];
        }

        Ok(x)
    }

    /// Simple precision summary string
    pub fn precision_info(&self) -> String {
        let kind = match self.current_precision {
            PrecisionType::Double => "double",
            PrecisionType::Extended => "extended",
            PrecisionType::Arbitrary { .. } => "arbitrary",
        };
        format!("Precision: {} bits ({})", self.precision_bits, kind)
    }

    /// Rough numerical error estimate
    pub fn estimate_numerical_error(&self, operation: &str) -> f64 {
        let epsilon = match self.precision_bits {
            32 => 1.19e-7,      // f32
            53 | 64 => 2.22e-16, // f64
            80 => 1.08e-19,     // extended
            bits => 2.0f64.powi(-(bits as i32)),
        };

        log::debug!(
            "Estimated numerical error for {}: {}",
            operation,
            epsilon
        );
        epsilon
    }
}

/// High-precision arithmetic wrapper for lattice operations
pub struct HighPrecisionLatticeOps {
    precision_manager: PrecisionManager,
}

/// Result of a high-precision Gram-Schmidt computation
pub struct HPGramSchmidt {
    pub mu: Vec<Vec<f64>>,
    pub orthogonal_basis: Vec<Vec<f64>>,
    pub norm_squared: Vec<f64>,
}

impl HighPrecisionLatticeOps {
    pub fn new(precision_bits: usize) -> Result<Self> {
        let mut pm = PrecisionManager::new(precision_bits);
        pm.initialize()?;
        Ok(Self { precision_manager: pm })
    }

    pub fn hp_gram_schmidt_coefficients(
        &self,
        basis: &[Vec<f64>],
    ) -> Result<HPGramSchmidt> {
        self.precision_manager.ensure_high_precision()?;

        if basis.is_empty() {
            return Ok(HPGramSchmidt {
                mu: vec![],
                orthogonal_basis: vec![],
                norm_squared: vec![],
            });
        }

        let n = basis.len();
        let dim = basis[0].len();
        for row in basis {
            if row.len() != dim {
                return Err(LatticeError::invalid_dimensions(
                    (dim, dim),
                    (row.len(), dim),
                ));
            }
        }

        let mut mu = vec![vec![0.0f64; n]; n];
        let mut orthogonal = vec![vec![0.0f64; dim]; n];
        let mut norm_sq = vec![0.0f64; n];

        for i in 0..n {
            orthogonal[i] = basis[i].clone();

            for j in 0..i {
                let denom = norm_sq[j];
                if denom.abs() < 1e-18 {
                    continue;
                }
                let dot = self
                    .precision_manager
                    .hp_dot_product(&basis[i], &orthogonal[j])?;
                let coeff = dot / denom;
                mu[i][j] = coeff;
                for k in 0..dim {
                    orthogonal[i][k] -= coeff * orthogonal[j][k];
                }
            }

            let norm = self.precision_manager.hp_norm(&orthogonal[i])?;
            norm_sq[i] = norm * norm;
            mu[i][i] = 1.0;
        }

        Ok(HPGramSchmidt {
            mu,
            orthogonal_basis: orthogonal,
            norm_squared: norm_sq,
        })
    }

    pub fn hp_check_lovasz_condition(
        &self,
        gs: &HPGramSchmidt,
        delta: f64,
    ) -> Result<bool> {
        self.precision_manager.ensure_high_precision()?;

        let n = gs.mu.len();
        if gs.norm_squared.len() != n {
            return Err(LatticeError::invalid_parameters(
                "Gram-Schmidt data must have matching dimensions",
            ));
        }

        for i in 1..n {
            let lhs = gs.norm_squared[i];
            let rhs = (delta - gs.mu[i][i - 1].powi(2)) * gs.norm_squared[i - 1];
            if lhs < rhs {
                return Ok(false);
            }
        }

        Ok(true)
    }

    pub fn precision_manager(&self) -> &PrecisionManager {
        &self.precision_manager
    }
}

/// High-precision float wrapper – rug backend
#[derive(Debug, Clone)]
pub enum HighPrecisionFloat {
    #[cfg(feature = "high-precision")]
    Rug(Float),
    Fallback(f64),
}

impl HighPrecisionFloat {
    pub fn to_f64(&self) -> f64 {
        match self {
            #[cfg(feature = "high-precision")]
            HighPrecisionFloat::Rug(f) => f.to_f64(),
            HighPrecisionFloat::Fallback(v) => *v,
        }
    }
}
