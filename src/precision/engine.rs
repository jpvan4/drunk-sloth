// precision/engine.rs
// Big integer and big float precision engine

use crate::core::architecture::*;
use rug::{Integer, Float, Assign};
use std::sync::Arc;

pub struct HighPrecisionEngine {
    precision: PrecisionMode,
    context: PrecisionContext,
}

pub struct PrecisionContext {
    pub int_ctx: IntegerContext,
    pub float_ctx: FloatContext,
}

pub struct IntegerContext {
    pub default_bits: usize,
}

pub struct FloatContext {
    pub default_bits: usize,
}

impl HighPrecisionEngine {
    pub fn new(precision: PrecisionMode) -> Result<Self> {
        let bits = precision.bits();
        
        let context = PrecisionContext {
            int_ctx: IntegerContext { default_bits: bits },
            float_ctx: FloatContext { default_bits: bits },
        };

        Ok(HighPrecisionEngine {
            precision,
            context,
        })
    }

    pub fn gram_schmidt(&self, basis: &LatticeBasis) -> Result<LatticeBasis> {
        match &basis.data {
            MatrixData::BigInt(data) => self.gram_schmidt_bigint(data),
            MatrixData::BigFloat(data) => self.gram_schmidt_bigfloat(data),
            _ => Err(LatticeCoreError::PrecisionError(
                "Unsupported precision for high-precision engine".to_string()
            )),
        }
    }

    fn gram_schmidt_bigint(&self, matrix: &[Vec<Integer>]) -> Result<LatticeBasis> {
        let n = matrix.len();
        let dim = matrix[0].len();
        
        let mut orthogonal = Vec::with_capacity(n);
        let mut norms = Vec::with_capacity(n);
        
        // Convert to floats for orthogonalization
        let float_matrix: Vec<Vec<Float>> = matrix
            .iter()
            .map(|row| {
                row.iter()
                    .map(|x| {
                        let mut f = Float::with_val(self.context.float_ctx.default_bits, 0.0);
                        f.assign(x);
                        f
                    })
                    .collect()
            })
            .collect();
        
        for i in 0..n {
            let mut vec = float_matrix[i].clone();
            
            // Subtract projections onto previous vectors
            for j in 0..i {
                let dot = self.dot_product(&vec, &orthogonal[j])?;
                let norm_sq = &norms[j];
                
                if !norm_sq.is_zero() {
                    let coefficient = &dot / norm_sq;
                    
                    for k in 0..dim {
                        vec[k] -= &coefficient * &orthogonal[j][k];
                    }
                }
            }
            
            // Compute norm squared
            let norm_sq = self.dot_product(&vec, &vec)?;
            norms.push(norm_sq);
            orthogonal.push(vec);
        }
        
        // Convert back to big integers
        let result: Vec<Vec<Integer>> = orthogonal
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|x| {
                        let mut int = Integer::new();
                        int.assign(x.round());
                        int
                    })
                    .collect()
            })
            .collect();
        
        Ok(LatticeBasis::new_bigint(result, self.precision.bits()))
    }

    fn gram_schmidt_bigfloat(&self, matrix: &[Vec<Float>]) -> Result<LatticeBasis> {
        let n = matrix.len();
        let dim = matrix[0].len();
        
        let mut orthogonal = matrix.to_vec();
        let mut norms = Vec::with_capacity(n);
        
        for i in 0..n {
            for j in 0..i {
                let dot = self.dot_product(&orthogonal[i], &orthogonal[j])?;
                let norm_sq = &norms[j];
                
                if !norm_sq.is_zero() {
                    let coefficient = &dot / norm_sq;
                    
                    for k in 0..dim {
                        orthogonal[i][k] -= &coefficient * &orthogonal[j][k];
                    }
                }
            }
            
            let norm_sq = self.dot_product(&orthogonal[i], &orthogonal[i])?;
            norms.push(norm_sq);
        }
        
        Ok(LatticeBasis {
            data: MatrixData::BigFloat(orthogonal),
            precision: self.precision,
        })
    }

    fn dot_product(&self, a: &[Float], b: &[Float]) -> Result<Float> {
        if a.len() != b.len() {
            return Err(LatticeCoreError::DimensionError {
                expected: (a.len(), 1),
                actual: (b.len(), 1),
            });
        }
        
        let mut result = Float::with_val(self.context.float_ctx.default_bits, 0.0);
        for (ai, bi) in a.iter().zip(b.iter()) {
            result += ai * bi;
        }
        
        Ok(result)
    }

    fn validate_basis(&self, basis: &LatticeBasis) -> Result<()> {
        let (rows, cols) = match &basis.data {
            MatrixData::BigInt(data) => {
                if data.is_empty() {
                    return Err(LatticeCoreError::ParameterError("Basis cannot be empty".to_string()));
                }
                let cols = data[0].len();
                (data.len(), cols)
            }
            MatrixData::BigFloat(data) => {
                if data.is_empty() {
                    return Err(LatticeCoreError::ParameterError("Basis cannot be empty".to_string()));
                }
                let cols = data[0].len();
                (data.len(), cols)
            }
            _ => return Err(LatticeCoreError::PrecisionError(
                "Unsupported matrix data type for high-precision engine".to_string()
            )),
        };
        if cols == 0 {
            return Err(LatticeCoreError::ParameterError("Basis vectors cannot have zero dimension".to_string()));
        }
        if rows != cols {
            return Err(LatticeCoreError::DimensionError {
                expected: (cols, cols),
                actual: (rows, cols),
            });
        }
        Ok(())
    }

    // Helper methods for LLL reduction
    fn size_reduce(&self, basis: &mut LatticeBasis, k: usize) -> Result<()> {
        // Implementation of size reduction for LLL
        match &mut basis.data {
            MatrixData::BigInt(data) => {
                // Size reduction for integer basis
                for j in (0..k).rev() {
                    let mu = self.compute_mu(data, k, j)?;
                    if !mu.is_zero() {
                        let rounded_mu = mu.round().to_integer().unwrap();
                        for i in 0..data[0].len() {
                            data[k][i] -= &rounded_mu * &data[j][i];
                        }
                    }
                }
            }
            MatrixData::BigFloat(data) => {
                // Size reduction for float basis
                for j in (0..k).rev() {
                    let mu = self.compute_mu_float(data, k, j)?;
                    if mu.abs() > Float::with_val(self.context.float_ctx.default_bits, 0.5) {
                        let rounded_mu = mu.round();
                        for i in 0..data[0].len() {
                            data[k][i] -= &rounded_mu * &data[j][i];
                        }
                    }
                }
            }
            _ => return Err(LatticeCoreError::PrecisionError(
                "Unsupported matrix data type for size reduction".to_string()
            )),
        }
        Ok(())
    }

    fn compute_mu(&self, basis: &[Vec<Integer>], i: usize, j: usize) -> Result<Float> {
        // Compute mu_ij = (b_i · b_j*) / (b_j* · b_j*)
        // For now, using a simplified approach - in practice would need GSO
        let mut numerator = Float::with_val(self.context.float_ctx.default_bits, 0.0);
        let mut denominator = Float::with_val(self.context.float_ctx.default_bits, 0.0);
        
        for k in 0..basis[0].len() {
            let bi_k = Float::with_val(self.context.float_ctx.default_bits, &basis[i][k]);
            let bj_k = Float::with_val(self.context.float_ctx.default_bits, &basis[j][k]);
            numerator += &bi_k * &bj_k;
            denominator += &bj_k * &bj_k;
        }
        
        if denominator.is_zero() {
            Ok(Float::with_val(self.context.float_ctx.default_bits, 0.0))
        } else {
            Ok(numerator / denominator)
        }
    }

    fn compute_mu_float(&self, basis: &[Vec<Float>], i: usize, j: usize) -> Result<Float> {
        // Compute mu_ij for float basis
        let mut numerator = Float::with_val(self.context.float_ctx.default_bits, 0.0);
        let mut denominator = Float::with_val(self.context.float_ctx.default_bits, 0.0);
        
        for k in 0..basis[0].len() {
            numerator += &basis[i][k] * &basis[j][k];
            denominator += &basis[j][k] * &basis[j][k];
        }
        
        if denominator.is_zero() {
            Ok(Float::with_val(self.context.float_ctx.default_bits, 0.0))
        } else {
            Ok(numerator / denominator)
        }
    }

    fn check_lovasz_condition(
        &self,
        basis: &LatticeBasis,
        k: usize,
        delta: &Float,
        eta: &Float,
    ) -> Result<bool> {
        match &basis.data {
            MatrixData::BigInt(data) => {
                // For BigInt, convert to floats for comparison
                let norm_k = self.compute_norm(data, k)?;
                let norm_k_minus_1 = self.compute_norm(data, k - 1)?;
                let mu_k_k_minus_1 = self.compute_mu(data, k, k - 1)?;
                
                let lhs = &norm_k + (&mu_k_k_minus_1 * &mu_k_k_minus_1) * &norm_k_minus_1;
                let rhs = delta * &norm_k_minus_1;
                
                Ok(lhs >= rhs)
            }
            MatrixData::BigFloat(data) => {
                let norm_k = self.compute_norm_float(data, k)?;
                let norm_k_minus_1 = self.compute_norm_float(data, k - 1)?;
                let mu_k_k_minus_1 = self.compute_mu_float(data, k, k - 1)?;
                
                let lhs = &norm_k + (&mu_k_k_minus_1 * &mu_k_k_minus_1) * &norm_k_minus_1;
                let rhs = delta * &norm_k_minus_1;
                
                Ok(lhs >= rhs)
            }
            _ => Err(LatticeCoreError::PrecisionError(
                "Unsupported matrix data type for Lovász condition".to_string()
            )),
        }
    }

    fn compute_norm(&self, basis: &[Vec<Integer>], i: usize) -> Result<Float> {
        let mut norm_sq = Float::with_val(self.context.float_ctx.default_bits, 0.0);
        for j in 0..basis[0].len() {
            let val = Float::with_val(self.context.float_ctx.default_bits, &basis[i][j]);
            norm_sq += &val * &val;
        }
        Ok(norm_sq.sqrt())
    }

    fn compute_norm_float(&self, basis: &[Vec<Float>], i: usize) -> Result<Float> {
        let mut norm_sq = Float::with_val(self.context.float_ctx.default_bits, 0.0);
        for j in 0..basis[0].len() {
            norm_sq += &basis[i][j] * &basis[i][j];
        }
        Ok(norm_sq.sqrt())
    }

    fn swap_vectors(&self, basis: &mut LatticeBasis, i: usize, j: usize) -> Result<()> {
        match &mut basis.data {
            MatrixData::BigInt(data) => {
                data.swap(i, j);
            }
            MatrixData::BigFloat(data) => {
                data.swap(i, j);
            }
            _ => return Err(LatticeCoreError::PrecisionError(
                "Unsupported matrix data type for vector swapping".to_string()
            )),
        }
        Ok(())
    }
}

#[async_trait]
impl LatticeEngine for HighPrecisionEngine {
    async fn reduce_lll(&self, basis: &LatticeBasis) -> Result<LatticeBasis> {
        // Validate input basis
        self.validate_basis(basis)?;

        // Perform Gram-Schmidt orthogonalization as first step
        let mut current_basis = self.gram_schmidt(basis)?;

        // LLL reduction parameters (standard values)
        let delta = Float::with_val(self.context.float_ctx.default_bits, 0.75);
        let eta = Float::with_val(self.context.float_ctx.default_bits, 0.5);

        // Main LLL reduction loop
        let mut k = 1;
        while k < current_basis.rows() {
            // Size reduction step
            self.size_reduce(&mut current_basis, k)?;

            // Check Lovász condition
            if self.check_lovasz_condition(&current_basis, k, &delta, &eta)? {
                k += 1;
            } else {
                // Swap vectors k-1 and k
                self.swap_vectors(&mut current_basis, k - 1, k)?;
                k = std::cmp::max(k - 1, 1);
            }
        }

        Ok(current_basis)
    }

    async fn reduce_bkz(&self, basis: &LatticeBasis, beta: usize) -> Result<LatticeBasis> {
    // Validate input
    self.validate_basis(basis)?;
    
    if beta < 2 || beta > basis.rows() {
        return Err(LatticeCoreError::ParameterError(
            format!("BKZ block size beta must be between 2 and {}, got {}", basis.rows(), beta)
        ));
    }
    
    let mut current_basis = basis.clone();
    let n = current_basis.rows();
    
    // BKZ main loop
    let mut k = 0;
    let max_loops = 10 * n; // Safety limit
    let mut loop_count = 0;
    
    while k < n - 1 && loop_count < max_loops {
        loop_count += 1;
        
        // Size reduce the current basis using LLL
        let lll_reduced = self.reduce_lll(&current_basis).await?;
        current_basis = lll_reduced;
        
        // Process blocks
        let z = std::cmp::min(k + beta, n);
        let block_start = k;
        let block_end = z;
        
        // Create sub-basis for the current block
        let sub_basis = self.extract_block(&current_basis, block_start, block_end)?;
        
        // Solve SVP on the block (using enumeration for small blocks)
        if block_end - block_start <= 20 { // Enumeration feasible up to dimension ~20
            match self.solve_svp(&sub_basis).await {
                Ok(svp_result) => {
                    // Insert the short vector if it's shorter than the first vector in the block
                    let block_first_norm = self.compute_vector_norm(&current_basis, block_start)?;
                    if svp_result.norm < block_first_norm {
                        // Create new basis with the short vector inserted
                        current_basis = self.insert_short_vector(&current_basis, &svp_result.vector, k)?;
                        k = std::cmp::max(k - 1, 0); // Move back to reprocess
                    } else {
                        k += 1;
                    }
                }
                Err(_) => {
                    // If SVP fails, just do LLL and move on
                    k += 1;
                }
            }
        } else {
            // For larger blocks, use approximate SVP (HKZ reduction on the block)
            let hkz_reduced = self.reduce_hkz_approx(&sub_basis).await?;
            current_basis = self.update_block(&current_basis, &hkz_reduced, block_start)?;
            k += 1;
        }
    }
    
    // Final LLL reduction
    self.reduce_lll(&current_basis).await
}

async fn solve_svp(&self, basis: &LatticeBasis) -> Result<SVPResult> {
    self.validate_basis(basis)?;
    
    let n = basis.rows();
    
    // For very small dimensions, use complete enumeration
    if n <= 10 {
        return self.enumerate_svp(basis).await;
    }
    
    // For medium dimensions, use Schnorr-Euchner enumeration with pruning
    if n <= 40 {
        return self.enumerate_pruned_svp(basis).await;
    }
    
    // For high dimensions, use probabilistic methods
    // This is an approximation - returns the shortest vector found by sampling
    self.approximate_svp(basis).await
}

async fn solve_cvp(&self, basis: &LatticeBasis, target: &[f64]) -> Result<CVPResult> {
    self.validate_basis(basis)?;
    
    if target.len() != basis.cols() {
        return Err(LatticeCoreError::DimensionError {
            expected: (basis.cols(), 1),
            actual: (target.len(), 1),
        });
    }
    
    // Convert target to high precision
    let target_hp = match &basis.data {
        MatrixData::BigInt(_) => {
            // For integer basis, convert to Integer
            let mut int_target = Vec::with_capacity(target.len());
            for &val in target {
                int_target.push(Integer::from(val.round() as i64));
            }
            int_target
        }
        MatrixData::BigFloat(_) => {
            // For float basis, convert to Float
            let mut float_target = Vec::with_capacity(target.len());
            for &val in target {
                let mut f = Float::with_val(self.context.float_ctx.default_bits, val);
                float_target.push(f);
            }
            float_target
        }
        _ => return Err(LatticeCoreError::PrecisionError(
            "Unsupported matrix data type for CVP".to_string()
        )),
    };
    
    // Use Babai's nearest plane algorithm
    self.babai_nearest_plane(basis, &target_hp).await
}

// ========== Helper Methods ==========

fn extract_block(&self, basis: &LatticeBasis, start: usize, end: usize) -> Result<LatticeBasis> {
    match &basis.data {
        MatrixData::BigInt(data) => {
            let block_data: Vec<Vec<Integer>> = data[start..end].to_vec();
            Ok(LatticeBasis::new_bigint(block_data, self.precision.bits()))
        }
        MatrixData::BigFloat(data) => {
            let block_data: Vec<Vec<Float>> = data[start..end].to_vec();
            Ok(LatticeBasis {
                data: MatrixData::BigFloat(block_data),
                precision: self.precision,
            })
        }
        _ => Err(LatticeCoreError::PrecisionError(
            "Unsupported matrix data type for block extraction".to_string()
        )),
    }
}

fn insert_short_vector(&self, basis: &LatticeBasis, vector: &[Integer], position: usize) -> Result<LatticeBasis> {
    match &basis.data {
        MatrixData::BigInt(data) => {
            let mut new_data = data.clone();
            
            // Convert vector to appropriate type
            let new_vector = vector.to_vec();
            
            // Insert at specified position
            new_data.insert(position, new_vector);
            
            // Remove last vector to maintain dimension
            new_data.pop();
            
            Ok(LatticeBasis::new_bigint(new_data, self.precision.bits()))
        }
        MatrixData::BigFloat(_) => {
            // For float basis, convert vector to floats
            let float_vector: Vec<Float> = vector
                .iter()
                .map(|x| {
                    let mut f = Float::with_val(self.context.float_ctx.default_bits, 0.0);
                    f.assign(x);
                    f
                })
                .collect();
            
            self.insert_short_vector_float(basis, &float_vector, position)
        }
        _ => Err(LatticeCoreError::PrecisionError(
            "Unsupported matrix data type for vector insertion".to_string()
        )),
    }
}

fn insert_short_vector_float(&self, basis: &LatticeBasis, vector: &[Float], position: usize) -> Result<LatticeBasis> {
    match &basis.data {
        MatrixData::BigFloat(data) => {
            let mut new_data = data.clone();
            new_data.insert(position, vector.to_vec());
            new_data.pop();
            
            Ok(LatticeBasis {
                data: MatrixData::BigFloat(new_data),
                precision: self.precision,
            })
        }
        _ => Err(LatticeCoreError::PrecisionError(
            "Cannot insert float vector into non-float basis".to_string()
        )),
    }
}

fn update_block(&self, basis: &LatticeBasis, block: &LatticeBasis, start: usize) -> Result<LatticeBasis> {
    match (&basis.data, &block.data) {
        (MatrixData::BigInt(basis_data), MatrixData::BigInt(block_data)) => {
            let mut new_data = basis_data.clone();
            
            for (i, row) in block_data.iter().enumerate() {
                if start + i < new_data.len() {
                    new_data[start + i] = row.clone();
                }
            }
            
            Ok(LatticeBasis::new_bigint(new_data, self.precision.bits()))
        }
        (MatrixData::BigFloat(basis_data), MatrixData::BigFloat(block_data)) => {
            let mut new_data = basis_data.clone();
            
            for (i, row) in block_data.iter().enumerate() {
                if start + i < new_data.len() {
                    new_data[start + i] = row.clone();
                }
            }
            
            Ok(LatticeBasis {
                data: MatrixData::BigFloat(new_data),
                precision: self.precision,
            })
        }
        _ => Err(LatticeCoreError::PrecisionError(
            "Type mismatch in block update".to_string()
        )),
    }
}

async fn reduce_hkz_approx(&self, basis: &LatticeBasis) -> Result<LatticeBasis> {
    // Approximate HKZ reduction using deep insertions
    let mut current = basis.clone();
    let n = current.rows();
    
    for i in 0..n {
        // Extract suffix basis
        if i < n - 1 {
            let suffix = self.extract_block(&current, i, n)?;
            
            // Approximate SVP on suffix
            let svp_result = self.approximate_svp(&suffix).await?;
            
            // Deep insertion if vector is shorter
            if i > 0 {
                let current_norm = self.compute_vector_norm(&current, i)?;
                if svp_result.norm < current_norm {
                    // Insert the short vector
                    match &svp_result.vector {
                        MatrixData::BigInt(vec) => {
                            current = self.insert_short_vector(&current, vec, i)?;
                        }
                        MatrixData::BigFloat(vec) => {
                            current = self.insert_short_vector_float(&current, vec, i)?;
                        }
                        _ => continue,
                    }
                }
            }
        }
        
        // LLL reduce the current basis
        current = self.reduce_lll(&current).await?;
    }
    
    Ok(current)
}

async fn enumerate_svp(&self, basis: &LatticeBasis) -> Result<SVPResult> {
    // Complete enumeration for small dimensions
    let gso = self.gram_schmidt(basis)?;
    
    match (&basis.data, &gso.data) {
        (MatrixData::BigInt(basis_data), MatrixData::BigFloat(gso_data)) => {
            let mut shortest_norm = Float::with_val(self.context.float_ctx.default_bits, f64::INFINITY);
            let mut shortest_vector: Option<Vec<Integer>> = None;
            let n = basis_data.len();
            
            // Simple bounding for enumeration radius
            let bound = self.compute_vector_norm(basis, 0)?;
            
            // TODO: Implement full enumeration algorithm
            // For now, return the first basis vector as placeholder
            shortest_vector = Some(basis_data[0].clone());
            shortest_norm = bound;
            
            Ok(SVPResult {
                vector: MatrixData::BigInt(shortest_vector.unwrap()),
                norm: shortest_norm,
            })
        }
        _ => Err(LatticeCoreError::PrecisionError(
            "SVP enumeration not implemented for this data type".to_string()
        )),
    }
}

async fn enumerate_pruned_svp(&self, basis: &LatticeBasis) -> Result<SVPResult> {
    // Pruned enumeration (Schnorr-Euchner)
    // Placeholder implementation - returns shortest basis vector
    let mut shortest_idx = 0;
    let mut shortest_norm = Float::with_val(self.context.float_ctx.default_bits, f64::INFINITY);
    
    for i in 0..basis.rows() {
        let norm = self.compute_vector_norm(basis, i)?;
        if norm < shortest_norm {
            shortest_norm = norm;
            shortest_idx = i;
        }
    }
    
    match &basis.data {
        MatrixData::BigInt(data) => {
            Ok(SVPResult {
                vector: MatrixData::BigInt(data[shortest_idx].clone()),
                norm: shortest_norm,
            })
        }
        MatrixData::BigFloat(data) => {
            Ok(SVPResult {
                vector: MatrixData::BigFloat(data[shortest_idx].clone()),
                norm: shortest_norm,
            })
        }
        _ => Err(LatticeCoreError::PrecisionError(
            "Unsupported data type for pruned enumeration".to_string()
        )),
    }
}

async fn approximate_svp(&self, basis: &LatticeBasis) -> Result<SVPResult> {
    // Probabilistic SVP approximation using Klein's algorithm or similar
    // Returns the shortest vector found by sampling
    
    let mut best_vector = match &basis.data {
        MatrixData::BigInt(data) => MatrixData::BigInt(data[0].clone()),
        MatrixData::BigFloat(data) => MatrixData::BigFloat(data[0].clone()),
        _ => return Err(LatticeCoreError::PrecisionError(
            "Unsupported data type for approximate SVP".to_string()
        )),
    };
    
    let mut best_norm = self.compute_vector_norm(basis, 0)?;
    
    // Sample random lattice points (simplified)
    let num_samples = 1000.min(10 * basis.rows());
    
    for _ in 0..num_samples {
        // Generate random coefficients
        let coeffs: Vec<i64> = (0..basis.rows())
            .map(|_| rand::random::<i8>() as i64)
            .collect();
        
        // Compute linear combination
        match &basis.data {
            MatrixData::BigInt(data) => {
                let mut candidate = vec![Integer::new(); data[0].len()];
                for (i, &coeff) in coeffs.iter().enumerate() {
                    if coeff != 0 {
                        let coeff_int = Integer::from(coeff);
                        for j in 0..data[0].len() {
                            candidate[j] += &coeff_int * &data[i][j];
                        }
                    }
                }
                
                let norm = self.compute_norm_integer(&candidate)?;
                if norm < best_norm {
                    best_norm = norm;
                    best_vector = MatrixData::BigInt(candidate);
                }
            }
            MatrixData::BigFloat(data) => {
                let mut candidate = vec![Float::with_val(self.context.float_ctx.default_bits, 0.0); data[0].len()];
                for (i, &coeff) in coeffs.iter().enumerate() {
                    if coeff != 0 {
                        let coeff_float = Float::with_val(self.context.float_ctx.default_bits, coeff);
                        for j in 0..data[0].len() {
                            candidate[j] += &coeff_float * &data[i][j];
                        }
                    }
                }
                
                let norm = self.compute_norm_float_slice(&candidate)?;
                if norm < best_norm {
                    best_norm = norm;
                    best_vector = MatrixData::BigFloat(candidate);
                }
            }
            _ => continue,
        }
    }
    
    Ok(SVPResult {
        vector: best_vector,
        norm: best_norm,
    })
}

async fn babai_nearest_plane(&self, basis: &LatticeBasis, target: &[Integer]) -> Result<CVPResult> {
    // Babai's nearest plane algorithm for CVP
    
    // Compute Gram-Schmidt orthogonalization
    let gso = self.gram_schmidt(basis)?;
    
    match (&basis.data, &gso.data) {
        (MatrixData::BigInt(basis_data), MatrixData::BigFloat(gso_data)) => {
            let mut w = target.to_vec();
            let mut coefficients = vec![Integer::new(); basis_data.len()];
            
            // Work backwards through the basis
            for i in (0..basis_data.len()).rev() {
                // Compute coefficient
                let mut numerator = Float::with_val(self.context.float_ctx.default_bits, 0.0);
                let mut denominator = Float::with_val(self.context.float_ctx.default_bits, 0.0);
                
                for j in 0..w.len() {
                    let w_j = Float::with_val(self.context.float_ctx.default_bits, &w[j]);
                    numerator += &w_j * &gso_data[i][j];
                    denominator += &gso_data[i][j] * &gso_data[i][j];
                }
                
                if !denominator.is_zero() {
                    let mu = &numerator / &denominator;
                    let c = mu.round().to_integer().unwrap();
                    coefficients[i] = c.clone();
                    
                    // Subtract c * b_i from w
                    for j in 0..w.len() {
                        w[j] -= &c * &basis_data[i][j];
                    }
                }
            }
            
            // Compute closest vector: sum(coefficients[i] * basis_data[i])
            let mut closest = vec![Integer::new(); target.len()];
            for i in 0..basis_data.len() {
                if !coefficients[i].is_zero() {
                    for j in 0..target.len() {
                        closest[j] += &coefficients[i] * &basis_data[i][j];
                    }
                }
            }
            
            // Compute distance
            let mut distance_sq = Float::with_val(self.context.float_ctx.default_bits, 0.0);
            for j in 0..target.len() {
                let diff = Float::with_val(self.context.float_ctx.default_bits, &target[j]) 
                    - Float::with_val(self.context.float_ctx.default_bits, &closest[j]);
                distance_sq += &diff * &diff;
            }
            
            Ok(CVPResult {
                closest_vector: MatrixData::BigInt(closest),
                distance: distance_sq.sqrt(),
                target: MatrixData::BigInt(target.to_vec()),
            })
        }
        _ => Err(LatticeCoreError::PrecisionError(
            "Babai's algorithm not implemented for this data type".to_string()
        )),
    }
}

// Additional helper method for float targets
async fn babai_nearest_plane_float(&self, basis: &LatticeBasis, target: &[Float]) -> Result<CVPResult> {
    // Similar implementation for float targets
    // ... (implementation would mirror the integer version)
    
    // Placeholder - convert to integer version
    let int_target: Vec<Integer> = target
        .iter()
        .map(|x| {
            let mut int = Integer::new();
            int.assign(x.round());
            int
        })
        .collect();
    
    self.babai_nearest_plane(basis, &int_target).await
}

fn compute_vector_norm(&self, basis: &LatticeBasis, idx: usize) -> Result<Float> {
    match &basis.data {
        MatrixData::BigInt(data) => {
            let mut norm_sq = Float::with_val(self.context.float_ctx.default_bits, 0.0);
            for val in &data[idx] {
                let f = Float::with_val(self.context.float_ctx.default_bits, val);
                norm_sq += &f * &f;
            }
            Ok(norm_sq.sqrt())
        }
        MatrixData::BigFloat(data) => {
            let mut norm_sq = Float::with_val(self.context.float_ctx.default_bits, 0.0);
            for val in &data[idx] {
                norm_sq += val * val;
            }
            Ok(norm_sq.sqrt())
        }
        _ => Err(LatticeCoreError::PrecisionError(
            "Cannot compute norm for unsupported data type".to_string()
        )),
    }
}

fn compute_norm_integer(&self, vec: &[Integer]) -> Result<Float> {
    let mut norm_sq = Float::with_val(self.context.float_ctx.default_bits, 0.0);
    for val in vec {
        let f = Float::with_val(self.context.float_ctx.default_bits, val);
        norm_sq += &f * &f;
    }
    Ok(norm_sq.sqrt())
}

fn compute_norm_float_slice(&self, vec: &[Float]) -> Result<Float> {
    let mut norm_sq = Float::with_val(self.context.float_ctx.default_bits, 0.0);
    for val in vec {
        norm_sq += val * val;
    }
    Ok(norm_sq.sqrt())
}

// Add to struct definition if needed:
#[derive(Clone)]
pub struct SVPResult {
    pub vector: MatrixData,
    pub norm: Float,
}

#[derive(Clone)]
pub struct CVPResult {
    pub closest_vector: MatrixData,
    pub distance: Float,
    pub target: MatrixData,
}

    fn precision(&self) -> PrecisionMode {
        self.precision
    }

    fn supports_gpu(&self) -> bool {
        false
    }
}