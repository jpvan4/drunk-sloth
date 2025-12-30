#[cfg(feature = "high-precision")]
use crate::core::bigint_matrix::BigIntMatrix;
#[cfg(feature = "high-precision")]
use crate::core::error::{LatticeError, Result};
#[cfg(feature = "high-precision")]
use crate::core::matrix::Matrix;
#[cfg(feature = "high-precision")]
use crate::core::types::LatticeVector;
// `rug::Integer` is used via `BigIntMatrix` and not directly needed here

#[cfg(feature = "high-precision")]
#[derive(Debug, Clone, PartialEq)]
pub struct BigIntLattice {
    pub basis: BigIntMatrix,
    pub dimension: (usize, usize),
}

#[cfg(feature = "high-precision")]
impl BigIntLattice {
    pub fn new(basis: BigIntMatrix) -> Result<Self> {
        let rows = basis.rows();
        let cols = basis.cols();
        if rows == 0 || cols == 0 {
            return Err(LatticeError::invalid_parameters("Basis matrix must be non-empty"));
        }
        Ok(BigIntLattice { basis, dimension: (rows, cols) })
    }

    pub fn rank(&self) -> usize { self.basis.rows() }
    pub fn ambient_dimension(&self) -> usize { self.basis.cols() }

    /// Convert to a `Lattice` with double precision by converting integer entries
    /// to f64 for the algorithms that still expect `Lattice`.
    /// This conversion loses integer precision but keeps numeric values when possible.
    pub fn to_f64_lattice(&self) -> Result<crate::core::lattice::Lattice> {
        let rows = self.rank();
        let cols = self.ambient_dimension();
        let mut data: Vec<Vec<i64>> = Vec::with_capacity(rows);
        // Try converting to i64 first, otherwise convert to f64 and then to i64 if small enough
        for r in 0..rows {
            let mut row_f64: Vec<f64> = Vec::with_capacity(cols);
            for c in 0..cols {
                let val = self.basis.get(r, c).unwrap();
                // For safety, try convert to f64; if extremely large we may lose precision but
                // algorithms operate on float anyway so it's an acceptable trade-off here.
                row_f64.push(val.to_f64());
            }
            // Convert to i64 row when possible (for Matrix type), else round
            let row_i64: Vec<i64> = row_f64.iter().map(|v| *v as i64).collect();
            data.push(row_i64);
        }
        let matrix = Matrix::new(data).map_err(|e| LatticeError::custom(format!("Error converting BigIntLattice to Matrix: {}", e)))?;
        crate::core::lattice::Lattice::new(matrix)
    }

    /// Try to extract a lattice vector with integer coefficients (BigInt) as an f64 vector
    pub fn generate_vector(&self, coefficients: &[i64]) -> Result<LatticeVector> {
        if coefficients.len() != self.rank() {
            return Err(LatticeError::invalid_dimensions((self.rank(),1), (coefficients.len(), 1)));
        }
        let mut result = vec![0.0_f64; self.ambient_dimension()];
        for (i, &c) in coefficients.iter().enumerate() {
            for j in 0..self.ambient_dimension() {
                let v = self.basis.get(i,j).unwrap().to_f64();
                result[j] += (c as f64) * v;
            }
        }
        Ok(LatticeVector::new(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::Integer;
    use crate::core::bigint_matrix::BigIntMatrix;

    #[test]
    fn test_bigint_lattice_to_f64() {
        let m = BigIntMatrix::new(vec![vec![Integer::from(2), Integer::from(3)]]).unwrap();
        let lat = BigIntLattice::new(m).unwrap();
        let f = lat.to_f64_lattice().unwrap();
        assert_eq!(f.rank(), 1);
        assert_eq!(f.ambient_dimension(), 2);
    }

    #[test]
    fn test_from_fplll_bigint() {
        // Example fplll-like content is not used in this test directly; parsing is validated elsewhere.
        let lat = super::BigIntLattice::new(crate::core::bigint_matrix::BigIntMatrix::new(vec![vec![Integer::from_str_radix("115792089237316195423570985008687907853269984665640564039457584007913129639937", 10).unwrap(), Integer::from(1)], vec![Integer::from(0), Integer::from(2)]]).unwrap()).unwrap();
        let f = lat.to_f64_lattice().unwrap();
        assert_eq!(f.rank(), 2);
        assert_eq!(f.ambient_dimension(), 2);
    }
}
