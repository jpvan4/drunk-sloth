// Big integer matrix implementation using `rug::Integer`
// This is a lightweight replacement for `Matrix` when high-precision
// integer arithmetic is required (basis files contain >64-bit integers).
#[cfg(feature = "high-precision")]
use crate::core::error::{LatticeError, Result};
#[cfg(feature = "high-precision")]
use rug::Integer;
#[cfg(feature = "high-precision")]
use std::fmt;

#[cfg(feature = "high-precision")]
#[derive(Debug, Clone, PartialEq)]
pub struct BigIntMatrix {
    pub data: Vec<Vec<Integer>>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "high-precision")]
impl BigIntMatrix {
    pub fn new(data: Vec<Vec<Integer>>) -> Result<Self> {
        if data.is_empty() {
            return Err(LatticeError::invalid_parameters("Matrix must have at least one row"));
        }

        let rows = data.len();
        let cols = data[0].len();
        for (i, r) in data.iter().enumerate() {
            if r.len() != cols {
                return Err(LatticeError::invalid_dimensions((rows, cols), (rows, r.len())));
            }
            let _ = i; // keep enumerator for future debug logging
        }
        Ok(BigIntMatrix { data, rows, cols })
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        let z = Integer::from(0);
        BigIntMatrix { data: vec![vec![z; cols]; rows], rows, cols }
    }

    pub fn identity(n: usize) -> Result<Self> {
        if n == 0 {
            return Err(LatticeError::invalid_parameters("Identity matrix requires n>0"));
        }
        let mut data = vec![vec![Integer::from(0); n]; n];
        for i in 0..n {
            data[i][i] = Integer::from(1);
        }
        Ok(BigIntMatrix { data, rows: n, cols: n })
    }

    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }

    pub fn get(&self, row: usize, col: usize) -> Option<&Integer> { self.data.get(row)?.get(col) }

    pub fn get_row(&self, row: usize) -> Result<Vec<Integer>> {
        if row >= self.rows { return Err(LatticeError::invalid_dimensions((self.rows, self.cols), (row, self.cols))); }
        Ok(self.data[row].clone())
    }

    pub fn get_col(&self, col: usize) -> Result<Vec<Integer>> {
        if col >= self.cols { return Err(LatticeError::invalid_dimensions((self.rows, self.cols), (self.rows, col))); }
        Ok(self.data.iter().map(|r| r[col].clone()).collect())
    }

    pub fn set(&mut self, row: usize, col: usize, value: Integer) -> Result<()> {
        if row >= self.rows || col >= self.cols {
            return Err(LatticeError::invalid_dimensions((self.rows, self.cols), (row, col)));
        }
        self.data[row][col] = value;
        Ok(())
    }

    pub fn transpose(&self) -> Self {
        let mut data = vec![vec![Integer::from(0); self.rows]; self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                data[j][i] = self.data[i][j].clone();
            }
        }
        BigIntMatrix { data, rows: self.cols, cols: self.rows }
    }

    /// Convert to a Vec<Vec<f64>> for algorithms that require floating point
    /// This may lose precision for numbers beyond f64 range.
    pub fn to_f64_vec(&self) -> Vec<Vec<f64>> {
        let mut out: Vec<Vec<f64>> = Vec::with_capacity(self.rows);
        for r in &self.data {
            let mut row = Vec::with_capacity(self.cols);
            for v in r {
                row.push(v.to_f64());
            }
            out.push(row);
        }
        out
    }

    /// Convert to flat float32 vector (row-major) for GPU use
    pub fn to_flat_f32(&self) -> Vec<f32> {
        let mut flat: Vec<f32> = Vec::with_capacity(self.rows * self.cols);
        for r in &self.data {
            for v in r {
                flat.push(v.to_f64() as f32);
            }
        }
        flat
    }

    /// Try to convert to the i64-based `Matrix` if numbers fit in `i64`.
    pub fn to_matrix_i64(&self) -> Result<crate::core::matrix::Matrix> {
        let mut data: Vec<Vec<i64>> = Vec::with_capacity(self.rows);
        for r in &self.data {
            let mut row: Vec<i64> = Vec::with_capacity(self.cols);
            for v in r {
                if let Some(i) = v.to_i64() {
                    row.push(i);
                } else {
                    return Err(LatticeError::precision_error("Value cannot be represented as i64"));
                }
            }
            data.push(row);
        }
        crate::core::matrix::Matrix::new(data)
    }

    /// Basic matrix multiplication - integer arithmetic
    pub fn mul(&self, other: &BigIntMatrix) -> Result<BigIntMatrix> {
        if self.cols != other.rows { return Err(LatticeError::invalid_dimensions((self.rows, self.cols), (other.rows, other.cols))); }
        let mut data = vec![vec![Integer::from(0); other.cols]; self.rows];
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut acc = Integer::from(0);
                for k in 0..self.cols {
                    let mut t = self.data[i][k].clone();
                    t *= &other.data[k][j];
                    acc += t;
                }
                data[i][j] = acc;
            }
        }
        Ok(BigIntMatrix { data, rows: self.rows, cols: other.cols })
    }
}

#[cfg(feature = "high-precision")]
impl fmt::Display for BigIntMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "BigIntMatrix {}x{}:", self.rows, self.cols)?;
        for row in &self.data {
            write!(f, "[")?;
            for (i, v) in row.iter().enumerate() {
                if i > 0 { write!(f, ", ")?; }
                write!(f, "{}", v)?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::Integer;

    #[test]
    fn test_bigint_matrix_basic() {
        let data = vec![vec![Integer::from(1), Integer::from(2)], vec![Integer::from(3), Integer::from(4)]];
        let m = BigIntMatrix::new(data).unwrap();
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 2);
        let t = m.transpose();
        assert_eq!(t.rows(), 2);
        assert_eq!(t.get(0, 1).unwrap().to_i32(), Some(3));
    }

    #[test]
    fn test_bigint_mul() {
        let a = BigIntMatrix::new(vec![vec![Integer::from(2), Integer::from(3)]]).unwrap();
        let b = BigIntMatrix::new(vec![vec![Integer::from(5)], vec![Integer::from(7)]]).unwrap();
        let c = a.mul(&b).unwrap();
        assert_eq!(c.rows(), 1);
        assert_eq!(c.cols(), 1);
        assert_eq!(c.get(0, 0).unwrap().to_i32(), Some(31));
    }
}
