//! Matrix operations and utilities

use crate::core::error::{LatticeError, Result};

/// Matrix represented as a vector of vectors (row-major)
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    data: Vec<Vec<i64>>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    /// Create a new matrix from 2D vector
    pub fn new(data: Vec<Vec<i64>>) -> Result<Self> {
        if data.is_empty() {
            return Err(LatticeError::invalid_parameters("Matrix cannot be empty"));
        }
        
        let rows = data.len();
        let cols = data[0].len();
        
        // Verify all rows have the same length
        for (i, row) in data.iter().enumerate() {
            if row.len() != cols {
                return Err(LatticeError::invalid_dimensions(
                    (rows, cols),
                    (i + 1, row.len())
                ));
            }
        }
        
        Ok(Matrix { data, rows, cols })
    }
    
    /// Create a matrix with given dimensions, filled with zeros
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            data: vec![vec![0; cols]; rows],
            rows,
            cols,
        }
    }
    
    /// Create a matrix with given dimensions, filled with ones
    pub fn ones(rows: usize, cols: usize) -> Self {
        Matrix {
            data: vec![vec![1; cols]; rows],
            rows,
            cols,
        }
    }
    
    /// Create an identity matrix
    pub fn identity(n: usize) -> Result<Self> {
        if n == 0 {
            return Err(LatticeError::invalid_parameters("Dimension cannot be zero"));
        }
        
        let mut data = vec![vec![0i64; n]; n];
        for i in 0..n {
            data[i][i] = 1;
        }
        
        Ok(Matrix { data, rows: n, cols: n })
    }
    
    /// Get the number of rows
    pub fn rows(&self) -> usize {
        self.rows
    }
    
    /// Get the number of columns
    pub fn cols(&self) -> usize {
        self.cols
    }
    
    /// Get a reference to a specific element
    pub fn get(&self, row: usize, col: usize) -> Option<&i64> {
        self.data.get(row)?.get(col)
    }
    
    /// Get a mutable reference to a specific element
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut i64> {
        self.data.get_mut(row)?.get_mut(col)
    }
    
    /// Set a specific element
    pub fn set(&mut self, row: usize, col: usize, value: i64) -> Result<()> {
        if row >= self.rows || col >= self.cols {
            return Err(LatticeError::invalid_dimensions(
                (self.rows, self.cols),
                (row + 1, col + 1)
            ));
        }
        
        self.data[row][col] = value;
        Ok(())
    }
    
    /// Get a row as a vector
    pub fn get_row(&self, row: usize) -> Result<Vec<i64>> {
        if row >= self.rows {
            return Err(LatticeError::invalid_dimensions(
                (self.rows, self.cols),
                (row + 1, self.cols)
            ));
        }
        
        Ok(self.data[row].clone())
    }
    
    /// Get a column as a vector
    pub fn get_col(&self, col: usize) -> Result<Vec<i64>> {
        if col >= self.cols {
            return Err(LatticeError::invalid_dimensions(
                (self.rows, self.cols),
                (self.rows, col + 1)
            ));
        }
        
        Ok(self.data.iter().map(|row| row[col]).collect())
    }
    
    /// Transpose the matrix
    pub fn transpose(&self) -> Self {
        let mut data = vec![vec![0i64; self.rows]; self.cols];
        
        for i in 0..self.rows {
            for j in 0..self.cols {
                data[j][i] = self.data[i][j];
            }
        }
        
        Matrix { data, rows: self.cols, cols: self.rows }
    }
    
    /// Matrix addition
    pub fn add(&self, other: &Matrix) -> Result<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(LatticeError::invalid_dimensions(
                (self.rows, self.cols),
                (other.rows, other.cols)
            ));
        }
        
        let mut data = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let mut row = Vec::with_capacity(self.cols);
            for j in 0..self.cols {
                row.push(self.data[i][j] + other.data[i][j]);
            }
            data.push(row);
        }
        
        Ok(Matrix { data, rows: self.rows, cols: self.cols })
    }
    
    /// Matrix subtraction
    pub fn sub(&self, other: &Matrix) -> Result<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(LatticeError::invalid_dimensions(
                (self.rows, self.cols),
                (other.rows, other.cols)
            ));
        }
        
        let mut data = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let mut row = Vec::with_capacity(self.cols);
            for j in 0..self.cols {
                row.push(self.data[i][j] - other.data[i][j]);
            }
            data.push(row);
        }
        
        Ok(Matrix { data, rows: self.rows, cols: self.cols })
    }
    
    /// Matrix multiplication
    pub fn mul(&self, other: &Matrix) -> Result<Self> {
        if self.cols != other.rows {
            return Err(LatticeError::invalid_dimensions(
                (self.rows, self.cols),
                (other.rows, other.cols)
            ));
        }
        
        let mut data = vec![vec![0i64; other.cols]; self.rows];
        
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0i64;
                for k in 0..self.cols {
                    sum = sum.wrapping_add(self.data[i][k].wrapping_mul(other.data[k][j]));
                }
                data[i][j] = sum;
            }
        }
        
        Ok(Matrix { data, rows: self.rows, cols: other.cols })
    }
    
    /// Compute the Gram matrix (B^T * B)
    pub fn gram(&self) -> Result<Self> {
        let transposed = self.transpose();
        self.mul(&transposed)
    }
    
    /// Compute the determinant (only for square matrices)
    pub fn determinant(&self) -> Result<i64> {
        if self.rows != self.cols {
            return Err(LatticeError::invalid_parameters("Determinant only defined for square matrices"));
        }
        
        self.det_recursive()
    }
    
    /// Recursive determinant computation
    fn det_recursive(&self) -> Result<i64> {
        let n = self.rows;
        
        if n == 1 {
            return Ok(self.data[0][0]);
        }
        
        if n == 2 {
            return Ok(self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]);
        }
        
        let mut det = 0i64;
        for j in 0..n {
            let mut minor_data = Vec::with_capacity(n - 1);
            for i in 1..n {
                let mut minor_row = Vec::with_capacity(n - 1);
                for k in 0..n {
                    if k != j {
                        minor_row.push(self.data[i][k]);
                    }
                }
                minor_data.push(minor_row);
            }
            
            let minor = Matrix::new(minor_data)?;
            let cofactor = if j % 2 == 0 { 1 } else { -1 };
            det = det.wrapping_add(cofactor * self.data[0][j] * minor.det_recursive()?);
        }
        
        Ok(det)
    }
    
    /// Compute the Frobenius norm
    pub fn frobenius_norm(&self) -> f64 {
        let mut sum_sq = 0f64;
        for row in &self.data {
            for &val in row {
                sum_sq += (val as f64).powi(2);
            }
        }
        sum_sq.sqrt()
    }
    
    /// Check if matrix is singular
    pub fn is_singular(&self) -> bool {
        self.determinant().unwrap_or(0) == 0
    }
    
    /// Convert to Vec<Vec<i64>>
    pub fn to_vec(&self) -> Vec<Vec<i64>> {
        self.data.clone()
    }
    
    /// Create from Vec<Vec<i64>>
    pub fn from_vec(data: Vec<Vec<i64>>) -> Result<Self> {
        Matrix::new(data)
    }
    
    /// Apply a function to all elements
    pub fn map<F>(&self, f: F) -> Self 
    where
        F: Fn(i64) -> i64,
    {
        let data = self.data.iter()
            .map(|row| row.iter().map(|&val| f(val)).collect())
            .collect();
        Matrix { data, rows: self.rows, cols: self.cols }
    }
    /// Get the dimension of the matrix
    pub fn dimension(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Swap two rows
    pub fn swap_rows(&mut self, i: usize, j: usize) -> Result<()> {
        if i >= self.rows || j >= self.rows {
            return Err(LatticeError::invalid_parameters(
                format!("Row indices {} and {} out of bounds for {} rows", i, j, self.rows)
            ));
        }
        self.data.swap(i, j);
        Ok(())
    }

    /// Get mutable reference to a row
    pub fn get_row_mut(&mut self, row: usize) -> Result<&mut Vec<i64>> {
        if row >= self.rows {
            return Err(LatticeError::invalid_parameters(
                format!("Row index {} out of bounds for {} rows", row, self.rows)
            ));
        }
        Ok(&mut self.data[row])
    }

    /// Convert to flat vector (row-major)
    pub fn to_flat_vec(&self) -> Vec<i64> {
        self.data.iter().flatten().cloned().collect()
    }

    /// Create from flat vector
    pub fn from_flat_vec(data: Vec<i64>, rows: usize, cols: usize) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(LatticeError::invalid_parameters(
                format!("Flat vector length {} doesn't match dimensions {}x{}", data.len(), rows, cols)
            ));
        }
        
        let mut matrix_data = Vec::with_capacity(rows);
        for i in 0..rows {
            let start = i * cols;
            let end = start + cols;
            matrix_data.push(data[start..end].to_vec());
        }
        
        Matrix::new(matrix_data)
    }
    /// Create a submatrix excluding given row and column
    pub fn submatrix(&self, exclude_row: usize, exclude_col: usize) -> Result<Self> {
        if exclude_row >= self.rows || exclude_col >= self.cols {
            return Err(LatticeError::invalid_dimensions(
                (self.rows, self.cols),
                (exclude_row + 1, exclude_col + 1)
            ));
        }
        
        let mut data = Vec::with_capacity(self.rows - 1);
        for i in 0..self.rows {
            if i != exclude_row {
                let mut row = Vec::with_capacity(self.cols - 1);
                for j in 0..self.cols {
                    if j != exclude_col {
                        row.push(self.data[i][j]);
                    }
                }
                data.push(row);
            }
        }
        
        Matrix::new(data)
    }

    /// Remove a row and return it
    pub fn remove_row(&mut self, index: usize) -> Result<Vec<i64>> {
        if index >= self.rows {
            return Err(LatticeError::invalid_parameters(format!(
                "row index {} out of bounds for {} rows",
                index, self.rows
            )));
        }
        self.rows -= 1;
        Ok(self.data.remove(index))
    }

    /// Insert a row at the specified position
    pub fn insert_row(&mut self, index: usize, row: Vec<i64>) -> Result<()> {
        if row.len() != self.cols {
            return Err(LatticeError::invalid_dimensions(
                (self.rows, self.cols),
                (1, row.len()),
            ));
        }
        if index > self.rows {
            return Err(LatticeError::invalid_parameters(format!(
                "row index {} out of bounds for {} rows",
                index, self.rows
            )));
        }
        self.data.insert(index, row);
        self.rows += 1;
        Ok(())
    }
}

impl std::fmt::Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Matrix {}x{}:", self.rows, self.cols)?;
        for row in &self.data {
            writeln!(f, "[{}]", row.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_matrix_creation() {
        let data = vec![vec![1, 2], vec![3, 4]];
        let matrix = Matrix::new(data).unwrap();
        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 2);
    }
    
    #[test]
    fn test_matrix_transpose() {
        let data = vec![vec![1, 2], vec![3, 4]];
        let matrix = Matrix::new(data).unwrap();
        let transposed = matrix.transpose();
        
        assert_eq!(transposed.get(0, 1), Some(&3));
        assert_eq!(transposed.get(1, 0), Some(&2));
    }
    
    #[test]
    fn test_matrix_addition() {
        let data1 = vec![vec![1, 2], vec![3, 4]];
        let data2 = vec![vec![5, 6], vec![7, 8]];
        
        let m1 = Matrix::new(data1).unwrap();
        let m2 = Matrix::new(data2).unwrap();
        let sum = m1.add(&m2).unwrap();
        
        assert_eq!(sum.get(0, 0), Some(&6));
        assert_eq!(sum.get(1, 1), Some(&12));
    }
    
    #[test]
    fn test_matrix_multiplication() {
        let data1 = vec![vec![1, 2], vec![3, 4]];
        let data2 = vec![vec![1, 0], vec![0, 1]];
        
        let m1 = Matrix::new(data1).unwrap();
        let m2 = Matrix::new(data2).unwrap();
        let product = m1.mul(&m2).unwrap();
        
        assert_eq!(product.get(0, 0), Some(&1));
        assert_eq!(product.get(1, 1), Some(&4));
    }
    
    #[test]
    fn test_determinant_2x2() {
        let data = vec![vec![2, 3], vec![1, 4]];
        let matrix = Matrix::new(data).unwrap();
        assert_eq!(matrix.determinant().unwrap(), 5);
    }
    
    #[test]
    fn test_determinant_3x3() {
        let data = vec![vec![1, 2, 3], vec![0, 1, 4], vec![5, 6, 0]];
        let matrix = Matrix::new(data).unwrap();
        assert_eq!(matrix.determinant().unwrap(), 1);
    }
}
