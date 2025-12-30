# Lattice Solver

An almost ready Rust crate implementing lattice reduction algorithms with LLL, BKZ, SVP, and CVP solvers, featuring GPU acceleration and high-precision arithmetic support.

## Features

### Core Algorithms
- **LLL (Lenstra-Lenstra-Lovász) Reduction**: Standard and extended variants including deep insertions and floating-point arithmetic
- **BKZ (Block Korkine-Zolotarev) Reduction**: Progressive BKZ with configurable block sizes and pruning
- **SVP (Shortest Vector Problem)**: Exact and approximation solvers using enumeration, Schnorr-Euchner, and Kannan's algorithm
- **CVP (Closest Vector Problem)**: Babai's algorithms, enumeration-based methods, and embedding techniques

### Performance & Accuracy
- **GPU Acceleration**: Optional GPU-based acceleration for parallel computations
- **High-Precision Arithmetic**: MPFR integration for arbitrary precision computations
- **Parallel Processing**: Multi-threaded operations using Rayon and manual threading
- **Comprehensive Error Handling**: Robust error management for numerical stability

### Input/Output Standards
- **fplll Format Compatibility**: Full support for fplll library input/output formats
- **Multiple Export Formats**: JSON, CSV, and custom formats
- **CLI Interface**: Comprehensive command-line tool for all operations

## Quick Start

### Basic Usage

```rust
use lattice_solver::{Lattice, LLLReducer, LLLParams};

// Create a lattice from matrix
let matrix = vec![
    vec![1, 2, 3],
    vec![4, 5, 6],
    vec![7, 8, 10],
];
let lattice = Lattice::from_matrix(matrix).unwrap();

// Perform LLL reduction
let params = LLLParams::default();
let reducer = LLLReducer::new(params);
let reduced = reducer.reduce(&lattice).unwrap();

println!("Original lattice:\n{}", lattice);
println!("Reduced lattice:\n{}", reduced);
```

### BKZ Reduction

```rust
use lattice_solver::{BKZReducer, BKZParams};

// BKZ with custom block size
let params = BKZParams::new(20); // Block size = 20
let reducer = BKZReducer::with_params(params);
let reduced = reducer.reduce(&lattice).unwrap();
```

### SVP Solving

```rust
use lattice_solver::{SVPSolver, SVPSolverParams, SVPAlgorithm};

// Solve SVP using enumeration
let solver = SVPSolver::with_params(
    SVPSolverParams::with_algorithm(SVPAlgorithm::Enumeration)
);
let result = solver.solve(&lattice).unwrap();

println!("Shortest vector: {}", result.solution);
println!("Norm: {}", result.norm);
```

### CVP Solving

```rust
use lattice_solver::{CVPSolver, CVPSolverParams, CVPAlgorithm};
use lattice_solver::core::types::LatticeVector;

// Solve CVP with target vector
let target = LatticeVector::new(vec![1.5, 2.5, 3.5]);
let solver = CVPSolver::with_params(
    CVPSolverParams::with_algorithm(CVPAlgorithm::BabaiNearestPlane)
);
let result = solver.solve(&lattice, &target).unwrap();

println!("Closest vector: {}", result.closest_vector);
println!("Distance: {}", result.distance);
```

## Command Line Interface

### Installation

```bash
# Build the CLI tool
cargo build --release

# Enable GPU support (optional)
cargo build --release --features gpu

# Enable high-precision arithmetic (optional)
cargo build --release --features high-precision
```

### Basic Commands

```bash
# LLL reduction
./lattice_solver lll --input lattice.mat --output reduced.mat --delta 0.99 --eta 0.51

# BKZ reduction
./lattice_solver bkz --input lattice.mat --beta 20 --progressive

# SVP solving
./lattice_solver svp --input lattice.mat --algorithm enumeration

# CVP solving
./lattice_solver cvp --input lattice.mat --target 1.5,2.5,3.5 --preprocessing

# Benchmark algorithms
./lattice_solver benchmark --algorithms lll,bkz --dimensions 3,4,5 --runs 10

# Generate test lattices
./lattice_solver generate --output-dir test_data --dimensions 2,3,4 --count 5

# Analyze lattice properties
./lattice_solver analyze --input lattice.mat --property all --detailed
```

### Advanced Options

```bash
# High precision arithmetic
./lattice_solver lll --input lattice.mat --high-precision --precision-bits 256

# GPU acceleration
./lattice_solver bkz --input lattice.mat --gpu --beta 30

# Custom output format
./lattice_solver svp --input lattice.mat --format json --output result.json

# Verbose logging
./lattice_solver bkz --input lattice.mat --verbose --log-level debug
```

## Input Format

The crate supports fplll-compatible matrix format:

```
# First line: dimensions (rows cols)
# Following lines: matrix rows (space-separated integers)

3 3
1 2 3
4 5 6
7 8 10
```

### Example Input Files

#### Identity Lattice (2D)
```
2 2
1 0
0 1
```

#### Random Lattice (4D)
```
4 4
15 23 8 12
7 19 31 14
22 6 27 9
11 35 4 18
```

#### Ill-conditioned Lattice
```
3 3
1 1000 1000
0 1 1000
0 0 1
```

## Algorithms Overview

### LLL Reduction

The Lenstra-Lenstra-Lovász algorithm reduces lattice bases while preserving the lattice. Key parameters:

- **δ (delta)**: Reduction parameter, typically 0.99
- **η (eta)**: Approximation parameter, typically 0.51
- **Variants**: Standard, Deep insertions, Recursive, Floating-point

**When to use**: Preprocessing for other algorithms, general lattice reduction

### BKZ Reduction

Block Korkine-Zolotarev provides stronger reduction by processing blocks of vectors:

- **β (block size)**: Typically 10-100, larger blocks give better reduction
- **Progressive BKZ**: Gradually increases block size
- **SVP subroutine**: Uses enumeration to find short vectors in blocks

**When to use**: Strong reduction needed, SVP approximation, preprocessing for CVP

### SVP Solving

Shortest Vector Problem solvers find the non-zero lattice vector with minimal norm:

- **Enumeration**: Exhaustive search, exact solution for small lattices
- **Schnorr-Euchner**: Improved enumeration with better pruning
- **Kannan's Algorithm**: Uses preprocessing and smaller enumeration
- **BKZ Approximation**: Uses BKZ-reduced basis for approximation

**When to use**: Finding shortest vectors, computing lattice constants

### CVP Solving

Closest Vector Problem finds the lattice vector closest to a target:

- **Babai Nearest Plane**: Fast approximation, exact for reduced bases
- **Babai Rounding**: Simple rounding approach
- **Enumeration**: Exact solution for small lattices
- **Embedding**: Transforms CVP to SVP

**When to use**: Lattice decoding, cryptography, optimization

## Performance Optimization

### Memory Usage

```rust
// Efficient matrix operations
let result = matrix_a.mul(&matrix_b)?; // Uses cache-friendly access patterns

// Avoid unnecessary allocations
let gs = lattice.gram_schmidt()?; // Computes in-place when possible
```

### GPU Acceleration

```rust
#[cfg(feature = "gpu")]
{
    use lattice_solver::gpu::GPUAcceleratedOperations;
    
    let gpu_ops = GPUAcceleratedOperations::new().await?;
    let result = gpu_ops.accelerated_lll_reduction(&lattice).await?;
}
```

### High Precision

```rust
// Use high precision for ill-conditioned problems
let params = LLLParams::with_high_precision(256);
let reducer = LLLReducer::with_params(params);
```

## Testing

```bash
# Run all tests
cargo test

# Run specific test categories
cargo test --lib          # Unit tests
cargo test --benches      # Benchmarks
cargo test --release      # Optimized tests

# Test with different features
cargo test --features gpu
cargo test --features high-precision
```

### Benchmarking

```bash
# Run comprehensive benchmarks
cargo bench

# Benchmark specific algorithms
cargo bench --bench lattice_benchmarks -- LLL
cargo bench --bench lattice_benchmarks -- BKZ

# Compare CPU vs GPU
cargo bench --features gpu -- --compare-cpu-gpu
```

## Examples

### Complete SVP Workflow

```rust
use lattice_solver::*;

fn solve_svp_example() -> Result<()> {
    // Load or create lattice
    let lattice = Lattice::load_from_file("example.mat")?;
    
    // Preprocess with LLL
    let lll_reducer = LLLReducer::new();
    let preprocessed = lll_reducer.reduce(&lattice)?;
    
    // Solve SVP
    let svp_solver = SVPSolver::new();
    let result = svp_solver.solve(&preprocessed)?;
    
    println!("Shortest vector found: {}", result.solution);
    println!("Norm: {}", result.norm);
    println!("Enumeration time: {:?}", result.execution_time);
    
    // Save result
    let output = format!(
        "SVP Solution:\nVector: {}\nNorm: {}\nTime: {:?}",
        result.solution, result.norm, result.execution_time
    );
    std::fs::write("svp_result.txt", output)?;
    
    Ok(())
}
```

### CVP with Preprocessing

```rust
fn solve_cvp_with_preprocessing() -> Result<()> {
    let lattice = Lattice::load_from_file("lattice.mat")?;
    let target = LatticeVector::new(vec![1.5, 2.5, 3.5, 4.5]);
    
    // CVP with LLL preprocessing
    let mut cvp_params = CVPSolverParams::default();
    cvp_params.enable_preprocessing = true;
    cvp_params.preprocessing_algorithm = PreprocessingAlgorithm::LLL;
    
    let solver = CVPSolver::with_params(cvp_params);
    let result = solver.solve(&lattice, &target)?;
    
    println!("Target: {}", target);
    println!("Closest vector: {}", result.closest_vector);
    println!("Distance: {}", result.distance);
    
    Ok(())
}
```

### Benchmark Comparison

```rust
fn compare_algorithms() -> Result<()> {
    let lattice = generate_random_lattice(10, 10, Some(42))?;
    
    // LLL
    let start = std::time::Instant::now();
    let lll_reducer = LLLReducer::new();
    let lll_result = lll_reducer.reduce(&lattice)?;
    let lll_time = start.elapsed();
    
    // BKZ
    let start = std::time::Instant::now();
    let bkz_reducer = BKZReducer::with_params(BKZParams::new(20));
    let bkz_result = bkz_reducer.reduce(&lattice)?;
    let bkz_time = start.elapsed();
    
    println!("LLL time: {:?}", lll_time);
    println!("BKZ time: {:?}", bkz_time);
    println!("Speedup: {}", bkz_time.as_secs_f64() / lll_time.as_secs_f64());
    
    Ok(())
}
```

## Configuration Options

### Cargo Features

```toml
[features]
default = ["std"]
std = []
gpu = ["wgpu"]           # GPU acceleration support
high-precision = ["mpfr"] # High-precision arithmetic
parallel = ["rayon", "crossbeam"] # Parallel processing
```

### Environment Variables

```bash
# Set number of threads
export RAYON_NUM_THREADS=8

# Enable GPU debugging
export WGPU_LOG=debug

# Precision settings
export LATTICE_PRECISION_BITS=256
```

## Error Handling

The crate provides comprehensive error handling:

```rust
use lattice_solver::core::error::LatticeError;

match lattice_reduction() {
    Ok(result) => println!("Success: {:?}", result),
    Err(LatticeError::NotFullRank(rank)) => {
        println!("Lattice is not full rank: {}", rank);
    }
    Err(LatticeError::NumericalInstability(msg)) => {
        println!("Numerical issues detected: {}", msg);
        // Consider using higher precision
    }
    Err(e) => println!("Error: {}", e),
}
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes with tests
4. Run benchmarks to ensure performance
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
cargo install cargo-watch cargo-audit cargo-tree

# Watch for changes and run tests
cargo watch -x test

# Audit for security issues
cargo audit

# Check dependency tree
cargo tree
```

## Performance Tips

1. **Use appropriate algorithms**: LLL for preprocessing, BKZ for strong reduction
2. **Choose optimal block sizes**: Larger blocks for BKZ, but watch performance
3. **Enable GPU acceleration**: For large matrices and parallel operations
4. **Use high precision**: For ill-conditioned problems
5. **Preprocess inputs**: LLL reduction often speeds up subsequent operations

## Limitations

- Matrix sizes up to 256x256 for GPU operations
- Exact SVP/CVP solutions practical up to dimension ~20
- High precision arithmetic may be slower than standard floating-point
- Some algorithms require full-rank lattices

## License

MIT License - see [LICENSE](LICENSE) file for details.

## References

- Lenstra, A. K., Lenstra, H. W., & Lovász, L. (1982). Factoring polynomials with rational coefficients
- Schnorr, C. P., & Euchner, M. (1994). Lattice basis reduction: improved practical algorithms
- Kannan, R. (1987). Minkowski's convex body theorem and integer programming
- Gama, N., & Nguyen, P. Q. (2008). Finding short lattice vectors within Mordell's inequality