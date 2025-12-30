//! Benchmarks for lattice reduction algorithms

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use lattice_solver::{
    Lattice, LLLReducer, LLLParams, BKZReducer, BKZParams,
    SVPSolver, SVPSolverParams, SVPAlgorithm,
    CVPSolver, CVPSolverParams, CVPAlgorithm,
    utils::*,
};

/// Generate test lattice with specific characteristics
fn generate_test_lattice(n: usize, ill_conditioned: bool) -> Lattice {
    if ill_conditioned {
        // Create ill-conditioned lattice for stress testing
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = vec![0i64; n];
            for j in 0..n {
                if i == j {
                    row[j] = 1;
                } else if i < j {
                    row[j] = (rand::random::<i64>() % 100) + 1000; // Large off-diagonal
                }
            }
            data.push(row);
        }
        Lattice::from_matrix(data).unwrap()
    } else {
        // Create well-conditioned lattice
        generate_random_lattice(n, n, Some(42)).unwrap()
    }
}

fn bench_lll_reduction(c: &mut Criterion) {
    let mut group = c.benchmark_group("LLL Reduction");
    
    for size in [3, 5, 10, 20, 30].iter() {
        group.bench_with_input(
            BenchmarkId::new("LLL", size),
            size,
            |b, &size| {
                let lattice = generate_test_lattice(size, false);
                let reducer = LLLReducer::new();
                
                b.iter(|| {
                    black_box(reducer.reduce(black_box(&lattice)).unwrap())
                });
            },
        );
    }
    
    group.finish();
}

fn bench_bkz_reduction(c: &mut Criterion) {
    let mut group = c.benchmark_group("BKZ Reduction");
    
    for size in [5, 10, 15, 20].iter() {
        group.bench_with_input(
            BenchmarkId::new("BKZ", size),
            size,
            |b, &size| {
                let lattice = generate_test_lattice(size, false);
                let params = BKZParams::new(size.min(20));
                let reducer = BKZReducer::with_params(params);
                
                b.iter(|| {
                    black_box(reducer.reduce(black_box(&lattice)).unwrap())
                });
            },
        );
    }
    
    group.finish();
}

fn bench_svp_solving(c: &mut Criterion) {
    let mut group = c.benchmark_group("SVP Solving");
    
    for size in [3, 5, 8, 10].iter() {
        group.bench_with_input(
            BenchmarkId::new("SVP Enumeration", size),
            size,
            |b, &size| {
                let lattice = generate_test_lattice(size, false);
                let solver = SVPSolver::with_params(
                    SVPSolverParams::with_algorithm(SVPAlgorithm::Enumeration)
                );
                
                b.iter(|| {
                    black_box(solver.solve(black_box(&lattice)).unwrap())
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("SVP BKZ Approximation", size),
            size,
            |b, &size| {
                let lattice = generate_test_lattice(size, false);
                let solver = SVPSolver::with_params(
                    SVPSolverParams::with_algorithm(SVPAlgorithm::BKZApproximation)
                );
                
                b.iter(|| {
                    black_box(solver.solve(black_box(&lattice)).unwrap())
                });
            },
        );
    }
    
    group.finish();
}

fn bench_cvp_solving(c: &mut Criterion) {
    let mut group = c.benchmark_group("CVP Solving");
    
    for size in [3, 5, 8, 10].iter() {
        group.bench_with_input(
            BenchmarkId::new("CVP Babai Nearest Plane", size),
            size,
            |b, &size| {
                let lattice = generate_test_lattice(size, false);
                let target = LatticeVector::zeros(size);
                let solver = CVPSolver::with_params(
                    CVPSolverParams::with_algorithm(CVPAlgorithm::BabaiNearestPlane)
                );
                
                b.iter(|| {
                    black_box(solver.solve(black_box(&lattice), black_box(&target)).unwrap())
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("CVP Rounding", size),
            size,
            |b, &size| {
                let lattice = generate_test_lattice(size, false);
                let target = LatticeVector::zeros(size);
                let solver = CVPSolver::with_params(
                    CVPSolverParams::with_algorithm(CVPAlgorithm::BabaiRounding)
                );
                
                b.iter(|| {
                    black_box(solver.solve(black_box(&lattice), black_box(&target)).unwrap())
                });
            },
        );
    }
    
    group.finish();
}

fn bench_matrix_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix Operations");
    
    for size in [10, 50, 100, 200].iter() {
        let matrix_a = Matrix::new(
            (0..*size)
                .map(|i| {
                    (0..*size)
                        .map(|j| rand::random::<i64>() % 100)
                        .collect()
                })
                .collect()
        ).unwrap();
        
        let matrix_b = Matrix::new(
            (0..*size)
                .map(|i| {
                    (0..*size)
                        .map(|j| rand::random::<i64>() % 100)
                        .collect()
                })
                .collect()
        ).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("Matrix Multiplication", size),
            size,
            |b, &size| {
                b.iter(|| {
                    black_box(matrix_a.mul(black_box(&matrix_b)).unwrap())
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("Gram-Schmidt", size),
            size,
            |b, &size| {
                let lattice = generate_test_lattice(size, false);
                b.iter(|| {
                    black_box(lattice.gram_schmidt().unwrap())
                });
            },
        );
    }
    
    group.finish();
}

fn bench_ill_conditioned_lattices(c: &mut Criterion) {
    let mut group = c.benchmark_group("Ill-conditioned Lattices");
    
    for size in [5, 10, 15].iter() {
        group.bench_with_input(
            BenchmarkId::new("LLL Ill-conditioned", size),
            size,
            |b, &size| {
                let lattice = generate_test_lattice(size, true);
                let reducer = LLLReducer::new();
                
                b.iter(|| {
                    black_box(reducer.reduce(black_box(&lattice)).unwrap())
                });
            },
        );
    }
    
    group.finish();
}

fn bench_algorithm_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Algorithm Comparison");
    
    let sizes = [5, 8, 10];
    
    for &size in &sizes {
        let lattice = generate_test_lattice(size, false);
        
        // LLL vs BKZ comparison
        group.bench_with_input(
            BenchmarkId::new("LLL vs BKZ", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    // LLL
                    let lll_reducer = LLLReducer::new();
                    let lll_result = black_box(lll_reducer.reduce(black_box(&lattice)).unwrap());
                    
                    // BKZ
                    let bkz_reducer = BKZReducer::with_params(BKZParams::new(size.min(15)));
                    let bkz_result = black_box(bkz_reducer.reduce(black_box(&lattice)).unwrap());
                    
                    (lll_result, bkz_result)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_precision_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Precision Comparison");
    
    for precision_bits in [64, 128, 256, 512].iter() {
        group.bench_with_input(
            BenchmarkId::new("LLL Precision", precision_bits),
            precision_bits,
            |b, &bits| {
                let lattice = generate_test_lattice(8, false);
                let mut params = LLLParams::default();
                params.precision = crate::core::types::PrecisionType::Arbitrary { bits };
                let reducer = LLLReducer::with_params(params);
                
                b.iter(|| {
                    black_box(reducer.reduce(black_box(&lattice)).unwrap())
                });
            },
        );
    }
    
    group.finish();
}

// Memory usage benchmarks
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Usage");
    
    for size in [50, 100, 200].iter() {
        group.bench_with_input(
            BenchmarkId::new("Large Matrix Operations", size),
            size,
            |b, &size| {
                let lattice = generate_test_lattice(size, false);
                
                b.iter(|| {
                    // Measure memory usage during various operations
                    let gs = black_box(lattice.gram_schmidt().unwrap());
                    let det = black_box(lattice.basis().determinant().unwrap());
                    let sv = black_box(lattice.shortest_vector().unwrap());
                    
                    (gs, det, sv)
                });
            },
        );
    }
    
    group.finish();
}

// Parallel processing benchmarks (if rayon is available)
#[cfg(feature = "parallel")]
fn bench_parallel_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Parallel Operations");
    
    for size in [20, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("Parallel Matrix Operations", size),
            size,
            |b, &size| {
                let matrices: Vec<_> = (0..4)
                    .map(|_| generate_test_lattice(size, false))
                    .collect();
                
                b.iter(|| {
                    use rayon::prelude::*;
                    
                    black_box(matrices.par_iter()
                        .map(|lattice| lattice.gram_schmidt().unwrap())
                        .collect::<Vec<_>>())
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_configuration() -> Criterion {
    Criterion::default()
        .sample_size(10)  // Reduce sample size for faster benchmarks
        .warm_up_time(std::time::Duration::from_millis(100))
        .measurement_time(std::time::Duration::from_secs(2))
        .confidence_level(0.95)
}

criterion_group!(
    benches,
    bench_lll_reduction,
    bench_bkz_reduction,
    bench_svp_solving,
    bench_cvp_solving,
    bench_matrix_operations,
    bench_ill_conditioned_lattices,
    bench_algorithm_comparison,
    bench_precision_comparison,
    bench_memory_usage
);

#[cfg(feature = "parallel")]
criterion_group!(
    parallel_benches,
    bench_parallel_operations
);

criterion_main!(benches);

#[cfg(feature = "parallel")]
criterion_main!(benches, parallel_benches);