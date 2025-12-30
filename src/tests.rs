//! Unit tests for lattice reduction algorithms
// In tests/integration_tests.rs
use lattice_solver::*;
use lattice_solver::features;

#[tokio::test]
async fn test_end_to_end_gpu_pipeline() {
    if !features::gpu_enabled() {
        eprintln!("GPU feature not enabled, skipping test");
        return;
    }
    
    let gpu_ops = GPUAcceleratedOperations::new().await.unwrap();
    let lattice = generate_random_lattice(50, 50, Some(42)).unwrap();
    
    // Test GPU-accelerated LLL
    let start = std::time::Instant::now();
    let reduced = gpu_ops.accelerated_lll_reduction(&lattice).await.unwrap();
    let gpu_time = start.elapsed();
    
    // Compare with CPU
    let start = std::time::Instant::now();
    let reducer = LLLReducer::new();
    let cpu_reduced = reducer.reduce(&lattice).unwrap();
    let cpu_time = start.elapsed();
    
    // Verify correctness
    assert_eq!(reduced.dimension(), cpu_reduced.dimension());
    assert!((reduced.determinant().unwrap() - cpu_reduced.determinant().unwrap()).abs() < 1e-6);
    
    println!("GPU: {:?}, CPU: {:?}, Speedup: {:.2}x", gpu_time, cpu_time, 
             cpu_time.as_secs_f64() / gpu_time.as_secs_f64());
}

#[test]
fn test_precision_consistency() {
    let lattice = generate_random_lattice(10, 10, Some(123)).unwrap();
    
    // Test with standard precision
    let reducer_std = LLLReducer::new();
    let reduced_std = reducer_std.reduce(&lattice).unwrap();
    let sv_std = reduced_std.shortest_vector().unwrap().norm();
    
    // Test with high precision
    if features::high_precision_enabled() {
        let mut params = LLLParams::default();
        params.precision = PrecisionType::Arbitrary { bits: 256 };
        let reducer_hp = LLLReducer::with_params(params);
        let reduced_hp = reducer_hp.reduce(&lattice).unwrap();
        let sv_hp = reduced_hp.shortest_vector().unwrap().norm();
        
        // Results should be very close
        assert!((sv_std - sv_hp).abs() < 1e-12);
    }
}
#[cfg(test)]
mod core_tests {
    use super::*;
    use lattice_solver::core::*;
    
    #[test]
    fn test_lattice_creation_and_properties() {
        // Test identity matrix
        let data = vec![vec![1, 0], vec![0, 1]];
        let lattice = Lattice::from_matrix(data).unwrap();
        
        assert!(lattice.is_full_rank());
        assert_eq!(lattice.rank(), 2);
        assert_eq!(lattice.ambient_dimension(), 2);
        
        let det = lattice.determinant().unwrap();
        assert!((det - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_invalid_lattice_creation() {
        // Empty matrix should fail
        let empty_data: Vec<Vec<i64>> = vec![];
        assert!(Lattice::from_matrix(empty_data).is_err());
        
        // Non-rectangular matrix should fail
        let irregular_data = vec![vec![1, 2, 3], vec![4, 5]];
        assert!(Lattice::from_matrix(irregular_data).is_err());
    }
    
    #[test]
    fn test_gram_schmidt_orthogonalization() {
        let data = vec![vec![2, 1], vec![1, 2]];
        let lattice = Lattice::from_matrix(data).unwrap();
        
        let gs = lattice.gram_schmidt().unwrap();
        assert_eq!(gs.b_star.len(), 2);
        assert_eq!(gs.mu.len(), 2);
        assert_eq!(gs.norm_squared.len(), 2);
        
        // Check orthogonality (simplified)
        let dot_product = gs.b_star[0].dot(&gs.b_star[1]).unwrap();
        assert!(dot_product.abs() < 1e-10); // Should be close to 0
    }
    
    #[test]
    fn test_matrix_operations() {
        let data1 = vec![vec![1, 2], vec![3, 4]];
        let data2 = vec![vec![5, 6], vec![7, 8]];
        
        let matrix1 = Matrix::new(data1).unwrap();
        let matrix2 = Matrix::new(data2).unwrap();
        
        // Addition
        let sum = matrix1.add(&matrix2).unwrap();
        assert_eq!(sum.get(0, 0), Some(&6));
        assert_eq!(sum.get(1, 1), Some(&12));
        
        // Multiplication
        let product = matrix1.mul(&matrix2).unwrap();
        assert_eq!(product.get(0, 0), Some(&19));
        assert_eq!(product.get(1, 1), Some(&52));
        
        // Determinant
        let det = matrix1.determinant().unwrap();
        assert_eq!(det, -2);
    }
    
    #[test]
    fn test_fplll_format_io() {
        let original_data = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let lattice = Lattice::from_matrix(original_data).unwrap();
        
        // Convert to fplll format
        let format_string = lattice.to_fplll_format();
        
        // Parse back from fplll format
        let loaded_lattice = Lattice::from_fplll_format(&format_string).unwrap();
        
        assert_eq!(lattice.dimension(), loaded_lattice.dimension());
        assert_eq!(lattice.rank(), loaded_lattice.rank());
    }
}

#[cfg(test)]
mod lll_tests {
    use super::*;
    use lattice_solver::lll::*;
    
    #[test]
    fn test_lll_params_validation() {
        // Valid parameters
        let valid_params = LLLParams::new(0.99, 0.51);
        assert!(valid_params.validate().is_ok());
        
        // Invalid delta
        let invalid_delta = LLLParams::new(0.4, 0.51);
        assert!(invalid_delta.validate().is_err());
        
        // Invalid eta
        let invalid_eta = LLLParams::new(0.99, 0.9);
        assert!(invalid_eta.validate().is_err());
    }
    
    #[test]
    fn test_lll_reduction_2d() {
        // Simple 2D lattice
        let data = vec![vec![1, 1], vec![1, 0]];
        let lattice = Lattice::from_matrix(data).unwrap();
        
        let reducer = LLLReducer::new();
        let reduced = reducer.reduce(&lattice).unwrap();
        
        assert!(reduced.is_reduced());
        assert_eq!(reduced.dimension(), lattice.dimension());
        
        // Verify reduction improved the basis
        let original_sv = lattice.shortest_vector().unwrap().norm();
        let reduced_sv = reduced.shortest_vector().unwrap().norm();
        
        // Shortest vector should not be longer after reduction
        assert!(reduced_sv <= original_sv + 1e-10);
    }
    
    #[test]
    fn test_lll_reduction_3d() {
        // 3D lattice with some structure
        let data = vec![
            vec![3, 1, 1],
            vec![1, 3, 1],
            vec![1, 1, 3],
        ];
        let lattice = Lattice::from_matrix(data).unwrap();
        
        let reducer = LLLReducer::new();
        let reduced = reducer.reduce(&lattice).unwrap();
        
        assert!(reduced.is_reduced());
        
        // Test extended LLL variants
        let variants = vec![
            LLLVariant::Standard,
            LLLVariant::Deep,
            LLLVariant::Recursive,
            LLLVariant::FloatingPoint,
        ];
        
        for variant in variants {
            let reducer = ExtendedLLLReducer::new(LLLParams::default(), variant);
            let reduced_variant = reducer.reduce_with_variant(&lattice).unwrap();
            assert!(reduced_variant.is_reduced());
        }
    }
    
    #[test]
    fn test_lll_ill_conditioned() {
        // Ill-conditioned lattice (close to singular)
        let data = vec![
            vec![1000000, 1, 0],
            vec![1, 1000000, 1],
            vec![0, 1, 1000000],
        ];
        let lattice = Lattice::from_matrix(data).unwrap();
        
        let reducer = LLLReducer::new();
        let reduced = reducer.reduce(&lattice).unwrap();
        
        assert!(reduced.is_reduced());
        assert!(reduced.determinant().unwrap() > 0.0);
    }
}

#[cfg(test)]
mod bkz_tests {
    use super::*;
    use lattice_solver::bkz::*;
    
    #[test]
    fn test_bkz_params_validation() {
        // Valid parameters
        let valid_params = BKZParams::new(20);
        assert!(valid_params.validate().is_ok());
        
        // Invalid block size
        let invalid_params = BKZParams::new(1);
        assert!(invalid_params.validate().is_err());
    }
    
    #[test]
    fn test_bkz_reduction() {
        let data = vec![vec![5, 3], vec>[2, 1]];
        let lattice = Lattice::from_matrix(data).unwrap();
        
        let params = BKZParams::new(3);
        let reducer = BKZReducer::with_params(params);
        let reduced = reducer.reduce(&lattice).unwrap();
        
        assert!(reduced.is_reduced());
        assert_eq!(reduced.dimension(), lattice.dimension());
    }
    
    #[test]
    fn test_progressive_bkz() {
        let data = vec![
            vec![10, 5, 2],
            vec>[5, 10, 1],
            vec>[2, 1, 10],
        ];
        let lattice = Lattice::from_matrix(data).unwrap();
        
        let reducer = ProgressiveBKZReducer::new(BKZParams::new(10));
        let reduced = reducer.reduce(&lattice).unwrap();
        
        assert!(reduced.is_reduced());
        
        // Check that schedule was generated
        let schedule = ProgressiveBKZReducer::generate_schedule(10);
        assert!(!schedule.is_empty());
        assert_eq!(schedule.last(), Some(&10));
    }
    
    #[test]
    fn test_optimized_bkz() {
        let data = vec![
            vec![7, 3, 1],
            vec>[3, 8, 2],
            vec>[1, 2, 9],
        ];
        let lattice = Lattice::from_matrix(data).unwrap();
        
        let reducer = OptimizedBKZReducer::new(BKZParams::default());
        let reduced = reducer.reduce(&lattice).unwrap();
        
        assert!(reduced.is_reduced());
    }
}

#[cfg(test)]
mod svp_tests {
    use super::*;
    use lattice_solver::svp::*;
    
    #[test]
    fn test_svp_params_validation() {
        let valid_params = SVPSolverParams::default();
        assert!(valid_params.validate().is_ok());
        
        let invalid_params = SVPSolverParams {
            precision_bits: 16,
            ..Default::default()
        };
        assert!(invalid_params.validate().is_err());
    }
    
    #[test]
    fn test_svp_enumeration() {
        // Simple diagonal lattice
        let data = vec![vec![3, 0], vec>[0, 4]];
        let lattice = Lattice::from_matrix(data).unwrap();
        
        let solver = SVPSolver::with_params(
            SVPSolverParams::with_algorithm(SVPAlgorithm::Enumeration)
        );
        let result = solver.solve(&lattice).unwrap();
        
        assert!(result.found);
        assert!(result.norm > 0.0);
        assert!(result.points_enumerated > 0);
        
        // For diagonal lattice [3,0; 0,4], shortest vector should have norm 3
        let expected_norm = 3.0;
        assert!((result.norm - expected_norm).abs() < 1e-10);
    }
    
    #[test]
    fn test_svp_random_sampling() {
        let data = vec![vec![1, 0], vec>[0, 1]];
        let lattice = Lattice::from_matrix(data).unwrap();
        
        let solver = SVPSolver::with_params(
            SVPSolverParams::with_algorithm(SVPAlgorithm::RandomSampling)
        );
        let result = solver.solve(&lattice).unwrap();
        
        assert!(result.found);
        // For standard basis, shortest vector should be unit vector
        assert!((result.norm - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_svp_bkz_approximation() {
        let data = vec![vec>[5, 3], vec>[2, 1]];
        let lattice = Lattice::from_matrix(data).unwrap();
        
        let solver = SVPSolver::with_params(
            SVPSolverParams::with_algorithm(SVPAlgorithm::BKZApproximation)
        );
        let result = solver.solve(&lattice).unwrap();
        
        assert!(result.found);
        assert!(result.norm > 0.0);
        assert!(result.points_enumerated >= 1);
    }
    
    #[test]
    fn test_advanced_svp_solver() {
        let data = vec![vec>[1, 0], vec>[0, 1]];
        let lattice = Lattice::from_matrix(data).unwrap();
        
        let solver = AdvancedSVPSolver::new(SVPSolverParams::default());
        let result = solver.solve_automatic(&lattice).unwrap();
        
        assert!(result.found);
        assert!(result.norm > 0.0);
    }
}

#[cfg(test)]
mod cvp_tests {
    use super::*;
    use lattice_solver::cvp::*;
    
    #[test]
    fn test_cvp_babai_nearest_plane() {
        let data = vec![vec>[2, 0], vec>[0, 3]];
        let lattice = Lattice::from_matrix(data).unwrap();
        
        let target = LatticeVector::new(vec![1.5, 2.5]);
        
        let solver = CVPSolver::with_params(
            CVPSolverParams::with_algorithm(CVPAlgorithm::BabaiNearestPlane)
        );
        let result = solver.solve(&lattice, &target).unwrap();
        
        assert!(result.found);
        assert!(result.distance >= 0.0);
        
        // For diagonal lattice, closest vector should be [2, 3] or similar
        assert!(result.distance < target.norm()); // Should be closer than origin
    }
    
    #[test]
    fn test_cvp_babai_rounding() {
        let data = vec![vec>[1, 0], vec>[0, 1]];
        let lattice = Lattice::from_matrix(data).unwrap();
        
        let target = LatticeVector::new(vec![2.3, 1.7]);
        
        let solver = CVPSolver::with_params(
            CVPSolverParams::with_algorithm(CVPAlgorithm::BabaiRounding)
        );
        let result = solver.solve(&lattice, &target).unwrap();
        
        assert!(result.found);
        assert_eq!(result.coefficient_vector, vec![2, 2]); // Should round to [2,2]
    }
    
    #[test]
    fn test_cvp_with_preprocessing() {
        let data = vec![vec>[5, 3], vec>[2, 1]];
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
    
    #[test]
    fn test_advanced_cvp_solver() {
        let data = vec![vec>[1, 0], vec>[0, 1]];
        let lattice = Lattice::from_matrix(data).unwrap();
        
        let target = LatticeVector::new(vec![1.5, 1.5]);
        
        let solver = AdvancedCVPSolver::new(CVPSolverParams::default());
        let result = solver.solve_adaptive(&lattice, &target).unwrap();
        
        assert!(result.found);
        assert!((result.distance - 0.70710678118).abs() < 1e-10); // sqrt(2)/2
    }
}

#[cfg(test)]
mod utils_tests {
    use super::*;
    use lattice_solver::utils::*;
    
    #[test]
    fn test_random_lattice_generation() {
        let lattice = generate_random_lattice(3, 3, Some(42)).unwrap();
        assert_eq!(lattice.dimension(), (3, 3));
        assert!(lattice.is_full_rank());
    }
    
    #[test]
    fn test_well_conditioned_lattice() {
        let lattice = create_well_conditioned_lattice(4).unwrap();
        assert_eq!(lattice.dimension(), (4, 4));
        assert!(lattice.is_full_rank());
        
        let basis = lattice.basis();
        let det = basis.determinant().unwrap();
        assert!(det.abs() > 0);
    }
    
    #[test]
    fn test_matrix_rank_computation() {
        let full_rank_data = vec![vec>[1, 2], vec>[3, 4]];
        let matrix = Matrix::new(full_rank_data).unwrap();
        let rank = matrix_rank(&matrix).unwrap();
        assert_eq!(rank, 2);
        
        let singular_data = vec![vec>[1, 2], vec>[2, 4]]; // Rank 1
        let matrix = Matrix::new(singular_data).unwrap();
        let rank = matrix_rank(&matrix).unwrap();
        assert_eq!(rank, 1);
    }
    
    #[test]
    fn test_vector_operations() {
        let v1 = LatticeVector::new(vec![1.0, 2.0, 3.0]);
        let v2 = LatticeVector::new(vec![4.0, 5.0, 6.0]);
        
        // Dot product
        let dot = v1.dot(&v2).unwrap();
        assert_eq!(dot, 32.0);
        
        // Addition
        let sum = v1.add(&v2).unwrap();
        assert_eq!(sum.get(0), Some(&5.0));
        
        // Subtraction
        let diff = v2.sub(&v1).unwrap();
        assert_eq!(diff.get(0), Some(&3.0));
        
        // Scalar multiplication
        let scaled = v1.scalar_mul(2.0);
        assert_eq!(scaled.get(0), Some(&2.0));
        
        // Normalization
        let normalized = normalize(&v1);
        assert!((normalized.norm() - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_lattice_statistics() {
        let v1 = LatticeVector::new(vec![1.0, 0.0]);
        let v2 = LatticeVector::new(vec>[0.0, 1.0]);
        let v3 = LatticeVector::new(vec![1.0, 1.0]);
        
        let stats = lattice_statistics(&[v1, v2, v3]);
        assert_eq!(stats.count, 3);
        assert!(stats.min_norm > 0.0);
        assert!(stats.max_norm > stats.min_norm);
    }
    
    #[test]
    fn test_file_io_operations() {
        let lattice = create_well_conditioned_lattice(2).unwrap();
        
        // Custom format
        let filename = "test_lattice.custom";
        save_lattice_custom(&lattice, filename).unwrap();
        let loaded = load_lattice_custom(filename).unwrap();
        assert_eq!(lattice.dimension(), loaded.dimension());
        
        // CSV export
        let csv_filename = "test_lattice.csv";
        export_lattice_csv(&lattice, csv_filename).unwrap();
        
        // Clean up
        let _ = std::fs::remove_file(filename);
        let _ = std::fs::remove_file(csv_filename);
    }
    
    #[test]
    fn test_profiling() {
        let mut profiler = Profiler::new();
        
        {
            let _timer = profiler.start_operation("test_op".to_string());
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        
        assert!(profiler.total_time().as_millis() >= 10);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use lattice_solver::{*, utils::*};
    
    #[test]
    fn test_complete_workflow() {
        // Generate test lattice
        let lattice = generate_random_lattice(4, 4, Some(42)).unwrap();
        
        // LLL reduction
        let lll_reducer = LLLReducer::new();
        let lll_reduced = lll_reducer.reduce(&lattice).unwrap();
        
        // BKZ reduction
        let bkz_reducer = BKZReducer::with_params(BKZParams::new(4));
        let bkz_reduced = bkz_reducer.reduce(&lattice).unwrap();
        
        // SVP solving
        let svp_solver = SVPSolver::new();
        let svp_result = svp_solver.solve(&lattice).unwrap();
        
        // CVP solving
        let target = LatticeVector::zeros(4);
        let cvp_solver = CVPSolver::new();
        let cvp_result = cvp_solver.solve(&lattice, &target).unwrap();
        
        // Verify all operations succeeded
        assert!(lll_reduced.is_reduced());
        assert!(bkz_reduced.is_reduced());
        assert!(svp_result.found);
        assert!(cvp_result.found);
        
        // Quality checks
        let original_quality = lattice.quality_metrics().unwrap();
        let lll_quality = lll_reduced.quality_metrics().unwrap();
        let bkz_quality = bkz_reduced.quality_metrics().unwrap();
        
        assert!(lll_quality.conditioning_number <= original_quality.conditioning_number * 1.1); // LLL should not worsen much
        assert!(bkz_quality.conditioning_number <= original_quality.conditioning_number); // BKZ should improve
    }
    
    #[test]
    fn test_ill_conditioned_problem() {
        // Create ill-conditioned lattice
        let mut data = Vec::with_capacity(3);
        for i in 0..3 {
            let mut row = Vec::with_capacity(3);
            for j in 0..3 {
                row.push(if i == j {
                    1000000
                } else {
                    rand::random::<i64>() % 1000 + 1000
                });
            }
            data.push(row);
        }
        
        let lattice = Lattice::from_matrix(data).unwrap();
        
        // Test with different precision levels
        let standard_reducer = LLLReducer::new();
        let standard_result = standard_reducer.reduce(&lattice);
        
        // Should succeed even with standard precision
        assert!(standard_result.is_ok());
        
        let reduced = standard_result.unwrap();
        assert!(reduced.is_reduced());
        assert!(reduced.determinant().unwrap() > 0.0);
    }
    
    #[test]
    fn test_performance_regression() {
        let lattice = generate_random_lattice(6, 6, Some(123)).unwrap();
        
        // Measure LLL performance
        let start = std::time::Instant::now();
        let lll_reducer = LLLReducer::new();
        let _lll_result = lll_reducer.reduce(&lattice).unwrap();
        let lll_time = start.elapsed();
        
        // Measure BKZ performance (smaller block size for speed)
        let start = std::time::Instant::now();
        let bkz_reducer = BKZReducer::with_params(BKZParams::new(4));
        let _bkz_result = bkz_reducer.reduce(&lattice).unwrap();
        let bkz_time = start.elapsed();
        
        // Log performance (for manual verification)
        println!("6x6 lattice: LLL {:?}, BKZ {:?}", lll_time, bkz_time);
        
        // Basic sanity checks
        assert!(lll_time.as_millis() < 1000); // Should complete within 1 second
        assert!(bkz_time.as_millis() < 5000); // BKZ can be slower
    }
}

// Test utilities
fn generate_test_lattice_with_seed(dimension: usize, seed: u64) -> Lattice {
    generate_random_lattice(dimension, dimension, Some(seed)).unwrap()
}

fn assert_lattice_properties(lattice: &Lattice) {
    assert!(lattice.rank() > 0);
    assert!(lattice.ambient_dimension() > 0);
    assert!(lattice.rank() <= lattice.ambient_dimension());
    
    if lattice.is_full_rank() {
        let det = lattice.determinant().unwrap();
        assert!(det != 0.0);
    }
}

fn assert_reduction_quality(original: &Lattice, reduced: &Lattice) {
    assert_eq!(original.dimension(), reduced.dimension());
    assert!(reduced.is_reduced());
    
    let original_sv = original.shortest_vector().unwrap().norm();
    let reduced_sv = reduced.shortest_vector().unwrap().norm();
    
    // Reduction should not make shortest vector significantly longer
    assert!(reduced_sv <= original_sv * 1.01);
    
    let original_condition = original.conditioning_number().unwrap_or(f64::INFINITY);
    let reduced_condition = reduced.conditioning_number().unwrap_or(f64::INFINITY);
    
    // BKZ should improve conditioning, LLL should not worsen it much
    assert!(reduced_condition <= original_condition * 2.0);
}