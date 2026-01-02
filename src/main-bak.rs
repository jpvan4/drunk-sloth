//! Command-line interface for lattice reduction algorithms

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use std::time::Instant;

use lattice_solver::{
    Lattice, LLLReducer, LLLParams, BKZReducer, BKZParams,
    SVPSolver, SVPSolverParams, SVPAlgorithm,
    CVPSolver, CVPSolverParams, CVPAlgorithm,
    core::types::{QualityMetrics, BenchmarkResult, LatticeVector, PrecisionType},
    utils::*,
};
use lattice_solver::gpu::GPUAcceleratedOperations; // Uncomment if GPUAcceleratedOperations is in gpu module
use lattice_solver::features;


/// Lattice reduction algorithms CLI
#[derive(Parser, Debug)]
#[clap(name = "lattice_solver")]
#[clap(about = "Production-ready lattice reduction crate with LLL, BKZ, SVP, and CVP solvers")]
#[clap(version = "1.0.0")]
struct Args {
    /// Input file containing lattice basis (fplll format)
    #[clap(short, long)]
    input: Option<PathBuf>,
    
    /// Output file for results
    #[clap(short, long)]
    output: Option<PathBuf>,
    
    /// Algorithm to use
    #[clap(subcommand)]
    command: Commands,
    
    /// Enable verbose logging
    #[clap(short, long)]
    verbose: bool,
    
    /// Set logging level (error, warn, info, debug, trace)
    #[clap(long, default_value = "info")]
    log_level: String,
    
    /// Number of threads for parallel processing
    #[clap(long)]
    threads: Option<usize>,
    
    /// Output format (plain, json, csv)
    #[clap(long, value_enum, default_value = "plain")]
    format: OutputFormat,
    
    /// Enable GPU acceleration if available
    #[clap(long)]
    gpu: bool,
    
    /// Use high precision arithmetic
    #[clap(long)]
    high_precision: bool,
    
    /// Precision in bits (when using high precision)
    #[clap(long, default_value = "256")]
    precision_bits: usize,
}

#[derive(Subcommand, Debug, Clone)]
enum Commands {
    /// LLL lattice reduction
    LLL {
        /// LLL delta parameter (0.5 < delta < 1.0)
        #[clap(long, default_value = "0.99")]
        delta: f64,
        
        /// LLL eta parameter (0.5 < eta < sqrt(delta))
        #[clap(long, default_value = "0.51")]
        eta: f64,
        
        /// LLL variant (standard, deep, recursive, floating)
        #[clap(long, value_enum, default_value = "standard")]
        variant: LLLVariantCli,
    },
    
    /// BKZ lattice reduction
    BKZ {
        /// BKZ block size
        #[clap(long, default_value = "20")]
        beta: usize,
        
        /// Maximum BKZ rounds
        #[clap(long, default_value = "10")]
        max_rounds: usize,
        
        /// Enable progressive BKZ
        #[clap(long)]
        progressive: bool,
        
        /// Enable pruning in SVP enumeration
        #[clap(long)]
        pruning: bool,
    },
    
    /// Solve Shortest Vector Problem (SVP)
    SVP {
        /// SVP algorithm (enumeration, schnorr_euchner, kannan, random_sampling, bkz_approximation)
        #[clap(long, value_enum, default_value = "enumeration")]
        algorithm: SVPAlgorithmCli,
        
        /// Maximum enumeration radius
        #[clap(long, default_value = "1e6")]
        max_radius: f64,
        
        /// Enable pruning
        #[clap(long)]
        pruning: bool,
    },
    
    /// Solve Closest Vector Problem (CVP)
    CVP {
        /// Target vector (comma-separated values)
        #[clap(long)]
        target: Option<Vec<f64>>,
        
        /// CVP algorithm (babai_nearest_plane, babai_rounding, enumeration, schnorr_euchner, vaidya, kannan, embedding)
        #[clap(long, value_enum, default_value = "babai_nearest_plane")]
        algorithm: CVPAlgorithmCli,
        
        /// Enable preprocessing with lattice reduction
        #[clap(long)]
        preprocessing: bool,
        
        /// Preprocessing algorithm (none, lll, bkz, deep_lll, combined)
        #[clap(long, value_enum, default_value = "lll")]
        preprocessing_algo: PreprocessingAlgorithm,
    },
    
    /// Benchmark algorithms
    Benchmark {
        /// Algorithms to benchmark
        #[clap(long, value_enum, value_delimiter = ',')]
        algorithms: Vec<BenchmarkAlgorithm>,
        
        /// Problem dimensions to test
        #[clap(long, value_delimiter = ',', default_values = &["2", "3", "4", "5"])]
        dimensions: Vec<usize>,
        
        /// Number of test runs per dimension
        #[clap(long, default_value = "10")]
        runs: usize,
        
        /// Compare CPU vs GPU performance
        #[clap(long)]
        compare_gpu: bool,
    },
    
    /// Generate test lattices
    Generate {
        /// Output directory for test files
        #[clap(long, default_value = "test_lattices")]
        output_dir: PathBuf,
        
        /// Dimensions to generate
        #[clap(long, value_delimiter = ',', default_values = &["2", "3", "4", "5", "6"])]
        dimensions: Vec<usize>,
        
        /// Number of random lattices per dimension
        #[clap(long, default_value = "5")]
        count: usize,
        
        /// Seed for reproducible random generation
        #[clap(long)]
        seed: Option<u64>,
    },
    
    /// Analyze lattice properties
    Analyze {
        /// Property to analyze (determinant, rank, conditioning, shortest_vector, all)
        #[clap(long, default_value = "all")]
        property: AnalysisProperty,
        
        /// Show detailed statistics
        #[clap(long)]
        detailed: bool,
    },
}

#[derive(ValueEnum, Clone, Copy, Debug)]
enum LLLVariantCli {
    Standard,
    Deep,
    Block,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
enum PreprocessingAlgorithm {
    None,
    LLL,
    BKZ,
    DeepLLL,
    Combined,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
enum BenchmarkAlgorithm {
    LLL,
    BKZ,
    SVP,
    CVP,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
enum AnalysisProperty {
    Determinant,
    Rank,
    Conditioning,
    ShortestVector,
    All,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
enum OutputFormat {
    Plain,
    Json,
    Csv,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
enum SVPAlgorithmCli {
    Enumeration,
    SchnorrEuchner,
    Kannan,
    RandomSampling,
    BkzApproximation,
}

impl From<SVPAlgorithmCli> for SVPAlgorithm {
    fn from(value: SVPAlgorithmCli) -> Self {
        match value {
            SVPAlgorithmCli::Enumeration => SVPAlgorithm::Enumeration,
            SVPAlgorithmCli::SchnorrEuchner => SVPAlgorithm::SchnorrEuchner,
            SVPAlgorithmCli::Kannan => SVPAlgorithm::Kannan,
            SVPAlgorithmCli::RandomSampling => SVPAlgorithm::RandomSampling,
            SVPAlgorithmCli::BkzApproximation => SVPAlgorithm::BKZApproximation,
        }
    }
}

#[derive(ValueEnum, Clone, Copy, Debug)]
enum CVPAlgorithmCli {
    BabaiNearestPlane,
    BabaiRounding,
    Enumeration,
    SchnorrEuchner,
    Vaidya,
    Kannan,
    Embedding,
}

impl From<CVPAlgorithmCli> for CVPAlgorithm {
    fn from(value: CVPAlgorithmCli) -> Self {
        match value {
            CVPAlgorithmCli::BabaiNearestPlane => CVPAlgorithm::BabaiNearestPlane,
            CVPAlgorithmCli::BabaiRounding => CVPAlgorithm::BabaiRounding,
            CVPAlgorithmCli::Enumeration => CVPAlgorithm::Enumeration,
            CVPAlgorithmCli::SchnorrEuchner => CVPAlgorithm::SchnorrEuchnerCVP,
            CVPAlgorithmCli::Vaidya => CVPAlgorithm::Vaidya,
            CVPAlgorithmCli::Kannan => CVPAlgorithm::KannanCVP,
            CVPAlgorithmCli::Embedding => CVPAlgorithm::Embedding,
        }
    }
}
// Add missing utility functions
fn generate_random_lattice_cli(
    rows: usize,
    cols: usize,
    seed: Option<u64>,
) -> Result<Lattice, Box<dyn std::error::Error>> {
    lattice_solver::utils::generate_random_lattice(rows, cols, seed).map_err(|e| e.into())
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    
    // Setup logging
    setup_logging(&args)?;
    // Combine compile-time features with runtime flags
    let compile_time_gpu = cfg!(feature = "gpu");
    let compile_time_hp = cfg!(feature = "high-precision");
    let gpu_enabled = compile_time_gpu && args.gpu;
    let hp_enabled = compile_time_hp && args.high_precision;
   //et parallel_enabled = cfg!(feature = "parallel_enabled") && args.parallel_enabled;
    log::info!("GPU feature compiled-in: {}", compile_time_gpu);
    log::info!("High precision feature compiled-in: {}", compile_time_hp);
    log::info!("GPU enabled at runtime: {}", gpu_enabled);
    log::info!("High precision enabled at runtime: {}", hp_enabled);

    // If user requested GPU/high-precision at runtime but the binary was not built
    // with corresponding features, explain how to rebuild with the features enabled.
    if args.gpu && !compile_time_gpu {
        log::warn!("Runtime flag --gpu requested but binary lacks 'gpu' feature. Rebuild with: cargo run --release --features 'gpu' -- --gpu");
    }
    if args.high_precision && !compile_time_hp {
        log::warn!("Runtime flag --high-precision requested but binary lacks 'high-precision' feature. Rebuild with: cargo run --release --features 'high-precision' -- --high-precision");
    }
   // log::info!("Parallel processing: {}", config.parallel);

    // Initialize features
  //log::info!("GPU enabled: {}", features::gpu_available());
  //log::info!("High precision: {}", features::high_precision_enabled());
  //log::info!("Parallel processing: {}", features::parallel_enabled());
    
    let command = args.command.clone();

    // Handle commands
    match command {
        Commands::LLL { delta, eta, variant } => {
            run_lll_reduction(&args, delta, eta, variant).await?;
        }
        Commands::BKZ { beta, max_rounds, progressive, pruning } => {
            run_bkz_reduction(&args, beta, max_rounds, progressive, pruning).await?;
        }
        Commands::SVP { algorithm, max_radius, pruning } => {
            run_svp_solver(&args, algorithm.into(), max_radius, pruning).await?;
        }
        Commands::CVP { target, algorithm, preprocessing, preprocessing_algo } => {
            run_cvp_solver(&args, target, algorithm.into(), preprocessing, preprocessing_algo).await?;
        }
        Commands::Benchmark { algorithms, dimensions, runs, compare_gpu } => {
            run_benchmark(&args, algorithms, dimensions, runs, compare_gpu).await?;
        }
        Commands::Generate { output_dir, dimensions, count, seed } => {
            run_generate(&args, output_dir, dimensions, count, seed)?;
        }
        Commands::Analyze { property, detailed } => {
            run_analyze(&args, property, detailed)?;
        }
    }
    
    Ok(())
}

// Custom error type for better error handling
#[derive(Debug, thiserror::Error)]
enum LatticeSolverError {
    #[error("GPU initialization failed: {0}")]
    GpuInitializationFailed(String),
    #[error("Lattice operation failed: {0}")]
    LatticeOperationFailed(String),
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("File operation failed: {0}")]
    FileOperationFailed(String),
    #[error("Benchmark failed: {0}")]
    BenchmarkFailed(String),
}

// Helper function to handle GPU/CPU execution with fallback
async fn execute_with_gpu_fallback<T, F, G, R>(
    use_gpu: bool,
    gpu_available: bool,
    gpu_operation: G,
    cpu_operation: F,
    operation_name: &str,
) -> Result<T, LatticeSolverError>
where
    F: FnOnce() -> Result<T, Box<dyn std::error::Error>>,
    G: std::future::Future<Output = Result<T, Box<dyn std::error::Error>>>,
{
    if use_gpu && gpu_available {
        log::info!("Using GPU-accelerated {}", operation_name);
        match gpu_operation.await {
            Ok(result) => Ok(result),
            Err(e) => {
                log::warn!("Failed to initialize GPU for {}: {}. Falling back to CPU.", operation_name, e);
                cpu_operation().map_err(|e| LatticeSolverError::LatticeOperationFailed(e.to_string()))
            }
        }
    } else {
        log::info!("Using CPU-only {}", operation_name);
        cpu_operation().map_err(|e| LatticeSolverError::LatticeOperationFailed(e.to_string()))
    }
}

// Helper function to create or validate target vector
fn create_or_validate_target_vector(
    lattice: &Lattice,
    target: Option<Vec<f64>>,
) -> Result<LatticeVector, LatticeSolverError> {
    match target {
        Some(v) => {
            if v.len() != lattice.ambient_dimension() {
                return Err(LatticeSolverError::DimensionMismatch {
                    expected: lattice.ambient_dimension(),
                    actual: v.len(),
                });
            }
            Ok(LatticeVector::new(v))
        }
        None => {
            // Generate random target if not provided
            let v: Vec<f64> = (0..lattice.ambient_dimension())
                .map(|_| rand::random::<f64>() * 10.0 - 5.0)
                .collect();
            Ok(LatticeVector::new(v))
        }
    }
}

// Helper function to convert preprocessing algorithm
fn convert_preprocessing_algorithm(algo: PreprocessingAlgorithm) -> lattice_solver::cvp::PreprocessingAlgorithm {
    match algo {
        PreprocessingAlgorithm::None => lattice_solver::cvp::PreprocessingAlgorithm::None,
        PreprocessingAlgorithm::LLL => lattice_solver::cvp::PreprocessingAlgorithm::LLL,
        PreprocessingAlgorithm::BKZ => lattice_solver::cvp::PreprocessingAlgorithm::BKZ,
        PreprocessingAlgorithm::DeepLLL => lattice_solver::cvp::PreprocessingAlgorithm::DeepLLL,
        PreprocessingAlgorithm::Combined => lattice_solver::cvp::PreprocessingAlgorithm::Combined,
    }
}

// Helper function to run benchmark algorithm
async fn run_benchmark_algorithm(
    algo: BenchmarkAlgorithm,
    test_lattice: &Lattice,
) -> Result<String, LatticeSolverError> {
    let result = match algo {
        BenchmarkAlgorithm::LLL => {
            let reducer = LLLReducer::new();
            reducer.reduce(test_lattice)?;
            "LLL".to_string()
        }
        BenchmarkAlgorithm::BKZ => {
            let dim = test_lattice.dimension().0;
            let reducer = BKZReducer::with_params(BKZParams::new(dim.min(20)));
            reducer.reduce(test_lattice)?;
            "BKZ".to_string()
        }
        BenchmarkAlgorithm::SVP => {
            let solver = SVPSolver::new();
            solver.solve(test_lattice)?;
            "SVP".to_string()
        }
        BenchmarkAlgorithm::CVP => {
            let solver = CVPSolver::new();
            let dim = test_lattice.dimension().0;
            let target = LatticeVector::zeros(dim);
            solver.solve(test_lattice, &target)?;
            "CVP".to_string()
        }
    };
    Ok(result)
}

// Helper function to run GPU benchmark comparison
async fn run_gpu_benchmark_comparison(
    gpu_ops: &GPUAcceleratedOperations,
    dimensions: &[usize],
) -> Result<(), LatticeSolverError> {
    for &dim in dimensions {
        let test_lattice = generate_random_lattice_cli(dim, dim, None)
            .map_err(|e| LatticeSolverError::LatticeOperationFailed(e.to_string()))?;
        
        let benchmark_result = gpu_ops
            .benchmark_gpu_performance(test_lattice.basis())
            .await
            .map_err(|e| LatticeSolverError::GpuInitializationFailed(e.to_string()))?;
        
        log::info!("GPU speedup: {:.2}x for {}x{} matrix",
                  benchmark_result.speedup, dim, dim);
    }
    Ok(())
}

async fn run_lll_reduction(
    args: &Args,
    delta: f64,
    eta: f64,
    variant: LLLVariantCli,
) -> Result<(), Box<dyn std::error::Error>> {
    log::info!("Running LLL reduction with delta={}, eta={}, variant={:?}", delta, eta, variant);
    
    let lattice = load_input_lattice(args)?;
    let original_lattice = lattice.clone();
    let params = LLLParams::new(delta, eta);
    
    let start_time = Instant::now();
    let reduced_lattice = execute_with_gpu_fallback(
        args.gpu,
        features::gpu_available(),
        async {
            let gpu_ops = GPUAcceleratedOperations::new().await?;
            gpu_ops.accelerated_lll_reduction(&lattice).await
        },
        || {
            let reducer = LLLReducer::with_params(params.clone());
            reducer.reduce(&lattice)
        },
        "LLL reduction",
    ).await?;
    let execution_time = start_time.elapsed();
    
    // Compute quality metrics
    let reduced_quality = reduced_lattice.quality_metrics()
        .map_err(|e| LatticeSolverError::LatticeOperationFailed(e.to_string()))?;
    
    // Output results
    let result = LLLResult {
        original_lattice,
        reduced_lattice,
        execution_time,
        quality_improvement: reduced_quality,
        algorithm: "LLL".to_string(),
        parameters: format!("delta={}, eta={}, variant={:?}", delta, eta, variant),
    };
    
    save_result(args, &result)?;
    
    log::info!("LLL reduction completed in {:?}", execution_time);
    Ok(())
}

async fn run_bkz_reduction(
    args: &Args,
    beta: usize,
    max_rounds: usize,
    progressive: bool,
    pruning: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    log::info!("Running BKZ reduction with beta={}, rounds={}, progressive={}, pruning={}",
               beta, max_rounds, progressive, pruning);
    
    let lattice = load_input_lattice(args)?;
    let original_lattice = lattice.clone();
    
    let mut params = BKZParams::new(beta);
    params.max_rounds = max_rounds;
    params.progressive = progressive;
    params.enable_pruning = pruning;
    
    // Setup precision if requested
    if args.high_precision {
        params.lll_params.precision = PrecisionType::Arbitrary { bits: args.precision_bits };
    }
    
    let start_time = Instant::now();
    let reduced_lattice = execute_with_gpu_fallback(
        args.gpu,
        features::gpu_available(),
        async {
            let gpu_ops = GPUAcceleratedOperations::new().await?;
            gpu_ops.accelerated_bkz_reduction(&lattice, &params).await
        },
        || {
            let reducer = BKZReducer::with_params(params);
            reducer.reduce(&lattice)
        },
        "BKZ reduction",
    ).await?;
    let execution_time = start_time.elapsed();
    
    let quality = reduced_lattice.quality_metrics()
        .map_err(|e| LatticeSolverError::LatticeOperationFailed(e.to_string()))?;
    
    let result = BKZResult {
        original_lattice,
        reduced_lattice,
        execution_time,
        quality_improvement: quality,
        algorithm: "BKZ".to_string(),
        parameters: format!("beta={}, rounds={}, progressive={}, pruning={}",
                           beta, max_rounds, progressive, pruning),
    };
    
    save_result(args, &result)?;
    log::info!("BKZ reduction completed in {:?}", execution_time);
    Ok(())
}

async fn run_svp_solver(
    args: &Args,
    algorithm: SVPAlgorithm,
    max_radius: f64,
    pruning: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    log::info!("Solving SVP with algorithm={:?}, max_radius={}, pruning={}",
               algorithm, max_radius, pruning);
    
    let lattice = load_input_lattice(args)?;
    
    let mut params = SVPSolverParams::with_algorithm(algorithm);
    params.max_radius = max_radius;
    params.enable_pruning = pruning;
    
    let start_time = Instant::now();
    let solver = SVPSolver::with_params(params);
    let result = solver.solve(&lattice)
        .map_err(|e| LatticeSolverError::LatticeOperationFailed(e.to_string()))?;
    let execution_time = start_time.elapsed();
    
    let svp_result = SVPResult {
        lattice,
        solution: result.solution,
        norm: result.norm,
        points_enumerated: result.points_enumerated,
        execution_time,
        algorithm: format!("{:?}", algorithm),
        parameters: format!("max_radius={}, pruning={}", max_radius, pruning),
    };
    
    save_result(args, &svp_result)?;
    log::info!("SVP solved in {:?}, found vector with norm {}", execution_time, result.norm);
    Ok(())
}

async fn run_cvp_solver(
    args: &Args,
    target: Option<Vec<f64>>,
    algorithm: CVPAlgorithm,
    preprocessing: bool,
    preprocessing_algo: PreprocessingAlgorithm,
) -> Result<(), Box<dyn std::error::Error>> {
    log::info!("Solving CVP with algorithm={:?}, preprocessing={}, preprocessing_algo={:?}",
               algorithm, preprocessing, preprocessing_algo);
    
    let lattice = load_input_lattice(args)?;
    
    let target_vector = create_or_validate_target_vector(&lattice, target)?;
    
    let mut params = CVPSolverParams::with_algorithm(algorithm);
    params.enable_preprocessing = preprocessing;
    params.preprocessing_algorithm = convert_preprocessing_algorithm(preprocessing_algo);
    
    let start_time = Instant::now();
    let solver = CVPSolver::with_params(params);
    let result = solver.solve(&lattice, &target_vector)
        .map_err(|e| LatticeSolverError::LatticeOperationFailed(e.to_string()))?;
    let execution_time = start_time.elapsed();
    
    let cvp_result = CVPResult {
        lattice,
        target: target_vector,
        closest_vector: result.closest_vector,
        distance: result.distance,
        points_examined: result.points_examined,
        execution_time,
        algorithm: format!("{:?}", algorithm),
        parameters: format!("preprocessing={}, preprocessing_algo={:?}", preprocessing, preprocessing_algo),
    };
    
    save_result(args, &cvp_result)?;
    log::info!("CVP solved in {:?}, distance {}", execution_time, result.distance);
    Ok(())
}

async fn run_benchmark(
    args: &Args,
    algorithms: Vec<BenchmarkAlgorithm>,
    dimensions: Vec<usize>,
    runs: usize,
    compare_gpu: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    log::info!("Running benchmark: {:?}, dimensions: {:?}, runs: {}",
               algorithms, dimensions, runs);
    
    let mut benchmark_results = Vec::new();
    
    // Pre-allocate test lattices to reduce memory allocation overhead
    let mut test_lattices: Vec<Lattice> = Vec::with_capacity(dimensions.len());
    for &dim in &dimensions {
        test_lattices.push(generate_random_lattice_cli(dim, dim, None)
            .map_err(|e| LatticeSolverError::BenchmarkFailed(e.to_string()))?);
    }
    
    for (i, &dim) in dimensions.iter().enumerate() {
        log::info!("Benchmarking dimension {}", dim);
        let test_lattice = &test_lattices[i];
        
        for run in 0..runs {
            // Use a fresh copy of the lattice for each run to avoid state contamination
            let lattice_copy = test_lattice.clone();
            
            for &algo in &algorithms {
                let start_time = Instant::now();
                
                let result = run_benchmark_algorithm(algo, &lattice_copy).await?;
                
                let execution_time = start_time.elapsed();
                
                let benchmark_result = BenchmarkResult {
                    algorithm: result,
                    input_size: dim,
                    execution_time_ms: execution_time.as_secs_f64() * 1000.0,
                    peak_memory_bytes: 0, // Would need proper memory tracking
                    quality_metrics: lattice_copy.quality_metrics()
                        .map_err(|e| LatticeSolverError::BenchmarkFailed(e.to_string()))?,
                    gpu_used: false,
                    precision: PrecisionType::Double,
                };
                
                benchmark_results.push(benchmark_result);
            }
            
            // Log progress for long-running benchmarks
            if runs > 10 && (run + 1) % (runs / 10) == 0 {
                log::info!("Completed {}/{} runs for dimension {}", run + 1, runs, dim);
            }
        }
    }
    
    // GPU comparison if requested
    if compare_gpu && args.gpu && features::gpu_available() {
        log::info!("Running GPU comparison");
        let gpu_accelerated_operations = GPUAcceleratedOperations::new().await
            .map_err(|e| LatticeSolverError::GpuInitializationFailed(e.to_string()))?;
        
        run_gpu_benchmark_comparison(&gpu_accelerated_operations, &dimensions).await?;
    }
    
    // Output benchmark results
    save_benchmark_results(args, &benchmark_results)?;
    
    Ok(())
}

fn run_generate(
    _args: &Args,
    output_dir: PathBuf,
    dimensions: Vec<usize>,
    count: usize,
    seed: Option<u64>,
) -> Result<(), Box<dyn std::error::Error>> {
    log::info!("Generating {} lattices in {:?}", count, dimensions);
    
    let mut generated_count = 0;
    
    for &dim in &dimensions {
        for i in 0..count {
            let lattice_seed = seed.map(|s| s + (i as u64) + (dim as u64) * 1000);
            let lattice = generate_random_lattice_cli(dim, dim, lattice_seed)?;
            let filename = output_dir.join(format!("random_{}d_{}.mat", dim, i));
            lattice.save_to_file(&filename.to_str().unwrap())?;
            generated_count += 1;
        }
    }
    
    log::info!("Generated {} lattices in {:?}", generated_count, output_dir);
    println!("Generated {} lattices in {:?}", generated_count, output_dir);
    
    Ok(())
}

fn run_analyze(
    args: &Args,
    property: AnalysisProperty,
    detailed: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let lattice = load_input_lattice(args)?;
    
    match property {
        AnalysisProperty::Determinant => {
            let det = lattice.determinant()?;
            println!("Determinant: {}", det);
        }
        AnalysisProperty::Rank => {
            let rank = lattice.rank();
            println!("Rank: {}", rank);
        }
        AnalysisProperty::Conditioning => {
            let condition = lattice.conditioning_number()?;
            println!("Conditioning number: {}", condition);
        }
        AnalysisProperty::ShortestVector => {
            let sv = lattice.shortest_vector()?;
            println!("Shortest vector: {}", sv);
            println!("Length: {}", sv.norm());
        }
        AnalysisProperty::All => {
            let rank = lattice.rank();
            let det = lattice.determinant()?;
            let condition = lattice.conditioning_number()?;
            let sv = lattice.shortest_vector()?;
            let metrics = lattice.quality_metrics()?;
            
            println!("Lattice Analysis:");
            println!("  Dimension: {}x{}", rank, lattice.ambient_dimension());
            println!("  Rank: {}", rank);
            println!("  Determinant: {}", det);
            println!("  Conditioning number: {}", condition);
            println!("  Shortest vector length: {}", sv.norm());
            println!("  Hermite constant: {}", metrics.hermite_constant);
            
            if detailed {
                println!("  Orthogonality defect: {}", metrics.orthogonality_defect);
                println!("  Covering radius: {}", lattice.covering_radius()?);
                println!("  Packing density: {}", lattice_utils::packing_density(&lattice)?);
            }
        }
    }
    
    Ok(())
}

fn setup_logging(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    use env_logger::Builder;
    use log::LevelFilter;
    
    let level_filter = match args.log_level.as_str() {
        "error" => LevelFilter::Error,
        "warn" => LevelFilter::Warn,
        "info" => LevelFilter::Info,
        "debug" => LevelFilter::Debug,
        "trace" => LevelFilter::Trace,
        _ => LevelFilter::Info,
    };
    
    let mut builder = Builder::from_default_env();
    builder.filter_level(level_filter);
    
    if args.verbose {
        builder.init();
    } else {
        builder.init();
    }
    
    Ok(())
}

fn load_input_lattice(args: &Args) -> Result<Lattice, Box<dyn std::error::Error>> {
    match &args.input {
        Some(path) => {
            if path.exists() {
                if args.high_precision && cfg!(feature = "high-precision") {
                    // Try to load as BigInt lattice and convert to double-precision Lattice
                    #[cfg(feature = "high-precision")]
                    {
                        match Lattice::load_from_file(path.to_str().unwrap()) {
                            Ok(m) => return Ok(m), // file parsed as i64 matrix
                            Err(_) => {
                                let big = Lattice::load_from_file_bigint(path.to_str().unwrap())?.to_f64_lattice()?;
                                return Ok(big);
                            }
                        }
                    }
                }
                Lattice::load_from_file(path.to_str().unwrap())
                    .map_err(|e| format!("Failed to load lattice from {}: {}", path.display(), e).into())
            } else {
                Err(format!("Input file {} does not exist", path.display()).into())
            }
        }
        None => {
            // Generate a default 3x3 lattice for testing
            log::warn!("No input file specified, using default test lattice");
            generate_random_lattice_cli(3, 3, Some(42))
                .map_err(|e| format!("Failed to generate test lattice: {}", e).into())
        }
    }
}

fn save_result(args: &Args, result: &dyn SerializableResult) -> Result<(), Box<dyn std::error::Error>> {
    match args.output {
        Some(ref path) => {
            let content = result.serialize(args.format)?;
            std::fs::write(path, content)
                .map_err(|e| format!("Failed to write output to {}: {}", path.display(), e).into())
        }
        None => {
            // Print to stdout
            let content = result.serialize(args.format)?;
            println!("{}", content);
            Ok(())
        }
    }
}

fn save_benchmark_results(args: &Args, results: &[BenchmarkResult]) -> Result<(), Box<dyn std::error::Error>> {
    match args.output {
        Some(ref path) => {
            let content = format_benchmark_results(results, args.format)?;
            std::fs::write(path, content)
                .map_err(|e| format!("Failed to write benchmark results to {}: {}", path.display(), e).into())
        }
        None => {
            let content = format_benchmark_results(results, args.format)?;
            println!("{}", content);
            Ok(())
        }
    }
}

fn format_benchmark_results(results: &[BenchmarkResult], format: OutputFormat) -> Result<String, Box<dyn std::error::Error>> {
    match format {
        OutputFormat::Plain => {
            let mut output = String::new();
            output.push_str("Benchmark Results:\n");
            for result in results {
                output.push_str(&format!("{}: {}ms (dimension {}, quality: {:.3})\n",
                    result.algorithm, result.execution_time_ms, result.input_size, 
                    result.quality_metrics.shortest_vector_length));
            }
            Ok(output)
        }
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(results)?;
            Ok(json)
        }
        OutputFormat::Csv => {
            let mut csv = String::new();
            csv.push_str("algorithm,input_size,execution_time_ms,shortest_vector_length\n");
            for result in results {
                csv.push_str(&format!("{},{},{},{}\n",
                    result.algorithm, result.input_size, result.execution_time_ms,
                    result.quality_metrics.shortest_vector_length));
            }
            Ok(csv)
        }
    }
}

// Result serialization trait
trait SerializableResult {
    fn serialize(&self, format: OutputFormat) -> Result<String, Box<dyn std::error::Error>>;
}

#[derive(Debug)]
struct LLLResult {
    original_lattice: Lattice,
    reduced_lattice: Lattice,
    execution_time: std::time::Duration,
    quality_improvement: QualityMetrics,
    algorithm: String,
    parameters: String,
}

impl SerializableResult for LLLResult {
    fn serialize(&self, format: OutputFormat) -> Result<String, Box<dyn std::error::Error>> {
        let original_metrics = self.original_lattice.quality_metrics().ok();
        let improvement_ratio = original_metrics
            .map(|orig| {
                if orig.shortest_vector_length.abs() > f64::EPSILON {
                    self.quality_improvement.shortest_vector_length / orig.shortest_vector_length
                } else {
                    0.0
                }
            })
            .unwrap_or(0.0);

        match format {
            OutputFormat::Plain => {
                Ok(format!(
                    "LLL Reduction Result:\n\
                     Algorithm: {}\n\
                     Parameters: {}\n\
                     Execution Time: {:?}\n\
                     Quality: {}\n\
                     Improvement ratio: {:.3}\n\
                     Reduced Basis:\n{}",
                    self.algorithm, self.parameters, self.execution_time,
                    self.quality_improvement.shortest_vector_length,
                    improvement_ratio,
                    self.reduced_lattice.to_fplll_format()
                ))
            }
            OutputFormat::Json => {
                #[derive(serde::Serialize)]
                struct LLLResultJson {
                    algorithm: String,
                    parameters: String,
                    execution_time_ms: f64,
                    quality: f64,
                    improvement_ratio: f64,
                    reduced_basis: Vec<Vec<i64>>,
                    original_basis: Vec<Vec<i64>>,
                }
                
                let json = LLLResultJson {
                    algorithm: self.algorithm.clone(),
                    parameters: self.parameters.clone(),
                    execution_time_ms: self.execution_time.as_secs_f64() * 1000.0,
                    quality: self.quality_improvement.shortest_vector_length,
                    improvement_ratio,
                    reduced_basis: self.reduced_lattice.basis().to_vec(),
                    original_basis: self.original_lattice.basis().to_vec(),
                };
                
                Ok(serde_json::to_string_pretty(&json)?)
            }
            OutputFormat::Csv => {
                Ok(format!(
                    "algorithm,execution_time_ms,quality,beta\n\
                     {},{},{},{}\n",
                    self.algorithm, self.execution_time.as_secs_f64() * 1000.0,
                    self.quality_improvement.shortest_vector_length, "N/A"
                ))
            }
        }
    }
}

#[derive(Debug)]
struct BKZResult {
    original_lattice: Lattice,
    reduced_lattice: Lattice,
    execution_time: std::time::Duration,
    quality_improvement: QualityMetrics,
    algorithm: String,
    parameters: String,
}

impl SerializableResult for BKZResult {
    fn serialize(&self, format: OutputFormat) -> Result<String, Box<dyn std::error::Error>> {
        let original_metrics = self.original_lattice.quality_metrics().ok();
        let improvement_ratio = original_metrics
            .map(|orig| {
                if orig.shortest_vector_length.abs() > f64::EPSILON {
                    self.quality_improvement.shortest_vector_length / orig.shortest_vector_length
                } else {
                    0.0
                }
            })
            .unwrap_or(0.0);

        match format {
            OutputFormat::Plain => {
                Ok(format!(
                    "BKZ Reduction Result:\n\
                     Algorithm: {}\n\
                     Parameters: {}\n\
                     Execution Time: {:?}\n\
                     Quality: {}\n\
                     Improvement ratio: {:.3}\n\
                     Reduced Basis:\n{}",
                    self.algorithm, self.parameters, self.execution_time,
                    self.quality_improvement.shortest_vector_length,
                    improvement_ratio,
                    self.reduced_lattice.to_fplll_format()
                ))
            }
            OutputFormat::Json => {
                #[derive(serde::Serialize)]
                struct BKZResultJson {
                    algorithm: String,
                    parameters: String,
                    execution_time_ms: f64,
                    quality: f64,
                    improvement_ratio: f64,
                    reduced_basis: Vec<Vec<i64>>,
                    original_basis: Vec<Vec<i64>>,
                }
                
                let json = BKZResultJson {
                    algorithm: self.algorithm.clone(),
                    parameters: self.parameters.clone(),
                    execution_time_ms: self.execution_time.as_secs_f64() * 1000.0,
                    quality: self.quality_improvement.shortest_vector_length,
                    improvement_ratio,
                    reduced_basis: self.reduced_lattice.basis().to_vec(),
                    original_basis: self.original_lattice.basis().to_vec(),
                };
                
                Ok(serde_json::to_string_pretty(&json)?)
            }
            OutputFormat::Csv => {
                Ok(format!(
                    "algorithm,execution_time_ms,quality\n\
                     {},{},{}\n",
                    self.algorithm, self.execution_time.as_secs_f64() * 1000.0,
                    self.quality_improvement.shortest_vector_length
                ))
            }
        }
    }
}

#[derive(Debug)]
struct SVPResult {
    lattice: Lattice,
    solution: LatticeVector,
    norm: f64,
    points_enumerated: u64,
    execution_time: std::time::Duration,
    algorithm: String,
    parameters: String,
}

impl SerializableResult for SVPResult {
    fn serialize(&self, format: OutputFormat) -> Result<String, Box<dyn std::error::Error>> {
        let dims = self.lattice.dimension();
        match format {
            OutputFormat::Plain => {
                Ok(format!(
                    "SVP Solution Result:\n\
                     Algorithm: {}\n\
                     Parameters: {}\n\
                     Execution Time: {:?}\n\
                     Lattice Dimension: {}x{}\n\
                     Solution Vector: {}\n\
                     Norm: {}\n\
                     Points Enumerated: {}",
                    self.algorithm, self.parameters, self.execution_time,
                    dims.0, dims.1,
                    self.solution, self.norm, self.points_enumerated
                ))
            }
            OutputFormat::Json => {
                #[derive(serde::Serialize)]
                struct SVPResultJson {
                    algorithm: String,
                    parameters: String,
                    execution_time_ms: f64,
                    lattice_size: (usize, usize),
                    solution: Vec<f64>,
                    norm: f64,
                    points_enumerated: u64,
                }
                
                let json = SVPResultJson {
                    algorithm: self.algorithm.clone(),
                    parameters: self.parameters.clone(),
                    execution_time_ms: self.execution_time.as_secs_f64() * 1000.0,
                    lattice_size: dims,
                    solution: self.solution.as_slice().to_vec(),
                    norm: self.norm,
                    points_enumerated: self.points_enumerated,
                };
                
                Ok(serde_json::to_string_pretty(&json)?)
            }
            OutputFormat::Csv => {
                Ok(format!(
                    "algorithm,execution_time_ms,norm,points_enumerated\n\
                     {},{},{},{}\n",
                    self.algorithm, self.execution_time.as_secs_f64() * 1000.0,
                    self.norm, self.points_enumerated
                ))
            }
        }
    }
}

#[derive(Debug)]
struct CVPResult {
    lattice: Lattice,
    target: LatticeVector,
    closest_vector: LatticeVector,
    distance: f64,
    points_examined: u64,
    execution_time: std::time::Duration,
    algorithm: String,
    parameters: String,
}

impl SerializableResult for CVPResult {
    fn serialize(&self, format: OutputFormat) -> Result<String, Box<dyn std::error::Error>> {
        let dims = self.lattice.dimension();
        match format {
            OutputFormat::Plain => {
                Ok(format!(
                    "CVP Solution Result:\n\
                     Algorithm: {}\n\
                     Parameters: {}\n\
                     Execution Time: {:?}\n\
                     Lattice Dimension: {}x{}\n\
                     Target Vector: {}\n\
                     Closest Vector: {}\n\
                     Distance: {}\n\
                     Points Examined: {}",
                    self.algorithm, self.parameters, self.execution_time,
                    dims.0, dims.1,
                    self.target, self.closest_vector, self.distance, self.points_examined
                ))
            }
            OutputFormat::Json => {
                #[derive(serde::Serialize)]
                struct CVPResultJson {
                    algorithm: String,
                    parameters: String,
                    execution_time_ms: f64,
                    lattice_size: (usize, usize),
                    target: Vec<f64>,
                    closest_vector: Vec<f64>,
                    distance: f64,
                    points_examined: u64,
                }
                
                let json = CVPResultJson {
                    algorithm: self.algorithm.clone(),
                    parameters: self.parameters.clone(),
                    execution_time_ms: self.execution_time.as_secs_f64() * 1000.0,
                    lattice_size: dims,
                    target: self.target.as_slice().to_vec(),
                    closest_vector: self.closest_vector.as_slice().to_vec(),
                    distance: self.distance,
                    points_examined: self.points_examined,
                };
                
                Ok(serde_json::to_string_pretty(&json)?)
            }
            OutputFormat::Csv => {
                Ok(format!(
                    "algorithm,execution_time_ms,distance,points_examined\n\
                     {},{},{},{}\n",
                    self.algorithm, self.execution_time.as_secs_f64() * 1000.0,
                    self.distance, self.points_examined
                ))
            }
        }
    }
}
