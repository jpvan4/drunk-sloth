use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lattice_solver::{utils::generate_random_lattice, gpu::GPUManager};
use std::time::Duration;

fn bench_gram_schmidt(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let gpu_manager: GPUManager = rt.block_on(GPUManager::new()).unwrap();
    
    let lattice = generate_random_lattice(100, 100, Some(42)).unwrap();
    let matrix = lattice.basis().clone();
    
    c.bench_function("gram_schmidt_gpu", |b| {
        b.iter(|| {
            rt.block_on(async {
                let result: _ = gpu_manager.gram_schmidt_gpu(&matrix).await.unwrap();
                black_box(result)
            })
        });
    });
    
    c.bench_function("gram_schmidt_cpu", |b| {
        b.iter(|| {
            black_box({
                let mut orthogonal: Vec<Vec<f64>> = Vec::new();
                for i in 0..matrix.rows() {
                    let row = matrix.get_row(i).unwrap();
                    let vec: Vec<f64> = row.into_iter().map(|x| x as f64).collect();
                    orthogonal.push(vec);
                }
            })
        });
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(30));
    targets = bench_gram_schmidt
);
criterion_main!(benches);