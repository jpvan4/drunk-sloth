use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    // Only attempt to compile CUDA kernels when the `gpu` feature is enabled.
    if env::var("CARGO_FEATURE_GPU").is_err() {
        return;
    }

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let src_path = Path::new("src/gpu/kernels.cu");
    let out_path = Path::new(&out_dir).join("kernels.ptx");

    // Attempt to find nvcc by trying to invoke it and checking for an error.
    let nvcc_check = Command::new("nvcc").arg("--version").output();
    if nvcc_check.is_err() {
        panic!("nvcc not found in PATH. Install CUDA toolkit to compile kernels.cu");
    }

    let status = Command::new("nvcc")
        .args(&["-ptx", src_path.to_str().unwrap(), "-o", out_path.to_str().unwrap()])
        .status()
        .expect("failed to spawn nvcc");

    if !status.success() {
        panic!("nvcc failed to compile kernels.cu to PTX");
    }

    // Invalidate the build when kernel source changes
    println!("cargo:rerun-if-changed={}", src_path.display());
    println!("cargo:rerun-if-env-changed=KERNELS_PTX");
}
