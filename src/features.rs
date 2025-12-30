// src/features.rs

/// Is GPU support compiled in?
pub fn gpu_available() -> bool {
    cfg!(feature = "gpu")
}

/// Is high-precision support compiled in?
pub fn high_precision_available() -> bool {
    cfg!(feature = "high-precision")
}

/// Is parallel mode compiled in?
pub fn parallel_available() -> bool {
    cfg!(feature = "parallel")
}
