//! Core module wiring: lattice structures, matrices, errors, and type utilities.

pub mod lattice;
pub mod matrix;
pub mod error;
pub mod types;
#[cfg(feature = "high-precision")]
pub mod bigint_matrix;
#[cfg(feature = "high-precision")]
pub mod bigint_lattice;

// Re-export the most commonly used items so downstream code can simply import
// `crate::core::*` without having to juggle individual submodules.
pub use error::*;
pub use lattice::*;
pub use matrix::*;
pub use types::*;
#[cfg(feature = "high-precision")]
pub use bigint_matrix::*;
#[cfg(feature = "high-precision")]
pub use bigint_lattice::*;
