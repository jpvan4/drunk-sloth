#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// ===============================
// Simple Gram-Schmidt (f64)
// ===============================
extern "C" {
__global__ void gram_schmidt_kernel(double* matrix,
                                    double* orthogonal,
                                    double* norm_sq,
                                    unsigned int rows,
                                    unsigned int cols) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows) return;

    // Copy row to orthogonal
    for (unsigned int j = 0; j < cols; ++j) {
        orthogonal[idx * cols + j] = matrix[idx * cols + j];
    }

    // Project and subtract previous orthogonal vectors
    for (unsigned int j = 0; j < idx; ++j) {
        double denom = norm_sq[j];
        if (fabs(denom) < 1e-18) continue;

        double dot = 0.0;
        for (unsigned int k = 0; k < cols; ++k) {
            dot += matrix[idx * cols + k] * orthogonal[j * cols + k];
        }

        double coeff = dot / denom;

        for (unsigned int k = 0; k < cols; ++k) {
            orthogonal[idx * cols + k] -= coeff * orthogonal[j * cols + k];
        }
    }

    // Compute norm squared
    double norm = 0.0;
    for (unsigned int k = 0; k < cols; ++k) {
        double val = orthogonal[idx * cols + k];
        norm += val * val;
    }
    norm_sq[idx] = norm;
}

// ===============================
// Matrix-vector multiply (f64)
// ===============================
__global__ void matrix_vector_kernel(double* matrix,
                                     double* vector,
                                     double* result,
                                     unsigned int rows,
                                     unsigned int cols) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows) return;

    double sum = 0.0;
    for (unsigned int j = 0; j < cols; ++j) {
        sum += matrix[idx * cols + j] * vector[j];
    }
    result[idx] = sum;
}

// ===============================
// Size-reduction of a single vector k
// ===============================
__global__ void size_reduce_kernel(double* basis,
                                   double* mu,
                                   unsigned int n,
                                   unsigned int m,
                                   unsigned int k,
                                   double eta) {
    unsigned int tid = threadIdx.x;

    for (int j = (int)k - 1; j >= 0; j--) {
        double mu_kj = mu[k * n + j];
        if (fabs(mu_kj) > eta) {
            double r = round(mu_kj);

            // basis[k] -= r * basis[j]
            for (unsigned int l = tid; l < m; l += blockDim.x) {
                basis[k * m + l] -= r * basis[j * m + l];
            }
            __syncthreads();

            // mu[k][0..j] -= r * mu[j][0..j]
            if (tid == 0) {
                for (int i = 0; i <= j; i++) {
                    mu[k * n + i] -= r * mu[j * n + i];
                }
            }
            __syncthreads();
        }
    }
}

// ===============================
// Parallel reduction helpers
// ===============================
// Requires dynamic shared memory: `blockDim.x * sizeof(double)`
__device__ double parallel_reduce(double val) {
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    sdata[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    return sdata[0];
}

// Parallel dot product <a, b>
__device__ double parallel_dot(const double* a,
                               const double* b,
                               unsigned int len) {
    double sum = 0.0;
    for (unsigned int i = threadIdx.x; i < len; i += blockDim.x) {
        sum += a[i] * b[i];
    }
    return parallel_reduce(sum);
}

// ===============================
// Full one-block LLL (f64)
// ===============================
__global__ void lll_kernel(double* basis,
                           double* orthogonal,
                           double* mu,
                           double* B,
                           unsigned int n,
                           unsigned int m,
                           double delta,
                           double eta) {
    unsigned int tid = threadIdx.x;
    if (n > blockDim.x) return;  // safety: this kernel assumes n <= blockDim.x

    // -------------------------------
    // Initial GSO (Gram-Schmidt)
    // -------------------------------
    for (unsigned int i = 0; i < n; i++) {
        // Copy basis[i] to orthogonal[i]
        for (unsigned int l = tid; l < m; l += blockDim.x) {
            orthogonal[i * m + l] = basis[i * m + l];
        }
        __syncthreads();

        for (unsigned int j = 0; j < i; j++) {
            double denom = B[j];
            if (fabs(denom) < 1e-18) continue;

            double dot = parallel_dot(&basis[i * m], &orthogonal[j * m], m);
            double coeff = dot / denom;

            if (tid == 0) {
                mu[i * n + j] = coeff;
            }
            __syncthreads();

            for (unsigned int l = tid; l < m; l += blockDim.x) {
                orthogonal[i * m + l] -= coeff * orthogonal[j * m + l];
            }
            __syncthreads();
        }

        double norm2 = parallel_dot(&orthogonal[i * m], &orthogonal[i * m], m);
        if (tid == 0) {
            B[i] = norm2;
            mu[i * n + i] = 1.0;
        }
        __syncthreads();
    }

    // Shared scalars for swap step
    __shared__ double sh_old_u;
    __shared__ double sh_new_u;
    __shared__ double sh_old_d, sh_old_e;


    unsigned int k = 1;
    int max_iter = (int)n * (int)n;
    int iter = 0;

    while (k < n && iter < max_iter) {
        iter++;

        // -------------------------------
        // Size reduction on b_k
        // -------------------------------
      //  for (int j = (int)k - 1; j >= 0; j--) {
      //      double mu_kj = mu[k * n + j];
      //      if (fabs(mu_kj) > eta) {    // use eta (e.g. 0.51)
      //          double r = round(mu_kj);
//
      //          // basis[k] -= r * basis[j]
      //          for (unsigned int l = tid; l < m; l += blockDim.x) {
      //              basis[k * m + l] -= r * basis[j * m + l];
      //          }
      //          __syncthreads();
//
      //          // mu[k][0..j] -= r * mu[j][0..j]
      //          if (tid == 0) {
      //              for (int i = 0; i <= j; i++) {
      //                  mu[k * n + i] -= r * mu[j * n + i];
      //              }
      //          }
      //          __syncthreads();
      //      }
      //  }
//
        // -------------------------------
        // NEW: Recompute GSO row k
        // -------------------------------
        // orthogonal[k] := basis[k]
        for (unsigned int l = tid; l < m; l += blockDim.x) {
            orthogonal[k * m + l] = basis[k * m + l];
        }
        __syncthreads();

        // Project against previous b*_j
        for (unsigned int j = 0; j < k; j++) {
            double denom = B[j];
            if (fabs(denom) < 1e-18) continue;

            // dot(b_k, b*_j)
            double dot = parallel_dot(&basis[k * m], &orthogonal[j * m], m);
            double coeff = dot / denom;

            if (tid == 0) {
                mu[k * n + j] = coeff;
            }
            __syncthreads();

            // b*_k -= coeff * b*_j
            for (unsigned int l = tid; l < m; l += blockDim.x) {
                orthogonal[k * m + l] -= coeff * orthogonal[j * m + l];
            }
            __syncthreads();
        }

        // Recompute B[k] = ||b*_k||^2 and set mu[k,k] = 1
        {
            double norm2 = parallel_dot(&orthogonal[k * m], &orthogonal[k * m], m);
            if (tid == 0) {
                B[k] = norm2;
                mu[k * n + k] = 1.0;
            }
            __syncthreads();
        }
        // -------------------------------
        // Size reduction on b_k using FRESH μ and orthogonal
        // -------------------------------
        for (int j = (int)k - 1; j >= 0; j--) {
            double mu_kj = mu[k * n + j];
            if (fabs(mu_kj) > eta) {
                double r = round(mu_kj);
            
                // basis[k] -= r * basis[j]
                for (unsigned int l = tid; l < m; l += blockDim.x) {
                    basis[k * m + l] -= r * basis[j * m + l];
                }
                __syncthreads();
            
                // orthogonal[k] -= r * orthogonal[j]
                for (unsigned int l = tid; l < m; l += blockDim.x) {
                    orthogonal[k * m + l] -= r * orthogonal[j * m + l];
                }
                __syncthreads();
            
                // Update μ[k][0..j] using μ relations:
                // μ_kj_new   = μ_kj - r
                // μ_ki_new   = μ_ki - r * μ_ji for i < j
                if (tid == 0) {
                    // update μ_kj
                    mu[k * n + j] = mu_kj - r;
                    // update μ_ki for i < j
                    for (int i = 0; i < j; i++) {
                        mu[k * n + i] -= r * mu[j * n + i];
                    }
                }
                __syncthreads();
            }
        }

        // Recompute B[k] from updated orthogonal[k]
        double norm2_after = parallel_dot(&orthogonal[k * m], &orthogonal[k * m], m);
        if (tid == 0) {
            B[k] = norm2_after;
        }
        __syncthreads();

        // -------------------------------
        // Lovász condition check
        // -------------------------------
        double u = mu[k * n + (k - 1)];
        double lhs = B[k];
        double rhs = (delta - u * u) * B[k - 1];
        bool do_swap = (lhs < rhs);
        __syncthreads();

        if (do_swap) {
            // ---------------------------
            // Swap basis[k-1] and basis[k]
            // ---------------------------
            for (unsigned int l = tid; l < m; l += blockDim.x) {
                double temp = basis[(k - 1) * m + l];
                basis[(k - 1) * m + l] = basis[k * m + l];
                basis[k * m + l]       = temp;
            }
            __syncthreads();

            // ---------------------------
            // Partial GSO update
            // ---------------------------
            if (tid == 0) {
                sh_old_u = mu[k * n + (k - 1)];
                sh_old_d = B[k - 1];
                sh_old_e = B[k];

                double new_d = sh_old_e + sh_old_u * sh_old_u * sh_old_d;
                if (fabs(new_d) < 1e-18) {
                    // Very degenerate; avoid NaN
                    new_d = (sh_old_d + sh_old_e) + 1e-18;
                }
                double new_u = sh_old_u * sh_old_d / new_d;
                double new_e = sh_old_d * sh_old_e / new_d;


                sh_new_u = new_u;
  

                B[k - 1] = new_d;
                B[k]     = new_e;
                mu[k * n + (k - 1)] = new_u;

                // Swap mu rows for j < k-1
                for (unsigned int j = 0; j < k - 1; j++) {
                    double temp = mu[(k - 1) * n + j];
                    mu[(k - 1) * n + j] = mu[k * n + j];
                    mu[k * n + j]       = temp;
                }
            }
            __syncthreads();

            double old_u_local = sh_old_u;
            double new_u_local = sh_new_u;

            // ---------------------------
            // Update orthogonal[k-1], orthogonal[k]
            // ---------------------------
            for (unsigned int l = tid; l < m; l += blockDim.x) {
                double ok1 = orthogonal[(k - 1) * m + l];
                double ok  = orthogonal[k * m + l];

                double new_ok1 = ok + old_u_local * ok1;
                double new_ok  = ok1 - new_u_local * new_ok1;

                orthogonal[(k - 1) * m + l] = new_ok1;
                orthogonal[k * m + l]       = new_ok;
            }
            __syncthreads();

            // ---------------------------
            // Update higher mu entries (i > k)
            // ---------------------------
            if (tid == 0) {
                for (unsigned int i = k + 1; i < n; i++) {
                    double alpha = mu[i * n + (k - 1)];
                    double beta  = mu[i * n + k];

                    double new_alpha = beta + old_u_local * alpha;
                    double new_beta  = alpha - new_u_local * new_alpha;

                    mu[i * n + (k - 1)] = new_alpha;
                    mu[i * n + k]       = new_beta;
                }
            }
            __syncthreads();

            if (k > 1) k--;
            else       k = 1;
        } else {
            k++;
        }
    }
}

// ==============================
// Iterative LLL GSO computation (f64)
// ==============================
__global__ void lll_gso_kernel(double* basis,
                               double* orthogonal,
                               double* mu,
                               double* B,
                               unsigned int n,
                               unsigned int m,
                               unsigned int start_i) {
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int bdim = blockDim.x;

    // Each block handles one vector i
    unsigned int i = bid + start_i;
    if (i >= n) return;

    // Copy basis[i] to orthogonal[i]
    for (unsigned int l = tid; l < m; l += bdim) {
        orthogonal[i * m + l] = basis[i * m + l];
    }
    __syncthreads();

    // Project against previous
    for (unsigned int j = 0; j < i; j++) {
        double denom = B[j];
        if (fabs(denom) < 1e-18) continue;

        // Compute dot product
        double dot = 0.0;
        for (unsigned int l = tid; l < m; l += bdim) {
            dot += basis[i * m + l] * orthogonal[j * m + l];
        }
        // Reduce dot
        sdata[tid] = dot;
        __syncthreads();
        for (unsigned int s = bdim / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        dot = sdata[0];

        double coeff = dot / denom;

        if (tid == 0) {
            mu[i * n + j] = coeff;
        }
        __syncthreads();

        // Subtract
        for (unsigned int l = tid; l < m; l += bdim) {
            orthogonal[i * m + l] -= coeff * orthogonal[j * m + l];
        }
        __syncthreads();
    }

    // Compute B[i]
    double norm2 = 0.0;
    for (unsigned int l = tid; l < m; l += bdim) {
        double val = orthogonal[i * m + l];
        norm2 += val * val;
    }
    // Reduce norm2
    extern __shared__ double sdata[];
    sdata[tid] = norm2;
    __syncthreads();
    for (unsigned int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        B[i] = sdata[0];
        mu[i * n + i] = 1.0;
    }
}

// ==============================
// Lovász condition check and swap (f64)
// ==============================
__global__ void lovasz_swap_kernel(double* basis,
                                   double* orthogonal,
                                   double* mu,
                                   double* B,
                                   unsigned int n,
                                   unsigned int m,
                                   unsigned int k,
                                   double delta,
                                   int* do_swap) {
    unsigned int tid = threadIdx.x;
    unsigned int bdim = blockDim.x;

    if (k == 0) {
        *do_swap = 0;
        return;
    }

    double u = mu[k * n + (k - 1)];
    double lhs = B[k];
    double rhs = (delta - u * u) * B[k - 1];
    *do_swap = (lhs < rhs) ? 1 : 0;

    if (*do_swap) {
        // Swap basis[k-1] and basis[k]
        for (unsigned int l = tid; l < m; l += bdim) {
            double temp = basis[(k - 1) * m + l];
            basis[(k - 1) * m + l] = basis[k * m + l];
            basis[k * m + l] = temp;
        }
        __syncthreads();

        // Update mu and B
        if (tid == 0) {
            double old_u = u;
            double old_d = B[k - 1];
            double old_e = B[k];

            double new_d = old_e + old_u * old_u * old_d;
            if (fabs(new_d) < 1e-18) {
                new_d = (old_d + old_e) + 1e-18;
            }
            double new_u = old_u * old_d / new_d;
            double new_e = old_d * old_e / new_d;

            B[k - 1] = new_d;
            B[k] = new_e;
            mu[k * n + (k - 1)] = new_u;

            // Swap mu rows for j < k-1
            for (unsigned int j = 0; j < k - 1; j++) {
                double temp = mu[(k - 1) * n + j];
                mu[(k - 1) * n + j] = mu[k * n + j];
                mu[k * n + j] = temp;
            }
        }
        __syncthreads();

        // Update orthogonal
        double old_u_local = u;
        double new_u_local = mu[k * n + (k - 1)];  // updated

        for (unsigned int l = tid; l < m; l += bdim) {
            double ok1 = orthogonal[(k - 1) * m + l];
            double ok = orthogonal[k * m + l];

            double new_ok1 = ok + old_u_local * ok1;
            double new_ok = ok1 - new_u_local * new_ok1;

            orthogonal[(k - 1) * m + l] = new_ok1;
            orthogonal[k * m + l] = new_ok;
        }
        __syncthreads();

        // Update higher mu
        if (tid == 0) {
            for (unsigned int i = k + 1; i < n; i++) {
                double alpha = mu[i * n + (k - 1)];
                double beta = mu[i * n + k];

                double new_alpha = beta + old_u_local * alpha;
                double new_beta = alpha - new_u_local * new_alpha;

                mu[i * n + (k - 1)] = new_alpha;
                mu[i * n + k] = new_beta;
            }
        }
    }
}
}