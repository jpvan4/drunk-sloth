// test_lll.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <iomanip>
#include "kernels.cuh"

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error (" << msg << "): "
                  << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}

void write_matrix_int(const std::string& path,
                      const std::vector<long long>& mat,
                      unsigned int n,
                      unsigned int m) {
    std::ofstream out(path);
    out << "[[\n";
    for (unsigned int i = 0; i < n; i++) {
        out << "[";
        for (unsigned int j = 0; j < m; j++) {
            out << mat[i * m + j];
            if (j + 1 < m) out << ", ";
        }
        out << "]";
        if (i + 1 < n) out << ",\n";
        else out << "\n";
    }
    out << "]]\n";
}

void write_matrix_double(const std::string& path,
                         const std::vector<double>& mat,
                         unsigned int n,
                         unsigned int m) {
    std::ofstream out(path);
    out << std::setprecision(17); // keep full double precision
    out << "[[\n";
    for (unsigned int i = 0; i < n; i++) {
        out << "[";
        for (unsigned int j = 0; j < m; j++) {
            out << mat[i * m + j];
            if (j + 1 < m) out << ", ";
        }
        out << "]";
        if (i + 1 < n) out << ",\n";
        else out << "\n";
    }
    out << "]]\n";
}

int main() {
    const unsigned int n = 80;
    const unsigned int m = 80;
    const double delta = 0.99;
    const double eta   = 0.49;

    // -------------------------
    // 1. Generate random integer basis
    // -------------------------
    std::mt19937_64 rng(123456789ull);
    std::uniform_int_distribution<int> dist(-50, 50);

    std::vector<long long> basis_int(n * m);
    std::vector<double>    basis_host(n * m);

    for (unsigned int i = 0; i < n * m; i++) {
        long long v = dist(rng);
        basis_int[i]  = v;
        basis_host[i] = static_cast<double>(v);
    }

    write_matrix_int("basis_original_20x20.basis", basis_int, n, m);

    // -------------------------
    // 2. Allocate device memory
    // -------------------------
    double *d_basis = nullptr, *d_orth = nullptr, *d_mu = nullptr, *d_B = nullptr;
    size_t size_mat = n * m * sizeof(double);
    size_t size_mu  = n * n * sizeof(double);
    size_t size_B   = n * sizeof(double);

    check_cuda(cudaMalloc(&d_basis, size_mat), "malloc d_basis");
    check_cuda(cudaMalloc(&d_orth,  size_mat), "malloc d_orth");
    check_cuda(cudaMalloc(&d_mu,    size_mu),  "malloc d_mu");
    check_cuda(cudaMalloc(&d_B,     size_B),   "malloc d_B");

    check_cuda(cudaMemcpy(d_basis, basis_host.data(),
                          size_mat, cudaMemcpyHostToDevice),
               "memcpy basis -> d_basis");

    check_cuda(cudaMemset(d_orth,  0, size_mat), "memset d_orth");
    check_cuda(cudaMemset(d_mu,    0, size_mu),  "memset d_mu");
    check_cuda(cudaMemset(d_B,     0, size_B),   "memset d_B");

    // -------------------------
    // 3. Launch lll_kernel
    // -------------------------
    int threads = 32; // power of 2 >= n (20)
    size_t shmem = threads * sizeof(double); // for parallel_reduce

    std::cout << "Launching LLL kernel with n=" << n
              << ", m=" << m << ", threads=" << threads << "\n";

    lll_kernel<<<1, threads, shmem>>>(d_basis, d_orth, d_mu, d_B, n, m, delta, eta);
    check_cuda(cudaGetLastError(), "lll_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "lll_kernel sync");

    // -------------------------
    // 4. Copy reduced basis back
    // -------------------------
    std::vector<double> basis_gpu(n * m);
    check_cuda(cudaMemcpy(basis_gpu.data(), d_basis,
                          size_mat, cudaMemcpyDeviceToHost),
               "memcpy d_basis -> host");

    // Also dump the double result and a rounded integer version
    write_matrix_double("basis_gpu_20x20_double.basis", basis_gpu, n, m);

    std::vector<long long> basis_gpu_int(n * m);
    for (unsigned int i = 0; i < n * m; i++) {
        basis_gpu_int[i] = static_cast<long long>(llround(basis_gpu[i]));
    }
    write_matrix_int("basis_gpu_20x20_int.basis", basis_gpu_int, n, m);

    // -------------------------
    // 5. Cleanup
    // -------------------------
    cudaFree(d_basis);
    cudaFree(d_orth);
    cudaFree(d_mu);
    cudaFree(d_B);

    std::cout << "Done. Wrote:\n"
              << "  basis_original_20x20.basis\n"
              << "  basis_gpu_20x20_double.basis\n"
              << "  basis_gpu_20x20_int.basis\n";
    return 0;
}
