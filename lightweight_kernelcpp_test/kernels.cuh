#pragma once

extern "C" {

__global__ void lll_kernel(double* basis,
                           double* orthogonal,
                           double* mu,
                           double* B,
                           unsigned int n,
                           unsigned int m,
                           double delta,
                           double eta);

}
