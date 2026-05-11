#include <iostream>
#include <tensor.hpp>
#include <operations.hpp>
#include <math_kernels.h>
#include <matmul.hpp>
#include <chrono>
#include "../core/src/gemm_config.h"

int main() {
    const std::size_t M = MC * 10;
    const std::size_t N = NC * 10;
    const std::size_t K = KC * 10;
    std::cout << "M: " << M << std::endl;
    std::cout << "N: " << N << std::endl;
    std::cout << "K: " << K << std::endl;

    float* A = (float*)std::malloc(M * K * 4);
    for (std::size_t i = 0; i < M * K; i++)
        A[i] = (float)(rand() % 10);
    float* B = (float*)std::malloc(K * N * 4);
    for (std::size_t i = 0; i < K * N; i++)
        B[i] = (float)(rand() % 10);
    float* C = (float*)std::malloc(M * N * 4);
    float* C_naive = (float*)std::malloc(M * N * 4);

    auto start = std::chrono::high_resolution_clock::now();
    gemm(A, B, C, M, N, K, false, false, 1.f, 0.f, K, N, N);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "GEMM: " << elapsed.count() << " ms" << std::endl;

    pml::matmul_kernel(A, B, C_naive, M, N, K);

    float max_err = 0.f;
    for (std::size_t i = 0; i < M * N; i++)
        max_err = std::max(fabsf(C[i] - C_naive[i]), max_err);
    std::cout << "Max err: " << max_err << std::endl;
    return 0;
}
