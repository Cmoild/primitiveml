#include <iostream>
#include <tensor.hpp>
#include <operations.hpp>
#include <math_kernels.h>
#include <matmul.hpp>
#include <chrono>
#include "../core/src/gemm_config.h"
#include <cblas.h>

const std::size_t num_of_runs = 100;

int main() {
    const std::size_t M = 1440;
    const std::size_t N = NC * 10;
    const std::size_t K = 2560;
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

    double all_time = 0.;
    for (std::size_t i = 0; i < num_of_runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        pml_sgemm(A, B, C, M, N, K, false, false, 1.f, 0.f, K, N, N);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "\rGEMM (i = " << i << "): " << elapsed.count() << " ms" << std::flush;
        all_time += elapsed.count();
    }
    std::cout << std::endl;
    std::cout << "Avg: " << all_time / num_of_runs << " ms" << std::endl;

    all_time = 0.;
    for (std::size_t i = 0; i < num_of_runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.f, A, K, B, N, 0.f, C, N);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "\rOpenBLAS SGEMM (i = " << i << "): " << elapsed.count() << " ms"
                  << std::flush;
        all_time += elapsed.count();
    }
    std::cout << std::endl;
    std::cout << "Avg (OpenBLAS): " << all_time / num_of_runs << " ms" << std::endl;

    return 0;
}
