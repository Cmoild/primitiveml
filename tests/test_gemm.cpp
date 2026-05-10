#include <assert.h>
#include <cmath>
#include <math_kernels.h>
#include "../core/src/gemm_config.h"
#include <matmul.hpp>
#include <catch2/catch_test_macros.hpp>

static inline float rand_f() {
    return (float)(rand() % 10);
}

static float max_diff(const float* a, const float* b, size_t n) {
    float m = 0.f;
    for (size_t i = 0; i < n; ++i)
        m = std::max(m, fabsf(a[i] - b[i]));
    return m;
}

static void fill_random(float* A, size_t n) {
    for (size_t i = 0; i < n; ++i)
        A[i] = rand_f();
}

static void fill_ones(float* A, size_t n) {
    for (size_t i = 0; i < n; ++i)
        A[i] = 1.f;
}

static void fill_sequential(float* A, size_t n) {
    for (size_t i = 0; i < n; ++i)
        A[i] = (float)(i % 13);
}

TEST_CASE("GEMM random correctness", "[gemm]") {
    const size_t M = MC * 10;
    const size_t N = NC * 10;
    const size_t K = KC * 10;

    float* A = (float*)std::malloc(M * K * sizeof(float));
    float* B = (float*)std::malloc(K * N * sizeof(float));
    float* C = (float*)std::malloc(M * N * sizeof(float));
    float* C_ref = (float*)std::malloc(M * N * sizeof(float));

    fill_random(A, M * K);
    fill_random(B, K * N);

    for (size_t i = 0; i < M * N; ++i) {
        C[i] = 0.f;
        C_ref[i] = 0.f;
    }

    gemm(A, B, C, M, N, K, false, false, 1.f, 0.f, K, N, N);
    pml::matmul_kernel(A, B, C_ref, M, N, K);

    float err = max_diff(C, C_ref, M * N);

    REQUIRE(err < 1e-3f);

    free(A);
    free(B);
    free(C);
    free(C_ref);
}

TEST_CASE("GEMM all ones", "[gemm]") {
    const size_t M = MC * 4 + 3;
    const size_t N = NC * 4 + 5;
    const size_t K = KC * 4 + 7;

    float* A = (float*)std::malloc(M * K * sizeof(float));
    float* B = (float*)std::malloc(K * N * sizeof(float));
    float* C = (float*)std::malloc(M * N * sizeof(float));
    float* C_ref = (float*)std::malloc(M * N * sizeof(float));

    fill_ones(A, M * K);
    fill_ones(B, K * N);

    std::fill(C, C + M * N, 0.f);
    std::fill(C_ref, C_ref + M * N, 0.f);

    gemm(A, B, C, M, N, K, false, false, 1.f, 0.f, K, N, N);
    pml::matmul_kernel(A, B, C_ref, M, N, K);

    float expected = (float)K;

    float err = 0.f;
    for (size_t i = 0; i < M * N; ++i)
        err = std::max(err, fabsf(C[i] - expected));

    REQUIRE(err < 1e-3f);

    free(A);
    free(B);
    free(C);
    free(C_ref);
}

TEST_CASE("GEMM sequential pattern", "[gemm]") {
    const size_t M = MC * 3 + 1;
    const size_t N = NC * 3 + 2;
    const size_t K = KC * 3 + 3;

    float* A = (float*)std::malloc(M * K * sizeof(float));
    float* B = (float*)std::malloc(K * N * sizeof(float));
    float* C = (float*)std::malloc(M * N * sizeof(float));
    float* C_ref = (float*)std::malloc(M * N * sizeof(float));

    fill_sequential(A, M * K);
    fill_sequential(B, K * N);

    std::fill(C, C + M * N, 0.f);
    std::fill(C_ref, C_ref + M * N, 0.f);

    gemm(A, B, C, M, N, K, false, false, 1.f, 0.f, K, N, N);
    pml::matmul_kernel(A, B, C_ref, M, N, K);

    float err = max_diff(C, C_ref, M * N);

    REQUIRE(err < 1e-3f);

    free(A);
    free(B);
    free(C);
    free(C_ref);
}

TEST_CASE("GEMM edge-heavy sizes", "[gemm]") {
    const size_t M = MC * 2 + MC - 1;
    const size_t N = NC * 2 + NC - 1;
    const size_t K = KC * 2 + KC - 1;

    float* A = (float*)std::malloc(M * K * sizeof(float));
    float* B = (float*)std::malloc(K * N * sizeof(float));
    float* C = (float*)std::malloc(M * N * sizeof(float));
    float* C_ref = (float*)std::malloc(M * N * sizeof(float));

    fill_random(A, M * K);
    fill_random(B, K * N);

    std::fill(C, C + M * N, 0.f);
    std::fill(C_ref, C_ref + M * N, 0.f);

    gemm(A, B, C, M, N, K, false, false, 1.f, 0.f, K, N, N);
    pml::matmul_kernel(A, B, C_ref, M, N, K);

    float err = max_diff(C, C_ref, M * N);

    REQUIRE(err < 1e-3f);

    free(A);
    free(B);
    free(C);
    free(C_ref);
}

TEST_CASE("GEMM stress large", "[gemm][stress]") {
    const size_t M = 1024;
    const size_t N = 1024;
    const size_t K = 1024;

    float* A = (float*)std::malloc(M * K * sizeof(float));
    float* B = (float*)std::malloc(K * N * sizeof(float));
    float* C = (float*)std::malloc(M * N * sizeof(float));
    float* C_ref = (float*)std::malloc(M * N * sizeof(float));

    fill_random(A, M * K);
    fill_random(B, K * N);

    std::fill(C, C + M * N, 0.f);
    std::fill(C_ref, C_ref + M * N, 0.f);

    gemm(A, B, C, M, N, K, false, false, 1.f, 0.f, K, N, N);
    pml::matmul_kernel(A, B, C_ref, M, N, K);

    float err = max_diff(C, C_ref, M * N);

    REQUIRE(err < 1e-3f);

    free(A);
    free(B);
    free(C);
    free(C_ref);
}
