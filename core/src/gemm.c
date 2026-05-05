#include <math_kernels.h>
#include <memory.h>
#include <omp.h>

#define KC 128
#define MC 256
#define NC 384

static inline void pack_A_tile(const float* A, float* pack_A);

static inline void pack_B_tile(const float* B, float* pack_B, const size_t row, const size_t col,
                               const size_t K, const size_t N, const size_t n_tile_rows,
                               const size_t n_tile_cols, bool b_transposed) {
    if (b_transposed) {
        for (size_t i = 0; i < n_tile_rows; i++) {
            memcpy(pack_B + KC * i, B + (row + i) * K + col, n_tile_cols * sizeof(float));
        }
    } else {
        for (size_t i = 0; i < n_tile_rows; i++) {
            for (size_t j = 0; j < n_tile_cols; j++) {
                pack_B[KC * i] = B[(row + i) * K + col + j];
            }
        }
    }
}

void gemm(const float* A, const float* B, float* C, size_t M, size_t N, size_t K,
          const bool a_transposed, const bool b_transposed, const float alpha, const float beta) {
#pragma omp parallel
    {
        float* pack_A = aligned_alloc(64, MC * KC * sizeof(float));
        float* pack_B = aligned_alloc(64, KC * NC * sizeof(float));

#pragma omp for schedule(static)
        for (size_t jc = 0; jc < N; jc += NC) {
            const size_t n = NC < (N - jc) ? NC : (N - jc);

            for (size_t pc = 0; pc < K; pc += KC) {
                const size_t k = KC < (K - pc) ? KC : (K - pc);

                pack_B_tile(B, pack_B, pc, jc, K, N, k, n, b_transposed);

                for (size_t ic = 0; ic < M; ic += MC) {
                    const size_t m = MC < (M - ic) ? MC : (M - ic);

                    pack_A_tile(...);

                    gemm_microkernel(...);
                }
            }
        }

        free(pack_A);
        free(pack_B);
    }
}
