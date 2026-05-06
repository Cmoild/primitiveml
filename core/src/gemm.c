#include <math_kernels.h>
#include <memory.h>
#include <omp.h>
#include "gemm_config.h"

static inline void pack_A_tile(const float* A, float* pack_A, const size_t row, const size_t col,
                               bool a_transposed, const size_t lda) {
    if (a_transposed) {
        for (size_t i = 0; i < MC; i++) {
            for (size_t j = 0; j < KC; j++) {
                pack_A[KC * i + j] = A[(col + j) * lda + row + i];
            }
        }
    } else {
        for (size_t i = 0; i < MC; i++) {
            memcpy(pack_A + KC * i, A + (row + i) * lda + col, KC * sizeof(float));
        }
    }
}

static inline void pack_B_tile(const float* B, float* pack_B, const size_t row, const size_t col,
                               bool b_transposed, const size_t ldb) {
    if (b_transposed) {
        for (size_t j = 0; j < NC; j++) {
            memcpy(pack_B + KC * j, B + (col + j) * ldb + row, KC * sizeof(float));
        }
    } else {
        for (size_t i = 0; i < KC; i++) {
            for (size_t j = 0; j < NC; j++) {
                pack_B[KC * j + i] = B[(row + i) * ldb + col + j];
            }
        }
    }
}

void gemm(const float* A, const float* B, float* C, size_t M, size_t N, size_t K,
          const bool a_transposed, const bool b_transposed, const float alpha, const float beta,
          const size_t lda, const size_t ldb) {
#pragma omp parallel
    {
        float* pack_A = aligned_alloc(64, MC * KC * sizeof(float));
        float* pack_B = aligned_alloc(64, KC * NC * sizeof(float));

#pragma omp for schedule(static)
        for (size_t jc = 0; jc + NC <= N; jc += NC) {

            for (size_t pc = 0; pc + KC <= K; pc += KC) {

                pack_B_tile(B, pack_B, pc, jc, b_transposed, ldb);

                for (size_t ic = 0; ic + MC <= M; ic += MC) {

                    pack_A_tile(A, pack_A, ic, pc, a_transposed, lda);

                    gemm_microkernel(...);
                }
            }
        }

        free(pack_A);
        free(pack_B);
    }

    // TODO: process tails
}
