#include <math_kernels.h>
#include <memory.h>
#include <stdlib.h>
#include <omp.h>
#include <stdio.h>
#include <threads.h>
#include "gemm_config.h"

static inline void pack_A_panel(const float* A, float* pack_A, const size_t ic, const size_t jc,
                                bool a_transposed, const size_t lda) {
    if (a_transposed) {
        for (size_t i = 0; i < MC; i++) {
            for (size_t j = 0; j < KC; j++) {
                pack_A[KC * i + j] = A[(jc + j) * lda + ic + i];
            }
        }
    } else {
        for (size_t i = 0; i < MC; i++) {
            memcpy(pack_A + KC * i, A + (ic + i) * lda + jc, KC * sizeof(float));
        }
    }
}

static inline void pack_B_panel(const float* B, float* pack_B, const size_t pc, const size_t jc,
                                bool b_transposed, const size_t ldb) {
    if (b_transposed) {
        for (size_t i = 0; i < KC; i++) {
            for (size_t j = 0; j < NC; j++) {
                pack_B[i * NC + j] = B[(jc + j) * ldb + (pc + i)];
            }
        }
    } else {
        for (size_t i = 0; i < KC; i++) {
            memcpy(pack_B + NC * i, B + (pc + i) * ldb + jc, NC * sizeof(float));
        }
    }
}

static inline void pack_A_panel_tail(const float* A, float* pack_A, size_t ic, size_t pc, size_t mb,
                                     size_t kb, bool a_transposed, size_t lda) {
    memset(pack_A, 0, MC * KC * sizeof(float));

    if (!a_transposed) {
        for (size_t i = 0; i < mb; ++i) {
            memcpy(pack_A + i * KC, A + (ic + i) * lda + pc, kb * sizeof(float));
        }
    } else {
        for (size_t i = 0; i < mb; ++i) {
            for (size_t k = 0; k < kb; ++k) {
                pack_A[i * KC + k] = A[(pc + k) * lda + (ic + i)];
            }
        }
    }
}

static inline void pack_B_panel_tail(const float* B, float* pack_B, size_t pc, size_t jc, size_t kb,
                                     size_t nb, bool b_transposed, size_t ldb) {
    memset(pack_B, 0, KC * NC * sizeof(float));

    if (!b_transposed) {
        for (size_t k = 0; k < kb; ++k) {
            memcpy(pack_B + k * NC, B + (pc + k) * ldb + jc, nb * sizeof(float));
        }
    } else {
        for (size_t k = 0; k < kb; ++k) {
            for (size_t j = 0; j < nb; ++j) {
                pack_B[k * NC + j] = B[(jc + j) * ldb + (pc + k)];
            }
        }
    }
}

static inline void scale_C(float* C, size_t M, size_t N, size_t ldc, float beta) {
    if (beta == 1.0f)
        return;

    for (size_t i = 0; i < M; ++i) {
        float* row = C + i * ldc;
        if (beta == 0.0f) {
            memset(row, 0, N * sizeof(float));
        } else {
            for (size_t j = 0; j < N; ++j) {
                row[j] *= beta;
            }
        }
    }
}

void gemm(const float* A, const float* B, float* C, size_t M, size_t N, size_t K,
          const bool a_transposed, const bool b_transposed, const float alpha, const float beta,
          const size_t lda, const size_t ldb, const size_t ldc) {

    scale_C(C, M, N, ldc, beta);

#pragma omp parallel
    {
        float* pack_A = aligned_alloc(64, MC * KC * sizeof(float));
        float* pack_B = aligned_alloc(64, KC * NC * sizeof(float));
        float* Cblk = (float*)aligned_alloc(64, MC * NC * sizeof(float));

#pragma omp for schedule(static)
        for (size_t jc = 0; jc < N; jc += NC) {
            const size_t nb = (jc + NC <= N) ? NC : (N - jc);

            for (size_t pc = 0; pc < K; pc += KC) {
                const size_t kb = (pc + KC <= K) ? KC : (K - pc);

                if (nb == NC && kb == KC) {
                    pack_B_panel(B, pack_B, pc, jc, b_transposed, ldb);
                } else {
                    pack_B_panel_tail(B, pack_B, pc, jc, kb, nb, b_transposed, ldb);
                }

                for (size_t ic = 0; ic < M; ic += MC) {
                    const size_t mb = (ic + MC <= M) ? MC : (M - ic);

                    if (mb == MC && kb == KC) {
                        pack_A_panel(A, pack_A, ic, pc, a_transposed, lda);
                    } else {
                        pack_A_panel_tail(A, pack_A, ic, pc, mb, kb, a_transposed, lda);
                    }

                    if (mb == MC && nb == NC && kb == KC) {
                        sgemm_microkernel_avx2(pack_A, pack_B, C + ic * ldc + jc, ldc, alpha);
                    } else {
                        for (size_t i = 0; i < mb; ++i) {
                            memcpy(Cblk + i * NC, C + (ic + i) * ldc + jc, nb * sizeof(float));
                            if (nb < NC) {
                                memset(Cblk + i * NC + nb, 0, (NC - nb) * sizeof(float));
                            }
                        }
                        if (mb < MC) {
                            memset(Cblk + mb * NC, 0, (MC - mb) * NC * sizeof(float));
                        }

                        sgemm_microkernel_avx2(pack_A, pack_B, Cblk, NC, alpha);

                        for (size_t i = 0; i < mb; ++i) {
                            memcpy(C + (ic + i) * ldc + jc, Cblk + i * NC, nb * sizeof(float));
                        }
                    }
                }
            }
        }

        free(pack_A);
        free(pack_B);
        free(Cblk);
    }
}
