#include <math_kernels.h>
#include <immintrin.h>
#include <algorithm>

void exp_avx2_intrin(float* operand, float* result, size_t n) {
    size_t i = 0;

    const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
    const __m256 a0 = _mm256_set1_ps(1.0f);
    const __m256 a1 = _mm256_set1_ps(0.6931471805599453f);
    const __m256 a2 = _mm256_set1_ps(0.2402265069591007f);
    const __m256 a3 = _mm256_set1_ps(0.05550410866482158f);
    const __m256 a4 = _mm256_set1_ps(0.009618129107628477f);
    const __m256 a5 = _mm256_set1_ps(0.001333355814642844f);

    const __m256i bias = _mm256_set1_epi32(127);
    const __m256i maxe = _mm256_set1_epi32(254);
    const __m256i zero = _mm256_setzero_si256();

    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(operand + i);

        // y = x * log2(e)
        __m256 y = _mm256_mul_ps(x, log2e);

        // n = floor(y)
        __m256 nf = _mm256_round_ps(y, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);

        // f = y - n
        __m256 f = _mm256_sub_ps(y, nf);

        // polynomial approximation for 2^f
        __m256 p = a4;
        p = _mm256_fmadd_ps(p, f, a5); // a4 + f * a5
        p = _mm256_fmadd_ps(p, f, a3); // a3 + f * (...)
        p = _mm256_fmadd_ps(p, f, a2); // a2 + f * (...)
        p = _mm256_fmadd_ps(p, f, a1); // a1 + f * (...)
        p = _mm256_fmadd_ps(p, f, a0); // a0 + f * (...)

        // int(n)
        __m256i ni = _mm256_cvtps_epi32(nf);

        // n + 127
        __m256i exp = _mm256_add_epi32(ni, bias);

        // clamp to [0, 254]
        exp = _mm256_max_epi32(exp, zero);
        exp = _mm256_min_epi32(exp, maxe);

        // build float bits for 2^n
        exp = _mm256_slli_epi32(exp, 23);
        __m256 pow2n = _mm256_castsi256_ps(exp);

        // result = 2^n * 2^f
        __m256 out = _mm256_mul_ps(p, pow2n);

        _mm256_storeu_ps(result + i, out);
    }

    for (; i < n; ++i) {
        __m128 x = _mm_load_ss(operand + i);

        __m128 y = _mm_mul_ss(x, _mm_set_ss(1.4426950408889634f));
        __m128 nf = _mm_round_ss(_mm_setzero_ps(), y, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
        __m128 f = _mm_sub_ss(y, nf);

        __m128 p = _mm_set_ss(0.009618129107628477f);
        p = _mm_fmadd_ss(p, f, _mm_set_ss(0.001333355814642844f));
        p = _mm_fmadd_ss(p, f, _mm_set_ss(0.05550410866482158f));
        p = _mm_fmadd_ss(p, f, _mm_set_ss(0.2402265069591007f));
        p = _mm_fmadd_ss(p, f, _mm_set_ss(0.6931471805599453f));
        p = _mm_fmadd_ss(p, f, _mm_set_ss(1.0f));

        int ni = _mm_cvtss_si32(nf);
        int exp = std::clamp(ni + 127, 0, 254);
        __m128 pow2n = _mm_castsi128_ps(_mm_cvtsi32_si128(exp << 23));

        __m128 out = _mm_mul_ss(p, pow2n);
        _mm_store_ss(result + i, out);
    }
}
