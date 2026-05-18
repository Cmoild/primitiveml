#ifndef CPU_FLAGS_H
#define CPU_FLAGS_H

#ifdef __cplusplus
extern "C" {
#endif

bool __attribute__((sysv_abi)) avx2_fma_supported();

#ifdef __cplusplus
}
#endif

#endif
