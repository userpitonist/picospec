// int8 kernel for CPUs with the Armv8.2-A DotProd extension (SDOT).
// This file alone is compiled with -march=armv8.2-a+dotprod; the rest of the
// library stays at the baseline so the binary still starts on older cores.
// The caller must check has_dotprod() first.
#if defined(__aarch64__)

#include <arm_neon.h>
#include <cstring>

#include "picospec/kernels.hpp"
#include "picospec/requant.hpp"

#define PICOSPEC_Q8_FN conv1d_q8_neon_dotprod
#define PICOSPEC_DOT16(acc, x, w) vdotq_s32(acc, x, w)
#include "conv_q8_neon_impl.inc"

#endif  // __aarch64__
