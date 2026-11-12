// Runtime CPU feature detection. A binary cannot assume the CPU it was
// compiled on is the CPU it runs on (a Cortex-A53 has NEON but no SDOT), so
// the choice of kernel is made when the program starts, not when it is built.
#include "picospec/kernels.hpp"

#if defined(__aarch64__) && defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1UL << 20)  // from <asm/hwcap.h>
#endif
#endif

namespace picospec {

bool has_neon() {
#if defined(__aarch64__)
    return true;  // Advanced SIMD is mandatory in AArch64
#else
    return false;
#endif
}

bool has_dotprod() {
#if defined(__aarch64__) && defined(__APPLE__)
    int value = 0;
    std::size_t size = sizeof(value);
    if (sysctlbyname("hw.optional.arm.FEAT_DotProd", &value, &size, nullptr, 0) == 0)
        return value != 0;
    return false;
#elif defined(__aarch64__) && defined(__linux__)
    return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#else
    return false;
#endif
}

}  // namespace picospec
