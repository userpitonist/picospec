// Hand-written NEON kernels that run on every AArch64 CPU (Armv8.0-A).
#if defined(__aarch64__)

#include <arm_neon.h>
#include <cstring>

#include "picospec/kernels.hpp"
#include "picospec/requant.hpp"

namespace picospec {

namespace {

// acc += dot(x, w) over 16 int8 pairs without the DotProd extension:
// SMULL/SMULL2 widen 8x8 -> 16 bit, SADALP adds neighbouring products into
// the 32-bit accumulator. (-128 * -128 = 16384 fits in int16.)
inline int32x4_t dot16_mull(int32x4_t acc, int8x16_t x, int8x16_t w) {
    const int16x8_t lo = vmull_s8(vget_low_s8(x), vget_low_s8(w));
    const int16x8_t hi = vmull_high_s8(x, w);
    acc = vpadalq_s16(acc, lo);
    return vpadalq_s16(acc, hi);
}

}  // namespace

void conv1d_f32_neon(const float* in, float* out, std::size_t out_len, const ConvF32& L) {
    const int kc = L.k * L.cin;
    const int kc16 = kc & ~15;
    const int kc4 = kc & ~3;
    for (std::size_t i = 0; i < out_len; ++i) {
        const float* win = in + (static_cast<std::ptrdiff_t>(i) * L.stride - L.pad) * L.cin;
        for (int co = 0; co < L.cout; ++co) {
            const float* w = L.w + static_cast<std::size_t>(co) * kc;
            // Four independent accumulators: an FMA has a latency of several
            // cycles, so a single accumulator would make every FMA wait for
            // the previous one. Four chains let the core overlap them.
            float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
            int j = 0;
            for (; j < kc16; j += 16) {
                a0 = vfmaq_f32(a0, vld1q_f32(win + j), vld1q_f32(w + j));
                a1 = vfmaq_f32(a1, vld1q_f32(win + j + 4), vld1q_f32(w + j + 4));
                a2 = vfmaq_f32(a2, vld1q_f32(win + j + 8), vld1q_f32(w + j + 8));
                a3 = vfmaq_f32(a3, vld1q_f32(win + j + 12), vld1q_f32(w + j + 12));
            }
            for (; j < kc4; j += 4) a0 = vfmaq_f32(a0, vld1q_f32(win + j), vld1q_f32(w + j));
            float acc = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3))) + L.b[co];
            for (; j < kc; ++j) acc += win[j] * w[j];
            if (L.relu && acc < 0.0f) acc = 0.0f;
            out[i * L.cout + co] = acc;
        }
    }
}

}  // namespace picospec

#define PICOSPEC_Q8_FN conv1d_q8_neon
#define PICOSPEC_DOT16(acc, x, w) dot16_mull(acc, x, w)
#include "conv_q8_neon_impl.inc"

#endif  // __aarch64__
