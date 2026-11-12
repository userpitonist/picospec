// Fixed-point requantisation shared by every int8 kernel.
//
// A convolution accumulates int32 sums at scale (s_in * s_w). The output
// wants scale s_out, so each sum is multiplied by M = s_in * s_w / s_out.
// M is a real number below 1; it is stored as M = mult * 2^-31 * 2^-shift
// with mult in [2^30, 2^31). The two steps below map one-to-one onto the
// NEON instructions SQRDMULH and SRSHL, which is why scalar and NEON results
// are bit-identical rather than merely close.
#pragma once

#include <cstdint>

namespace picospec {

// round(a * b / 2^31), ties rounded towards +infinity.
inline int32_t rounding_doubling_high_mul(int32_t a, int32_t b) {
    const int64_t p = static_cast<int64_t>(a) * static_cast<int64_t>(b);
    return static_cast<int32_t>((p + (int64_t{1} << 30)) >> 31);
}

// round(v / 2^n), ties rounded towards +infinity.
inline int32_t rounding_shift_right(int32_t v, int32_t n) {
    if (n == 0) return v;
    const int64_t r = static_cast<int64_t>(v) + (int64_t{1} << (n - 1));
    return static_cast<int32_t>(r >> n);
}

inline int8_t requantize(int32_t acc, int32_t mult, int32_t shift, int32_t out_zp,
                         int32_t act_min, int32_t act_max) {
    int32_t v = rounding_shift_right(rounding_doubling_high_mul(acc, mult), shift) + out_zp;
    if (v < act_min) v = act_min;
    if (v > act_max) v = act_max;
    return static_cast<int8_t>(v);
}

}  // namespace picospec
