// Reference kernels in plain C++. Every optimised kernel is tested against
// these. This file is compiled twice (see CMakeLists.txt): once normally and
// once with auto-vectorisation disabled, selected by PICOSPEC_VARIANT.
#include "picospec/kernels.hpp"
#include "picospec/requant.hpp"

#ifndef PICOSPEC_VARIANT
#define PICOSPEC_VARIANT scalar
#endif
#define PICOSPEC_CAT2(a, b) a##b
#define PICOSPEC_CAT(a, b) PICOSPEC_CAT2(a, b)
#define PICOSPEC_NAME(base) PICOSPEC_CAT(base, PICOSPEC_VARIANT)

namespace picospec {

void PICOSPEC_NAME(conv1d_f32_)(const float* in, float* out, std::size_t len,
                                const ConvF32& L) {
    const int kc = L.k * L.cin;  // length of one receptive field
    const int pad = L.k / 2;
    for (std::size_t i = 0; i < len; ++i) {
        const float* win = in + (static_cast<std::ptrdiff_t>(i) - pad) * L.cin;
        for (int co = 0; co < L.cout; ++co) {
            const float* w = L.w + static_cast<std::size_t>(co) * kc;
            float acc = L.b[co];
            for (int j = 0; j < kc; ++j) acc += win[j] * w[j];
            if (L.relu && acc < 0.0f) acc = 0.0f;
            out[i * L.cout + co] = acc;
        }
    }
}

void PICOSPEC_NAME(conv1d_q8_)(const int8_t* in, int8_t* out, std::size_t len,
                               const ConvQ8& L) {
    const int kc = L.k * L.cin;
    const int pad = L.k / 2;
    for (std::size_t i = 0; i < len; ++i) {
        const int8_t* win = in + (static_cast<std::ptrdiff_t>(i) - pad) * L.cin;
        for (int co = 0; co < L.cout; ++co) {
            const int8_t* w = L.w + static_cast<std::size_t>(co) * kc;
            int32_t acc = L.bias[co];
            for (int j = 0; j < kc; ++j)
                acc += static_cast<int32_t>(win[j]) * static_cast<int32_t>(w[j]);
            out[i * L.cout + co] =
                requantize(acc, L.mult[co], L.shift[co], L.out_zp, L.act_min, L.act_max);
        }
    }
}

}  // namespace picospec
