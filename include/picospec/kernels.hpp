// Convolution kernels: the hot loops of the engine.
//
// Data layout is channel-last ("NLC"): sample i of channel c lives at
// buf[i * channels + c]. With this layout the receptive field of one output
// sample (k taps x cin channels) is a single contiguous run of k*cin values,
// so a convolution becomes a plain dot product between that run and the
// filter. That is what makes the SIMD versions simple and fast.
//
// Padding contract: `in` points at sample 0, and the caller guarantees that
// (k/2) samples before it and after the last sample are readable and hold
// the "zero" value (0.0f for float, the input zero-point for int8).
#pragma once

#include <cstddef>
#include <cstdint>

namespace picospec {

// Non-owning description of a float convolution layer.
struct ConvF32 {
    int cin = 0, cout = 0, k = 0;
    bool relu = false;
    const float* w = nullptr;  // [cout][k][cin]
    const float* b = nullptr;  // [cout]
};

// Non-owning description of an int8 convolution layer.
// Real value = scale * (q - zero_point). The scales never appear here: they
// are folded offline into a fixed-point multiplier and a right shift, so the
// kernel needs integer arithmetic only.
struct ConvQ8 {
    int cin = 0, cout = 0, k = 0;
    int32_t out_zp = 0;                     // output zero-point
    int32_t act_min = -128, act_max = 127;  // clamp range (ReLU is fused here)
    const int8_t* w = nullptr;       // [cout][k][cin], symmetric per-channel
    const int32_t* bias = nullptr;   // [cout], input zero-point already folded in
    const int32_t* mult = nullptr;   // [cout], Q0.31 multiplier in [2^30, 2^31)
    const int32_t* shift = nullptr;  // [cout], right shift >= 0
};

using ConvF32Fn = void (*)(const float* in, float* out, std::size_t len, const ConvF32& layer);
using ConvQ8Fn = void (*)(const int8_t* in, int8_t* out, std::size_t len, const ConvQ8& layer);

// Portable reference kernels. "novec" is the same source compiled with the
// compiler's auto-vectoriser switched off, so benchmarks can separate what
// the compiler does on its own from what the hand-written NEON adds.
void conv1d_f32_novec(const float*, float*, std::size_t, const ConvF32&);
void conv1d_q8_novec(const int8_t*, int8_t*, std::size_t, const ConvQ8&);
void conv1d_f32_scalar(const float*, float*, std::size_t, const ConvF32&);
void conv1d_q8_scalar(const int8_t*, int8_t*, std::size_t, const ConvQ8&);

#if defined(__aarch64__)
// Armv8.0-A NEON: widening multiply (SMULL) + pairwise accumulate (SADALP).
void conv1d_f32_neon(const float*, float*, std::size_t, const ConvF32&);
void conv1d_q8_neon(const int8_t*, int8_t*, std::size_t, const ConvQ8&);
// Armv8.2-A DotProd extension: SDOT does 16 int8 multiplies and 4 sums in
// one instruction. Only call after has_dotprod() returns true.
void conv1d_q8_neon_dotprod(const int8_t*, int8_t*, std::size_t, const ConvQ8&);
#endif

// Runtime CPU feature detection.
bool has_neon();
bool has_dotprod();

}  // namespace picospec
