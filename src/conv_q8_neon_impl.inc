// Shared body of the two NEON int8 kernels. The including file defines
//   PICOSPEC_Q8_FN     the function name
//   PICOSPEC_DOT16     acc += dot(x, w) for one 16-byte chunk
// so the Armv8.0 and the DotProd kernels differ in exactly one instruction
// sequence and nothing else.
//
// Strategy: four output channels at a time. Each channel keeps its own
// int32x4 accumulator; after the taps are consumed the four accumulators are
// reduced into one vector [sum0, sum1, sum2, sum3], and the whole
// requantise / add zero-point / clamp / narrow tail runs once on that vector.

namespace picospec {

namespace {

inline int32_t tail_dot(const int8_t* a, const int8_t* b, int from, int to) {
    int32_t s = 0;
    for (int j = from; j < to; ++j) s += static_cast<int32_t>(a[j]) * static_cast<int32_t>(b[j]);
    return s;
}

}  // namespace

void PICOSPEC_Q8_FN(const int8_t* in, int8_t* out, std::size_t len, const ConvQ8& L) {
    const int kc = L.k * L.cin;
    const int kc16 = kc & ~15;  // part of the receptive field handled 16 bytes at a time
    const int pad = L.k / 2;
    const int cout4 = L.cout & ~3;

    const int32x4_t v_zp = vdupq_n_s32(L.out_zp);
    const int32x4_t v_min = vdupq_n_s32(L.act_min);
    const int32x4_t v_max = vdupq_n_s32(L.act_max);

    for (std::size_t i = 0; i < len; ++i) {
        const int8_t* win = in + (static_cast<std::ptrdiff_t>(i) - pad) * L.cin;
        int8_t* o = out + i * L.cout;

        int co = 0;
        for (; co < cout4; co += 4) {
            const int8_t* w0 = L.w + static_cast<std::size_t>(co) * kc;
            const int8_t* w1 = w0 + kc;
            const int8_t* w2 = w1 + kc;
            const int8_t* w3 = w2 + kc;
            int32x4_t a0 = vdupq_n_s32(0), a1 = a0, a2 = a0, a3 = a0;
            for (int j = 0; j < kc16; j += 16) {
                const int8x16_t x = vld1q_s8(win + j);
                a0 = PICOSPEC_DOT16(a0, x, vld1q_s8(w0 + j));
                a1 = PICOSPEC_DOT16(a1, x, vld1q_s8(w1 + j));
                a2 = PICOSPEC_DOT16(a2, x, vld1q_s8(w2 + j));
                a3 = PICOSPEC_DOT16(a3, x, vld1q_s8(w3 + j));
            }
            // Horizontal reduction: [a0 a1 a2 a3] -> [sum(a0) sum(a1) sum(a2) sum(a3)].
            int32x4_t acc = vpaddq_s32(vpaddq_s32(a0, a1), vpaddq_s32(a2, a3));
            if (kc16 != kc) {  // leftover taps (e.g. the 1-channel first layer)
                const int32_t t[4] = {tail_dot(win, w0, kc16, kc), tail_dot(win, w1, kc16, kc),
                                      tail_dot(win, w2, kc16, kc), tail_dot(win, w3, kc16, kc)};
                acc = vaddq_s32(acc, vld1q_s32(t));
            }
            acc = vaddq_s32(acc, vld1q_s32(L.bias + co));

            // Requantise: SQRDMULH then rounding right shift (SRSHL by -shift).
            acc = vqrdmulhq_s32(acc, vld1q_s32(L.mult + co));
            acc = vrshlq_s32(acc, vnegq_s32(vld1q_s32(L.shift + co)));
            acc = vaddq_s32(acc, v_zp);
            acc = vminq_s32(vmaxq_s32(acc, v_min), v_max);

            // Values are already inside [-128, 127], so narrowing cannot saturate.
            const int16x4_t n16 = vmovn_s32(acc);
            const int8x8_t n8 = vmovn_s16(vcombine_s16(n16, n16));
            const uint32_t packed = vget_lane_u32(vreinterpret_u32_s8(n8), 0);
            std::memcpy(o + co, &packed, 4);
        }
        for (; co < L.cout; ++co) {  // remaining 1-3 output channels
            const int8_t* w = L.w + static_cast<std::size_t>(co) * kc;
            int32x4_t a = vdupq_n_s32(0);
            for (int j = 0; j < kc16; j += 16)
                a = PICOSPEC_DOT16(a, vld1q_s8(win + j), vld1q_s8(w + j));
            const int32_t acc = vaddvq_s32(a) + tail_dot(win, w, kc16, kc) + L.bias[co];
            o[co] = requantize(acc, L.mult[co], L.shift[co], L.out_zp, L.act_min, L.act_max);
        }
    }
}

}  // namespace picospec
