// Unit tests. No framework: a CHECK macro and a list of functions keeps the
// whole test suite readable in one sitting.
//
// What is being protected:
//   1. the fixed-point rounding rules (requant.hpp)
//   2. every optimised kernel agrees with the plain reference kernel -
//      bit-for-bit for int8, within float rounding for float32 - for
//      "same", strided, "valid" and dense-shaped layers
//   3. stride and padding give the output lengths and values worked out by hand
//   4. the engine reproduces the Python reference on real model files
//   5. malformed model files are rejected with an error, never a crash
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "picospec/model.hpp"
#include "picospec/requant.hpp"

using namespace picospec;

namespace {

int g_checks = 0, g_failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        ++g_checks;                                                            \
        if (!(cond)) {                                                         \
            ++g_failures;                                                      \
            std::printf("  FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
        }                                                                      \
    } while (0)

std::string g_root = ".";

template <typename T>
std::vector<T> read_file(const std::string& rel) {
    std::ifstream f(g_root + "/" + rel, std::ios::binary);
    if (!f) throw std::runtime_error("missing test file " + rel);
    std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<T> v(bytes.size() / sizeof(T));
    std::memcpy(v.data(), bytes.data(), v.size() * sizeof(T));
    return v;
}

// --------------------------------------------------------------- 1. rounding

void test_requant_rounding() {
    // 0.5 in Q0.31 is 2^30: multiplying by it halves, rounding ties upward.
    const int32_t half = int32_t{1} << 30;
    CHECK(rounding_doubling_high_mul(4, half) == 2);
    CHECK(rounding_doubling_high_mul(3, half) == 2);    //  1.5 ->  2
    CHECK(rounding_doubling_high_mul(-3, half) == -1);  // -1.5 -> -1
    CHECK(rounding_doubling_high_mul(1, half) == 1);    //  0.5 ->  1
    CHECK(rounding_doubling_high_mul(-1, half) == 0);   // -0.5 ->  0
    CHECK(rounding_shift_right(5, 0) == 5);
    CHECK(rounding_shift_right(5, 1) == 3);    //  2.5 ->  3
    CHECK(rounding_shift_right(-5, 1) == -2);  // -2.5 -> -2
    CHECK(rounding_shift_right(7, 2) == 2);    //  1.75 -> 2
    CHECK(rounding_shift_right(-7, 2) == -2);
    // Clamp and zero-point.
    CHECK(requantize(1000000, half, 0, 0, -128, 127) == 127);
    CHECK(requantize(-1000000, half, 0, 0, -128, 127) == -128);
    CHECK(requantize(-40, half, 0, -128, -128, 127) == -128);  // ReLU floor at zp
    CHECK(requantize(40, half, 2, 10, -128, 127) == 15);

    // Against double arithmetic on random values: fixed point must land
    // within one unit of the exact real-valued result.
    std::mt19937 rng(1);
    std::uniform_int_distribution<int32_t> acc_d(-2000000, 2000000), sh_d(0, 12);
    std::uniform_int_distribution<int32_t> m_d(int32_t{1} << 30, INT32_MAX);
    for (int i = 0; i < 20000; ++i) {
        const int32_t acc = acc_d(rng), m = m_d(rng), sh = sh_d(rng);
        const double exact = static_cast<double>(acc) * m / 2147483648.0 / std::ldexp(1.0, sh);
        const int32_t got = rounding_shift_right(rounding_doubling_high_mul(acc, m), sh);
        CHECK(std::fabs(got - exact) <= 1.0);
    }
}

// ------------------------------------------------------ 2. kernel agreement

struct Geometry {
    int cin, cout, k, stride, pad;
};

const Geometry kGeometries[] = {
    // "same" layers: stride 1, pad k/2 (the denoiser)
    {1, 16, 5, 1, 2}, {16, 16, 5, 1, 2}, {16, 1, 5, 1, 2}, {3, 7, 3, 1, 1}, {8, 5, 7, 1, 3},
    {1, 1, 1, 1, 0},  {5, 4, 9, 1, 4},   {32, 6, 3, 1, 1}, {2, 3, 15, 1, 7}, {7, 9, 5, 1, 2},
    // down-sampling layers: stride 2 or more (the classifier's convolutions)
    {1, 16, 9, 2, 4}, {16, 32, 7, 2, 3}, {32, 64, 5, 2, 2}, {3, 5, 4, 2, 2}, {4, 6, 5, 3, 1},
    {3, 2, 8, 4, 3},
    // "valid" layers: pad 0; with k equal to the input length this is a dense layer
    {6, 7, 5, 1, 0},  {5, 3, 6, 2, 0},   {64, 30, 17, 1, 0}, {16, 30, 64, 1, 0}};

const std::size_t kLengths[] = {1, 2, 17, 64, 256};

struct RandomQ8Layer {
    std::vector<int8_t> w;
    std::vector<int32_t> bias, mult, shift;
    ConvQ8 view;
    RandomQ8Layer(const Geometry& g, bool relu, std::mt19937& rng) {
        std::uniform_int_distribution<int> wd(-127, 127), bd(-20000, 20000), sd(4, 11), zd(-128, 20);
        std::uniform_int_distribution<int32_t> md(int32_t{1} << 30, INT32_MAX);
        w.resize(static_cast<std::size_t>(g.cin) * g.cout * g.k);
        for (auto& v : w) v = static_cast<int8_t>(wd(rng));
        for (int c = 0; c < g.cout; ++c) {
            bias.push_back(bd(rng));
            mult.push_back(md(rng));
            shift.push_back(sd(rng));
        }
        const int zp = zd(rng);
        view = {g.cin, g.cout, g.k, g.stride, g.pad, zp, relu ? zp : -128, 127,
                w.data(), bias.data(), mult.data(), shift.data()};
    }
};

void test_q8_kernels_bit_exact() {
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> xd(-128, 127);
    const ConvQ8Fn ref = conv1d_q8_novec;
    int cases = 0, dense_cases = 0;
    for (const Geometry& g : kGeometries) {
        for (std::size_t len : kLengths) {
            const std::size_t out_len = conv_out_len(len, g.k, g.stride, g.pad);
            if (out_len == 0) continue;  // kernel longer than the padded input
            for (bool relu : {false, true}) {
                RandomQ8Layer L(g, relu, rng);
                std::vector<int8_t> buf((len + 2 * g.pad) * g.cin);
                for (auto& v : buf) v = static_cast<int8_t>(xd(rng));
                const int8_t* in = buf.data() + g.pad * g.cin;
                std::vector<int8_t> expect(out_len * g.cout), got(out_len * g.cout);
                ref(in, expect.data(), out_len, L.view);

                std::vector<ConvQ8Fn> fns = {conv1d_q8_scalar};
#if defined(__aarch64__)
                fns.push_back(conv1d_q8_neon);
                if (has_dotprod()) fns.push_back(conv1d_q8_neon_dotprod);
#endif
                for (ConvQ8Fn fn : fns) {
                    std::fill(got.begin(), got.end(), int8_t{0});
                    fn(in, got.data(), out_len, L.view);
                    CHECK(got == expect);
                }
                ++cases;
                if (g.pad == 0 && static_cast<std::size_t>(g.k) == len) ++dense_cases;
            }
        }
    }
    CHECK(cases > 100);
    CHECK(dense_cases >= 4);  // the dense-shaped layers really were exercised
}

void test_f32_kernels_agree() {
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (const Geometry& g : kGeometries) {
        for (std::size_t len : kLengths) {
            const std::size_t out_len = conv_out_len(len, g.k, g.stride, g.pad);
            if (out_len == 0) continue;
            std::vector<float> w(static_cast<std::size_t>(g.cin) * g.cout * g.k), b(g.cout);
            for (auto& v : w) v = d(rng);
            for (auto& v : b) v = d(rng);
            const ConvF32 L{g.cin, g.cout, g.k, g.stride, g.pad, true, w.data(), b.data()};
            std::vector<float> buf((len + 2 * g.pad) * g.cin);
            for (auto& v : buf) v = d(rng);
            const float* in = buf.data() + g.pad * g.cin;
            std::vector<float> expect(out_len * g.cout), got(out_len * g.cout);
            conv1d_f32_novec(in, expect.data(), out_len, L);

            std::vector<ConvF32Fn> fns = {conv1d_f32_scalar};
#if defined(__aarch64__)
            fns.push_back(conv1d_f32_neon);
#endif
            for (ConvF32Fn fn : fns) {
                std::fill(got.begin(), got.end(), 0.0f);
                fn(in, got.data(), out_len, L);
                float worst = 0.0f;
                for (std::size_t i = 0; i < got.size(); ++i)
                    worst = std::max(worst, std::fabs(got[i] - expect[i]));
                // Different summation order -> not bit-identical, but tiny.
                CHECK(worst < 1e-3f);
            }
        }
    }
}

// --------------------------------------------- 3. stride / padding / shapes

bool throws(const std::function<void()>& f) {
    try {
        f();
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

void test_stride_and_valid_shapes() {
    // floor((len + 2*pad - k) / stride) + 1, or 0 when the kernel does not fit.
    CHECK(conv_out_len(1000, 9, 2, 4) == 500);  // the classifier's five layers
    CHECK(conv_out_len(500, 7, 2, 3) == 250);
    CHECK(conv_out_len(250, 5, 2, 2) == 125);
    CHECK(conv_out_len(125, 5, 2, 2) == 63);
    CHECK(conv_out_len(63, 63, 1, 0) == 1);
    CHECK(conv_out_len(5, 3, 1, 1) == 5);
    CHECK(conv_out_len(5, 3, 2, 1) == 3);
    CHECK(conv_out_len(6, 3, 3, 0) == 2);
    CHECK(conv_out_len(5, 5, 1, 0) == 1);
    CHECK(conv_out_len(4, 5, 1, 0) == 0);
    CHECK(conv_out_len(0, 1, 1, 0) == 0);

    Workspace ws;
    {
        // One stride-2 layer, taps 1, 10, 100, pad 1. By hand:
        //   out[i] = x[2i-1] + 10 x[2i] + 100 x[2i+1], zero outside the input.
        FloatModel::Layer L;
        L.cin = 1, L.cout = 1, L.k = 3, L.stride = 2, L.pad = 1;
        L.w = {1.0f, 10.0f, 100.0f};
        L.b = {0.0f};
        const FloatModel m({L});
        const float x[5] = {1, 2, 3, 4, 5};
        CHECK(m.output_size(5) == 3);
        CHECK(m.output_size(1) == 1);
        for (Backend b : available_backends()) {
            if (!m.supports(b)) continue;
            float y[3] = {-1, -1, -1};
            m.run(x, y, 5, b, ws);
            CHECK(y[0] == 210.0f && y[1] == 432.0f && y[2] == 54.0f);
        }
    }
    {
        // A two-layer classifier. Layer 1 (k=1) makes channels (x, -x) and
        // ReLU zeroes the second. Layer 2 is dense: a "valid" layer whose
        // kernel spans all 4 samples, 3 classes:
        //   class 0 = sum of x,  class 1 = sum of t * x[t],  class 2 = 7.
        FloatModel::Layer a, d;
        a.cin = 1, a.cout = 2, a.k = 1, a.relu = true;
        a.w = {1.0f, -1.0f};
        a.b = {0.0f, 0.0f};
        d.cin = 2, d.cout = 3, d.k = 4;  // stride 1, pad 0
        d.w = {1, 0, 1, 0, 1, 0, 1, 0,   // [class][tap][channel]
               0, 0, 1, 0, 2, 0, 3, 0,
               0, 5, 0, 5, 0, 5, 0, 5};
        d.b = {0.0f, 0.0f, 7.0f};
        const FloatModel m({a, d});
        const float x[4] = {1, 2, 3, 4};
        CHECK(m.output_channels() == 3);
        CHECK(m.output_size(4) == 3);  // one output sample x 3 classes
        CHECK(m.output_size(5) == 6);  // a longer input gives two score vectors
        CHECK(throws([&] { m.output_size(3); }));  // shorter than the dense kernel
        for (Backend b : available_backends()) {
            if (!m.supports(b)) continue;
            float y[3] = {-1, -1, -1};
            m.run(x, y, 4, b, ws);
            CHECK(y[0] == 10.0f && y[1] == 20.0f && y[2] == 7.0f);
            CHECK(argmax(y, 3) == 1);
            float none[3];
            CHECK(throws([&] { m.run(x, none, 3, b, ws); }));
        }
    }
    {
        // argmax: the lowest index wins a tie, like numpy.argmax.
        const float f[4] = {1, 3, 3, 2};
        const int8_t q[3] = {-128, -128, -128};
        CHECK(argmax(f, 4) == 1);
        CHECK(argmax(q, 3) == 0);
        CHECK(argmax(f, 1) == 0);
    }
    {
        // Layers that do not chain, or with impossible geometry, are refused.
        FloatModel::Layer L;
        L.cin = 1, L.cout = 1, L.k = 3, L.pad = 1;
        L.w = {1, 1, 1};
        L.b = {0};
        auto bad = L;
        bad.stride = 0;
        CHECK(throws([&] { FloatModel m({bad}); }));
        bad = L;
        bad.pad = 3;
        CHECK(throws([&] { FloatModel m({bad}); }));
        bad = L;
        bad.w.pop_back();
        CHECK(throws([&] { FloatModel m({bad}); }));
        bad = L;
        bad.cin = 2;
        bad.w.resize(6);
        CHECK(throws([&] { FloatModel m({L, bad}); }));
    }
}

// --------------------------------------------------- 4. golden model outputs

// `len` input samples per spectrum; the model decides how many values come out.
void check_golden_int8(const std::string& name, const std::string& prefix, std::size_t len) {
    const auto model = Model::load(g_root + "/models/" + name);
    const auto* q = dynamic_cast<const QuantizedModel*>(model.get());
    CHECK(q != nullptr);
    if (!q) return;
    const auto in = read_file<float>("tests/data/" + prefix + "_in.f32");
    const auto expect = read_file<int8_t>("tests/data/" + prefix + "_out_int8.i8");
    const std::size_t out = q->output_size(len), count = in.size() / len;
    CHECK(!in.empty() && in.size() % len == 0 && expect.size() == count * out);
    if (expect.size() != count * out) return;

    Workspace ws;
    for (Backend b : available_backends()) {
        std::vector<int8_t> qin(len), qout(out);
        std::vector<float> scores(out);
        bool identical = true, same_class = true;
        for (std::size_t n = 0; n < count; ++n) {
            for (std::size_t i = 0; i < len; ++i) qin[i] = q->quantize_input(in[n * len + i]);
            q->run_q8(qin.data(), qout.data(), len, b, ws);
            identical = identical && std::equal(qout.begin(), qout.end(), expect.begin() + n * out);
            // The float entry point must rank the outputs the same way.
            q->run(in.data() + n * len, scores.data(), len, b, ws);
            same_class = same_class &&
                         argmax(scores.data(), out) == argmax(expect.data() + n * out, out);
        }
        std::printf("  %-20s %-13s: %s\n", name.c_str(), backend_name(b),
                    identical ? "bit-identical to the Python integer reference" : "MISMATCH");
        CHECK(identical);
        CHECK(same_class);
    }
}

void check_golden_f32(const std::string& name, const std::string& prefix, std::size_t len,
                      float rel_tol) {
    const auto model = Model::load(g_root + "/models/" + name);
    const auto in = read_file<float>("tests/data/" + prefix + "_in.f32");
    const auto expect = read_file<float>("tests/data/" + prefix + "_out_f32.f32");
    const std::size_t out = model->output_size(len), count = in.size() / len;
    CHECK(!in.empty() && in.size() % len == 0 && expect.size() == count * out);
    if (expect.size() != count * out) return;
    float biggest = 0.0f;
    for (float v : expect) biggest = std::max(biggest, std::fabs(v));

    Workspace ws;
    for (Backend b : available_backends()) {
        if (!model->supports(b)) continue;
        std::vector<float> y(out);
        float worst = 0.0f;
        for (std::size_t n = 0; n < count; ++n) {
            model->run(in.data() + n * len, y.data(), len, b, ws);
            for (std::size_t i = 0; i < out; ++i)
                worst = std::max(worst, std::fabs(y[i] - expect[n * out + i]));
        }
        std::printf("  %-20s %-13s: max |diff| vs PyTorch %.2e (largest output %.1f)\n",
                    name.c_str(), backend_name(b), worst, biggest);
        CHECK(worst < rel_tol * (1.0f + biggest));
    }
}

void test_golden_int8_matches_python_reference() {
    check_golden_int8("classifier_int8.psm", "classifier", 1000);
    check_golden_int8("denoiser_int8.psm", "golden", 256);
}

void test_golden_f32_matches_pytorch() {
    // The classifier's dense layer sums 4032 products, so float rounding
    // differences are larger than in the denoiser; both stay tiny next to
    // the size of the outputs.
    check_golden_f32("classifier_f32.psm", "classifier", 1000, 1e-4f);
    check_golden_f32("denoiser_f32.psm", "golden", 256, 1e-5f);
}

void test_model_shapes() {
    for (const char* name : {"models/classifier_f32.psm", "models/classifier_int8.psm"}) {
        const auto m = Model::load(g_root + "/" + name);
        CHECK(m->num_layers() == 5);
        CHECK(m->num_params() == 155614);
        CHECK(m->output_channels() == 30);
        CHECK(m->output_size(1000) == 30);  // one score per class
        CHECK(throws([&] { m->output_size(100); }));  // too short for the dense layer
    }
    for (const char* name : {"models/denoiser_f32.psm", "models/denoiser_int8.psm"}) {
        const auto m = Model::load(g_root + "/" + name);
        CHECK(m->output_channels() == 1);
        CHECK(m->output_size(256) == 256 && m->output_size(1) == 1 && m->output_size(777) == 777);
    }
}

void test_any_length_and_reuse() {
    // Workspace reuse across different lengths and different models must not
    // leak state.
    const auto denoiser = Model::load(g_root + "/models/denoiser_int8.psm");
    const auto classifier = Model::load(g_root + "/models/classifier_int8.psm");
    Workspace shared;
    for (std::size_t len : {std::size_t{300}, std::size_t{1}, std::size_t{64}, std::size_t{300}}) {
        std::vector<float> x(len), a(len), b(len);
        for (std::size_t i = 0; i < len; ++i) x[i] = 0.5f + 0.4f * std::sin(0.1f * static_cast<float>(i));
        Workspace fresh;
        denoiser->run(x.data(), a.data(), len, Backend::Scalar, shared);
        denoiser->run(x.data(), b.data(), len, Backend::Scalar, fresh);
        CHECK(a == b);

        std::vector<float> s(1000), ca(30), cb(30);
        for (std::size_t i = 0; i < s.size(); ++i)
            s[i] = 0.5f + 0.4f * std::sin(0.02f * static_cast<float>(i * (len + 1)));
        Workspace fresh2;
        classifier->run(s.data(), ca.data(), s.size(), best_backend(), shared);
        classifier->run(s.data(), cb.data(), s.size(), best_backend(), fresh2);
        CHECK(ca == cb);
    }
}

// --------------------------------------------------------- 5. bad model files

bool parse_fails(std::vector<unsigned char> bytes) {
    return throws([&] { Model::parse(std::move(bytes)); });
}

void put_u32(std::vector<unsigned char>& v, std::size_t at, uint32_t x) {
    std::memcpy(v.data() + at, &x, 4);
}

template <typename T>
void append(std::vector<unsigned char>& v, T x, std::size_t count = 1) {
    unsigned char raw[sizeof(T)];
    std::memcpy(raw, &x, sizeof(T));
    for (std::size_t i = 0; i < count; ++i) v.insert(v.end(), raw, raw + sizeof(T));
}

// A minimal one-layer int8 model file with `k` taps, built by hand.
std::vector<unsigned char> tiny_int8_file(uint32_t k) {
    std::vector<unsigned char> v = {'P', 'S', 'M', '2'};
    append<uint32_t>(v, 1);     // dtype int8
    append<uint32_t>(v, 1);     // one layer
    append<float>(v, 0.01f);    // input scale
    append<float>(v, 0.01f);    // output scale
    for (uint32_t field : {1u, 1u, k, 1u, 0u, 0u}) append<uint32_t>(v, field);  // cin cout k stride pad relu
    append<int32_t>(v, 0, 2);   // zero-points
    append<int8_t>(v, 1, k);    // weights
    append<int32_t>(v, 0);      // bias
    append<int32_t>(v, int32_t{1} << 30);  // multiplier
    append<int32_t>(v, 8);      // shift
    return v;
}

void test_loader_rejects_malformed_files() {
    for (const char* name : {"models/denoiser_f32.psm", "models/denoiser_int8.psm",
                             "models/classifier_f32.psm", "models/classifier_int8.psm"}) {
        const auto good = read_file<unsigned char>(name);
        CHECK(!parse_fails(good));
        // Truncations must be rejected cleanly: every length in the header
        // region, then a spread of lengths over the rest of the file.
        bool all_rejected = true;
        const std::size_t step = std::max<std::size_t>(7, good.size() / 400 | 1);
        for (std::size_t n = 0; n < good.size(); n += n < 96 ? 1 : step)
            all_rejected = all_rejected &&
                           parse_fails(std::vector<unsigned char>(good.begin(), good.begin() + n));
        CHECK(all_rejected);
        auto bad = good;
        bad[0] = 'X';  // wrong magic
        CHECK(parse_fails(bad));
        bad = good;
        bad[3] = '1';  // the previous format version
        CHECK(parse_fails(bad));
        bad = good;
        bad.push_back(0);  // trailing garbage
        CHECK(parse_fails(bad));
        // First layer's shape fields start after the 12-byte header (plus
        // two float scales in the int8 format): cin, cout, k, stride, pad, relu.
        const std::size_t shape = good[4] == 0 ? 12 : 20;
        bad = good;
        put_u32(bad, shape, 0);  // cin = 0
        CHECK(parse_fails(bad));
        bad = good;
        put_u32(bad, shape + 12, 0);  // stride = 0
        CHECK(parse_fails(bad));
        bad = good;
        put_u32(bad, shape + 12, 1000);  // absurd stride
        CHECK(parse_fails(bad));
        bad = good;
        uint32_t k = 0;
        std::memcpy(&k, good.data() + shape + 8, 4);
        put_u32(bad, shape + 16, k);  // pad = k: a window made only of padding
        CHECK(parse_fails(bad));
        bad = good;
        put_u32(bad, shape + 20, 2);  // activation flag
        CHECK(parse_fails(bad));
    }
    // An int8 layer with so many inputs that an int32 sum could overflow is
    // refused; one just under the limit loads and runs.
    CHECK(!parse_fails(tiny_int8_file(16)));
    CHECK(!parse_fails(tiny_int8_file(65535)));
    CHECK(parse_fails(tiny_int8_file(65536)));
    {
        auto bad = tiny_int8_file(16);
        const int32_t huge = INT32_MAX;
        std::memcpy(bad.data() + bad.size() - 12, &huge, 4);  // bias
        CHECK(parse_fails(bad));
    }
    CHECK(throws([] { Model::load("definitely/not/a/file.psm"); }));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1) g_root = argv[1];
    const std::pair<const char*, std::function<void()>> tests[] = {
        {"requant rounding", test_requant_rounding},
        {"int8 kernels bit-exact vs reference", test_q8_kernels_bit_exact},
        {"float kernels agree with reference", test_f32_kernels_agree},
        {"stride, padding and dense shapes", test_stride_and_valid_shapes},
        {"int8 models == Python integer reference", test_golden_int8_matches_python_reference},
        {"float models == PyTorch", test_golden_f32_matches_pytorch},
        {"model shapes", test_model_shapes},
        {"any length, workspace reuse", test_any_length_and_reuse},
        {"loader rejects malformed files", test_loader_rejects_malformed_files},
    };
    std::printf("cpu: neon=%d dotprod=%d\n", has_neon(), has_dotprod());
    for (const auto& t : tests) {
        const int before = g_failures;
        try {
            t.second();
        } catch (const std::exception& e) {
            ++g_failures;
            std::printf("  EXCEPTION: %s\n", e.what());
        }
        std::printf("[%s] %s\n", g_failures == before ? " ok " : "FAIL", t.first);
    }
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
