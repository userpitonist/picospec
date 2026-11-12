// Unit tests. No framework: a CHECK macro and a list of functions keeps the
// whole test suite readable in one sitting.
//
// What is being protected:
//   1. the fixed-point rounding rules (requant.hpp)
//   2. every optimised kernel agrees with the plain reference kernel -
//      bit-for-bit for int8, within float rounding for float32
//   3. the engine reproduces the Python reference on real model files
//   4. malformed model files are rejected with an error, never a crash
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

struct RandomQ8Layer {
    std::vector<int8_t> w;
    std::vector<int32_t> bias, mult, shift;
    ConvQ8 view;
    RandomQ8Layer(int cin, int cout, int k, bool relu, std::mt19937& rng) {
        std::uniform_int_distribution<int> wd(-127, 127), bd(-20000, 20000), sd(4, 11), zd(-128, 20);
        std::uniform_int_distribution<int32_t> md(int32_t{1} << 30, INT32_MAX);
        w.resize(static_cast<std::size_t>(cin) * cout * k);
        for (auto& v : w) v = static_cast<int8_t>(wd(rng));
        for (int c = 0; c < cout; ++c) {
            bias.push_back(bd(rng));
            mult.push_back(md(rng));
            shift.push_back(sd(rng));
        }
        const int zp = zd(rng);
        view = {cin, cout, k, zp, relu ? zp : -128, 127, w.data(), bias.data(), mult.data(),
                shift.data()};
    }
};

void test_q8_kernels_bit_exact() {
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> xd(-128, 127);
    const int shapes[][3] = {{1, 16, 5}, {16, 16, 5}, {16, 1, 5}, {3, 7, 3},  {8, 5, 7},
                             {1, 1, 1},  {5, 4, 9},   {32, 6, 3}, {2, 3, 15}, {7, 9, 5}};
    const ConvQ8Fn ref = conv1d_q8_novec;
    for (const auto& s : shapes) {
        for (std::size_t len : {std::size_t{1}, std::size_t{2}, std::size_t{17}, std::size_t{256}}) {
            for (bool relu : {false, true}) {
                const int cin = s[0], cout = s[1], k = s[2], pad = k / 2;
                RandomQ8Layer L(cin, cout, k, relu, rng);
                std::vector<int8_t> buf((len + 2 * pad) * cin);
                for (auto& v : buf) v = static_cast<int8_t>(xd(rng));
                const int8_t* in = buf.data() + pad * cin;
                std::vector<int8_t> expect(len * cout), got(len * cout);
                ref(in, expect.data(), len, L.view);

                std::vector<ConvQ8Fn> fns = {conv1d_q8_scalar};
#if defined(__aarch64__)
                fns.push_back(conv1d_q8_neon);
                if (has_dotprod()) fns.push_back(conv1d_q8_neon_dotprod);
#endif
                for (ConvQ8Fn fn : fns) {
                    std::fill(got.begin(), got.end(), int8_t{0});
                    fn(in, got.data(), len, L.view);
                    CHECK(got == expect);
                }
            }
        }
    }
}

void test_f32_kernels_agree() {
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    const int shapes[][3] = {{1, 16, 5}, {16, 16, 5}, {16, 1, 5}, {3, 7, 3}, {6, 5, 9}, {1, 1, 1}};
    for (const auto& s : shapes) {
        for (std::size_t len : {std::size_t{1}, std::size_t{33}, std::size_t{256}}) {
            const int cin = s[0], cout = s[1], k = s[2], pad = k / 2;
            std::vector<float> w(static_cast<std::size_t>(cin) * cout * k), b(cout);
            for (auto& v : w) v = d(rng);
            for (auto& v : b) v = d(rng);
            const ConvF32 L{cin, cout, k, true, w.data(), b.data()};
            std::vector<float> buf((len + 2 * pad) * cin);
            for (auto& v : buf) v = d(rng);
            const float* in = buf.data() + pad * cin;
            std::vector<float> expect(len * cout), got(len * cout);
            conv1d_f32_novec(in, expect.data(), len, L);

            std::vector<ConvF32Fn> fns = {conv1d_f32_scalar};
#if defined(__aarch64__)
            fns.push_back(conv1d_f32_neon);
#endif
            for (ConvF32Fn fn : fns) {
                fn(in, got.data(), len, L);
                float worst = 0.0f;
                for (std::size_t i = 0; i < got.size(); ++i)
                    worst = std::max(worst, std::fabs(got[i] - expect[i]));
                // Different summation order -> not bit-identical, but tiny.
                CHECK(worst < 1e-4f);
            }
        }
    }
}

// --------------------------------------------------- 3. golden model outputs

void test_golden_int8_matches_python_reference() {
    const auto model = Model::load(g_root + "/models/denoiser_int8.psm");
    const auto* q = dynamic_cast<const QuantizedModel*>(model.get());
    CHECK(q != nullptr);
    if (!q) return;
    const auto in = read_file<float>("tests/data/golden_in.f32");
    const auto expect = read_file<int8_t>("tests/data/golden_out_int8.i8");
    const std::size_t len = 256;
    CHECK(in.size() == expect.size() && in.size() % len == 0 && !in.empty());

    Workspace ws;
    for (Backend b : available_backends()) {
        std::vector<int8_t> qin(len), qout(len);
        bool identical = true;
        for (std::size_t off = 0; off + len <= in.size(); off += len) {
            for (std::size_t i = 0; i < len; ++i) qin[i] = q->quantize_input(in[off + i]);
            q->run_q8(qin.data(), qout.data(), len, b, ws);
            for (std::size_t i = 0; i < len; ++i) identical = identical && qout[i] == expect[off + i];
        }
        std::printf("  int8 golden, backend %-13s: %s\n", backend_name(b),
                    identical ? "bit-identical" : "MISMATCH");
        CHECK(identical);
    }
}

void test_golden_f32_matches_pytorch() {
    const auto model = Model::load(g_root + "/models/denoiser_f32.psm");
    const auto in = read_file<float>("tests/data/golden_in.f32");
    const auto expect = read_file<float>("tests/data/golden_out_f32.f32");
    const std::size_t len = 256;
    CHECK(in.size() == expect.size());
    Workspace ws;
    for (Backend b : available_backends()) {
        if (!model->supports(b)) continue;
        std::vector<float> out(len);
        float worst = 0.0f;
        for (std::size_t off = 0; off + len <= in.size(); off += len) {
            model->run(in.data() + off, out.data(), len, b, ws);
            for (std::size_t i = 0; i < len; ++i)
                worst = std::max(worst, std::fabs(out[i] - expect[off + i]));
        }
        std::printf("  f32 golden,  backend %-13s: max |diff| vs PyTorch %.2e\n", backend_name(b),
                    worst);
        CHECK(worst < 1e-5f);
    }
}

void test_any_length_and_reuse() {
    // Workspace reuse across different lengths must not leak state.
    const auto model = Model::load(g_root + "/models/denoiser_int8.psm");
    Workspace shared;
    for (std::size_t len : {std::size_t{300}, std::size_t{1}, std::size_t{64}, std::size_t{300}}) {
        std::vector<float> x(len), a(len), b(len);
        for (std::size_t i = 0; i < len; ++i) x[i] = 0.5f + 0.4f * std::sin(0.1f * static_cast<float>(i));
        Workspace fresh;
        model->run(x.data(), a.data(), len, Backend::Scalar, shared);
        model->run(x.data(), b.data(), len, Backend::Scalar, fresh);
        CHECK(a == b);
    }
}

// --------------------------------------------------------- 4. bad model files

bool load_fails(const std::vector<char>& bytes) {
    const std::string path = "picospec_bad_model.tmp";
    {
        std::ofstream f(path, std::ios::binary);
        f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
    bool failed = false;
    try {
        Model::load(path);
    } catch (const std::runtime_error&) {
        failed = true;
    }
    std::remove(path.c_str());
    return failed;
}

void test_loader_rejects_malformed_files() {
    for (const char* name : {"models/denoiser_f32.psm", "models/denoiser_int8.psm"}) {
        const auto good = read_file<char>(name);
        CHECK(!load_fails(good));
        // Every possible truncation must be rejected cleanly.
        bool all_rejected = true;
        for (std::size_t n = 0; n < good.size(); n += 7)
            all_rejected = all_rejected && load_fails(std::vector<char>(good.begin(), good.begin() + n));
        CHECK(all_rejected);
        auto bad = good;
        bad[0] = 'X';  // wrong magic
        CHECK(load_fails(bad));
        bad = good;
        bad.push_back(0);  // trailing garbage
        CHECK(load_fails(bad));
        bad = good;
        // First layer's cin field (after the 12-byte header, plus two
        // float scales in the int8 format): set it to 0.
        bad[good[4] == 0 ? 12 : 20] = 0;
        CHECK(load_fails(bad));
    }
    bool threw = false;
    try {
        Model::load("definitely/not/a/file.psm");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1) g_root = argv[1];
    const std::pair<const char*, std::function<void()>> tests[] = {
        {"requant rounding", test_requant_rounding},
        {"int8 kernels bit-exact vs reference", test_q8_kernels_bit_exact},
        {"float kernels agree with reference", test_f32_kernels_agree},
        {"int8 model == Python integer reference", test_golden_int8_matches_python_reference},
        {"float model == PyTorch", test_golden_f32_matches_pytorch},
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
