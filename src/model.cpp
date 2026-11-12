#include "picospec/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace picospec {

// ---------------------------------------------------------------- backends

const char* backend_name(Backend b) {
    switch (b) {
        case Backend::ScalarNoVec: return "scalar-novec";
        case Backend::Scalar: return "scalar";
        case Backend::Neon: return "neon";
        case Backend::NeonDot: return "neon-dotprod";
    }
    return "?";
}

bool backend_from_name(const std::string& name, Backend& out) {
    for (Backend b : {Backend::ScalarNoVec, Backend::Scalar, Backend::Neon, Backend::NeonDot}) {
        if (name == backend_name(b)) {
            out = b;
            return true;
        }
    }
    return false;
}

bool backend_available(Backend b) {
    switch (b) {
        case Backend::ScalarNoVec:
        case Backend::Scalar: return true;
        case Backend::Neon: return has_neon();
        case Backend::NeonDot: return has_neon() && has_dotprod();
    }
    return false;
}

Backend best_backend() {
    if (backend_available(Backend::NeonDot)) return Backend::NeonDot;
    if (backend_available(Backend::Neon)) return Backend::Neon;
    return Backend::Scalar;
}

std::vector<Backend> available_backends() {
    std::vector<Backend> v;
    for (Backend b : {Backend::ScalarNoVec, Backend::Scalar, Backend::Neon, Backend::NeonDot})
        if (backend_available(b)) v.push_back(b);
    return v;
}

namespace {

ConvF32Fn f32_kernel(Backend b) {
    switch (b) {
        case Backend::ScalarNoVec: return conv1d_f32_novec;
        case Backend::Scalar: return conv1d_f32_scalar;
#if defined(__aarch64__)
        case Backend::Neon: return conv1d_f32_neon;
#endif
        default: break;
    }
    throw std::runtime_error(std::string("float32 model has no backend '") + backend_name(b) + "'");
}

ConvQ8Fn q8_kernel(Backend b) {
    if (!backend_available(b))
        throw std::runtime_error(std::string("backend '") + backend_name(b) +
                                 "' is not available on this CPU");
    switch (b) {
        case Backend::ScalarNoVec: return conv1d_q8_novec;
        case Backend::Scalar: return conv1d_q8_scalar;
#if defined(__aarch64__)
        case Backend::Neon: return conv1d_q8_neon;
        case Backend::NeonDot: return conv1d_q8_neon_dotprod;
#endif
        default: break;
    }
    throw std::runtime_error("unreachable backend");
}

// ------------------------------------------------------------- file reader

// Bounds-checked little-endian reader. Every read is validated because a
// model file is untrusted input: a truncated or corrupt file must produce an
// error message, never an out-of-bounds read.
class Reader {
public:
    explicit Reader(std::vector<unsigned char> bytes) : buf_(std::move(bytes)) {}

    template <typename T>
    T scalar() {
        T v;
        copy(&v, sizeof(T));
        return v;
    }
    template <typename T>
    std::vector<T> array(std::size_t n) {
        if (n > remaining() / sizeof(T)) throw std::runtime_error("model file is truncated");
        std::vector<T> v(n);
        if (n) copy(v.data(), n * sizeof(T));
        return v;
    }
    std::size_t remaining() const { return buf_.size() - pos_; }

private:
    void copy(void* dst, std::size_t n) {
        if (n > remaining()) throw std::runtime_error("model file is truncated");
        std::memcpy(dst, buf_.data() + pos_, n);
        pos_ += n;
    }
    std::vector<unsigned char> buf_;
    std::size_t pos_ = 0;
};

struct Shape {
    int cin, cout, k;
    bool relu;
};

Shape read_shape(Reader& r, int expected_cin) {
    const uint32_t cin = r.scalar<uint32_t>(), cout = r.scalar<uint32_t>();
    const uint32_t k = r.scalar<uint32_t>(), relu = r.scalar<uint32_t>();
    if (cin == 0 || cout == 0 || cin > 1024 || cout > 1024)
        throw std::runtime_error("model file: channel count out of range");
    if (k == 0 || k > 63 || k % 2 == 0) throw std::runtime_error("model file: kernel must be odd and <= 63");
    if (relu > 1) throw std::runtime_error("model file: bad activation flag");
    if (static_cast<int>(cin) != expected_cin)
        throw std::runtime_error("model file: layer input channels do not match previous layer");
    return {static_cast<int>(cin), static_cast<int>(cout), static_cast<int>(k), relu == 1};
}

template <typename LayerT>
void check_io_shape(const std::vector<LayerT>& layers) {
    if (layers.empty()) throw std::runtime_error("model has no layers");
    if (layers.front().cin != 1 || layers.back().cout != 1)
        throw std::runtime_error("model must map one input channel to one output channel");
}

// Copies `len` samples into the middle of a padded buffer and fills both
// margins with `zero`. Returns a pointer to sample 0.
template <typename T>
T* prepare(std::vector<T>& buf, std::size_t len, int channels, int max_pad, int max_ch, T zero) {
    const std::size_t need = (len + 2 * static_cast<std::size_t>(max_pad)) * max_ch;
    if (buf.size() < need) buf.resize(need);
    T* p = buf.data() + static_cast<std::size_t>(max_pad) * channels;
    std::fill(buf.data(), p, zero);
    std::fill(p + len * channels, p + (len + max_pad) * channels, zero);
    return p;
}

}  // namespace

// ------------------------------------------------------------------ loader

std::unique_ptr<Model> Model::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open model file: " + path);
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(f)),
                                     std::istreambuf_iterator<char>());
    Reader r(std::move(bytes));

    const auto magic = r.array<char>(4);
    if (std::memcmp(magic.data(), "PSM1", 4) != 0)
        throw std::runtime_error("not a picospec model (bad magic): " + path);
    const uint32_t dtype = r.scalar<uint32_t>();
    const uint32_t n_layers = r.scalar<uint32_t>();
    if (n_layers == 0 || n_layers > 64) throw std::runtime_error("model file: bad layer count");

    if (dtype == 0) {
        std::vector<FloatModel::Layer> layers;
        int cin = 1;
        for (uint32_t i = 0; i < n_layers; ++i) {
            const Shape s = read_shape(r, cin);
            FloatModel::Layer L;
            L.cin = s.cin, L.cout = s.cout, L.k = s.k, L.relu = s.relu;
            L.w = r.array<float>(static_cast<std::size_t>(s.cout) * s.k * s.cin);
            L.b = r.array<float>(s.cout);
            for (float v : L.w) if (!std::isfinite(v)) throw std::runtime_error("model file: non-finite weight");
            for (float v : L.b) if (!std::isfinite(v)) throw std::runtime_error("model file: non-finite bias");
            cin = s.cout;
            layers.push_back(std::move(L));
        }
        if (r.remaining() != 0) throw std::runtime_error("model file: trailing bytes");
        return std::make_unique<FloatModel>(std::move(layers));
    }
    if (dtype == 1) {
        const float in_scale = r.scalar<float>();
        const float out_scale = r.scalar<float>();
        if (!(in_scale > 0.0f) || !(out_scale > 0.0f) || !std::isfinite(in_scale) ||
            !std::isfinite(out_scale))
            throw std::runtime_error("model file: bad quantisation scale");
        std::vector<QuantizedModel::Layer> layers;
        int cin = 1;
        int32_t prev_zp = 0;
        for (uint32_t i = 0; i < n_layers; ++i) {
            const Shape s = read_shape(r, cin);
            QuantizedModel::Layer L;
            L.cin = s.cin, L.cout = s.cout, L.k = s.k;
            L.in_zp = r.scalar<int32_t>();
            L.out_zp = r.scalar<int32_t>();
            if (L.in_zp < -128 || L.in_zp > 127 || L.out_zp < -128 || L.out_zp > 127)
                throw std::runtime_error("model file: zero-point out of int8 range");
            if (i > 0 && L.in_zp != prev_zp)
                throw std::runtime_error("model file: zero-points of adjacent layers disagree");
            // ReLU is a clamp at "real zero", which in the quantised domain
            // is the output zero-point.
            L.act_min = s.relu ? L.out_zp : -128;
            L.act_max = 127;
            L.w = r.array<int8_t>(static_cast<std::size_t>(s.cout) * s.k * s.cin);
            L.bias = r.array<int32_t>(s.cout);
            L.mult = r.array<int32_t>(s.cout);
            L.shift = r.array<int32_t>(s.cout);
            for (int c = 0; c < s.cout; ++c) {
                if (L.mult[c] < (int32_t{1} << 30)) throw std::runtime_error("model file: multiplier not normalised");
                if (L.shift[c] < 0 || L.shift[c] > 31) throw std::runtime_error("model file: shift out of range");
            }
            prev_zp = L.out_zp;
            cin = s.cout;
            layers.push_back(std::move(L));
        }
        if (r.remaining() != 0) throw std::runtime_error("model file: trailing bytes");
        return std::make_unique<QuantizedModel>(std::move(layers), in_scale, out_scale);
    }
    throw std::runtime_error("model file: unknown dtype");
}

// ------------------------------------------------------------- float model

FloatModel::FloatModel(std::vector<Layer> layers) : layers_(std::move(layers)) {
    check_io_shape(layers_);
    for (const Layer& L : layers_) {
        max_pad_ = std::max(max_pad_, L.k / 2);
        max_ch_ = std::max({max_ch_, L.cin, L.cout});
    }
}

std::size_t FloatModel::num_params() const {
    std::size_t n = 0;
    for (const Layer& L : layers_) n += L.w.size() + L.b.size();
    return n;
}

bool FloatModel::supports(Backend b) const {
    return b != Backend::NeonDot && backend_available(b);
}

void FloatModel::run(const float* in, float* out, std::size_t len, Backend backend,
                     Workspace& ws) const {
    const ConvF32Fn conv = f32_kernel(backend);
    float* cur = prepare(ws.f32[0], len, 1, max_pad_, max_ch_, 0.0f);
    std::copy(in, in + len, cur);
    int which = 0;
    for (const Layer& L : layers_) {
        which ^= 1;
        float* next = prepare(ws.f32[which], len, L.cout, max_pad_, max_ch_, 0.0f);
        conv(cur, next, len, L.view());
        cur = next;
    }
    std::copy(cur, cur + len, out);
}

// --------------------------------------------------------- quantised model

QuantizedModel::QuantizedModel(std::vector<Layer> layers, float in_scale, float out_scale)
    : layers_(std::move(layers)), in_scale_(in_scale), out_scale_(out_scale) {
    check_io_shape(layers_);
    for (const Layer& L : layers_) {
        max_pad_ = std::max(max_pad_, L.k / 2);
        max_ch_ = std::max({max_ch_, L.cin, L.cout});
    }
}

std::size_t QuantizedModel::num_params() const {
    std::size_t n = 0;
    for (const Layer& L : layers_) n += L.w.size() + L.bias.size();
    return n;
}

std::size_t QuantizedModel::weight_bytes() const {
    std::size_t n = 0;
    for (const Layer& L : layers_)
        n += L.w.size() + sizeof(int32_t) * (L.bias.size() + L.mult.size() + L.shift.size());
    return n;
}

bool QuantizedModel::supports(Backend b) const { return backend_available(b); }

int8_t QuantizedModel::quantize_input(float x) const {
    // Round-half-to-even in double precision, the same rule as numpy.rint,
    // so the Python reference and the engine quantise identically.
    const double q = std::nearbyint(static_cast<double>(x) / static_cast<double>(in_scale_)) +
                     layers_.front().in_zp;
    return static_cast<int8_t>(std::min(127.0, std::max(-128.0, q)));
}

float QuantizedModel::dequantize_output(int8_t q) const {
    return out_scale_ * static_cast<float>(static_cast<int32_t>(q) - layers_.back().out_zp);
}

void QuantizedModel::run_q8(const int8_t* in, int8_t* out, std::size_t len, Backend backend,
                            Workspace& ws) const {
    const ConvQ8Fn conv = q8_kernel(backend);
    // Margins hold the zero-point, i.e. the int8 code for real 0.0, so
    // "same" padding needs no special case inside the kernels.
    int8_t* cur = prepare(ws.q8[0], len, 1, max_pad_, max_ch_,
                          static_cast<int8_t>(layers_.front().in_zp));
    std::copy(in, in + len, cur);
    int which = 0;
    for (const Layer& L : layers_) {
        which ^= 1;
        int8_t* next = prepare(ws.q8[which], len, L.cout, max_pad_, max_ch_,
                               static_cast<int8_t>(L.out_zp));
        conv(cur, next, len, L.view());
        cur = next;
    }
    std::copy(cur, cur + len, out);
}

void QuantizedModel::run(const float* in, float* out, std::size_t len, Backend backend,
                         Workspace& ws) const {
    std::vector<int8_t>& qin = ws.io[0];
    std::vector<int8_t>& qout = ws.io[1];
    if (qin.size() < len) qin.resize(len);
    if (qout.size() < len) qout.resize(len);
    for (std::size_t i = 0; i < len; ++i) qin[i] = quantize_input(in[i]);
    run_q8(qin.data(), qout.data(), len, backend, ws);
    for (std::size_t i = 0; i < len; ++i) out[i] = dequantize_output(qout[i]);
}

}  // namespace picospec
