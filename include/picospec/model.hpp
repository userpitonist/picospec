// Model loading and inference.
//
// A model is a stack of 1-D convolution layers read from a small binary
// ".psm" file written by python/export.py. Each layer has its own stride and
// padding, so the same stack can be a spectrum-to-spectrum filter (all
// layers "same", output as long as the input) or a classifier (stride-2
// layers that shrink the spectrum, then a dense layer that produces one
// score per class). Two concrete models share one interface: FloatModel
// (float32 weights) and QuantizedModel (int8 weights and activations,
// integer-only arithmetic).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "picospec/kernels.hpp"

namespace picospec {

enum class Backend { ScalarNoVec, Scalar, Neon, NeonDot };

const char* backend_name(Backend b);
bool backend_from_name(const std::string& name, Backend& out);
bool backend_available(Backend b);  // compiled in and supported by this CPU
Backend best_backend();
std::vector<Backend> available_backends();

// Index of the largest value; the lowest index wins a tie (the same rule as
// numpy.argmax). For a classifier this is the predicted class.
template <typename T>
std::size_t argmax(const T* v, std::size_t n) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < n; ++i)
        if (v[i] > v[best]) best = i;
    return best;
}

// Scratch memory for one inference. Owned by the caller so that run() does
// no heap allocation once the buffers have grown to size, and so that one
// model can be shared between threads (one Workspace per thread).
struct Workspace {
    std::vector<float> f32[2];
    std::vector<int8_t> q8[2];
    std::vector<int8_t> io[2];  // quantised input / output staging
};

class Model {
public:
    virtual ~Model() = default;

    // Reads a .psm file. Throws std::runtime_error on any malformed input.
    static std::unique_ptr<Model> load(const std::string& path);
    // The same, from bytes already in memory (no file system needed).
    static std::unique_ptr<Model> parse(std::vector<unsigned char> bytes);

    virtual const char* dtype() const = 0;
    virtual int num_layers() const = 0;
    virtual std::size_t num_params() const = 0;
    virtual std::size_t weight_bytes() const = 0;
    // The int8 model has no float kernel and vice versa; DotProd is int8-only.
    virtual bool supports(Backend b) const = 0;

    // Channels of the last layer: 1 for a denoiser, the number of classes
    // for a classifier.
    virtual int output_channels() const = 0;
    // How many values run() writes for an input of `len` samples:
    // (output samples) x output_channels(). Throws std::runtime_error if the
    // input is too short for the layers.
    virtual std::size_t output_size(std::size_t len) const = 0;

    // Run one spectrum: `in` holds `len` samples, `out` receives
    // output_size(len) values, channel-last.
    virtual void run(const float* in, float* out, std::size_t len, Backend backend,
                     Workspace& ws) const = 0;
};

class FloatModel final : public Model {
public:
    struct Layer {
        int cin = 0, cout = 0, k = 0, stride = 1, pad = 0;
        bool relu = false;
        std::vector<float> w, b;  // w is [cout][k][cin]
        ConvF32 view() const { return {cin, cout, k, stride, pad, relu, w.data(), b.data()}; }
    };

    explicit FloatModel(std::vector<Layer> layers);

    const char* dtype() const override { return "float32"; }
    int num_layers() const override { return static_cast<int>(layers_.size()); }
    std::size_t num_params() const override;
    std::size_t weight_bytes() const override { return num_params() * sizeof(float); }
    bool supports(Backend b) const override;
    int output_channels() const override { return layers_.back().cout; }
    std::size_t output_size(std::size_t len) const override;
    void run(const float* in, float* out, std::size_t len, Backend backend,
             Workspace& ws) const override;

private:
    std::vector<Layer> layers_;
    int max_pad_ = 0;
};

class QuantizedModel final : public Model {
public:
    struct Layer {
        int cin = 0, cout = 0, k = 0, stride = 1, pad = 0;
        int32_t in_zp = 0, out_zp = 0, act_min = -128, act_max = 127;
        std::vector<int8_t> w;  // [cout][k][cin]
        std::vector<int32_t> bias, mult, shift;
        ConvQ8 view() const {
            return {cin, cout, k, stride, pad, out_zp, act_min, act_max, w.data(), bias.data(),
                    mult.data(), shift.data()};
        }
    };

    QuantizedModel(std::vector<Layer> layers, float in_scale, float out_scale);

    const char* dtype() const override { return "int8"; }
    int num_layers() const override { return static_cast<int>(layers_.size()); }
    std::size_t num_params() const override;
    std::size_t weight_bytes() const override;
    bool supports(Backend b) const override;
    int output_channels() const override { return layers_.back().cout; }
    std::size_t output_size(std::size_t len) const override;
    void run(const float* in, float* out, std::size_t len, Backend backend,
             Workspace& ws) const override;

    // Integer-only entry point: `len` int8 samples in, output_size(len) int8
    // values out. This is the path that is compared bit-for-bit against the
    // Python integer reference.
    void run_q8(const int8_t* in, int8_t* out, std::size_t len, Backend backend,
                Workspace& ws) const;

    int8_t quantize_input(float x) const;
    float dequantize_output(int8_t q) const;

private:
    std::vector<Layer> layers_;
    float in_scale_ = 1.0f, out_scale_ = 1.0f;
    int max_pad_ = 0;
};

}  // namespace picospec
