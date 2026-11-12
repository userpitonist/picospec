// Command-line front end: inspect a model, run it on raw float32 spectra,
// print the predicted class of each spectrum, or benchmark every kernel
// this CPU supports.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "picospec/model.hpp"

using namespace picospec;

namespace {

void usage() {
    std::puts(
        "picospec - tiny 1-D CNN inference engine\n"
        "\n"
        "  picospec info  <model.psm>\n"
        "  picospec run      --model M --in X.f32 --out Y.f32 [--len 256] [--backend NAME]\n"
        "  picospec classify --model M --in X.f32 [--len 1000] [--backend NAME]\n"
        "  picospec bench    --model M [--model M2 ...] [--len 256] [--samples 300] [--json FILE]\n"
        "\n"
        "run writes the raw model output (a spectrum for a denoiser, one score per\n"
        "class for a classifier). classify prints one class index per input spectrum.\n"
        "\n"
        "Backends: scalar-novec, scalar, neon, neon-dotprod (default: best available).\n"
        "Spectra files are raw little-endian float32, <count> x <len> values.");
}

std::vector<float> read_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    const std::streamsize bytes = f.tellg();
    if (bytes % 4 != 0) throw std::runtime_error(path + ": size is not a multiple of 4 bytes");
    std::vector<float> v(static_cast<std::size_t>(bytes) / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), bytes);
    return v;
}

// Deterministic spectrum-like test signal (two peaks plus pseudo-random
// noise from a fixed-seed generator) so that benchmarks are repeatable.
std::vector<float> synthetic_input(std::size_t len) {
    std::vector<float> x(len);
    uint32_t state = 12345u;
    for (std::size_t i = 0; i < len; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(len);
        state = state * 1664525u + 1013904223u;
        const double noise = (static_cast<double>(state >> 8) / 16777216.0 - 0.5) * 0.08;
        const double v = 0.15 + 0.6 * std::exp(-0.5 * std::pow((t - 0.35) / 0.06, 2)) +
                         0.3 * std::exp(-0.5 * std::pow((t - 0.7) / 0.1, 2)) + noise;
        x[i] = static_cast<float>(std::min(1.0, std::max(0.0, v)));
    }
    return x;
}

struct Stats {
    double min_us, median_us, p90_us;
};

Stats time_backend(const Model& m, Backend b, const std::vector<float>& x, int samples) {
    using clock = std::chrono::steady_clock;
    Workspace ws;
    std::vector<float> y(m.output_size(x.size()));
    volatile float sink = 0.0f;  // keeps the optimiser from deleting the work

    // Choose a repeat count so one timing sample lasts about 2 ms: long
    // enough to drown timer resolution, short enough to collect hundreds.
    int reps = 1;
    for (;;) {
        const auto t0 = clock::now();
        for (int r = 0; r < reps; ++r) m.run(x.data(), y.data(), x.size(), b, ws);
        const double us = std::chrono::duration<double, std::micro>(clock::now() - t0).count();
        sink = sink + y[0];
        if (us >= 2000.0 || reps >= (1 << 20)) break;
        reps *= 2;
    }
    std::vector<double> t(static_cast<std::size_t>(samples));
    for (int s = 0; s < samples; ++s) {
        const auto t0 = clock::now();
        for (int r = 0; r < reps; ++r) m.run(x.data(), y.data(), x.size(), b, ws);
        t[static_cast<std::size_t>(s)] =
            std::chrono::duration<double, std::micro>(clock::now() - t0).count() / reps;
        sink = sink + y[y.size() / 2];
    }
    std::sort(t.begin(), t.end());
    return {t.front(), t[t.size() / 2], t[t.size() * 9 / 10]};
}

int cmd_info(const std::string& path) {
    const auto m = Model::load(path);
    std::printf("model      : %s\n", path.c_str());
    std::printf("dtype      : %s\n", m->dtype());
    std::printf("layers     : %d\n", m->num_layers());
    std::printf("parameters : %zu\n", m->num_params());
    std::printf("weights    : %zu bytes\n", m->weight_bytes());
    std::printf("outputs    : %d per output sample\n", m->output_channels());
    std::printf("cpu        : neon=%d dotprod=%d\n", has_neon(), has_dotprod());
    std::printf("backends   :");
    for (Backend b : available_backends())
        if (m->supports(b)) std::printf(" %s", backend_name(b));
    std::printf("\n");
    return 0;
}

using Args = std::map<std::string, std::vector<std::string>>;

const std::string& need(const Args& a, const char* k) {
    const auto it = a.find(k);
    if (it == a.end()) throw std::runtime_error(std::string("missing --") + k);
    return it->second.back();
}

Backend pick_backend(const Args& a, const Model& m) {
    Backend b = best_backend();
    if (a.count("backend") && !backend_from_name(a.at("backend").back(), b))
        throw std::runtime_error("unknown backend " + a.at("backend").back());
    if (!m.supports(b)) {
        if (a.count("backend")) throw std::runtime_error("backend not supported for this model/CPU");
        b = backend_available(Backend::Neon) ? Backend::Neon : Backend::Scalar;
    }
    return b;
}

// Runs the model on every spectrum in the --in file. Returns the outputs,
// `out_size` values per spectrum.
std::vector<float> run_all(const Args& a, std::size_t default_len, std::size_t& out_size) {
    const auto m = Model::load(need(a, "model"));
    const std::size_t len = a.count("len") ? std::stoul(a.at("len").back()) : default_len;
    const Backend b = pick_backend(a, *m);
    const std::vector<float> x = read_f32(need(a, "in"));
    if (len == 0 || x.size() % len != 0)
        throw std::runtime_error("input size is not a multiple of --len");
    out_size = m->output_size(len);
    const std::size_t count = x.size() / len;
    std::vector<float> y(count * out_size);
    Workspace ws;
    for (std::size_t i = 0; i < count; ++i)
        m->run(x.data() + i * len, y.data() + i * out_size, len, b, ws);
    std::fprintf(stderr, "%zu spectra x %zu samples -> %zu values each, %s, backend %s\n", count,
                 len, out_size, m->dtype(), backend_name(b));
    return y;
}

int cmd_run(const Args& a) {
    std::size_t out_size = 0;
    const std::vector<float> y = run_all(a, 256, out_size);
    std::ofstream f(need(a, "out"), std::ios::binary);
    f.write(reinterpret_cast<const char*>(y.data()),
            static_cast<std::streamsize>(y.size() * sizeof(float)));
    if (!f) throw std::runtime_error("cannot write " + need(a, "out"));
    return 0;
}

int cmd_classify(const Args& a) {
    std::size_t out_size = 0;
    const std::vector<float> y = run_all(a, 1000, out_size);
    for (std::size_t off = 0; off < y.size(); off += out_size)
        std::printf("%zu\n", argmax(y.data() + off, out_size));
    return 0;
}

// "models/classifier_int8.psm" -> "classifier_int8"
std::string stem(const std::string& path) {
    const std::size_t slash = path.find_last_of("/\\");
    const std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
    return name.substr(0, name.find_last_of('.'));
}

int cmd_bench(const Args& a) {
    if (!a.count("model")) throw std::runtime_error("missing --model");
    const std::size_t len = a.count("len") ? std::stoul(a.at("len").back()) : 256;
    const int samples = a.count("samples") ? std::stoi(a.at("samples").back()) : 300;
    const std::vector<float> x = synthetic_input(len);

    std::string json = "[\n";
    std::printf("| model | backend | median (us) | min (us) | p90 (us) | spectra/s |\n");
    std::printf("|---|---|---:|---:|---:|---:|\n");
    bool first = true;
    for (const std::string& path : a.at("model")) {
        const auto m = Model::load(path);
        for (Backend b : available_backends()) {
            if (!m->supports(b)) continue;
            const Stats s = time_backend(*m, b, x, samples);
            const std::string name = stem(path);
            std::printf("| %s | %s | %.2f | %.2f | %.2f | %.0f |\n", name.c_str(), backend_name(b),
                        s.median_us, s.min_us, s.p90_us, 1e6 / s.median_us);
            char line[384];
            std::snprintf(line, sizeof(line),
                          "%s  {\"model\": \"%s\", \"dtype\": \"%s\", \"backend\": \"%s\", "
                          "\"len\": %zu, \"median_us\": %.3f, \"min_us\": %.3f, \"p90_us\": %.3f}",
                          first ? "" : ",\n", name.c_str(), m->dtype(), backend_name(b), len,
                          s.median_us, s.min_us, s.p90_us);
            json += line;
            first = false;
        }
    }
    json += "\n]\n";
    if (a.count("json")) {
        std::ofstream f(a.at("json").back());
        f << json;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || !std::strcmp(argv[1], "-h") || !std::strcmp(argv[1], "--help")) {
            usage();
            return argc < 2 ? 2 : 0;
        }
        const std::string cmd = argv[1];
        if (cmd == "info") {
            if (argc != 3) throw std::runtime_error("usage: picospec info <model.psm>");
            return cmd_info(argv[2]);
        }
        Args args;
        for (int i = 2; i < argc; i += 2) {
            if (std::strncmp(argv[i], "--", 2) != 0 || i + 1 >= argc)
                throw std::runtime_error(std::string("bad argument: ") + argv[i]);
            args[argv[i] + 2].push_back(argv[i + 1]);
        }
        if (cmd == "run") return cmd_run(args);
        if (cmd == "classify") return cmd_classify(args);
        if (cmd == "bench") return cmd_bench(args);
        throw std::runtime_error("unknown command: " + cmd);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
