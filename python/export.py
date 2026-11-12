"""Export the trained models for the C++ engine.

    python python/export.py [--data ../raman-data]

Writes, for the bacteria classifier and for the denoiser,
  models/<name>_f32.psm     float32 weights
  models/<name>_int8.psm    post-training-quantised int8 model
  tests/data/*              golden inputs/outputs for the C++ tests

The classifier is calibrated on 1,000 spectra from the Raman reference
(training) set, so exporting it needs the dataset (see train_classifier.py).
Its golden inputs are synthetic, made by a fixed-seed generator: no dataset
spectrum is stored in this repository. Without the dataset or a trained
classifier, only the denoiser is exported.

It also contains `IntModel`, an integer-only NumPy implementation of the
quantised network. The C++ kernels must reproduce it bit for bit.

Quantisation scheme (the usual one for int8 inference on Arm CPUs):
  real = scale * (q - zero_point)
  activations  asymmetric int8, one (scale, zero_point) per tensor
  weights      symmetric int8, one scale per output channel, zero_point 0
  bias         int32 at scale (s_in * s_w)
  rescale      multiply by M = s_in * s_w / s_out as a Q0.31 fixed-point
               multiplier followed by a rounding right shift
"""
from __future__ import annotations

import argparse
import math
import pathlib
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # siblings, also under python -I

from spectra import SEED_CALIB, SEED_TEST, make_spectra  # noqa: E402
from train import build_model  # noqa: E402
from train_classifier import DEFAULT_DATA, LENGTH, build_classifier, load_split  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def conv_layers(model) -> list[tuple[np.ndarray, np.ndarray, bool, int, int]]:
    """[(weights [cout, k, cin], bias [cout], followed_by_relu, stride, pad), ...]

    A Linear layer after Flatten becomes a "valid" convolution whose kernel
    is as long as its input. PyTorch flattens [channels, length] with the
    channel as the slow index, so the weight row reshapes to [cin, k] exactly
    like a Conv1d weight and then gets the same transpose to [k, cin].
    """
    mods, out, channels = list(model), [], 1
    for i, m in enumerate(mods):
        relu = i + 1 < len(mods) and isinstance(mods[i + 1], torch.nn.ReLU)
        if isinstance(m, torch.nn.Conv1d):
            w, stride, pad = m.weight.detach().numpy(), m.stride[0], m.padding[0]
        elif isinstance(m, torch.nn.Linear):
            w = m.weight.detach().numpy().reshape(m.out_features, channels, -1)
            stride, pad = 1, 0
        else:
            continue
        w = np.ascontiguousarray(w.transpose(0, 2, 1).astype(np.float32))  # -> [cout, k, cin]
        out.append((w, m.bias.detach().numpy().astype(np.float32), relu, int(stride), int(pad)))
        channels = w.shape[0]
    return out


def write_f32(layers, path: pathlib.Path) -> None:
    with open(path, "wb") as f:
        f.write(b"PSM2" + struct.pack("<II", 0, len(layers)))
        for w, b, relu, stride, pad in layers:
            cout, k, cin = w.shape
            f.write(struct.pack("<IIIIII", cin, cout, k, stride, pad, int(relu)))
            f.write(w.astype("<f4").tobytes() + b.astype("<f4").tobytes())


def quantize_multiplier(m: float) -> tuple[int, int]:
    """m in (0, 1)  ->  (mult, shift) with m ~= mult * 2^-31 * 2^-shift, mult in [2^30, 2^31)."""
    assert 0.0 < m < 1.0, m
    frac, exp = math.frexp(m)  # m = frac * 2^exp, frac in [0.5, 1)
    mult = round(frac * (1 << 31))
    if mult == (1 << 31):
        mult //= 2
        exp += 1
    assert exp <= 0 and -exp <= 31
    return mult, -exp


def affine_params(lo: float, hi: float) -> tuple[float, int]:
    """Scale and zero-point so that [lo, hi] (widened to include 0) maps onto [-128, 127]."""
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    scale = (hi - lo) / 255.0
    zp = int(round(-128 - lo / scale))
    return scale, max(-128, min(127, zp))


class IntModel:
    """Integer-only reference of the quantised network (NumPy int64)."""

    def __init__(self, layers, act_ranges, in_range=(0.0, 1.0)):
        # Scales are rounded to float32 because that is what the file stores.
        self.in_scale, self.in_zp = affine_params(*in_range)
        self.in_scale = float(np.float32(self.in_scale))
        self.layers = []
        s_in, zp_in = self.in_scale, self.in_zp
        for (w, b, relu, stride, pad), (lo, hi) in zip(layers, act_ranges):
            cout, k, cin = w.shape
            s_out, zp_out = affine_params(lo, hi)
            s_out = float(np.float32(s_out))
            s_w = np.maximum(np.abs(w).reshape(cout, -1).max(axis=1), 1e-8) / 127.0
            w_q = np.clip(np.rint(w / s_w[:, None, None]), -127, 127).astype(np.int64)
            bias_q = np.rint(b.astype(np.float64) / (s_in * s_w)).astype(np.int64)
            # Fold the input zero-point into the bias:
            #   sum((x - zp) * w) = sum(x * w) - zp * sum(w)
            # so the kernel can do a plain int8 x int8 dot product.
            bias_folded = bias_q - zp_in * w_q.reshape(cout, -1).sum(axis=1)
            ms = [quantize_multiplier(float(s_in * s_w[c] / s_out)) for c in range(cout)]
            self.layers.append(dict(
                cin=cin, cout=cout, k=k, stride=stride, pad=pad, relu=relu,
                in_zp=zp_in, out_zp=zp_out,
                w=w_q, bias=bias_folded, mult=np.array([m for m, _ in ms], dtype=np.int64),
                shift=np.array([s for _, s in ms], dtype=np.int64)))
            s_in, zp_in = s_out, zp_out
        self.out_scale, self.out_zp = s_in, zp_in

    def quantize(self, x: np.ndarray) -> np.ndarray:
        q = np.rint(x.astype(np.float64) / np.float64(np.float32(self.in_scale))) + self.in_zp
        return np.clip(q, -128, 127).astype(np.int64)

    def dequantize(self, q: np.ndarray) -> np.ndarray:
        return np.float32(self.out_scale) * (q - self.out_zp).astype(np.float32)

    def forward_q(self, q: np.ndarray) -> np.ndarray:
        """q: int array [length] -> int array [out_length * out_channels] (channel-last),
        all values in int8 range. For the classifier that is one score per class."""
        h = q.reshape(-1, 1).astype(np.int64)  # [length, channels]
        for L in self.layers:
            pad = L["pad"]
            hp = np.pad(h, ((pad, pad), (0, 0)), constant_values=L["in_zp"])
            win = np.lib.stride_tricks.sliding_window_view(hp, L["k"], axis=0)  # [len, cin, k]
            win = win[:: L["stride"]]                                         # [out_len, cin, k]
            win = win.transpose(0, 2, 1).reshape(win.shape[0], -1)             # [out_len, k*cin]
            acc = win @ L["w"].reshape(L["cout"], -1).T + L["bias"]
            assert np.abs(acc).max() < 2**31
            v = (acc * L["mult"] + (1 << 30)) >> 31                 # SQRDMULH
            v = (v + ((1 << L["shift"]) >> 1)) >> L["shift"]        # rounding right shift
            v = v + L["out_zp"]
            h = np.clip(v, L["out_zp"] if L["relu"] else -128, 127)
        return h.reshape(-1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.stack([self.dequantize(self.forward_q(self.quantize(row))) for row in x])

    def write(self, path: pathlib.Path) -> None:
        with open(path, "wb") as f:
            f.write(b"PSM2" + struct.pack("<II", 1, len(self.layers)))
            f.write(struct.pack("<ff", self.in_scale, self.out_scale))
            for L in self.layers:
                f.write(struct.pack("<IIIIII", L["cin"], L["cout"], L["k"], L["stride"], L["pad"],
                                    int(L["relu"])))
                f.write(struct.pack("<ii", L["in_zp"], L["out_zp"]))
                f.write(L["w"].astype("<i1").tobytes())
                for key in ("bias", "mult", "shift"):
                    f.write(L[key].astype("<i4").tobytes())


def calibrate(model, calib: np.ndarray) -> list[tuple[float, float]]:
    """Observe each layer output (after its ReLU) on calibration data -> value ranges."""
    ranges, h = [], torch.from_numpy(calib).unsqueeze(1)
    mods = list(model)
    with torch.no_grad():
        for i, m in enumerate(mods):
            h = m(h)
            last = i == len(mods) - 1
            if isinstance(m, torch.nn.ReLU) or last:
                a = h.numpy().ravel()
                # ReLU outputs: clip the top 0.01 % so a few outliers do not
                # waste the int8 range. Final output: keep the full range.
                ranges.append((float(a.min()), float(a.max())) if last
                              else (0.0, float(np.quantile(a, 0.9999))))
    return ranges


def load_models():
    model = build_model()
    model.load_state_dict(torch.load(ROOT / "artifacts" / "denoiser.pt"))
    model.eval()
    layers = conv_layers(model)
    calib, _ = make_spectra(1000, SEED_CALIB)
    return model, layers, IntModel(layers, calibrate(model, calib))


def load_classifier(data_dir=DEFAULT_DATA):
    model = build_classifier()
    model.load_state_dict(torch.load(ROOT / "artifacts" / "classifier.pt"))
    model.eval()
    layers = conv_layers(model)
    # Calibrate on training data only: every 60th reference spectrum, which
    # is 1,000 spectra spread evenly over the 30 isolates.
    calib = load_split(data_dir, "reference")[0][::60]
    return model, layers, IntModel(layers, calibrate(model, calib))


def synthetic_inputs(count: int, seed: int = 2026, length: int = LENGTH) -> np.ndarray:
    """Spectrum-shaped test inputs from a fixed seed: a sloping baseline, a few
    narrow peaks and noise, scaled to [0, 1]. They only have to exercise the
    arithmetic; they are not Raman spectra and no accuracy is measured on them."""
    rng, x = np.random.default_rng(seed), np.linspace(0.0, 1.0, length)
    out = np.empty((count, length), dtype=np.float32)
    for i in range(count):
        y = rng.uniform(0.05, 0.3) + rng.uniform(-0.1, 0.1) * x
        for _ in range(int(rng.integers(3, 9))):
            y += rng.uniform(0.1, 0.8) * np.exp(-0.5 * ((x - rng.uniform(0.05, 0.95)) / rng.uniform(0.005, 0.04)) ** 2)
        y += rng.normal(0.0, 0.02, length)
        out[i] = (y - y.min()) / (y.max() - y.min())
    return out


def write_golden(model, qmodel, x: np.ndarray, prefix: str) -> None:
    """Golden vectors for the C++ tests: inputs, PyTorch outputs, integer-reference outputs."""
    data = ROOT / "tests" / "data"
    data.mkdir(parents=True, exist_ok=True)
    x.astype("<f4").tofile(data / f"{prefix}_in.f32")
    with torch.no_grad():
        ref = model(torch.from_numpy(x).unsqueeze(1)).numpy().reshape(len(x), -1)
    ref.astype("<f4").tofile(data / f"{prefix}_out_f32.f32")
    q = np.stack([qmodel.forward_q(qmodel.quantize(row)) for row in x])
    q.astype("<i1").tofile(data / f"{prefix}_out_int8.i8")


def export(name: str, layers, qmodel) -> None:
    (ROOT / "models").mkdir(exist_ok=True)
    write_f32(layers, ROOT / "models" / f"{name}_f32.psm")
    qmodel.write(ROOT / "models" / f"{name}_int8.psm")
    for kind in ("f32", "int8"):
        path = ROOT / "models" / f"{name}_{kind}.psm"
        print(f"models/{path.name}: {path.stat().st_size} bytes")
    print("  zero-points:", [(L["in_zp"], L["out_zp"]) for L in qmodel.layers],
          "shifts:", sorted({int(s) for L in qmodel.layers for s in L["shift"]}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    args = ap.parse_args()

    model, layers, qmodel = load_models()
    export("denoiser", layers, qmodel)
    write_golden(model, qmodel, make_spectra(16, SEED_TEST + 1)[0], "golden")  # 16 unseen spectra

    if (ROOT / "artifacts" / "classifier.pt").exists() and (pathlib.Path(args.data) / "X_reference.npy").exists():
        model, layers, qmodel = load_classifier(args.data)
        export("classifier", layers, qmodel)
        write_golden(model, qmodel, synthetic_inputs(8), "classifier")
    else:
        print("classifier: skipped (needs artifacts/classifier.pt and the dataset in --data)")


if __name__ == "__main__":
    main()
