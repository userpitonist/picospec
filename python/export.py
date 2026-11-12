"""Export the trained model for the C++ engine.

    python python/export.py

Writes
  models/denoiser_f32.psm    float32 weights
  models/denoiser_int8.psm   post-training-quantised int8 model
  tests/data/*               golden inputs/outputs for the C++ tests

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

import math
import pathlib
import struct

import numpy as np
import torch

from spectra import SEED_CALIB, SEED_TEST, make_spectra
from train import build_model

ROOT = pathlib.Path(__file__).resolve().parent.parent


def conv_layers(model) -> list[tuple[np.ndarray, np.ndarray, bool]]:
    """[(weights [cout, k, cin], bias [cout], followed_by_relu), ...]"""
    mods, out = list(model), []
    for i, m in enumerate(mods):
        if isinstance(m, torch.nn.Conv1d):
            relu = i + 1 < len(mods) and isinstance(mods[i + 1], torch.nn.ReLU)
            w = m.weight.detach().numpy().transpose(0, 2, 1).astype(np.float32)  # -> [cout, k, cin]
            out.append((np.ascontiguousarray(w), m.bias.detach().numpy().astype(np.float32), relu))
    return out


def write_f32(layers, path: pathlib.Path) -> None:
    with open(path, "wb") as f:
        f.write(b"PSM1" + struct.pack("<II", 0, len(layers)))
        for w, b, relu in layers:
            cout, k, cin = w.shape
            f.write(struct.pack("<IIII", cin, cout, k, int(relu)))
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
        for (w, b, relu), (lo, hi) in zip(layers, act_ranges):
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
                cin=cin, cout=cout, k=k, relu=relu, in_zp=zp_in, out_zp=zp_out,
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
        """q: int array [length] -> int array [length], all values in int8 range."""
        h = q.reshape(-1, 1).astype(np.int64)  # [length, channels]
        for L in self.layers:
            pad = L["k"] // 2
            hp = np.pad(h, ((pad, pad), (0, 0)), constant_values=L["in_zp"])
            win = np.lib.stride_tricks.sliding_window_view(hp, L["k"], axis=0)  # [len, cin, k]
            win = win.transpose(0, 2, 1).reshape(h.shape[0], -1)               # [len, k*cin]
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
            f.write(b"PSM1" + struct.pack("<II", 1, len(self.layers)))
            f.write(struct.pack("<ff", self.in_scale, self.out_scale))
            for L in self.layers:
                f.write(struct.pack("<IIII", L["cin"], L["cout"], L["k"], int(L["relu"])))
                f.write(struct.pack("<ii", L["in_zp"], L["out_zp"]))
                f.write(L["w"].astype("<i1").tobytes())
                for key in ("bias", "mult", "shift"):
                    f.write(L[key].astype("<i4").tobytes())


def calibrate(model, calib: np.ndarray) -> list[tuple[float, float]]:
    """Observe each conv output (after its ReLU) on calibration data -> value ranges."""
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


def main() -> None:
    model, layers, qmodel = load_models()
    (ROOT / "models").mkdir(exist_ok=True)
    write_f32(layers, ROOT / "models" / "denoiser_f32.psm")
    qmodel.write(ROOT / "models" / "denoiser_int8.psm")

    # Golden vectors for the C++ tests: 16 unseen spectra.
    noisy, _ = make_spectra(16, SEED_TEST + 1)
    data = ROOT / "tests" / "data"
    data.mkdir(parents=True, exist_ok=True)
    noisy.astype("<f4").tofile(data / "golden_in.f32")
    with torch.no_grad():
        ref = model(torch.from_numpy(noisy).unsqueeze(1)).squeeze(1).numpy()
    ref.astype("<f4").tofile(data / "golden_out_f32.f32")
    q = np.stack([qmodel.forward_q(qmodel.quantize(row)) for row in noisy])
    q.astype("<i1").tofile(data / "golden_out_int8.i8")

    for name in ("denoiser_f32.psm", "denoiser_int8.psm"):
        print(f"models/{name}: {(ROOT / 'models' / name).stat().st_size} bytes")
    print("zero-points:", [(L["in_zp"], L["out_zp"]) for L in qmodel.layers],
          "shifts:", sorted({int(s) for L in qmodel.layers for s in L["shift"]}))


if __name__ == "__main__":
    main()
