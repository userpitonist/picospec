"""Score every implementation of the denoiser on the same 4000 unseen simulated spectra.

    python python/evaluate.py [--binary build/picospec]

Compares, as RMSE against the noise-free spectrum:
  - the raw noisy input
  - two classical filters, each tuned on the validation split
  - the float32 network (PyTorch, and the C++ engine)
  - the int8 network (Python integer reference, and the C++ engine)
and checks that every C++ backend reproduces its reference.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # siblings, also under python -I

from export import load_models  # noqa: E402
from spectra import SEED_TEST, SEED_VAL, make_spectra  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)))


def median3(x: np.ndarray) -> np.ndarray:
    p = np.pad(x, ((0, 0), (1, 1)), mode="edge")
    return np.median(np.stack([p[:, :-2], p[:, 1:-1], p[:, 2:]]), axis=0)


def gaussian_filter(x: np.ndarray, sigma: float) -> np.ndarray:
    r = max(1, int(round(4 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    p = np.pad(median3(x), ((0, 0), (r, r)), mode="edge")
    return np.stack([np.convolve(row, k / k.sum(), mode="valid") for row in p])


def savgol_filter(x: np.ndarray, window: int, order: int = 3) -> np.ndarray:
    half = window // 2
    A = np.vander(np.arange(-half, half + 1), order + 1, increasing=True)
    coeff = np.linalg.pinv(A)[0]  # least-squares polynomial fit, value at centre
    p = np.pad(median3(x), ((0, 0), (half, half)), mode="edge")
    return np.stack([np.convolve(row, coeff[::-1], mode="valid") for row in p])


def run_cpp(binary: str, model: str, backend: str, x: np.ndarray) -> np.ndarray:
    """Run spectra [n, len] through the C++ engine -> raw model outputs [n, values per spectrum]."""
    with tempfile.TemporaryDirectory() as tmp:
        fin, fout = pathlib.Path(tmp) / "in.f32", pathlib.Path(tmp) / "out.f32"
        x.astype("<f4").tofile(fin)
        subprocess.run([binary, "run", "--model", str(ROOT / "models" / model), "--in", str(fin),
                        "--out", str(fout), "--len", str(x.shape[1]), "--backend", backend],
                       check=True, capture_output=True)
        return np.fromfile(fout, dtype="<f4").reshape(len(x), -1)


def backends(binary: str, model: str) -> list[str]:
    out = subprocess.run([binary, "info", str(ROOT / "models" / model)], check=True,
                         capture_output=True, text=True).stdout
    return next(l for l in out.splitlines() if l.startswith("backends")).split(":")[1].split()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=str(ROOT / "build" / "picospec"))
    args = ap.parse_args()

    model, _, qmodel = load_models()
    val_n, val_c = make_spectra(2000, SEED_VAL)
    test_n, test_c = make_spectra(4000, SEED_TEST)

    # Tune the classical filters on validation data, score on test data.
    sigma = min(np.arange(0.5, 6.01, 0.25), key=lambda s: rmse(gaussian_filter(val_n, s), val_c))
    window = min(range(5, 42, 2), key=lambda w: rmse(savgol_filter(val_n, w), val_c))

    with torch.no_grad():
        torch_out = model(torch.from_numpy(test_n).unsqueeze(1)).squeeze(1).numpy()
    int_ref = qmodel(test_n)

    rows = [
        ("noisy input (no processing)", rmse(test_n, test_c)),
        (f"median-3 + Gaussian smoothing (sigma={sigma:g}, tuned)", rmse(gaussian_filter(test_n, sigma), test_c)),
        (f"median-3 + Savitzky-Golay (window={window}, order 3, tuned)", rmse(savgol_filter(test_n, window), test_c)),
        ("CNN float32, PyTorch", rmse(torch_out, test_c)),
        ("CNN int8, Python integer reference", rmse(int_ref, test_c)),
    ]
    checks = {}
    for b in backends(args.binary, "denoiser_f32.psm"):
        out = run_cpp(args.binary, "denoiser_f32.psm", b, test_n)
        checks[f"f32/{b}: max |C++ - PyTorch|"] = float(np.abs(out - torch_out).max())
        if b == "neon" or len(checks) == 1:
            cpp_f32 = out
    for b in backends(args.binary, "denoiser_int8.psm"):
        out = run_cpp(args.binary, "denoiser_int8.psm", b, test_n)
        checks[f"int8/{b}: outputs identical to integer reference"] = bool(np.array_equal(out, int_ref))
        cpp_int8 = out
    rows.insert(4, ("CNN float32, C++ engine", rmse(cpp_f32, test_c)))
    rows.append(("CNN int8, C++ engine", rmse(cpp_int8, test_c)))

    print(f"Test set: {len(test_n)} simulated spectra x {test_n.shape[1]} samples\n")
    print("| method | RMSE vs clean |\n|---|---:|")
    for name, value in rows:
        print(f"| {name} | {value:.5f} |")
    print()
    for name, value in checks.items():
        print(f"{name}: {value}")
    assert all(v is True or (isinstance(v, float) and v < 1e-5) for v in checks.values()), checks

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "accuracy.json").write_text(json.dumps(
        {"test_spectra": len(test_n), "rmse": dict(rows), "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
