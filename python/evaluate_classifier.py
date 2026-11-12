"""Score every implementation of the bacteria classifier on the Raman test set.

    python python/evaluate_classifier.py [--data ../raman-data] [--binary build/picospec]

Reports test accuracy (3,000 spectra, 30 isolates, 100 spectra each) for
  - two logistic-regression baselines (see train_classifier.py)
  - the float32 network: PyTorch, and the C++ engine
  - the int8 network: the Python integer reference, and the C++ engine
and checks that every C++ backend reproduces its reference: float32 within
rounding error, int8 exactly (every one of the 3,000 x 30 scores identical).

Writes results/classifier_accuracy.json; the printed text is
results/classifier_accuracy.md.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # siblings, also under python -I

from evaluate import backends, run_cpp  # noqa: E402
from export import load_classifier  # noqa: E402
from train_classifier import DEFAULT_DATA, build_logreg, load_split  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def classify_cpp(binary: str, model: str, x: np.ndarray) -> np.ndarray:
    """Class index per spectrum from `picospec classify` (the engine's own argmax)."""
    with tempfile.TemporaryDirectory() as tmp:
        fin = pathlib.Path(tmp) / "in.f32"
        x.astype("<f4").tofile(fin)
        out = subprocess.run([binary, "classify", "--model", str(ROOT / "models" / model),
                              "--in", str(fin), "--len", str(x.shape[1])],
                             check=True, capture_output=True, text=True).stdout
        return np.array(out.split(), dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--binary", default=str(ROOT / "build" / "picospec"))
    args = ap.parse_args()

    x, y = load_split(args.data, "test")
    xt = torch.from_numpy(x).unsqueeze(1)
    acc = lambda scores: float((np.argmax(scores, axis=1) == y).mean())  # noqa: E731

    rows = []
    for name, standardise, label in (
            ("logreg_raw", False, "logistic regression, raw spectra"),
            ("logreg_std", True, "logistic regression, standardised inputs (settings picked on the test set)")):
        m = build_logreg(standardise)
        m.load_state_dict(torch.load(ROOT / "artifacts" / f"{name}.pt"))
        m.eval()
        with torch.no_grad():
            rows.append((label, acc(m(xt).numpy())))

    model, _, qmodel = load_classifier(args.data)
    with torch.no_grad():
        torch_scores = model(xt).numpy()
    int_q = np.stack([qmodel.forward_q(qmodel.quantize(row)) for row in x])  # int8 codes [n, 30]
    int_scores = qmodel.dequantize(int_q)

    checks, cpp = {}, {}
    tol = 1e-4 * (1.0 + float(np.abs(torch_scores).max()))
    for b in backends(args.binary, "classifier_f32.psm"):
        out = run_cpp(args.binary, "classifier_f32.psm", b, x)
        cpp[f"float32/{b}"] = acc(out)
        checks[f"f32/{b}: max |C++ - PyTorch| (scores reach {np.abs(torch_scores).max():.0f})"] = \
            float(np.abs(out - torch_scores).max())
        checks[f"f32/{b}: same predicted class as PyTorch on every spectrum"] = \
            bool(np.array_equal(out.argmax(1), torch_scores.argmax(1)))
    for b in backends(args.binary, "classifier_int8.psm"):
        out = run_cpp(args.binary, "classifier_int8.psm", b, x)
        cpp[f"int8/{b}"] = acc(out)
        checks[f"int8/{b}: all {out.size} scores identical to integer reference"] = \
            bool(np.array_equal(out, int_scores))
    for kind in ("f32", "int8"):
        ref = torch_scores if kind == "f32" else int_q
        checks[f"{kind}: `picospec classify` labels == argmax of reference scores"] = \
            bool(np.array_equal(classify_cpp(args.binary, f"classifier_{kind}.psm", x), ref.argmax(1)))

    f32_cpp, int8_cpp = set(v for k, v in cpp.items() if k.startswith("float32")), \
        set(v for k, v in cpp.items() if k.startswith("int8"))
    assert len(int8_cpp) == 1, cpp
    rows += [
        ("CNN float32, PyTorch", acc(torch_scores)),
        ("CNN float32, C++ engine", max(f32_cpp)),
        ("CNN int8, Python integer reference", acc(int_q)),
        ("CNN int8, C++ engine (all backends)", int8_cpp.pop()),
    ]
    agree = float((int_q.argmax(1) == torch_scores.argmax(1)).mean())
    se = 100 * math.sqrt(rows[2][1] * (1 - rows[2][1]) / len(y))

    print(f"Test set: {len(x)} measured Raman spectra x {x.shape[1]} samples, {int(y.max()) + 1} classes "
          f"(chance = {100 / (int(y.max()) + 1):.1f} %)\n")
    print("| method | test accuracy |\n|---|---:|")
    for name, value in rows:
        print(f"| {name} | {100 * value:.2f} % |")
    print(f"\nint8 vs float32 (PyTorch): {100 * (rows[4][1] - rows[2][1]):+.2f} points; "
          f"same predicted class on {100 * agree:.2f} % of spectra")
    print(f"sampling error of one accuracy figure on {len(y)} spectra: about +/- {se:.1f} points (one standard error)\n")
    for name, value in checks.items():
        print(f"{name}: {value}")
    bad = {k: v for k, v in checks.items() if not (v is True or (isinstance(v, float) and v < tol))}
    assert not bad, bad

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "classifier_accuracy.json").write_text(json.dumps(
        {"test_spectra": len(x), "classes": int(y.max()) + 1, "accuracy": dict(rows),
         "cpp_accuracy_by_backend": cpp, "int8_minus_float32_points": 100 * (rows[4][1] - rows[2][1]),
         "int8_float32_agreement": agree, "standard_error_points": se, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
