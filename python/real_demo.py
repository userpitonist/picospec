"""Run the int8 engine on the two real traces from my 2024 spectrometer.

    python python/real_demo.py [--binary build/picospec]

data/real_traces.csv holds the grey-value profiles I extracted with ImageJ
from phone photos of the spectrum: torch light through water (reference) and
through diluted Ribena. This is a qualitative check only: there is no
noise-free ground truth for these traces, so no accuracy number is claimed.
"""
from __future__ import annotations

import argparse
import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate import run_cpp
from spectra import LENGTH

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=str(ROOT / "build" / "picospec"))
    args = ap.parse_args()

    with open(ROOT / "data" / "real_traces.csv") as f:
        rows = list(csv.DictReader(f))
    px = np.array([float(r["pixel"]) for r in rows])
    grid = np.linspace(px.min(), px.max(), LENGTH)  # resample to the model's 256 samples
    traces = {}
    for name in ("water_grey", "ribena_grey"):
        grey = np.array([float(r[name]) for r in rows])
        traces[name] = np.interp(grid, px, grey) / 255.0  # 8-bit grey -> [0, 1]
    x = np.stack(list(traces.values())).astype(np.float32)
    y = run_cpp(args.binary, "denoiser_int8.psm", "scalar", x)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, title, raw, den in zip(axes, ("Torch through water", "Torch through diluted Ribena"), x, y):
        ax.plot(grid, raw, color="0.6", lw=1.0, label="measured (phone camera)")
        ax.plot(grid, den, color="#0b6e99", lw=1.6, label="int8 CNN output")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("pixel along the dispersion axis (blue to red)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("grey value / 255")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = ROOT / "results" / "real_traces.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
