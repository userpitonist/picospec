"""Simulated spectra for training and scoring the denoiser.

Why simulated: my 2024 home-made spectrometer (CD diffraction grating, phone
camera as detector, torch as light source) produced only a handful of real
traces and no noise-free ground truth, which is not enough to train or score
a model. The generator below imitates the kind of trace that instrument
gives: a white-LED-like source (narrow blue peak + broad phosphor band),
optionally passed through a sample with one or two absorption bands
(Beer-Lambert: transmitted = source * 10^-A), plus camera noise.

Every accuracy number in this repository is measured on these simulated
spectra. Nothing here claims accuracy on a real instrument.
"""
from __future__ import annotations

import numpy as np

LENGTH = 256  # samples per spectrum

# Fixed, non-overlapping seeds for each split.
SEED_TRAIN, SEED_VAL, SEED_CALIB, SEED_TEST = 101, 202, 303, 404


def _gauss(x: np.ndarray, centre: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - centre) / width) ** 2)


def make_spectra(count: int, seed: int, length: int = LENGTH) -> tuple[np.ndarray, np.ndarray]:
    """Return (noisy, clean) float32 arrays of shape [count, length], values in [0, 1]."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, length)
    clean = np.empty((count, length), dtype=np.float32)
    noisy = np.empty((count, length), dtype=np.float32)

    for i in range(count):
        if rng.random() < 0.7:
            # White LED: blue emitter plus broad phosphor emission.
            source = rng.uniform(0.4, 1.0) * _gauss(x, rng.uniform(0.14, 0.24), rng.uniform(0.025, 0.05))
            source += rng.uniform(0.5, 1.0) * _gauss(x, rng.uniform(0.45, 0.65), rng.uniform(0.11, 0.2))
        else:
            # Some other source: a few arbitrary emission bands.
            source = np.zeros(length)
            for _ in range(int(rng.integers(1, 5))):
                source += rng.uniform(0.2, 1.0) * _gauss(x, rng.uniform(0.08, 0.92), rng.uniform(0.02, 0.12))

        if rng.random() < 0.7:
            # Sample in the beam: absorbance bands, Beer-Lambert attenuation.
            absorbance = np.zeros(length)
            for _ in range(int(rng.integers(1, 3))):
                absorbance += rng.uniform(0.1, 1.5) * _gauss(x, rng.uniform(0.1, 0.9), rng.uniform(0.04, 0.15))
            source = source * 10.0 ** (-absorbance)

        signal = source + rng.uniform(0.0, 0.05)  # stray-light offset
        signal = signal / max(float(signal.max()), 1e-6) * rng.uniform(0.5, 0.95)

        # Camera noise: read noise (constant) + shot noise (grows with signal).
        read = rng.uniform(0.005, 0.03)
        shot = rng.uniform(0.01, 0.05)
        observed = signal + rng.normal(0.0, 1.0, length) * np.sqrt(read**2 + shot**2 * signal)
        if rng.random() < 0.5:  # hot pixels / dust glints
            idx = rng.choice(length, size=int(rng.integers(1, 4)), replace=False)
            observed[idx] += rng.uniform(0.1, 0.3, len(idx)) * rng.choice([-1.0, 1.0], len(idx))

        clean[i] = signal
        noisy[i] = np.clip(observed, 0.0, 1.0)
    return noisy, clean
