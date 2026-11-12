"""Train the bacteria classifier on real Raman spectra.

    python python/train_classifier.py [--data ../raman-data]

Data: the public bacteria-ID set from Ho et al., "Rapid identification of
pathogenic bacteria using Raman spectroscopy and deep learning", Nature
Communications 10, 4927 (2019). Code: https://github.com/csho33/bacteria-ID
(the data are on the Dropbox link in that repository's README). The dataset
is not part of this repository. The script needs these files in --data:

    X_reference.npy  y_reference.npy    60,000 spectra, 2,000 per isolate
    X_finetune.npy   y_finetune.npy      3,000 spectra,   100 per isolate
    X_test.npy       y_test.npy          3,000 spectra,   100 per isolate

Each spectrum has 1000 samples scaled to [0, 1]; labels are 30 isolates
(0-29). The three sets were measured on separate occasions, so the test set
is a fair check that the model has learnt the bacteria and not one session.

Protocol (the one the dataset's authors describe): train on the reference
set, fine-tune on the fine-tune set, score once on the test set.

Model: four stride-2 convolutions with ReLU, then a dense layer. No
batch-norm and no pooling, so every layer is something the C++ engine runs
directly: a dense layer is a convolution whose kernel covers its whole input.

Also trains two baselines on the same data with the same protocol, so the
comparison in the README is like for like. Both are multinomial logistic
regression (one linear layer, 30,030 parameters):
  logreg_raw   raw spectra, the same optimiser settings as the CNN
  logreg_std   every sample position standardised with the mean and standard
               deviation of the reference set, and stronger weight decay.
               These two settings were chosen by looking at test accuracy,
               which favours this baseline.

    python python/train_classifier.py --only baselines     # a few seconds

Writes artifacts/classifier.pt, artifacts/logreg_raw.pt, artifacts/logreg_std.pt
and artifacts/classifier_training.json.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch
from torch import nn

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT.parent / "raman-data"
LENGTH, CLASSES = 1000, 30
CONVS = [(16, 9), (32, 7), (64, 5), (64, 5)]  # (output channels, kernel), all stride 2
STRIDE = 2


def load_split(data_dir, name: str) -> tuple[np.ndarray, np.ndarray]:
    """-> (spectra float32 [n, 1000], labels int64 [n])"""
    d = pathlib.Path(data_dir)
    try:
        x = np.load(d / f"X_{name}.npy", mmap_mode="r", allow_pickle=False)
        y = np.load(d / f"y_{name}.npy", allow_pickle=False)
    except FileNotFoundError as e:
        raise SystemExit(f"{e}\nDownload the dataset first: see the docstring of {__file__}") from None
    assert x.ndim == 2 and x.shape[1] == LENGTH and len(x) == len(y), (x.shape, y.shape)
    return np.asarray(x, dtype=np.float32), np.asarray(y).astype(np.int64)


def build_classifier() -> nn.Sequential:
    mods, cin, n = [], 1, LENGTH
    for cout, k in CONVS:
        mods += [nn.Conv1d(cin, cout, k, stride=STRIDE, padding=k // 2), nn.ReLU()]
        n = (n + 2 * (k // 2) - k) // STRIDE + 1
        cin = cout
    # Flatten, not global average pooling: in a spectrum the position of a
    # peak is the information, and averaging over positions throws it away.
    return nn.Sequential(*mods, nn.Flatten(), nn.Dropout(0.3), nn.Linear(cin * n, CLASSES))


class Standardise(nn.Module):
    """(x - mean) / std per sample position; mean and std are fixed, not trained."""

    def __init__(self, mean=None, std=None):
        super().__init__()
        self.register_buffer("mean", torch.zeros(LENGTH) if mean is None else mean)
        self.register_buffer("std", torch.ones(LENGTH) if std is None else std)

    def forward(self, x):
        return (x - self.mean) / self.std


def build_logreg(standardise: bool, reference=None) -> nn.Sequential:
    """`reference` [n, 1, 1000]: where the standardisation statistics come from."""
    pre = []
    if standardise:
        pre = [Standardise()] if reference is None else [
            Standardise(reference.mean(dim=(0, 1)), reference.std(dim=(0, 1)) + 1e-6)]
    return nn.Sequential(nn.Flatten(), *pre, nn.Linear(LENGTH, CLASSES))


def fit(model, x, y, epochs: int, lr: float, tag: str, weight_decay: float = 1e-5,
        batch: int = 256) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        perm, total, t0 = torch.randperm(len(x)), 0.0, time.time()
        for i in range(0, len(x), batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss = loss_fn(model(x[idx]), y[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if epochs <= 10 or epoch % 10 == 0:
            print(f"  {tag} epoch {epoch:3d}  loss {total / len(x):.4f}  {time.time() - t0:.0f} s", flush=True)


def accuracy(model, x, y) -> float:
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(x[i : i + 1000]).argmax(1) for i in range(0, len(x), 1000)])
    return float((pred == y).float().mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--ref-epochs", type=int, default=4)
    ap.add_argument("--finetune-epochs", type=int, default=50)
    ap.add_argument("--only", choices=["classifier", "baselines"], help="train just this part")
    args = ap.parse_args()
    torch.set_num_threads(4)

    sets = {}
    for name in ("reference", "finetune", "test"):
        x, y = load_split(args.data, name)
        sets[name] = (torch.from_numpy(x).unsqueeze(1), torch.from_numpy(y))
        print(f"{name}: {tuple(x.shape)}, labels {int(y.min())}-{int(y.max())}")
    (xr, yr), (xf, yf), (xt, yt) = sets["reference"], sets["finetune"], sets["test"]
    (ROOT / "artifacts").mkdir(exist_ok=True)
    log = ROOT / "artifacts" / "classifier_training.json"
    info = json.loads(log.read_text()) if log.exists() else {}
    info.update({"torch": torch.__version__, "numpy": np.__version__, "seed": 0,
                 "reference_epochs": args.ref_epochs, "finetune_epochs": args.finetune_epochs})

    # The baselines get five times the epochs on the reference set: one epoch
    # takes a fraction of a second and they need more of them to settle.
    jobs = [
        ("logreg_raw", lambda: build_logreg(False), 5 * args.ref_epochs, 1e-5),
        ("logreg_std", lambda: build_logreg(True, xr), 5 * args.ref_epochs, 1e-3),
        ("classifier", build_classifier, args.ref_epochs, 1e-5),
    ]
    for name, build, ref_epochs, weight_decay in jobs:
        if args.only and (name == "classifier") != (args.only == "classifier"):
            continue
        torch.manual_seed(0)
        model, t0 = build(), time.time()
        print(f"{name}: {sum(p.numel() for p in model.parameters())} parameters")
        fit(model, xr, yr, ref_epochs, 1e-3, "reference", weight_decay)
        before = accuracy(model, xt, yt)
        fit(model, xf, yf, args.finetune_epochs, 3e-4, "fine-tune", weight_decay)
        after = accuracy(model, xt, yt)
        torch.save(model.state_dict(), ROOT / "artifacts" / f"{name}.pt")
        info[name] = {"parameters": sum(p.numel() for p in model.parameters()),
                      "reference_epochs": ref_epochs, "weight_decay": weight_decay,
                      "test_accuracy_before_finetune": before, "test_accuracy": after,
                      "seconds": round(time.time() - t0, 1)}
        print(f"{name}: test accuracy {after:.4f} (before fine-tune {before:.4f}), {info[name]['seconds']} s")
    info.pop("logreg", None)
    log.write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
