"""Train the float32 denoiser and write artifacts/denoiser.pt.

    python python/train.py

A deliberately small 1-D CNN: 4 convolutions, kernel 5, 16 channels, ReLU.
Small enough to read every weight, large enough to beat classical filters.
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import torch
from torch import nn

from spectra import SEED_TRAIN, SEED_VAL, make_spectra

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHANNELS, KERNEL = 16, 5


def build_model() -> nn.Sequential:
    pad = KERNEL // 2
    return nn.Sequential(
        nn.Conv1d(1, CHANNELS, KERNEL, padding=pad), nn.ReLU(),
        nn.Conv1d(CHANNELS, CHANNELS, KERNEL, padding=pad), nn.ReLU(),
        nn.Conv1d(CHANNELS, CHANNELS, KERNEL, padding=pad), nn.ReLU(),
        nn.Conv1d(CHANNELS, 1, KERNEL, padding=pad),
    )


def main() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(4)
    epochs, batch, n_train, n_val = 40, 128, 16000, 2000

    xn, xc = make_spectra(n_train, SEED_TRAIN)
    vn, vc = make_spectra(n_val, SEED_VAL)
    xn, xc, vn, vc = (torch.from_numpy(a).unsqueeze(1) for a in (xn, xc, vn, vc))

    model = build_model()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    history, best, t0 = [], float("inf"), time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss = loss_fn(model(xn[idx]), xc[idx])
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            val_rmse = float(torch.sqrt(loss_fn(model(vn), vc)))
        history.append({"epoch": epoch, "val_rmse": val_rmse})
        if val_rmse < best:
            best = val_rmse
            (ROOT / "artifacts").mkdir(exist_ok=True)
            torch.save(model.state_dict(), ROOT / "artifacts" / "denoiser.pt")
        print(f"epoch {epoch:2d}  val RMSE {val_rmse:.5f}")

    info = {
        "channels": CHANNELS, "kernel": KERNEL, "epochs": epochs, "train_count": n_train,
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_val_rmse": best, "seconds": round(time.time() - t0, 1),
        "torch": torch.__version__, "numpy": np.__version__, "history": history,
    }
    (ROOT / "artifacts" / "training.json").write_text(json.dumps(info, indent=2))
    print(f"best val RMSE {best:.5f}, {info['parameters']} parameters, {info['seconds']} s")


if __name__ == "__main__":
    main()
