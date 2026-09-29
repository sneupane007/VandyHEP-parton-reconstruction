"""Real training run: JSON -> transform -> train/val split -> GNN -> Chamfer -> checkpoint.

Run from the repo root:  .jupyter_venv/bin/python3 scripts/train.py

Unlike experiment.py (a plumbing smoke test with no held-out data), 

this holds out a validation split and saves a checkpoint per checkpoints/README.md's convention: weights,
optimizer state, epoch, and a copy of the run's config, in their own timestamped
directory, plus a training log.

Only chamfer_set_loss exists as a loss so far
"""

import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from parton_recon.data import JetPairDataset, LogPt
from parton_recon.losses import chamfer_set_loss
from parton_recon.models import JetGNN

GENERATED = REPO_ROOT / "parton_recon" / "data" / "generate"
CHECKPOINTS = REPO_ROOT / "checkpoints"

SEED = 0
VAL_FRACTION = 0.2
EPOCHS = 1000
LR = 1e-3
LOG_EVERY = 50

ARCH = dict(in_channels=3, hidden=64, n_slots=24, out_channels=3)


def split_dataset(dataset, val_fraction, seed):
    """Index-level train/val split, shuffled once with a fixed seed."""
    n = len(dataset)
    n_val = max(1, round(n * val_fraction))
    indices = np.random.default_rng(seed).permutation(n)
    val_idx, train_idx = indices[:n_val], indices[n_val:]
    return Subset(dataset, train_idx.tolist()), Subset(dataset, val_idx.tolist())


def save_checkpoint(path, epoch, model, optimizer, train_loss, val_loss):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "arch_config": ARCH,
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        path,
    )


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    dataset = JetPairDataset(
        GENERATED / "model1_hadron_level_graphs_.json",
        GENERATED / "model1_parton_level_graphs_.json",
        transform=LogPt(),
    )
    train_set, val_set = split_dataset(dataset, VAL_FRACTION, SEED)
    print(f"{len(dataset)} events -> {len(train_set)} train / {len(val_set)} val")

    # Full-batch per split — same pattern as experiment.py; the datasets are small
    # enough that this is not a meaningful simplification over mini-batching.
    train_batch = next(iter(DataLoader(
        train_set, batch_size=len(train_set), follow_batch=["parton_x"], shuffle=False
    )))
    val_batch = next(iter(DataLoader(
        val_set, batch_size=len(val_set), follow_batch=["parton_x"], shuffle=False
    )))
    train_batch, val_batch = train_batch.to(device), val_batch.to(device)

    model = JetGNN(**ARCH).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"training {sum(p.numel() for p in model.parameters()):,} parameters, {EPOCHS} epochs")

    run_dir = CHECKPOINTS / f"gnn_chamfer_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True)
    log_path = run_dir / "train_log.csv"
    log_path.write_text("epoch,train_loss,val_loss\n")

    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(train_batch)
        train_loss = chamfer_set_loss(pred, train_batch.parton_x, train_batch.parton_x_batch)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_batch)
            val_loss = chamfer_set_loss(val_pred, val_batch.parton_x, val_batch.parton_x_batch)

        with log_path.open("a") as f:
            f.write(f"{epoch},{train_loss.item():.6f},{val_loss.item():.6f}\n")

        if epoch == 1 or epoch % LOG_EVERY == 0:
            print(f"  epoch {epoch:4d}  train {train_loss.item():.4f}  val {val_loss.item():.4f}")

        if val_loss.item() < best_val:
            best_val = val_loss.item()
            save_checkpoint(run_dir / "checkpoint_best.pt", epoch, model, optimizer,
                             train_loss.item(), val_loss.item())

    save_checkpoint(run_dir / "checkpoint_final.pt", EPOCHS, model, optimizer,
                     train_loss.item(), val_loss.item())

    config = dict(
        seed=SEED, val_fraction=VAL_FRACTION, epochs=EPOCHS, lr=LR,
        loss="chamfer", arch=ARCH,
        n_train=len(train_set), n_val=len(val_set),
        data=str(GENERATED),
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"\nbest val loss {best_val:.4f} (epoch of best checkpoint recorded inside it)")
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    main()
