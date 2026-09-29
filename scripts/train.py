"""Real training run: JSON -> transform -> train/val split -> GNN -> Chamfer -> checkpoint.

Run from the repo root:  .venv/bin/python3 scripts/train.py [--epochs N] [--batch-size N] ...
Run `--help` for the full list of flags (data paths, epochs, batch size, lr, val split, seed).

Unlike experiment.py (a plumbing smoke test with no held-out data),

this holds out a validation split and saves a checkpoint per checkpoints/README.md's convention: weights,
optimizer state, epoch, and a copy of the run's config, in their own timestamped
directory, plus a training log.

Only chamfer_set_loss exists as a loss so far.

Trains in mini-batches (default batch size 32) so this scales past whatever fits in one GPU
batch — see notebooks/01_cluster_gpu_readiness.md task T7. To reproduce the old full-batch
behavior on a small dataset, pass --batch-size >= the number of training events.
"""

import argparse
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

ARCH = dict(in_channels=3, hidden=64, n_slots=24, out_channels=3)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hadron-json", type=Path, default=GENERATED / "model1_hadron_level_graphs_.json")
    p.add_argument("--parton-json", type=Path, default=GENERATED / "model1_parton_level_graphs_.json")
    p.add_argument("--max-events", type=int, default=None,
                    help="cap the number of events loaded, for a quick local run")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args()


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


def run_epoch(loader, model, optimizer, device, train):
    """One pass over `loader`. Returns the loss averaged per event (not per batch), so it's
    comparable across runs with different batch sizes."""
    model.train(train)
    total_loss, total_events = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        n_events = batch.num_graphs
        with torch.set_grad_enabled(train):
            pred = model(batch)
            loss = chamfer_set_loss(pred, batch.parton_x, batch.parton_x_batch)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * n_events
        total_events += n_events
    return total_loss / total_events


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    dataset = JetPairDataset(args.hadron_json, args.parton_json, transform=LogPt())
    if args.max_events is not None:
        dataset = Subset(dataset, range(min(args.max_events, len(dataset))))
    train_set, val_set = split_dataset(dataset, args.val_fraction, args.seed)
    print(f"{len(dataset)} events -> {len(train_set)} train / {len(val_set)} val, "
          f"batch size {args.batch_size}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, follow_batch=["parton_x"],
        shuffle=True, generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, follow_batch=["parton_x"], shuffle=False,
    )

    model = JetGNN(**ARCH).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"training {sum(p.numel() for p in model.parameters()):,} parameters, {args.epochs} epochs")

    run_dir = CHECKPOINTS / f"gnn_chamfer_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True)
    log_path = run_dir / "train_log.csv"
    log_path.write_text("epoch,train_loss,val_loss\n")

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(train_loader, model, optimizer, device, train=True)
        val_loss = run_epoch(val_loader, model, optimizer, device, train=False)

        with log_path.open("a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f}\n")

        if epoch == 1 or epoch % args.log_every == 0:
            print(f"  epoch {epoch:4d}  train {train_loss:.4f}  val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(run_dir / "checkpoint_best.pt", epoch, model, optimizer,
                             train_loss, val_loss)

    save_checkpoint(run_dir / "checkpoint_final.pt", args.epochs, model, optimizer,
                     train_loss, val_loss)

    config = dict(
        seed=args.seed, val_fraction=args.val_fraction, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, loss="chamfer", arch=ARCH,
        n_train=len(train_set), n_val=len(val_set),
        hadron_json=str(args.hadron_json), parton_json=str(args.parton_json),
        max_events=args.max_events,
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"\nbest val loss {best_val:.4f} (epoch of best checkpoint recorded inside it)")
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    main()
