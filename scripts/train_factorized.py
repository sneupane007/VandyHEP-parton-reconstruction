"""Training run for FactorizedJetGNN: JSON -> LogPt -> train/val split -> GNN -> matched fraction loss.

Variant of train2.py where the model predicts each event's jet shape (pT fractions z_i
summing to 1, plus eta/phi per slot) and its total pT (as a correction to the hadron pT
sum) separately, trained with matched_fraction_loss. Everything else is identical.

Run from the repo root:  .venv/bin/python3 scripts/train_factorized.py [--epochs N] [--n-slots K] ...
Run `--help` for the full list of flags (data paths, epochs, batch size, lr, val split, seed).

Unlike experiment.py (a plumbing smoke test with no held-out data),

this holds out a validation split and saves a checkpoint per checkpoints/README.md's convention: weights,
optimizer state, epoch, and a copy of the run's config, in their own timestamped
directory, plus a training log.

Unmatched slots are pushed to pT fraction z = 0, so no padding pT is left behind. The
training log has the shape and energy parts of the loss as separate columns.

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

from parton_recon.data import JetPairDataset
from parton_recon.data import LogPt
from parton_recon.losses import matched_fraction_loss

from parton_recon.models import FactorizedJetGNN

GENERATED = REPO_ROOT / "parton_recon" / "data" / "generate"
CHECKPOINTS = REPO_ROOT / "checkpoints"

ARCH = dict(model="FactorizedJetGNN", in_channels=3, hidden=64, n_slots=24, out_channels=3)


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
    p.add_argument("--n-slots", type=int, default=24)
    p.add_argument("--energy-weight", type=float, default=1.0,
                    help="weight of the total-pT term relative to the shape term")
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


def run_epoch(loader, model, optimizer, device, train, energy_weight):
    """One pass over `loader`. Returns (loss, shape, energy), each averaged per event (not
    per batch), so they're comparable across runs with different batch sizes."""
    model.train(train)
    total_loss, total_shape, total_energy, total_events = 0.0, 0.0, 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        n_events = batch.num_graphs
        with torch.set_grad_enabled(train):
            loss, shape, energy = matched_fraction_loss(
                model.parts(batch), batch.parton_x, batch.parton_x_batch, energy_weight)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * n_events
        total_shape += shape.item() * n_events
        total_energy += energy.item() * n_events
        total_events += n_events
    return total_loss / total_events, total_shape / total_events, total_energy / total_events


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

    ARCH["n_slots"] = args.n_slots
    model = FactorizedJetGNN(**{k: v for k, v in ARCH.items() if k != "model"}).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"training {sum(p.numel() for p in model.parameters()):,} parameters, {args.epochs} epochs")

    run_dir = CHECKPOINTS / f"gnn_factorized_k{args.n_slots}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True)
    log_path = run_dir / "train_log.csv"
    log_path.write_text("epoch,train_loss,val_loss,train_shape,val_shape,train_energy,val_energy\n")

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_shape, train_energy = run_epoch(
            train_loader, model, optimizer, device, train=True, energy_weight=args.energy_weight)
        val_loss, val_shape, val_energy = run_epoch(
            val_loader, model, optimizer, device, train=False, energy_weight=args.energy_weight)

        with log_path.open("a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{train_shape:.6f},{val_shape:.6f},"
                    f"{train_energy:.3e},{val_energy:.3e}\n")

        if epoch == 1 or epoch % args.log_every == 0:
            print(f"  epoch {epoch:4d}  train {train_loss:.4f}  val {val_loss:.4f}  "
                  f"(val shape {val_shape:.4f}, val energy {val_energy:.2e})")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(run_dir / "checkpoint_best.pt", epoch, model, optimizer,
                             train_loss, val_loss)

    save_checkpoint(run_dir / "checkpoint_final.pt", args.epochs, model, optimizer,
                     train_loss, val_loss)

    config = dict(
        seed=args.seed, val_fraction=args.val_fraction, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, loss="matched_fraction", energy_weight=args.energy_weight, arch=ARCH,
        n_train=len(train_set), n_val=len(val_set),
        hadron_json=str(args.hadron_json), parton_json=str(args.parton_json),
        max_events=args.max_events,
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"\nbest val loss {best_val:.4f} (epoch of best checkpoint recorded inside it)")
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    main()
