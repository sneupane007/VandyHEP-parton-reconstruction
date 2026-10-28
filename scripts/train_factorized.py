"""Training run for FactorizedJetGNN: JSON -> LogPt -> train/val split -> GNN -> matched fraction loss.

Variant of train2.py where the model predicts each event's jet shape (pT fractions z_i
summing to 1, plus eta/phi per slot) and its total pT (as a correction to the hadron pT
sum) separately, trained with matched_fraction_loss. Everything else is identical.

Run from the repo root:  .venv/bin/python3 scripts/train_factorized.py [--epochs N] [--n-slots K] ...
  --model attention  uses AttentionJetGNN (cross-attention slot decoder) instead; its size
  is set by --hidden/--heads/--n-conv/--n-dec, and --absolute turns off jet-axis-relative
  coordinates.

Loss options (matched_fraction_loss): --angle-weight pt weights each true parton's eta/phi
error by its pT fraction; --mass-weight L adds L * (log m_pred - log m_true)^2. --hadron-json
can be a compact .npz from scripts/convert_to_arrays.py (then pass the same path, or omit
--parton-json). --patience N stops after N epochs without a new best val loss. A long run
killed by the wall-clock limit continues with --resume checkpoints/<run_dir>.
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

from parton_recon.models import build_model

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
    p.add_argument("--model", choices=["factorized", "attention"], default="factorized")
    p.add_argument("--hidden", type=int, default=None,
                    help="hidden width (default 64 for factorized, 128 for attention)")
    p.add_argument("--heads", type=int, default=4, help="attention only")
    p.add_argument("--n-conv", type=int, default=3, help="attention only: GATv2 layers")
    p.add_argument("--n-dec", type=int, default=3, help="attention only: decoder layers")
    p.add_argument("--dropout", type=float, default=0.1, help="attention only")
    p.add_argument("--absolute", action="store_true",
                    help="attention only: absolute eta/phi instead of offsets from the jet axis")
    p.add_argument("--angle-weight", choices=["uniform", "pt"], default="uniform")
    p.add_argument("--mass-weight", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=0, help="early-stop patience in epochs (0 = off)")
    p.add_argument("--workers", type=int, default=0, help="DataLoader worker processes")
    p.add_argument("--tag", default="", help="appended to the run directory name")
    p.add_argument("--resume", type=Path, default=None,
                    help="run directory to continue; all other arguments are taken from its run_args.json")
    args = p.parse_args()
    if args.resume is not None:
        saved = json.loads((args.resume / "run_args.json").read_text())
        args = argparse.Namespace(**{**saved, "resume": args.resume})
        for key in ("hadron_json", "parton_json"):
            args.__dict__[key] = Path(args.__dict__[key]) if args.__dict__[key] else None
    return args


def split_dataset(dataset, val_fraction, seed):
    """Index-level train/val split, shuffled once with a fixed seed."""
    n = len(dataset)
    n_val = max(1, round(n * val_fraction))
    indices = np.random.default_rng(seed).permutation(n)
    val_idx, train_idx = indices[:n_val], indices[n_val:]
    return Subset(dataset, train_idx.tolist()), Subset(dataset, val_idx.tolist())


def save_checkpoint(path, epoch, model, optimizer, train_loss, val_loss, **extra):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "arch_config": ARCH,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **extra,
        },
        path,
    )


def run_epoch(loader, model, optimizer, device, train, loss_kwargs):
    """One pass over `loader`. Returns (loss, shape, energy, mass), each averaged per event
    (not per batch), so they're comparable across runs with different batch sizes."""
    model.train(train)
    totals, total_events = np.zeros(4), 0
    for batch in loader:
        batch = batch.to(device)
        n_events = batch.num_graphs
        with torch.set_grad_enabled(train):
            terms = matched_fraction_loss(
                model.parts(batch), batch.parton_x, batch.parton_x_batch, **loss_kwargs)
        if train:
            optimizer.zero_grad()
            terms[0].backward()
            optimizer.step()
        totals += np.array([t.item() for t in terms]) * n_events
        total_events += n_events
    return tuple(totals / total_events)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    if str(args.hadron_json).endswith(".npz") and args.parton_json == GENERATED / "model1_parton_level_graphs_.json":
        args.parton_json = args.hadron_json       # .npz holds both levels; the JSON default does not apply
    dataset = JetPairDataset(args.hadron_json, args.parton_json, transform=LogPt())
    if args.max_events is not None:
        dataset = Subset(dataset, range(min(args.max_events, len(dataset))))
    train_set, val_set = split_dataset(dataset, args.val_fraction, args.seed)
    print(f"{len(dataset)} events -> {len(train_set)} train / {len(val_set)} val, "
          f"batch size {args.batch_size}")

    shuffle_gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, follow_batch=["parton_x"],
        shuffle=True, generator=shuffle_gen, num_workers=args.workers,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, follow_batch=["parton_x"], shuffle=False,
        num_workers=args.workers,
    )

    ARCH["n_slots"] = args.n_slots
    if args.model == "attention":
        ARCH.update(model="AttentionJetGNN", hidden=args.hidden or 128, heads=args.heads,
                    n_conv=args.n_conv, n_dec=args.n_dec, dropout=args.dropout,
                    relative=not args.absolute)
    elif args.hidden is not None:
        ARCH["hidden"] = args.hidden
    model = build_model(ARCH).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"training {sum(p.numel() for p in model.parameters()):,} parameters, {args.epochs} epochs")

    loss_kwargs = dict(energy_weight=args.energy_weight, angle_weight=args.angle_weight,
                       mass_weight=args.mass_weight)
    best_val, since_best, start = float("inf"), 0, 1
    if args.resume is not None:
        run_dir = args.resume
        last = torch.load(run_dir / "checkpoint_last.pt", map_location=device, weights_only=False)
        model.load_state_dict(last["model_state_dict"])
        optimizer.load_state_dict(last["optimizer_state_dict"])
        shuffle_gen.set_state(last["shuffle_rng"])
        torch.set_rng_state(last["torch_rng"])
        best_val, since_best, start = last["best_val"], last["since_best"], last["epoch"] + 1
        log_path = run_dir / "train_log.csv"
        print(f"resuming {run_dir} at epoch {start} (best val so far {best_val:.4f})")
    else:
        name = (f"gnn_attn{'abs' if args.absolute else 'rel'}_h{ARCH['hidden']}_k{args.n_slots}"
                if args.model == "attention" else f"gnn_factorized_k{args.n_slots}")
        run_dir = CHECKPOINTS / f"{name}{args.tag}_{datetime.now():%Y%m%d_%H%M%S}"
        run_dir.mkdir(parents=True)
        saved_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "resume"}
        (run_dir / "run_args.json").write_text(json.dumps(saved_args, indent=2))
        log_path = run_dir / "train_log.csv"
        log_path.write_text("epoch,train_loss,val_loss,train_shape,val_shape,train_energy,val_energy"
                            + (",train_mass,val_mass" if args.mass_weight > 0 else "") + "\n")

    epoch = start - 1
    for epoch in range(start, args.epochs + 1):
        train_loss, train_shape, train_energy, train_mass = run_epoch(
            train_loader, model, optimizer, device, True, loss_kwargs)
        val_loss, val_shape, val_energy, val_mass = run_epoch(
            val_loader, model, optimizer, device, False, loss_kwargs)

        with log_path.open("a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{train_shape:.6f},{val_shape:.6f},"
                    f"{train_energy:.3e},{val_energy:.3e}"
                    + (f",{train_mass:.6f},{val_mass:.6f}" if args.mass_weight > 0 else "") + "\n")

        if epoch == 1 or epoch % args.log_every == 0:
            print(f"  epoch {epoch:4d}  train {train_loss:.4f}  val {val_loss:.4f}  "
                  f"(val shape {val_shape:.4f}, val energy {val_energy:.2e}"
                  + (f", val mass {val_mass:.4f}" if args.mass_weight > 0 else "") + ")", flush=True)

        if val_loss < best_val:
            best_val, since_best = val_loss, 0
            save_checkpoint(run_dir / "checkpoint_best.pt", epoch, model, optimizer,
                             train_loss, val_loss)
        else:
            since_best += 1
        save_checkpoint(run_dir / "checkpoint_last.pt", epoch, model, optimizer, train_loss, val_loss,
                        best_val=best_val, since_best=since_best,
                        shuffle_rng=shuffle_gen.get_state(), torch_rng=torch.get_rng_state())
        if args.patience and since_best >= args.patience:
            print(f"  no val improvement for {args.patience} epochs, stopping at epoch {epoch}")
            break

    save_checkpoint(run_dir / "checkpoint_final.pt", epoch, model, optimizer,
                     train_loss, val_loss)

    config = dict(
        seed=args.seed, val_fraction=args.val_fraction, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, loss="matched_fraction", energy_weight=args.energy_weight,
        angle_weight=args.angle_weight, mass_weight=args.mass_weight, patience=args.patience,
        epochs_run=epoch, arch=ARCH,
        n_train=len(train_set), n_val=len(val_set),
        hadron_json=str(args.hadron_json), parton_json=str(args.parton_json),
        max_events=args.max_events,
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"\nbest val loss {best_val:.4f} (epoch of best checkpoint recorded inside it)")
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    main()
