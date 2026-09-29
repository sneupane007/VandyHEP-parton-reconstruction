"""Evaluate a trained checkpoint's EMD against the true partons, and plot the distribution.

Run from the repo root:
  .venv/bin/python3 scripts/evaluate.py checkpoints/<run_dir>/checkpoint_best.pt

Rebuilds the same val split train.py used (same seed/val_fraction/data by default — override
with the matching flags if the checkpoint was trained with non-default ones), runs the model
on it, undoes LogPt and wraps phi, then computes per-event EMD between predicted and true
partons. Reports ln(EMD) against the March 2025 slide thresholds (good <= 4, fair 4-5.5,
bad >= 5.5; see notebooks/03_emd_metric.md) and saves a histogram next to the checkpoint.

Known caveats, not fixed here (see notebooks/03_emd_metric.md and 04_bug_audit.md):
- All 24 predicted slots are used uncritically; there is no slot-selection rule yet, so
  numbers are not directly comparable to a model with an "exists" head.
- The underlying data still has the photon-contamination and hadron/parton jet-mismatch
  issues from brief 04 — a perfect model still pays for those.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from parton_recon.data import JetPairDataset, LogPt
from parton_recon.metrics.emd import dataset_emd
from parton_recon.models import JetGNN

GENERATED = REPO_ROOT / "parton_recon" / "data" / "generate"

# Good/fair/bad thresholds on ln(EMD), from the March 2025 slides (slide 23) and
# Summary/rotations.ipynb — see notebooks/03_emd_metric.md.
GOOD_MAX = 4.0
FAIR_MAX = 5.5


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path, nargs="?",
                    default=REPO_ROOT / "checkpoints" / "gnn_chamfer_20260928_235508" / "checkpoint_best.pt",
                    help="path to a checkpoint_*.pt file (default: the in-progress run)")
    p.add_argument("--hadron-json", type=Path, default=GENERATED / "model1_hadron_level_graphs_.json")
    p.add_argument("--parton-json", type=Path, default=GENERATED / "model1_parton_level_graphs_.json")
    p.add_argument("--val-fraction", type=float, default=0.2,
                    help="must match the value train.py was run with")
    p.add_argument("--seed", type=int, default=0, help="must match the value train.py was run with")
    
    p.add_argument("--out", type=Path, default=None,
                    help="where to save the histogram (default: alongside the checkpoint)")
    return p.parse_args()


def split_dataset(dataset, val_fraction, seed):
    """Index-level train/val split, shuffled once with a fixed seed. Mirrors train.py exactly
    so the same events land in val here as they did during training."""
    n = len(dataset)
    n_val = max(1, round(n * val_fraction))
    indices = np.random.default_rng(seed).permutation(n)
    val_idx = indices[:n_val]
    return torch.utils.data.Subset(dataset, val_idx.tolist())


def wrap_phi(phi):
    """Wrap into (-pi, pi]. The model has no sin/cos encoding, so its phi output is
    unbounded; the true phi is already wrapped by the generator but wrapping again is a
    no-op for it."""
    return (phi + torch.pi) % (2 * torch.pi) - torch.pi


def to_physical_events(pred_logpt, true_logpt, true_batch, n_events):
    """[B, n_slots, 3] LogPt predictions + ragged LogPt truth -> lists of physical
    (N_i, 3) numpy [pT, eta, phi] arrays, one pair per event."""
    pred_phys = LogPt.inverse(pred_logpt)
    true_phys = LogPt.inverse(true_logpt)
    pred_phys[..., 2] = wrap_phi(pred_phys[..., 2])
    true_phys[..., 2] = wrap_phi(true_phys[..., 2])

    pred_events = [pred_phys[i].detach().cpu().numpy() for i in range(n_events)]
    true_events = [
        true_phys[true_batch == i].detach().cpu().numpy() for i in range(n_events)
    ]
    return pred_events, true_events


def main():
    args = parse_args()
    out_path = args.out or args.checkpoint.parent / "emd_distribution.png"

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = JetGNN(**ckpt["arch_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"loaded {args.checkpoint} (epoch {ckpt['epoch']}, val chamfer {ckpt['val_loss']:.4f})")

    dataset = JetPairDataset(args.hadron_json, args.parton_json, transform=LogPt())
    val_set = split_dataset(dataset, args.val_fraction, args.seed)
    val_loader = DataLoader(val_set, batch_size=len(val_set), follow_batch=["parton_x"], shuffle=False)
    batch = next(iter(val_loader)).to(device)

    with torch.no_grad():
        pred = model(batch)

    pred_events, true_events = to_physical_events(
        pred, batch.parton_x, batch.parton_x_batch, batch.num_graphs
    )

    print(f"computing EMD for {len(true_events)} val events (periodic_phi=True)...")
    raw_emd = dataset_emd(true_events, pred_events, periodic_phi=True, verbose=False)
    ln_emd = np.log(raw_emd)

    n = len(ln_emd)
    good = int((ln_emd <= GOOD_MAX).sum())
    fair = int(((ln_emd > GOOD_MAX) & (ln_emd <= FAIR_MAX)).sum())
    bad = int((ln_emd > FAIR_MAX).sum())
    print(f"mean ln(EMD) {ln_emd.mean():.2f}, median {np.median(ln_emd):.2f}")
    print("------------------------------------------------------------------")
    print(f"good (<={GOOD_MAX:.1f}): {good}/{n} ({100*good/n:.0f}%)  "
          f"fair ({GOOD_MAX:.1f}-{FAIR_MAX:.1f}): {fair}/{n} ({100*fair/n:.0f}%)  "
          f"bad (>{FAIR_MAX:.1f}): {bad}/{n} ({100*bad/n:.0f}%)")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(ln_emd, bins=min(15, max(5, n // 2)), color="#4C72B0", edgecolor="white")
    ax.axvline(GOOD_MAX, color="seagreen", linestyle="--", label=f"good <= {GOOD_MAX:.1f}")
    ax.axvline(FAIR_MAX, color="darkorange", linestyle="--", label=f"fair <= {FAIR_MAX:.1f}")
    ax.set_xlabel("ln(EMD)")
    ax.set_ylabel("events")
    ax.set_title(f"EMD: predicted vs. true partons\n{n} val events, checkpoint epoch {ckpt['epoch']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved plot to {out_path}")


if __name__ == "__main__":
    main()
