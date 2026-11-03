"""Number of partons per event: truth vs model prediction.

Truth counts every parton in the event. The model fills all n_slots slots every time, so a
predicted parton is a slot with pT > PT_CUT. Predicted slot pTs fall off smoothly down to
~0.01 GeV with no gap, so the predicted count depends on the cut; ~1% of true partons sit
below 0.1 GeV. The prediction can never exceed n_slots (24); truth goes up to 37.

Left: predicted vs true count per event (marker area ~ number of events at that point).
Right: overlaid count histograms.

Run from the repo root:
  .venv/bin/python3 scripts/nparton_compare_model.py [checkpoints/<run_dir>/checkpoint_best.pt]

Same val split as train.py/evaluate.py (match --val-fraction and --seed to the training run).
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator
from torch_geometric.loader import DataLoader

from evaluate import split_dataset
from parton_recon.data import JetPairDataset, LogPt
from parton_recon.models import JetGNN

GENERATED = REPO_ROOT / "parton_recon" / "data" / "generate"

# must match the checkpoint's config.json "transform"; comment/uncomment one
TRANSFORM = LogPt()
# TRANSFORM = None

PT_CUT = 0.1  # GeV; a predicted slot counts as a parton above this

TRUE_COLOR, PRED_COLOR = "#55A868", "#C44E52"  # same as jetmass_compare_model.py


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path, nargs="?",
                   default=REPO_ROOT / "checkpoints" / "gnn_mse_20261026_221312" / "checkpoint_best.pt")
    p.add_argument("--hadron-json", type=Path, default=GENERATED / "model1_hadron_level_graphs_.json")
    p.add_argument("--parton-json", type=Path, default=GENERATED / "model1_parton_level_graphs_.json")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None,
                   help="where to save the plot (default: nparton_compare.png next to the checkpoint)")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = args.out or args.checkpoint.parent / "nparton_compare.png"

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = JetGNN(**ckpt["arch_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    n_slots = ckpt["arch_config"]["n_slots"]

    dataset = JetPairDataset(args.hadron_json, args.parton_json, transform=TRANSFORM)
    val_set = split_dataset(dataset, args.val_fraction, args.seed)
    idx = val_set.indices

    batch = next(iter(DataLoader(val_set, batch_size=len(val_set), shuffle=False))).to(device)
    with torch.no_grad():
        pred = model(batch)
    if TRANSFORM is not None:
        pred = TRANSFORM.inverse(pred)
    pt = pred[..., 0].clamp(min=0)  # [B, n_slots] GeV; clamp only matters for raw-GeV models

    n_true = np.array([len(dataset.parton[i]["node_features"]) for i in idx])
    n_pred = (pt > PT_CUT).sum(dim=1).cpu().numpy()

    diff = n_pred - n_true
    print(f"{len(idx)} val events, checkpoint epoch {ckpt['epoch']}, pred cut pT > {PT_CUT} GeV")
    print(f"  true partons/event: mean {n_true.mean():.1f}, range {n_true.min()}-{n_true.max()}")
    print(f"  pred partons/event: mean {n_pred.mean():.1f}, range {n_pred.min()}-{n_pred.max()}")
    print(f"  pred - true: mean {diff.mean():+.1f}, median {np.median(diff):+.0f}, "
          f"exact {int((diff == 0).sum())}/{len(diff)}")

    fig, (ax_sc, ax_h) = plt.subplots(1, 2, figsize=(10, 4.5))
    hi = max(n_true.max(), n_pred.max(), n_slots) + 1

    pairs, counts = np.unique(np.stack([n_true, n_pred], axis=1), axis=0, return_counts=True)
    ax_sc.plot([0, hi], [0, hi], color="0.6", linestyle="--", linewidth=1, label="pred = true")
    ax_sc.axhline(n_slots, color="0.6", linestyle=":", linewidth=1, label=f"{n_slots} slots (max)")
    ax_sc.scatter(pairs[:, 0], pairs[:, 1], s=40 * counts, color=PRED_COLOR, edgecolor="white",
                  linewidth=1, zorder=3)
    ax_sc.set_xlim(0, hi)
    ax_sc.set_ylim(0, hi)
    ax_sc.set_aspect("equal")
    ax_sc.set_xlabel("true partons per event")
    ax_sc.set_ylabel(f"predicted partons (pT > {PT_CUT} GeV)")
    ax_sc.legend(loc="upper left", frameon=False)

    bins = np.arange(0, hi + 1) - 0.5
    ax_h.hist(n_true, bins=bins, histtype="step", linewidth=2, color=TRUE_COLOR, label="true")
    ax_h.hist(n_pred, bins=bins, histtype="step", linewidth=2, linestyle="--", color=PRED_COLOR,
              label=f"predicted (pT > {PT_CUT} GeV)")
    ax_h.set_xlabel("partons per event")
    ax_h.set_ylabel("events")
    ax_h.legend(loc="upper right", frameon=False)

    for ax in (ax_sc, ax_h):
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(f"Parton multiplicity: {len(idx)} val events, checkpoint epoch {ckpt['epoch']}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nsaved plot to {out_path}")


if __name__ == "__main__":
    main()
