"""Look at what a trained checkpoint actually generates, in physical units (GeV, eta, phi).

Run from the repo root:
  .venv/bin/python3 scripts/inspect_predictions.py checkpoints/<run_dir>/checkpoint_best.pt [--events 0 1 2]

Prints aggregate stats (real vs padded slots, summed pT) and, for each chosen val event, a
table pairing every true parton with its nearest-cost predicted slot. Saves an eta-phi
overlay plot next to the checkpoint. Assumes the checkpoint was trained with LogPt.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch_geometric.loader import DataLoader

from parton_recon.data import JetPairDataset, LogPt
from parton_recon.models import build_model

GENERATED = REPO_ROOT / "parton_recon" / "data" / "generate"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--hadron-json", type=Path, default=GENERATED / "model1_hadron_level_graphs_.json")
    p.add_argument("--parton-json", type=Path, default=GENERATED / "model1_parton_level_graphs_.json")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--events", type=int, nargs="+", default=[0, 1, 2, 3],
                    help="val-set event indices to print and plot")
    return p.parse_args()


def wrap_phi(phi):
    return (phi + np.pi) % (2 * np.pi) - np.pi


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ckpt["arch_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dataset = JetPairDataset(args.hadron_json, args.parton_json, transform=LogPt())
    n_val = max(1, round(len(dataset) * args.val_fraction))
    val_idx = np.random.default_rng(args.seed).permutation(len(dataset))[:n_val]
    val_set = torch.utils.data.Subset(dataset, val_idx.tolist())
    batch = next(iter(DataLoader(val_set, batch_size=len(val_set), follow_batch=["parton_x"])))

    with torch.no_grad():
        pred = LogPt.inverse(model(batch)).numpy()                          # [B, K, 3] physical
    true = LogPt.inverse(batch.parton_x).numpy()
    tb = batch.parton_x_batch.numpy()
    B, K, _ = pred.shape

    true_events = [true[tb == i][np.argsort(-true[tb == i][:, 0])] for i in range(B)]
    n_true = np.array([len(t) for t in true_events])

    # Match with the same cost as matched_mse_loss, but in physical units for reporting.
    real_pt, pad_pt, matched_pairs = [], [], []
    for i in range(B):
        t = true_events[i][:K]
        lp, lt = np.log(pred[i][:, 0]), np.log(t[:, 0])
        d = np.stack([lp[:, None] - lt[None], pred[i][:, 1, None] - t[None, :, 1],
                      wrap_phi(pred[i][:, 2, None] - t[None, :, 2])])
        s, j = linear_sum_assignment((d ** 2).sum(0))
        matched_pairs.append((s, j))
        mask = np.zeros(K, bool)
        mask[s] = True
        real_pt.extend(pred[i][mask, 0])
        pad_pt.extend(pred[i][~mask, 0])

    print(f"{B} val events | true partons/event: median {np.median(n_true):.0f}, max {n_true.max()}")
    print(f"true pT:  median {np.median(true[:, 0]):.1f} GeV, max {true[:, 0].max():.0f} GeV")
    print(f"true sum pT/event: median {np.median([t[:, 0].sum() for t in true_events]):.0f} GeV")
    print(f"pred sum pT/event: median {np.median(pred[:, :, 0].sum(1)):.0f} GeV")
    print(f"pred pT on matched slots: median {np.median(real_pt):.1f} GeV | "
          f"on unmatched slots: median {np.median(pad_pt):.2f} GeV (padding target 1.0)")
    print(f"unmatched slots above 5 GeV: {100 * np.mean(np.array(pad_pt) > 5):.0f}%")

    fig, axes = plt.subplots(1, len(args.events), figsize=(4.2 * len(args.events), 4), squeeze=False)
    for ax, i in zip(axes[0], args.events):
        t, (s, j) = true_events[i][:K], matched_pairs[i]
        print(f"\nevent {i}: {len(true_events[i])} true partons")
        print(f"  {'true pT':>8} {'eta':>6} {'phi':>6} | {'pred pT':>8} {'eta':>6} {'phi':>6}")
        for si, ti in sorted(zip(s, j), key=lambda p: p[1]):
            print(f"  {t[ti, 0]:8.1f} {t[ti, 1]:6.2f} {t[ti, 2]:6.2f} | "
                  f"{pred[i][si, 0]:8.1f} {pred[i][si, 1]:6.2f} {pred[i][si, 2]:6.2f}")
        ax.scatter(t[:, 1], t[:, 2], s=t[:, 0] * 2 + 10, c="k", alpha=0.6, label="true")
        ax.scatter(pred[i][:, 1], wrap_phi(pred[i][:, 2]), s=pred[i][:, 0] * 2 + 10,
                   c="tab:orange", alpha=0.6, label="pred")
        ax.set(title=f"event {i}", xlabel="eta", ylabel="phi")
    axes[0][0].legend(title="marker area ~ pT")
    fig.tight_layout()
    out = args.checkpoint.parent / "inspect_predictions.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
