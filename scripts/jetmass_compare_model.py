"""Jet mass of true-parton vs model-predicted-parton jets, drawn like the diagonal of
pca_compare_model.py's corner plot (one Gaussian per level), so the two plots can be read
side by side.

Jet mass sums the [pT, eta, phi] constituents as massless 4-vectors
(parton_recon.metrics.jet_mass). Predicted jets use all of the model's output slots, same
as the PCA features; surplus slots sit near pT = 1 GeV and add a little mass.

Run from the repo root:
  .venv/bin/python3 scripts/jetmass_compare_model.py [checkpoints/<run_dir>/checkpoint_best.pt]

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
from torch_geometric.loader import DataLoader

from evaluate import split_dataset, wrap_phi
from parton_recon.data import JetPairDataset, LogPt
from parton_recon.metrics.jetmass import jet_mass
from parton_recon.models import JetGNN

GENERATED = REPO_ROOT / "parton_recon" / "data" / "generate"

# comment/uncomment a line to toggle that level in the plot
LEVELS = [
    # ("hadron", "#4C72B0"),
    ("true parton", "#55A868"),
    ("predicted parton", "#C44E52"),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path, nargs="?",
                   default=REPO_ROOT / "checkpoints" / "gnn_chamfer_20260928_235508" / "checkpoint_best.pt")
    p.add_argument("--hadron-json", type=Path, default=GENERATED / "model1_hadron_level_graphs_.json")
    p.add_argument("--parton-json", type=Path, default=GENERATED / "model1_parton_level_graphs_.json")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None,
                   help="where to save the plot (default: jetmass_compare.png next to the checkpoint)")
    return p.parse_args()


def jet_masses(events):
    return np.array([jet_mass(np.asarray(e, dtype=float)) for e in events])


def main():
    args = parse_args()
    out_path = args.out or args.checkpoint.parent / "jetmass_compare.png"

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = JetGNN(**ckpt["arch_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dataset = JetPairDataset(args.hadron_json, args.parton_json, transform=None)
    val_set = split_dataset(dataset, args.val_fraction, args.seed)
    idx = val_set.indices

    batch = next(iter(DataLoader(val_set, batch_size=len(val_set), shuffle=False))).to(device)
    with torch.no_grad():
        pred = LogPt.inverse(model(batch))
    pred[..., 2] = wrap_phi(pred[..., 2])

    # one jet mass per val event, in val-split order; raw physical hadron/parton straight from the JSON
    masses = {
        "hadron": jet_masses(dataset.hadron[i]["node_features"] for i in idx),
        "true parton": jet_masses(dataset.parton[i]["node_features"] for i in idx),
        "predicted parton": jet_masses(pred.cpu().numpy()),
    }

    print(f"{len(idx)} val events, checkpoint epoch {ckpt['epoch']}")
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, color in LEVELS:
        m = masses[name]
        mu, sigma = m.mean(), m.std()
        print(f"  {name:>16}: mean {mu:6.1f} GeV, std {sigma:6.1f} GeV")
        # ax.hist(m, bins=10, density=True, histtype="step", color=color)
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
        pdf = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        ax.plot(x, pdf, color=color, label=name)
    ax.set_xlabel("jet mass [GeV]")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nsaved plot to {out_path}")


if __name__ == "__main__":
    main()
