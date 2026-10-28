"""Jet mass of predicted vs true parton jets (and the input hadron jet), on the val split.

Usage: .venv/bin/python repro/jet_mass_plot.py checkpoints/<run>/checkpoint_best.pt [data prefix | file.npz] [--out PNG]
       add --all-events to use every event of the data (e.g. data/veto_test.npz) instead of the val split

The data prefix defaults to the one in the run's config.json. Masses use
parton_recon.metrics.jet_mass: constituents summed as massless 4-vectors (the JSON
carries no masses, so the hadron jet is massless-summed too). Left: the three
distributions overlaid, with a pred/true ratio panel. Right: per-event predicted vs
true mass. Saves jet_mass.png next to the checkpoint.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from evaluate import split_dataset, wrap_phi
from parton_recon.data import JetPairDataset, LogPt
from parton_recon.metrics import jet_mass
from parton_recon.models import build_model, predict_batched

p = argparse.ArgumentParser()
p.add_argument("checkpoint", type=Path)
p.add_argument("prefix", nargs="?", default=None)
p.add_argument("--out", type=Path, default=None)
p.add_argument("--all-events", action="store_true", help="use every event instead of the run's val split")
p.add_argument("--label", default=None, help="title prefix")
args = p.parse_args()

cfg_path = args.checkpoint.parent / "config.json"
cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
prefix = args.prefix or cfg["hadron_json"].replace("_hadron_level_graphs_.json", "")
paths = (prefix, prefix) if prefix.endswith(".npz") else (prefix + "_hadron_level_graphs_.json",
                                                         prefix + "_parton_level_graphs_.json")
val_fraction, seed = cfg.get("val_fraction", 0.2), cfg.get("seed", 0)

ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
model = build_model(ck["arch_config"]); model.load_state_dict(ck["model_state_dict"]); model.eval()
ds = JetPairDataset(*paths, transform=LogPt())
events = ds if args.all_events else split_dataset(ds, val_fraction, seed)
b = next(iter(DataLoader(events, batch_size=10**6, follow_batch=["parton_x"])))
with torch.no_grad():
    pred = LogPt.inverse(predict_batched(model, b))
pred[..., 2] = wrap_phi(pred[..., 2])
true, hadron = LogPt.inverse(b.parton_x).numpy(), LogPt.inverse(b.x).numpy()
tb, hb = b.parton_x_batch.numpy(), b.batch.numpy()
n = b.num_graphs

m_true = np.array([jet_mass(true[tb == i]) for i in range(n)])
m_pred = np.array([jet_mass(pred[i].numpy()) for i in range(n)])
m_had = np.array([jet_mass(hadron[hb == i]) for i in range(n)])

rel = (m_pred - m_true) / m_true
kind = "test" if args.all_events else "val"
print(f"{n} {kind} events, checkpoint epoch {ck['epoch']}, data {prefix}")
for name, m in [("true parton", m_true), ("predicted parton", m_pred), ("hadron (input)", m_had)]:
    print(f"  {name:17s} mass: median {np.median(m):6.1f} GeV, mean {m.mean():6.1f}, 16-84% [{np.percentile(m, 16):.1f}, {np.percentile(m, 84):.1f}]")
print(f"  pred vs true: corr {np.corrcoef(m_pred, m_true)[0, 1]:.2f}, median (pred-true)/true {np.median(rel):+.3f}, "
      f"16-84% [{np.percentile(rel, 16):+.3f}, {np.percentile(rel, 84):+.3f}]")
print(f"  hadron vs true parton: corr {np.corrcoef(m_had, m_true)[0, 1]:.2f}")

hi = np.percentile(np.concatenate([m_true, m_pred, m_had]), 99.5)
bins = np.linspace(0, hi, 41)
fig = plt.figure(figsize=(12, 5.5))
gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[1.25, 1], hspace=0.05, wspace=0.25)
ax, axr, axs = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])

h_true, _, _ = ax.hist(m_true, bins, histtype="step", lw=2, color="#55A868", label="true parton")
h_pred, _, _ = ax.hist(m_pred, bins, histtype="step", lw=2, color="#C44E52", label="predicted parton")
ax.hist(m_had, bins, histtype="step", lw=1.2, ls="--", color="gray", label="hadron (input)")
ax.set_ylabel(f"{kind} events"); ax.legend(); ax.tick_params(labelbottom=False)
ax.set_title((args.label + ": " if args.label else "") + f"jet mass, {n} {kind} events, epoch {ck['epoch']}")

centers = 0.5 * (bins[1:] + bins[:-1])
ok = h_true > 0
axr.axhline(1, color="k", lw=0.8)
axr.errorbar(centers[ok], h_pred[ok] / h_true[ok], yerr=np.sqrt(h_pred[ok]) / h_true[ok], fmt="o", ms=3, color="#C44E52")
axr.set(xlabel="jet mass [GeV]", ylabel="pred / true", ylim=(0, 2))

axs.scatter(m_true, m_pred, s=5, alpha=0.4, color="#C44E52")
axs.plot([0, hi], [0, hi], "k--", lw=0.8)
axs.set(xlim=(0, hi), ylim=(0, hi), xlabel="true parton jet mass [GeV]", ylabel="predicted parton jet mass [GeV]",
        title=f"per event: corr {np.corrcoef(m_pred, m_true)[0, 1]:.2f}")
out = args.out or args.checkpoint.parent / "jet_mass.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
