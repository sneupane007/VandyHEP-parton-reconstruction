"""Overlay ln(EMD) and shape-only EMD for several LogPt checkpoints on one figure.

Usage: .venv/bin/python repro/compare_runs.py out.png label=checkpoints/<run> [label=...]
Each run's data prefix is read from its config.json; the seed-0 val split is rebuilt.
"""
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
from parton_recon.metrics.emd import event_emd
from parton_recon.models import build_model, predict_batched


def run_emds(run):
    cfg = json.loads((run / "config.json").read_text())
    ck = torch.load(run / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    m = build_model(ck["arch_config"]); m.load_state_dict(ck["model_state_dict"]); m.eval()
    ds = JetPairDataset(cfg["hadron_json"], cfg["parton_json"], transform=LogPt())
    b = next(iter(DataLoader(split_dataset(ds, cfg["val_fraction"], cfg["seed"]),
                             batch_size=10**6, follow_batch=["parton_x"])))
    with torch.no_grad():
        pred = LogPt.inverse(predict_batched(m, b))
    pred[..., 2] = wrap_phi(pred[..., 2])
    true, tb = LogPt.inverse(b.parton_x), b.parton_x_batch
    pairs = [(true[tb == i].numpy(), pred[i].numpy()) for i in range(b.num_graphs)]
    full = np.array([event_emd(t, p, periodic_phi=True) for t, p in pairs])
    shape = np.array([event_emd(t, p, periodic_phi=True, norm=True) for t, p in pairs])
    return np.log(full), shape


out = Path(sys.argv[1])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
for arg in sys.argv[2:]:
    label, run = arg.split("=", 1)
    ln_emd, shape = run_emds(Path(run))
    print(f"{label:28s} median ln(EMD) {np.median(ln_emd):.2f}   median shape EMD {np.median(shape):.3f}")
    a1.hist(ln_emd, bins=np.linspace(3, 10.5, 61), histtype="step", lw=1.6, label=f"{label} (median {np.median(ln_emd):.2f})")
    a2.hist(shape, bins=np.linspace(0, 0.4, 61), histtype="step", lw=1.6, label=f"{label} (median {np.median(shape):.3f})")
for x, c in [(4.0, "seagreen"), (5.5, "darkorange")]:
    a1.axvline(x, color=c, ls="--", lw=1)
a1.set(xlabel="ln(EMD)  [full EMD, includes |ΣpT| mismatch]", ylabel="val events", title="Full EMD")
a2.set(xlabel="EMD with norm=True  [shape only, energy-normalised]", title="Shape-only EMD")
a1.legend(fontsize=8); a2.legend(fontsize=8)
fig.suptitle("1345 val events per run, best checkpoint")
fig.tight_layout(); fig.savefig(out, dpi=150)
print(f"saved {out}")
