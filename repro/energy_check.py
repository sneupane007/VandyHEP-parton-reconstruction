"""Split a LogPt checkpoint's val-set EMD into its energy-mismatch and shape parts.

Usage: .venv/bin/python repro/energy_check.py checkpoints/<run>/checkpoint_best.pt [data prefix]

energyflow's EMD (norm=False) includes |sum pT_pred - sum pT_true|, so a model that
over-counts energy scores badly even with the right shape. norm=True gives the shape-only EMD.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from evaluate import split_dataset, wrap_phi
from parton_recon.data import JetPairDataset, LogPt
from parton_recon.metrics.emd import event_emd
from parton_recon.models import build_model, predict_batched

ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
pre = sys.argv[2] if len(sys.argv) > 2 else "data/repro6725"
m = build_model(ck["arch_config"]); m.load_state_dict(ck["model_state_dict"]); m.eval()
ds = JetPairDataset(pre + "_hadron_level_graphs_.json", pre + "_parton_level_graphs_.json", transform=LogPt())
b = next(iter(DataLoader(split_dataset(ds, 0.2, 0), batch_size=10**6, follow_batch=["parton_x"])))
with torch.no_grad():
    pred = LogPt.inverse(predict_batched(m, b))
pred[..., 2] = wrap_phi(pred[..., 2])
true, tb = LogPt.inverse(b.parton_x), b.parton_x_batch
n = b.num_graphs
sp = pred[..., 0].sum(1).numpy()
st = np.array([true[tb == i, 0].sum().item() for i in range(n)])
emd = np.array([event_emd(true[tb == i].numpy(), pred[i].numpy(), periodic_phi=True) for i in range(n)])
shape = np.array([event_emd(true[tb == i].numpy(), pred[i].numpy(), periodic_phi=True, norm=True) for i in range(n)])
print(f"epoch {ck['epoch']}, {n} val events")
print(f"sum pT/event: true median {np.median(st):.0f}, pred median {np.median(sp):.0f} GeV (5-95%: {np.percentile(sp, 5):.0f}-{np.percentile(sp, 95):.0f})")
print(f"leading slot pT median {np.median(pred[..., 0].max(1).values):.0f} GeV")
print(f"EMD median {np.median(emd):.0f} GeV (ln {np.median(np.log(emd)):.2f}); median |d sum pT| {np.median(abs(sp - st)):.0f} GeV = {100 * np.median(abs(sp - st) / emd):.0f}% of EMD")
print(f"shape-only EMD (norm=True) median {np.median(shape):.3f}")
