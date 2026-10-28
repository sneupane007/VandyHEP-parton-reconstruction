"""Score several checkpoints on one held-out test set and overlay the results.

Usage:
  .venv/bin/python repro/evaluate_test.py data/veto_test.npz out_prefix label=checkpoints/<run> [label=...]

Every event of the test file is used (no split). For each run (its checkpoint_best.pt):
median ln(EMD), shape-only EMD (norm=True), predicted/true sum pT, and the jet mass
(median relative error, 16-84% band, correlation with the true parton mass). Writes
<out_prefix>.txt (the table) and <out_prefix>.png (ln EMD, shape EMD, jet mass and
pred/true mass ratio, all runs overlaid, with the true parton and input hadron masses).

Seeds 191-200 (data/veto_test.npz) are generated but used by no training run, so models
trained on the 6725-event set and on the full sample are compared on the same jets.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from parton_recon.data import JetPairDataset, LogPt
from parton_recon.metrics import jet_mass
from parton_recon.metrics.emd import event_emd
from parton_recon.models import build_model, predict_batched


def wrap(phi):
    return (phi + np.pi) % (2 * np.pi) - np.pi


test_path, out = sys.argv[1], Path(sys.argv[2])
runs = [a.split("=", 1) for a in sys.argv[3:]]

ds = JetPairDataset(test_path, test_path, transform=LogPt())
batch = next(iter(DataLoader(ds, batch_size=len(ds), follow_batch=["parton_x"])))
true = LogPt.inverse(batch.parton_x).numpy()
had = LogPt.inverse(batch.x).numpy()
tb, hb, n = batch.parton_x_batch.numpy(), batch.batch.numpy(), batch.num_graphs
true_ev = [true[tb == i] for i in range(n)]
m_true = np.array([jet_mass(t) for t in true_ev])
m_had = np.array([jet_mass(had[hb == i]) for i in range(n)])
st = np.array([t[:, 0].sum() for t in true_ev])

rows, results = [], {}
for label, run in runs:
    ck = torch.load(Path(run) / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    model = build_model(ck["arch_config"]); model.load_state_dict(ck["model_state_dict"]); model.eval()
    with torch.no_grad():
        pred = LogPt.inverse(predict_batched(model, batch)).numpy()
    pred[..., 2] = wrap(pred[..., 2])
    ln_emd = np.log([event_emd(true_ev[i], pred[i], periodic_phi=True) for i in range(n)])
    shape = np.array([event_emd(true_ev[i], pred[i], periodic_phi=True, norm=True) for i in range(n)])
    m_pred = np.array([jet_mass(pred[i]) for i in range(n)])
    rel = (m_pred - m_true) / m_true
    sp = pred[..., 0].sum(1)
    results[label] = dict(ln_emd=ln_emd, shape=shape, m_pred=m_pred)
    rows.append(f"{label:34s} {ck['epoch']:5d} {np.median(ln_emd):8.2f} {100 * (ln_emd <= 4).mean():6.0f}% "
                f"{np.median(shape):7.3f} {np.median(sp / st):7.3f} {np.median(rel):+8.3f} "
                f"[{np.percentile(rel, 16):+.2f}, {np.percentile(rel, 84):+.2f}] {np.corrcoef(m_pred, m_true)[0, 1]:6.3f}")
    print(rows[-1], flush=True)

header = (f"{n} test events ({test_path}); true parton mass median {np.median(m_true):.1f} GeV, "
          f"hadron-vs-parton mass corr {np.corrcoef(m_had, m_true)[0, 1]:.3f}\n"
          f"{'run':34s} {'epoch':>5s} {'ln(EMD)':>8s} {'good':>7s} {'shape':>7s} {'pT r':>7s} "
          f"{'dm/m':>8s} {'16-84%':>14s} {'m corr':>6s}")
out.with_suffix(".txt").write_text(header + "\n" + "\n".join(rows) + "\n")
print(header)

fig, ax = plt.subplots(2, 2, figsize=(13, 9))
mbins = np.linspace(0, np.percentile(m_true, 99.5), 41)
ax[1, 0].hist(m_true, mbins, histtype="step", lw=2.2, color="k", label="true parton")
ax[1, 0].hist(m_had, mbins, histtype="step", lw=1.2, ls="--", color="gray", label="hadron (input)")
for label, r in results.items():
    ax[0, 0].hist(r["ln_emd"], np.linspace(1, 7, 61), histtype="step", lw=1.5, label=f"{label} ({np.median(r['ln_emd']):.2f})")
    ax[0, 1].hist(r["shape"], np.linspace(0, 0.2, 61), histtype="step", lw=1.5, label=f"{label} ({np.median(r['shape']):.3f})")
    ax[1, 0].hist(r["m_pred"], mbins, histtype="step", lw=1.5, label=label)
    ax[1, 1].hist(r["m_pred"] / m_true, np.linspace(0, 2, 61), histtype="step", lw=1.5,
                  label=f"{label} (median {np.median(r['m_pred'] / m_true):.2f})")
ax[0, 0].axvline(4, color="seagreen", ls="--", lw=1)
ax[0, 0].set(xlabel="ln(EMD)", ylabel="test events", title="Full EMD (median)")
ax[0, 1].set(xlabel="EMD, norm=True", title="Shape-only EMD (median)")
ax[1, 0].set(xlabel="jet mass [GeV]", ylabel="test events", title="Jet mass")
ax[1, 1].axvline(1, color="k", lw=0.8)
ax[1, 1].set(xlabel="predicted / true parton jet mass", title="Per-event mass ratio")
for a in ax.flat:
    a.legend(fontsize=7)
fig.suptitle(f"Held-out test set: {n} photon-vetoed jets (seeds 191-200), best checkpoints")
fig.tight_layout()
fig.savefig(out.with_suffix(".png"), dpi=140)
print(f"saved {out.with_suffix('.txt')} and {out.with_suffix('.png')}")
