"""End-to-end smoke run: JSON -> transform -> batch -> GNN -> Chamfer -> backward.

Run from the repo root:  .jupyter_venv/bin/python3 scripts/experiment.py

With 6 events this can only show that the loop closes and the model can memorise them.
It says nothing whatsoever about generalisation.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
from torch_geometric.loader import DataLoader

from parton_recon.data import JetPairDataset, LogPt
from parton_recon.losses import chamfer_set_loss
from parton_recon.models import JetGNN

GENERATED = REPO_ROOT / "parton_recon" / "data" / "generate"
STEPS = 300
SEED = 0


def main():
    torch.manual_seed(SEED)

    dataset = JetPairDataset(
        GENERATED / "model1_hadron_level_graphs_.json",
        GENERATED / "model1_parton_level_graphs_.json",
        transform=LogPt(),
    )
    print(f"{len(dataset)} events")
    # for i, sample in enumerate(dataset):
    #     print(
    #         f"  event {i}: {sample.x.shape[0]:3d} hadrons -> "
    #         f"{sample.parton_x.shape[0]:2d} partons, {sample.edge_index.shape[1]:5d} edges"
    #     )

    loader = DataLoader(
        dataset, batch_size=len(dataset), follow_batch=["parton_x"], shuffle=False
    )
    batch = next(iter(loader))
    print(f"\nbatched: x={tuple(batch.x.shape)}, parton_x={tuple(batch.parton_x.shape)}")
    print(f"partons per event: {torch.bincount(batch.parton_x_batch).tolist()}")

    model = JetGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"\ntraining {sum(p.numel() for p in model.parameters()):,} parameters")
    for step in range(1, STEPS + 1):
        optimizer.zero_grad()
        pred = model(batch)
        loss = chamfer_set_loss(pred, batch.parton_x, batch.parton_x_batch)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 25 == 0:
            print(f"  step {step:3d}  chamfer {loss.item():.4f}")

    with torch.no_grad():
        pred = model(batch)
    event = 0
    truth = batch.parton_x[batch.parton_x_batch == event]
    print(f"\nevent {event} — true partons (log pT, eta, phi):")
    for row in truth:
        print(f"  {row[0]:8.3f} {row[1]:8.3f} {row[2]:8.3f}")
    print(f"event {event} — {model.n_slots} predicted slots, nearest {len(truth)} shown:")
    nearest = torch.cdist(truth, pred[event]).min(dim=1).indices
    for row in pred[event][nearest]:
        print(f"  {row[0]:8.3f} {row[1]:8.3f} {row[2]:8.3f}")

    print("\n6 events only — this is a plumbing check, not a result.")


if __name__ == "__main__":
    main()
