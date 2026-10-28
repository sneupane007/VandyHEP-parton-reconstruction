"""Convert per-seed generator JSON into one compact .npz (node features only).

Run from the repo root:
  .venv/bin/python3 scripts/convert_to_arrays.py data/raw_veto data/veto_train.npz --seeds 101-190
  .venv/bin/python3 scripts/convert_to_arrays.py data/raw_veto data/veto_test.npz  --seeds 191-200

Reads seedNNN_{hadron,parton}_level_graphs_.json one seed at a time (so memory stays at
one file's worth), keeps only node_features, and writes
  hadron_x [N_h, 3] float32, hadron_offsets [E+1] int64, parton_x, parton_offsets,
  seed_of_event [E] int32, meta (JSON string: source files, seeds, event counts).
Edges are not stored: JetPairDataset rebuilds the fully connected graph and its dR on
access. For the first event of each seed the rebuilt edges are checked against the
stored ones, so a change in the generator's edge convention fails loudly here.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from parton_recon.data.dataset import full_graph_edges


def parse_seeds(text):
    lo, _, hi = text.partition("-")
    return range(int(lo), int(hi or lo) + 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("raw_dir", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--seeds", required=True, help="inclusive range, e.g. 101-190")
    args = p.parse_args()

    parts = {"hadron": [], "parton": []}
    counts = {"hadron": [], "parton": []}
    seed_of_event, used = [], []
    for seed in parse_seeds(args.seeds):
        paths = {lvl: args.raw_dir / f"seed{seed:03d}_{lvl}_level_graphs_.json" for lvl in parts}
        try:
            events = {lvl: json.loads(path.read_text()) for lvl, path in paths.items()}
        except (FileNotFoundError, json.JSONDecodeError) as err:
            print(f"skipping seed {seed}: {err}")
            continue
        if len(events["hadron"]) != len(events["parton"]):
            raise ValueError(f"seed {seed}: {len(events['hadron'])} hadron vs {len(events['parton'])} parton events")

        first = events["hadron"][0]
        x = torch.tensor(first["node_features"], dtype=torch.float)
        edge_index, edge_attr = full_graph_edges(x)
        stored_index = torch.tensor(first["edge_index"], dtype=torch.long).reshape(-1, 2).t()
        stored_attr = torch.tensor(first["edge_features"], dtype=torch.float).reshape(-1, 1)
        if not torch.equal(edge_index, stored_index) or not torch.allclose(edge_attr, stored_attr, atol=1e-5):
            raise ValueError(f"seed {seed}: rebuilt edges differ from the stored ones")

        for lvl in parts:
            for e in events[lvl]:
                arr = np.asarray(e["node_features"], dtype=np.float32).reshape(-1, 3)
                parts[lvl].append(arr)
                counts[lvl].append(len(arr))
        seed_of_event += [seed] * len(events["hadron"])
        used.append(seed)
        print(f"seed {seed}: {len(events['hadron'])} events (total {len(seed_of_event)})")

    out = {}
    for lvl in parts:
        out[f"{lvl}_x"] = np.concatenate(parts[lvl])
        out[f"{lvl}_offsets"] = np.concatenate([[0], np.cumsum(counts[lvl])]).astype(np.int64)
    meta = dict(raw_dir=str(args.raw_dir), seeds=used, n_events=len(seed_of_event),
                generator="main214.cc, vetoAndMatch per the raw_dir's sbatch script")
    np.savez(args.out, seed_of_event=np.array(seed_of_event, dtype=np.int32),
             meta=json.dumps(meta), **out)
    print(f"wrote {len(seed_of_event)} events from {len(used)} seeds to {args.out}")


if __name__ == "__main__":
    main()
