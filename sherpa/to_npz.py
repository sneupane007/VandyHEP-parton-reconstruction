"""sherpa_jets text output -> the compact .npz that JetPairDataset reads.

Usage (repo root):
  .venv/bin/python sherpa/to_npz.py data/raw_sherpa ahadic 1-90  data/sherpa_ahadic_train.npz
  .venv/bin/python sherpa/to_npz.py data/raw_sherpa ahadic 91-100 data/sherpa_ahadic_test.npz

Same arrays as scripts/convert_to_arrays.py: hadron_x / parton_x [N, 3] float32 (pT, eta,
phi), *_offsets [E+1], seed_of_event [E], meta (JSON).
"""
import json
import sys
from pathlib import Path

import numpy as np

raw, frag, seeds, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
lo, _, hi = seeds.partition("-")
parts = {"hadron": [], "parton": []}
counts = {"hadron": [], "parton": []}
seed_of_event, used = [], []
for seed in range(int(lo), int(hi or lo) + 1):
    path = raw / f"{frag}_seed{seed:03d}.txt"
    if not path.exists():
        print(f"missing {path.name}, skipped")
        continue
    lines = path.read_text().split("\n")
    i, n_before = 0, len(seed_of_event)
    while i < len(lines) and lines[i].startswith("E "):
        _, nh, npart = lines[i].split()
        nh, npart = int(nh), int(npart)
        block = np.array([l.split() for l in lines[i + 1:i + 1 + nh + npart]], dtype=np.float32).reshape(-1, 3)
        parts["hadron"].append(block[:nh]); counts["hadron"].append(nh)
        parts["parton"].append(block[nh:]); counts["parton"].append(npart)
        seed_of_event.append(seed)
        i += 1 + nh + npart
    used.append(seed)
    print(f"{path.name}: {len(seed_of_event) - n_before} events")

arrays = {}
for lvl in parts:
    arrays[f"{lvl}_x"] = np.concatenate(parts[lvl])
    arrays[f"{lvl}_offsets"] = np.concatenate([[0], np.cumsum(counts[lvl])]).astype(np.int64)
meta = dict(raw_dir=str(raw), seeds=used, n_events=len(seed_of_event),
            generator=f"Sherpa 3.0.1, FRAGMENTATION={frag}, sherpa/{frag}.yaml + sherpa/sherpa_jets.cc")
np.savez(out, seed_of_event=np.array(seed_of_event, dtype=np.int32), meta=json.dumps(meta), **arrays)
print(f"wrote {len(seed_of_event)} events from {len(used)} seeds to {out}")
