"""Concatenate per-seed generator outputs into one hadron/parton JSON pair.

Run from the repo root:
  .venv/bin/python3 scripts/merge_generated.py data/raw data/model1 [--max-events N]

Reads data/raw/seedNNN_{hadron,parton}_level_graphs_.json in seed order, checks that each
seed's two files have the same number of events (row i of each must be the same event), and
writes <out_prefix>_{hadron,parton}_level_graphs_.json. Seeds whose JSON is incomplete (job
still running or killed) are skipped with a warning.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("raw_dir", type=Path)
    p.add_argument("out_prefix", type=Path)
    p.add_argument("--max-events", type=int, default=None)
    args = p.parse_args()

    hadron, parton = [], []
    for h_path in sorted(args.raw_dir.glob("seed*_hadron_level_graphs_.json")):
        p_path = h_path.with_name(h_path.name.replace("_hadron_", "_parton_"))
        try:
            h, pa = json.loads(h_path.read_text()), json.loads(p_path.read_text())
        except (json.JSONDecodeError, FileNotFoundError) as err:
            print(f"skipping {h_path.name}: {err}")
            continue
        if len(h) != len(pa):
            raise ValueError(f"{h_path.name}: {len(h)} hadron vs {len(pa)} parton events")
        hadron += h
        parton += pa
        if args.max_events is not None and len(hadron) >= args.max_events:
            break

    hadron, parton = hadron[: args.max_events], parton[: args.max_events]
    for level, events in [("hadron", hadron), ("parton", parton)]:
        out = Path(f"{args.out_prefix}_{level}_level_graphs_.json")
        out.write_text(json.dumps(events, separators=(",", ":")))
        print(f"wrote {len(events)} events to {out}")


if __name__ == "__main__":
    main()
