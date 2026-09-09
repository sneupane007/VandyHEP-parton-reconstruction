"""Paired hadron/parton samples, read straight from the generator's JSON."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class JetPairDataset(Dataset):
    """One sample per event: the hadron graph as input, the parton set as target.

    The target carries no edges. The parton level is only ever compared as a set, so
    nothing message-passes over it and an edge list would go unused.

    Batch these with `torch_geometric.loader.DataLoader(..., follow_batch=['parton_x'])`,
    which emits `parton_x_batch` — the per-event segmentation a set loss needs to know
    which target particles belong to which event.
    """

    def __init__(self, hadron_json, parton_json, transform=None):
        self.hadron = json.loads(Path(hadron_json).read_text())
        self.parton = json.loads(Path(parton_json).read_text())
        self.transform = transform

        # Row i of each file has to describe the same event; the two are written by
        # separate appends in main214.cc, so nothing in the format enforces it.
        if len(self.hadron) != len(self.parton):
            raise ValueError(
                f"event count mismatch: {len(self.hadron)} hadron rows vs "
                f"{len(self.parton)} parton rows — the two files are not aligned"
            )

    def __len__(self):
        return len(self.hadron)

    def __getitem__(self, i):
        hadron, parton = self.hadron[i], self.parton[i]
        data = Data(
            x=torch.tensor(hadron["node_features"], dtype=torch.float),
            edge_index=torch.tensor(hadron["edge_index"], dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(hadron["edge_features"], dtype=torch.float),
            parton_x=torch.tensor(parton["node_features"], dtype=torch.float),
        )
        return data if self.transform is None else self.transform(data)
