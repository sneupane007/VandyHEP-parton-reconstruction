"""Paired hadron/parton samples, read from the generator's JSON or from compact arrays.

The compact format (scripts/convert_to_arrays.py) is one .npz holding only node features
plus per-event offsets; the fully connected edges and their dR are rebuilt on access in
the same order main214.cc writes them. The 100-seed vetoed sample is 19 GB as JSON and
~40 MB as .npz, and the JSON version does not fit in memory once parsed.
"""

import json
from pathlib import Path

import numpy as np
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

    Pass a `.npz` from scripts/convert_to_arrays.py as `hadron_json` (and the same path,
    or None, as `parton_json`) to read the compact format instead; `.hadron[i]` /
    `.parton[i]` then still return {"node_features": array} for scripts that read them.
    """

    def __init__(self, hadron_json, parton_json, transform=None):
        self.transform = transform
        if str(hadron_json).endswith(".npz"):
            if parton_json is not None and str(parton_json) != str(hadron_json):
                raise ValueError("an .npz holds both levels; pass the same path (or None) as parton_json")
            arrays = np.load(hadron_json)
            self.hadron = _NodeView(arrays["hadron_x"], arrays["hadron_offsets"])
            self.parton = _NodeView(arrays["parton_x"], arrays["parton_offsets"])
            self.meta = json.loads(str(arrays["meta"]))
            self._compact = True
        else:
            self.hadron = json.loads(Path(hadron_json).read_text())
            self.parton = json.loads(Path(parton_json).read_text())
            self._compact = False

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
        if self._compact:
            x = torch.from_numpy(self.hadron.array(i))
            edge_index, edge_attr = full_graph_edges(x)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                        parton_x=torch.from_numpy(self.parton.array(i)))
            return data if self.transform is None else self.transform(data)

        hadron, parton = self.hadron[i], self.parton[i]
        # A single-hadron event has no edges, so edge_index/edge_features come back as
        # `[]` rather than `[[i, j], ...]` — reshape before transposing so the tensor
        # still ends up [2, 0] / [0, 1] instead of the degenerate [0].
        data = Data(
            x=torch.tensor(hadron["node_features"], dtype=torch.float),
            edge_index=torch.tensor(hadron["edge_index"], dtype=torch.long)
            .reshape(-1, 2)
            .t()
            .contiguous(),
            edge_attr=torch.tensor(hadron["edge_features"], dtype=torch.float).reshape(-1, 1),
            parton_x=torch.tensor(parton["node_features"], dtype=torch.float),
        )
        return data if self.transform is None else self.transform(data)


class _NodeView:
    """Per-event slices of one concatenated [N_total, 3] array, JSON-row lookalike."""

    def __init__(self, x, offsets):
        self.x, self.offsets = x, offsets

    def __len__(self):
        return len(self.offsets) - 1

    def array(self, i):
        return self.x[self.offsets[i]:self.offsets[i + 1]]

    def __getitem__(self, i):
        return {"node_features": self.array(i)}


def full_graph_edges(x):
    """Edges i -> j for every ordered pair i != j (i-major, as main214.cc writes them) and
    their dR from columns 1, 2 (eta, phi), with dphi wrapped. Returns ([2, E] long, [E, 1])."""
    n = x.size(0)
    src, dst = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    keep = src != dst
    src, dst = src[keep], dst[keep]
    eta, phi = x[:, 1].double(), x[:, 2].double()
    dphi = torch.abs(phi[src] - phi[dst])
    dphi = torch.where(dphi > torch.pi, 2 * torch.pi - dphi, dphi)
    dr = torch.sqrt((eta[src] - eta[dst]) ** 2 + dphi ** 2).float()
    return torch.stack([src, dst]), dr.reshape(-1, 1)
