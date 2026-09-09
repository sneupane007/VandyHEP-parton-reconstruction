"""A small GNN for looking at the reconstruction problem. Not a contender."""

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool


class JetGNN(nn.Module):
    """Encode the hadron graph, emit a fixed set of `n_slots` candidate partons.

    GATv2Conv because it consumes `edge_attr` directly, and dR is the one edge feature
    the data actually carries.

    There is no "exists" head — every slot always fires, and the set loss tolerates the
    resulting size mismatch. Predicting multiplicity is the natural next step once you
    have watched this run.
    """

    def __init__(self, in_channels=3, hidden=64, n_slots=24, out_channels=3):
        super().__init__()
        self.n_slots = n_slots
        self.out_channels = out_channels

        self.conv1 = GATv2Conv(in_channels, hidden, edge_dim=1)
        self.conv2 = GATv2Conv(hidden, hidden, edge_dim=1)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_slots * out_channels),
        )

    def forward(self, batch):
        h = self.conv1(batch.x, batch.edge_index, batch.edge_attr).relu()
        h = self.conv2(h, batch.edge_index, batch.edge_attr).relu()
        pooled = torch.cat(
            [global_mean_pool(h, batch.batch), global_max_pool(h, batch.batch)], dim=1
        )
        return self.head(pooled).view(-1, self.n_slots, self.out_channels)
