"""A small GNN for looking at the reconstruction problem. Not a contender."""

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch


def _wrap_phi(phi):
    return (phi + torch.pi) % (2 * torch.pi) - torch.pi


def _leading_eta_phi(x, batch):
    """[B, 2] eta/phi of each event's highest-pT hadron. argmax is the same for pT and
    log(pT), so this works with or without LogPt."""
    dense, present = to_dense_batch(x, batch)
    pt = dense[..., 0].masked_fill(~present, float("-inf"))
    lead = pt.argmax(dim=1)
    return dense[torch.arange(dense.size(0), device=x.device), lead, 1:3]


class JetGNN(nn.Module):
    """Encode the hadron graph, emit a fixed set of `n_slots` candidate partons.

    GATv2Conv because it consumes `edge_attr` directly, and dR is the one edge feature
    the data actually carries.

    There is no "exists" head — every slot always fires, and the set loss tolerates the
    resulting size mismatch. Predicting multiplicity is the natural next step once you
    have watched this run.

    relative_angles: work in coordinates centred on each event's leading (highest-pT)
    hadron. Input eta/phi become offsets from it, the head predicts offsets, and the
    leading hadron's eta/phi is added back (phi wrapped) before returning. Inputs and
    outputs stay absolute, so nothing outside the model changes; the network only has to
    learn where partons sit inside the jet, not where the jet points. dR edges are
    unchanged by the shift. Off by default so older checkpoints load and behave as before.
    """

    def __init__(self, in_channels=3, hidden=64, n_slots=24, out_channels=3, relative_angles=False):
        super().__init__()
        self.n_slots = n_slots
        self.out_channels = out_channels
        self.relative_angles = relative_angles

        self.conv1 = GATv2Conv(in_channels, hidden, edge_dim=1)
        self.conv2 = GATv2Conv(hidden, hidden, edge_dim=1)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_slots * out_channels),
        )

    def forward(self, batch):
        x = batch.x
        if self.relative_angles:
            ref = _leading_eta_phi(x, batch.batch)                       # [B, 2]
            node_ref = ref[batch.batch]                                  # [N, 2]
            x = torch.cat(
                [x[:, :1], x[:, 1:2] - node_ref[:, :1], _wrap_phi(x[:, 2:3] - node_ref[:, 1:])], dim=1
            )
        h = self.conv1(x, batch.edge_index, batch.edge_attr).relu()
        h = self.conv2(h, batch.edge_index, batch.edge_attr).relu()
        pooled = torch.cat(
            [global_mean_pool(h, batch.batch), global_max_pool(h, batch.batch)], dim=1
        )
        out = self.head(pooled).view(-1, self.n_slots, self.out_channels)
        if self.relative_angles:
            eta = out[..., 1] + ref[:, None, 0]
            phi = _wrap_phi(out[..., 2] + ref[:, None, 1])
            out = torch.stack([out[..., 0], eta, phi], dim=-1)
        return out
