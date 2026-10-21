"""JetGNN variant that predicts jet shape and jet energy separately."""

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, global_add_pool, global_max_pool, global_mean_pool


class FactorizedJetGNN(nn.Module):
    """Same encoder as JetGNN; the decoder splits each parton's pT into fraction x total.

    Shape head: per slot a logit, eta and phi. A softmax over the slots gives pT fractions
    z_i >= 0 with sum_i z_i = 1, so a slot that should not exist goes to z ~ 0 instead of
    carrying a fixed padding pT.

    Energy head: one scalar delta per event, with
        pT_total = (sum of hadron pT) * exp(delta).
    The hadron sum already matches the parton sum to < 1% (ratio median 1.000, 16-84%
    0.996-1.004 on the photon-vetoed sample), so delta starts near its target and only
    has to learn the hadronization / out-of-cone correction.

    Inputs are expected under LogPt (column 0 of batch.x is log pT), as for JetGNN.

    forward() returns [B, n_slots, 3] in the same LogPt space as JetGNN, column 0 being
    log(z_i * pT_total), so every evaluation script that does LogPt.inverse(model(batch))
    works unchanged. The loss uses parts() instead.
    """

    def __init__(self, in_channels=3, hidden=64, n_slots=24, out_channels=3):
        super().__init__()
        if out_channels != 3:
            raise ValueError("FactorizedJetGNN outputs [pT, eta, phi]; out_channels must be 3")
        self.n_slots = n_slots
        self.out_channels = out_channels

        self.conv1 = GATv2Conv(in_channels, hidden, edge_dim=1)
        self.conv2 = GATv2Conv(hidden, hidden, edge_dim=1)
        self.shape_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_slots * 3),     # logit, eta, phi per slot
        )
        self.energy_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # Start at delta = 0, i.e. pT_total = hadron sum, which is already right to < 1%.
        nn.init.zeros_(self.energy_head[-1].weight)
        nn.init.zeros_(self.energy_head[-1].bias)

    def parts(self, batch):
        """Return (log_z [B, K], eta [B, K], phi [B, K], delta [B], log_hadron_sum [B])."""
        h = self.conv1(batch.x, batch.edge_index, batch.edge_attr).relu()
        h = self.conv2(h, batch.edge_index, batch.edge_attr).relu()
        pooled = torch.cat(
            [global_mean_pool(h, batch.batch), global_max_pool(h, batch.batch)], dim=1
        )
        slots = self.shape_head(pooled).view(-1, self.n_slots, 3)
        log_z = torch.log_softmax(slots[..., 0], dim=1)
        delta = self.energy_head(pooled).squeeze(-1)
        log_hadron_sum = torch.log(global_add_pool(torch.exp(batch.x[:, 0:1]), batch.batch)).squeeze(-1)
        return log_z, slots[..., 1], slots[..., 2], delta, log_hadron_sum

    def forward(self, batch):
        log_z, eta, phi, delta, log_hadron_sum = self.parts(batch)
        log_pt = log_z + (log_hadron_sum + delta)[:, None]
        return torch.stack([log_pt, eta, phi], dim=-1)
