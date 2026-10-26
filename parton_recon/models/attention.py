"""FactorizedJetGNN with a deeper encoder and a cross-attention (DETR-style) slot decoder."""

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch


def _wrap(phi):
    return (phi + torch.pi) % (2 * torch.pi) - torch.pi


class AttentionJetGNN(nn.Module):
    """Same output as FactorizedJetGNN (pT fractions x total pT), different decoder.

    FactorizedJetGNN decodes every slot from one pooled vector, which in practice collapses
    to an "average jet": the predicted jet mass misses the whole tail above ~120 GeV even
    though the hadron jet mass correlates 0.98 with the parton one. Here each of the
    n_slots learned queries cross-attends to every hadron node embedding, so a slot can
    take its position and pT from the hadrons it attends to.

    Encoder: input projection, then n_conv GATv2 layers (multi-head, residual, LayerNorm).
    Decoder: n_dec nn.TransformerDecoder layers — slot self-attention, then cross-attention
    to the nodes of that event (padded per event, masked).
    Heads: per slot a logit (softmax over slots -> pT fraction z), eta, phi; per event the
    total-pT correction delta, with pT_total = (sum hadron pT) * exp(delta), as before.

    relative=True: node eta/phi are fed as offsets from the hadron jet axis (pT-weighted
    eta; phi of the summed pT vector), and slots predict offsets from the same axis, which
    is added back at the output. Translation in eta and rotation in phi then cost the
    network nothing. Edge features (dR) are unaffected.

    Inputs under LogPt, as for JetGNN. forward() returns [B, n_slots, 3] in LogPt space;
    parts() feeds matched_fraction_loss.
    """

    def __init__(self, in_channels=3, hidden=128, n_slots=24, out_channels=3, heads=4,
                 n_conv=3, n_dec=3, dropout=0.1, relative=True):
        super().__init__()
        if out_channels != 3:
            raise ValueError("AttentionJetGNN outputs [pT, eta, phi]; out_channels must be 3")
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.n_slots = n_slots
        self.out_channels = out_channels
        self.relative = relative

        self.embed = nn.Linear(in_channels, hidden)
        self.convs = nn.ModuleList(
            GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=1) for _ in range(n_conv)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(n_conv))

        self.queries = nn.Parameter(torch.randn(n_slots, hidden) * 0.1)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=2 * hidden, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_dec)
        self.slot_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 3))  # logit, eta, phi

        self.energy_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # Start at delta = 0, i.e. pT_total = hadron sum, which is already right to < 1%.
        nn.init.zeros_(self.energy_head[-1].weight)
        nn.init.zeros_(self.energy_head[-1].bias)

    def _axis(self, x, batch):
        """pT-weighted jet axis (eta, phi) per event from LogPt node features."""
        pt = torch.exp(x[:, 0:1])
        sums = global_add_pool(torch.cat([pt, pt * x[:, 1:2], pt * torch.cos(x[:, 2:3]),
                                          pt * torch.sin(x[:, 2:3])], dim=1), batch)
        return sums[:, 1] / sums[:, 0], torch.atan2(sums[:, 3], sums[:, 2])

    def parts(self, batch):
        """Return (log_z [B, K], eta [B, K], phi [B, K], delta [B], log_hadron_sum [B])."""
        x = batch.x
        if self.relative:
            axis_eta, axis_phi = self._axis(x, batch.batch)
            x = torch.stack([x[:, 0], x[:, 1] - axis_eta[batch.batch],
                             _wrap(x[:, 2] - axis_phi[batch.batch])], dim=1)

        h = self.embed(x)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + conv(h, batch.edge_index, batch.edge_attr).relu())

        nodes, mask = to_dense_batch(h, batch.batch)                    # [B, N_max, H], [B, N_max]
        queries = self.queries.unsqueeze(0).expand(nodes.size(0), -1, -1)
        slots = self.slot_head(self.decoder(queries, nodes, memory_key_padding_mask=~mask))

        log_z = torch.log_softmax(slots[..., 0], dim=1)
        eta, phi = slots[..., 1], slots[..., 2]
        if self.relative:
            eta = eta + axis_eta[:, None]
            phi = _wrap(phi + axis_phi[:, None])

        pooled = torch.cat([global_mean_pool(h, batch.batch), global_max_pool(h, batch.batch)], dim=1)
        delta = self.energy_head(pooled).squeeze(-1)
        log_hadron_sum = torch.log(global_add_pool(torch.exp(batch.x[:, 0:1]), batch.batch)).squeeze(-1)
        return log_z, eta, phi, delta, log_hadron_sum

    def forward(self, batch):
        log_z, eta, phi, delta, log_hadron_sum = self.parts(batch)
        log_pt = log_z + (log_hadron_sum + delta)[:, None]
        return torch.stack([log_pt, eta, phi], dim=-1)
