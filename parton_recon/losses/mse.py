"""Fixed-slot mean squared error — the baseline objective that needs a slot-to-particle pairing."""

import torch
from torch_geometric.utils import to_dense_batch


def _wrap_dphi(dphi):
    return (dphi + torch.pi) % (2 * torch.pi) - torch.pi


def slot_mse_loss(pred, true, true_batch, pad_pt=0.0):
    """Mean squared error between predicted slots and the pT-sorted true partons.

    pred:       [B, K, F]       fixed-size candidate set per event, F = [pT, eta, phi]
    true:       [M_total, F]    every true parton in the batch, concatenated
    true_batch: [M_total]       which event each true parton belongs to
    pad_pt:     pT target for empty slots, in the same space as pred's pT column

    MSE needs a one-to-one pairing, so truth is sorted by pT descending and placed in
    slots 0..n-1. Slots beyond the true multiplicity are targeted at pT = pad_pt and only pT
    is penalised there — eta/phi of a particle that does not exist carry no meaning. This
    is what pushes surplus slots toward zero energy instead of copying the hard particle.
    Events with more than K true partons are truncated to the K hardest. phi differences
    are wrapped, so +3.1 vs -3.1 is a small miss, as in matched_mse_loss.

    pad_pt = 0 is 0 GeV on raw pT, but 1 GeV under LogPt (log(1) = 0), which leaves every
    empty slot as a 1 GeV particle that inflates jet mass. Under LogPt pass the log of a pT
    below any real parton instead, e.g. math.log(0.01).
    """
    B, K, F = pred.shape
    # Sort all partons in the batch by pT (highest first), then regroup them by event.
    # The second sort is stable, so each event's partons stay in pT order.
    order = true[:, 0].argsort(descending=True, stable=True)
    true, true_batch = true[order], true_batch[order]
    order = true_batch.argsort(stable=True)
    true, true_batch = true[order], true_batch[order]

    # [B, K, F] target: each event's partons in slots 0..n-1, pT = pad_pt after, anything past K dropped.
    padded, present = to_dense_batch(true, true_batch, batch_size=B, max_num_nodes=K)
    padded[~present, 0] = pad_pt

    # Real slots: all three features count. Empty slots: only pT counts.
    mask = present.to(pred.dtype).unsqueeze(-1).repeat(1, 1, F)
    mask[:, :, 0] = 1.0

    diff = pred - padded
    diff = torch.cat([diff[..., :2], _wrap_dphi(diff[..., 2:])], dim=-1)
    squared_error = diff ** 2 * mask
    return (squared_error.sum(dim=(1, 2)) / mask.sum(dim=(1, 2))).mean()
