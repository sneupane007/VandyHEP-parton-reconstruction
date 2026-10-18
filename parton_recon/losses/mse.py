"""Fixed-slot mean squared error — the baseline objective that needs a slot-to-particle pairing."""

import torch


def slot_mse_loss(pred, true, true_batch):
    """Mean squared error between predicted slots and the pT-sorted true partons.

    pred:       [B, K, F]       fixed-size candidate set per event
    true:       [M_total, F]    every true parton in the batch, concatenated
    true_batch: [M_total]       which event each true parton belongs to

    MSE needs a one-to-one pairing, so truth is sorted by pT descending and placed in
    slots 0..n-1. Slots beyond the true multiplicity are targeted at pT = 0 and only pT
    is penalised there — eta/phi of a particle that does not exist carry no meaning. This
    is what pushes surplus slots toward zero energy instead of copying the hard particle.
    Events with more than K true partons are truncated to the K hardest.

    The padding target of pT = 0 is physical GeV. Do not combine with LogPt: log(0) is
    -inf and a log-space zero means 1 GeV. Train without the transform.
    """
    per_event = []
    for event in range(pred.size(0)):
        target = true[true_batch == event]
        target = target[target[:, 0].argsort(descending=True)][: pred.size(1)]
        n = target.size(0)

        padded = torch.zeros_like(pred[event])
        padded[:n] = target
        mask = torch.zeros_like(pred[event])
        mask[:n] = 1.0
        mask[:, 0] = 1.0

        squared_error = (pred[event] - padded) ** 2 * mask
        per_event.append(squared_error.sum() / mask.sum())
    return torch.stack(per_event).mean()
