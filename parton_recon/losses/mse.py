"""Fixed-slot mean squared error — the baseline objective that needs a slot-to-particle pairing."""

from torch_geometric.utils import to_dense_batch


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
    B, K, F = pred.shape
    # Sort all partons in the batch by pT (highest first), then regroup them by event.
    # The second sort is stable, so each event's partons stay in pT order.
    order = true[:, 0].argsort(descending=True, stable=True)
    true, true_batch = true[order], true_batch[order]
    order = true_batch.argsort(stable=True)
    true, true_batch = true[order], true_batch[order]

    # [B, K, F] target: each event's partons in slots 0..n-1, zeros after, anything past K dropped.
    padded, present = to_dense_batch(true, true_batch, batch_size=B, max_num_nodes=K)

    # Real slots: all three features count. Empty slots: only pT counts.
    mask = present.to(pred.dtype).unsqueeze(-1).repeat(1, 1, F)
    mask[:, :, 0] = 1.0

    squared_error = (pred - padded) ** 2 * mask
    return (squared_error.sum(dim=(1, 2)) / mask.sum(dim=(1, 2))).mean()
