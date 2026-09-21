"""Chamfer set distance — a diagnostic objective, not a settled one."""

import torch


def chamfer_set_loss(pred, true, true_batch):
    """Mean bidirectional Chamfer distance between predicted and true parton sets.

    pred:       [B, K, F]       fixed-size candidate set per event
    true:       [M_total, F]    every true parton in the batch, concatenated
    true_batch: [M_total]       which event each true parton belongs to

    Handles K != M by construction: each prediction is charged its distance to the
    nearest truth, and each truth its distance to the nearest prediction. Loops over
    events because batches here are small and the loop reads far better than the
    masked-tensor equivalent.
    generated with sonnet 5 high
    """
    per_event = []
    for event in range(pred.size(0)):
        target = true[true_batch == event]
        distance = torch.cdist(pred[event], target)
        per_event.append(
            distance.min(dim=1).values.mean() + distance.min(dim=0).values.mean()
        )
    return torch.stack(per_event).mean()
