"""Hungarian-matched slot MSE — slot_mse_loss with the pairing chosen by cost, not pT rank."""

import torch
from scipy.optimize import linear_sum_assignment


def _wrap_dphi(dphi):
    return (dphi + torch.pi) % (2 * torch.pi) - torch.pi


def matched_mse_loss(pred, true, true_batch):
    """Squared error under the optimal one-to-one slot <-> parton assignment.

    pred:       [B, K, F]       fixed-size candidate set per event, F = [pT, eta, phi]
    true:       [M_total, F]    every true parton in the batch, concatenated
    true_batch: [M_total]       which event each true parton belongs to

    Same objective as slot_mse_loss — matched slots regress their parton, surplus slots are
    pushed to pT = 0 with eta/phi unpenalised — but each event's pairing comes from
    scipy's linear_sum_assignment on the squared-error cost instead of sorting truth by
    pT. The assignment is computed on detached values; gradients flow through the
    squared error of the chosen pairs, as in DETR-style set losses. phi differences are
    wrapped, both for matching and for the error, so +3.1 vs -3.1 is a small miss.
    Events with more than K true partons are truncated to the K hardest.

    The padding target is 0 in whatever space pT is in: 1 GeV under LogPt, 0 GeV without.
    """
    per_event = []
    for event in range(pred.size(0)):
        target = true[true_batch == event]
        target = target[target[:, 0].argsort(descending=True)][: pred.size(1)]
        n = target.size(0)

        diff = pred[event][:, None, :] - target[None, :, :]        # [K, n, F]
        diff = torch.cat([diff[..., :2], _wrap_dphi(diff[..., 2:])], dim=-1)
        cost = (diff ** 2).sum(dim=-1)                             # [K, n]
        slot_idx, true_idx = linear_sum_assignment(cost.detach().cpu().numpy())

        matched_error = cost[slot_idx, true_idx].sum()

        unmatched = torch.ones(pred.size(1), dtype=torch.bool, device=pred.device)
        unmatched[torch.as_tensor(slot_idx, device=pred.device)] = False
        pad_error = (pred[event][unmatched, 0] ** 2).sum()

        # Same normalisation as slot_mse_loss: n matched slots x F, plus one pT per padded slot.
        n_terms = n * pred.size(2) + int(unmatched.sum())
        per_event.append((matched_error + pad_error) / n_terms)
    return torch.stack(per_event).mean()
