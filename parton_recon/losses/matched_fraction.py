"""Hungarian-matched loss on pT fractions, plus a total-pT term — for FactorizedJetGNN."""

import torch
from scipy.optimize import linear_sum_assignment

from parton_recon.losses.matched import _wrap_dphi


def matched_fraction_loss(parts, true, true_batch, energy_weight=1.0):
    """matched_mse_loss applied to the jet's shape, with its energy fitted separately.

    parts:      FactorizedJetGNN.parts(batch) -> (log_z, eta, phi, delta, log_hadron_sum)
    true:       [M_total, 3]  every true parton in the batch under LogPt: [log pT, eta, phi]
    true_batch: [M_total]     which event each true parton belongs to

    Shape term: per event, truth is turned into fractions z = pT / sum pT (over all true
    partons, so ones dropped by the slot limit still count in the denominator) and matched
    one-to-one to slots with linear_sum_assignment, as in matched_mse_loss. The per-pair
    cost is (sqrt z_pred - sqrt z_true)^2 + d_eta^2 + d_phi^2 (wrapped). sqrt(z) sits
    between z, where the error is dominated by the hardest parton, and log z, where it is
    dominated by the softest. Unmatched slots are pushed to z = 0 through their
    (sqrt z)^2 = z; their eta/phi are not penalised. Events with more true partons than
    slots keep the hardest K. Normalised per event like matched_mse_loss.

    Energy term: energy_weight * (delta - delta_true)^2 with
    delta_true = log(sum parton pT / sum hadron pT), averaged over events.

    Returns (total, shape, energy) so the two parts can be logged separately.
    """
    log_z, eta, phi, delta, log_hadron_sum = parts
    n_slots = log_z.size(1)
    sqrt_z = torch.exp(0.5 * log_z)

    shape_terms, delta_true = [], []
    for event in range(log_z.size(0)):
        target = true[true_batch == event]
        pt = torch.exp(target[:, 0])
        delta_true.append(torch.log(pt.sum()) - log_hadron_sum[event])
        order = pt.argsort(descending=True)[:n_slots]
        t_sqrt_z = torch.sqrt(pt[order] / pt.sum())
        t_eta, t_phi = target[order, 1], target[order, 2]
        n = order.numel()

        cost = (
            (sqrt_z[event][:, None] - t_sqrt_z[None, :]) ** 2
            + (eta[event][:, None] - t_eta[None, :]) ** 2
            + _wrap_dphi(phi[event][:, None] - t_phi[None, :]) ** 2
        )                                                           # [K, n]
        slot_idx, true_idx = linear_sum_assignment(cost.detach().cpu().numpy())
        matched_error = cost[slot_idx, true_idx].sum()

        unmatched = torch.ones(n_slots, dtype=torch.bool, device=log_z.device)
        unmatched[torch.as_tensor(slot_idx, device=log_z.device)] = False
        pad_error = (sqrt_z[event][unmatched] ** 2).sum()

        n_terms = n * 3 + int(unmatched.sum())
        shape_terms.append((matched_error + pad_error) / n_terms)

    shape = torch.stack(shape_terms).mean()
    energy = ((delta - torch.stack(delta_true).detach()) ** 2).mean()
    return shape + energy_weight * energy, shape, energy
