"""Hungarian-matched loss on pT fractions, plus total-pT and jet-mass terms — for FactorizedJetGNN."""

import torch
from scipy.optimize import linear_sum_assignment

from parton_recon.losses.matched import _wrap_dphi


def _log_mass2(pt, eta, phi):
    """log m^2 of massless constituents summed along the last dim. float64 so that
    E^2 - p^2 (both ~ (800 GeV)^2) does not lose the ~(70 GeV)^2 difference."""
    pt, eta, phi = pt.double(), eta.double(), phi.double()
    e = (pt * torch.cosh(eta)).sum(-1)
    px, py, pz = (pt * torch.cos(phi)).sum(-1), (pt * torch.sin(phi)).sum(-1), (pt * torch.sinh(eta)).sum(-1)
    return torch.log((e ** 2 - px ** 2 - py ** 2 - pz ** 2).clamp(min=1e-6))


def matched_fraction_loss(parts, true, true_batch, energy_weight=1.0, angle_weight="uniform",
                          mass_weight=0.0):
    """matched_mse_loss applied to the jet's shape, with its energy fitted separately.

    parts:      FactorizedJetGNN.parts(batch) -> (log_z, eta, phi, delta, log_hadron_sum)
    true:       [M_total, 3]  every true parton in the batch under LogPt: [log pT, eta, phi]
    true_batch: [M_total]     which event each true parton belongs to

    Shape term: per event, truth is turned into fractions z = pT / sum pT (over all true
    partons, so ones dropped by the slot limit still count in the denominator) and matched
    one-to-one to slots with linear_sum_assignment, as in matched_mse_loss. The per-pair
    cost is (sqrt z_pred - sqrt z_true)^2 + w * (d_eta^2 + d_phi^2) (d_phi wrapped).
    sqrt(z) sits between z, where the error is dominated by the hardest parton, and log z,
    where it is dominated by the softest. Unmatched slots are pushed to z = 0 through their
    (sqrt z)^2 = z; their eta/phi are not penalised. Events with more true partons than
    slots keep the hardest K. Normalised per event like matched_mse_loss.

    angle_weight: "uniform" -> w = 1 for every parton (the original loss). "pt" -> w = n *
    z_true, the parton's pT fraction scaled so the weights average ~1. That is how the EMD
    weighs position errors, and it stops a soft parton whose position the hadrons barely
    constrain from costing as much as the hardest one.

    Energy term: energy_weight * (delta - delta_true)^2 with
    delta_true = log(sum parton pT / sum hadron pT), averaged over events.

    Mass term (mass_weight > 0): mass_weight * (log m_pred - log m_true)^2, with both jet
    masses summed from massless constituents (the predicted pT_i = z_i * pT_total), averaged
    over events. Squared-error position fits pull uncertain partons toward the jet axis,
    which makes predicted jets too narrow (median mass ~25% low); this ties the width back
    to the truth at the level of the whole jet.

    Returns (total, shape, energy, mass) so the parts can be logged separately; mass is 0
    when mass_weight is 0.
    """
    if angle_weight not in ("uniform", "pt"):
        raise ValueError(f"angle_weight must be 'uniform' or 'pt', not {angle_weight!r}")
    log_z, eta, phi, delta, log_hadron_sum = parts
    n_slots = log_z.size(1)
    sqrt_z = torch.exp(0.5 * log_z)

    shape_terms, delta_true, log_m2_true = [], [], []
    for event in range(log_z.size(0)):
        target = true[true_batch == event]
        pt = torch.exp(target[:, 0])
        delta_true.append(torch.log(pt.sum()) - log_hadron_sum[event])
        if mass_weight > 0:
            log_m2_true.append(_log_mass2(pt, target[:, 1], target[:, 2]))
        order = pt.argsort(descending=True)[:n_slots]
        t_z = pt[order] / pt.sum()
        t_eta, t_phi = target[order, 1], target[order, 2]
        n = order.numel()
        w = n * t_z if angle_weight == "pt" else torch.ones_like(t_z)

        cost = (
            (sqrt_z[event][:, None] - torch.sqrt(t_z)[None, :]) ** 2
            + w[None, :] * (
                (eta[event][:, None] - t_eta[None, :]) ** 2
                + _wrap_dphi(phi[event][:, None] - t_phi[None, :]) ** 2
            )
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
    total = shape + energy_weight * energy
    mass = torch.zeros((), device=log_z.device)
    if mass_weight > 0:
        pt_pred = torch.exp(log_z + (log_hadron_sum + delta)[:, None])
        log_m2_pred = _log_mass2(pt_pred, eta, phi)
        # (log m)^2 difference = (0.5 * log m^2 difference)^2
        mass = ((0.5 * (log_m2_pred - torch.stack(log_m2_true).detach())) ** 2).mean().float()
        total = total + mass_weight * mass
    return total, shape, energy, mass
