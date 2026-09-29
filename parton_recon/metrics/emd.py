"""Exact Earth Mover's Distance via `energyflow`, ported from
../../../Summary/jetmass.ipynb cell 22.

EMD treats each event as a pile of energy spread over the (eta, phi) plane and asks the
minimum work to reshape the predicted pile into the true one — the right metric here
because it compares two *sets* with no need to pair particles up one-to-one, and it
doesn't require equal cardinality between the two sets.

Slow and CPU-bound (O(n^2) per pair) — cap the number of events evaluated. The prior
work used 5,000 and it took minutes.

Operates on physical-space particles — [pT, eta, phi]. Undo any `LogPt` transform
before calling these.
"""

import numpy as np
import energyflow
import wasserstein

# The OpenMP build of `wasserstein` (energyflow's EMD backend) has been observed to fail
# `dlopen` on macOS even with Homebrew's libomp installed (symbol not found:
# '___kmpc_barrier'). Switching to the non-OpenMP backend is a one-line workaround, done
# once here so callers don't need to know about it. See notebooks/03_emd_metric.md.
wasserstein.without_openmp()


def event_emd(true_particles, pred_particles, **kwargs):
    """EMD between one event's true and predicted [pT, eta, phi] particle sets."""
    return energyflow.emd.emd(true_particles, pred_particles, **kwargs)


def dataset_emd(true_events, pred_events, max_events=None, verbose=True, **kwargs):
    """EMD per event over a dataset.

    true_events / pred_events: sequences of (N_i, 3) arrays, one per event, same length.
    Returns a (n,) float64 array of raw EMD values. Conventionally reported as ln(EMD),
    since raw values span orders of magnitude — take the log at the call site.

    **kwargs forwards to `energyflow.emd.emd` per event, e.g. `periodic_phi=True`.
    """
    # if len(true_events) != len(pred_events):
    #     raise ValueError(
    #         f"event count mismatch: {len(true_events)} true vs {len(pred_events)} pred"
    #     )
    n = len(true_events) if max_events is None else min(max_events, len(true_events))
    emds = np.empty(n, dtype=np.float64)
    for i in range(n):
        emds[i] = event_emd(true_events[i], pred_events[i], **kwargs)
        if verbose and (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n} done")
    return emds
