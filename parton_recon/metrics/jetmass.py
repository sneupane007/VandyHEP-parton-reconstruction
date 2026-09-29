"""Jet invariant mass, ported from ../../../Summary/jetmass.ipynb cell 10.

Operates on physical-space particles — [pT, eta, phi], massless constituents. Undo any
`LogPt` transform (exponentiate column 0) before calling these.
"""

import numpy as np
import vector


def clip_negative_pt(particles):
    """Clamp pT (column 0) to >= 0, in place. Leave eta/phi untouched.

    A regressor can emit unphysical negative pT. Ported from
    `../../../Summary/rotations.ipynb` cell 5 (`_clip_negative_pt_inplace`).
    """
    np.maximum(particles[:, 0], 0.0, out=particles[:, 0])
    return particles


def jet_mass(particles):
    """Invariant mass of a set of [pT, eta, phi] particles, summed as massless 4-vectors.

    particles: (N, 3) array. A row with pT <= 0 contributes nothing — under the ragged
    schema this is only ever a clamped negative prediction, never padding.
    """
    jet = vector.obj(px=0.0, py=0.0, pz=0.0, energy=0.0)
    for pt, eta, phi in particles:
        if pt <= 0.0:
            continue
        jet = jet + vector.obj(pt=float(pt), eta=float(eta), phi=float(phi), mass=0.0)
    return float(jet.mass)
