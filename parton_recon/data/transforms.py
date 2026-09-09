"""Data preparation. Add new transforms here.

A transform subclasses `BaseTransform` and implements `forward(data) -> data`. That
buys composition for free — `torch_geometric.transforms.Compose([A(), B()])` — and any
transform can be handed to `JetPairDataset(transform=...)` to run on every access.

Two things to know before writing one:

`BaseTransform.__call__` shallow-copies the sample before calling `forward`, so the
tensors are still shared with the dataset's cached copy. Build a new tensor and assign
it; writing into an existing one in place corrupts the sample for every later access.

A sample carries both levels — `x` is the hadron input, `parton_x` the parton target. A
transform that changes what a feature column means has to change it on both, or the
loss ends up comparing two different spaces.

`edge_attr` holds dR, computed from eta/phi when the JSON was generated. `LogPt` leaves
eta/phi untouched so it stays valid, but a transform that moves either one makes the
stored edges stale.

Not implemented, in rough order of likely usefulness: sin/cos encoding of phi (it is
periodic, so raw phi has a false discontinuity at +/-pi), per-feature standardisation,
a pT floor to drop soft particles, recomputing edge_attr from transformed coordinates.
"""

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class LogPt(BaseTransform):
    """Replace pT (column 0) with log(pT), on both levels.

    pT spans roughly four orders of magnitude while eta and phi are O(1), which leaves
    the first linear layer trying to fit wildly different scales at once.
    """

    def forward(self, data: Data) -> Data:
        x = data.x.clone()
        x[:, 0] = torch.log(x[:, 0])
        data.x = x

        parton_x = data.parton_x.clone()
        parton_x[:, 0] = torch.log(parton_x[:, 0])
        data.parton_x = parton_x

        return data
