"""Training objectives. Differentiable by construction — see metrics/ for evaluation-only scores."""

from parton_recon.losses.chamfer import chamfer_set_loss
from parton_recon.losses.matched import matched_mse_loss
from parton_recon.losses.mse import slot_mse_loss

__all__ = ["chamfer_set_loss", "matched_mse_loss", "slot_mse_loss"]
