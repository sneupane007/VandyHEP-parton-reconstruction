"""Training objectives. Differentiable by construction — see metrics/ for evaluation-only scores."""

from parton_recon.losses.chamfer import chamfer_set_loss

__all__ = ["chamfer_set_loss"]
