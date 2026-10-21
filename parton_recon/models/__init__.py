"""Encoder, decoder, and baseline architectures."""

from parton_recon.models.factorized import FactorizedJetGNN
from parton_recon.models.gnn import JetGNN

MODELS = {"JetGNN": JetGNN, "FactorizedJetGNN": FactorizedJetGNN}


def build_model(arch_config):
    """Instantiate from a checkpoint's arch_config. Checkpoints written before the "model"
    key existed are all JetGNN."""
    arch = dict(arch_config)
    return MODELS[arch.pop("model", "JetGNN")](**arch)


__all__ = ["JetGNN", "FactorizedJetGNN", "build_model"]
