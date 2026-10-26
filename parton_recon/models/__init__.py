"""Encoder, decoder, and baseline architectures."""

import torch
from torch_geometric.data import Batch

from parton_recon.models.attention import AttentionJetGNN
from parton_recon.models.factorized import FactorizedJetGNN
from parton_recon.models.gnn import JetGNN

MODELS = {"JetGNN": JetGNN, "FactorizedJetGNN": FactorizedJetGNN, "AttentionJetGNN": AttentionJetGNN}


def build_model(arch_config):
    """Instantiate from a checkpoint's arch_config. Checkpoints written before the "model"
    key existed are all JetGNN."""
    arch = dict(arch_config)
    return MODELS[arch.pop("model", "JetGNN")](**arch)


def predict_batched(model, batch, chunk=128):
    """model(batch) evaluated `chunk` events at a time, concatenated along the event dim.

    The evaluation scripts load the whole val split as one batch. That is fine for JetGNN,
    but AttentionJetGNN on ~3.5M edges at once runs out of memory on a login node.
    """
    if batch.num_graphs <= chunk:
        return model(batch)
    events = batch.to_data_list()
    return torch.cat([model(Batch.from_data_list(events[i:i + chunk]))
                      for i in range(0, len(events), chunk)])


__all__ = ["JetGNN", "FactorizedJetGNN", "AttentionJetGNN", "build_model", "predict_batched"]
