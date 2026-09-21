# checkpoints

Trained models. **Gitignored.**

Every checkpoint must save the architecture config alongside the weights, plus the optimizer state
and epoch for resuming after an ACCRE wall-clock kill.

The previous project's `.pth` files are bare `state_dict`s — weights and nothing else. Loading one
meant retyping the model class by hand and inferring dimensions from tensor shapes, and since the
class had drifted between two notebooks there was no authoritative version to retype. Combined with
the missing training script, none of those seven checkpoints can be reproduced or reliably
interpreted. That is the failure this directory's convention exists to prevent.

Suggested layout: one directory per run, named for its config and timestamp, containing the
checkpoint, a copy of the config, and the training log.
