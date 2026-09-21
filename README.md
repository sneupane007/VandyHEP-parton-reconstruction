# parton-recon

Reconstructing parton-level jet constituents from hadron-level constituents, using a graph neural
network.

Data comes from Pythia 8 (`qg → qγ`, anti-kT R=0.8 jets). See `parton_recon/data/generate/`.

## Status

**Skeleton and a few scripts.** Directories exist with a stated purpose; few code, basic environment, few data for experimentation.
Direction is pending a conversation with Dr. Rithya — see "Open questions" below.

## Background

This replaces an earlier effort archived in `../Summary/`. That work is kept for reference but is
not built on, for these reasons:

- No training script exists for any of its seven checkpoints, so none of its results are
  reproducible. The only recoverable detail is that `StandardScaler` was fit on the first 8,000 rows
  with `shuffle=False`.
- The model architecture was defined inline in two separate notebooks and drifted between them
  (`fc5` outputs 61 in one, 60 in the other).
- Data was headerless CSV in two mutually incompatible column conventions with no schema. This
  caused a silent bug: `jetmass.ipynb` reads column 0 as a particle count when under that file's
  convention it is the leading pT, inflating reconstructed jet mass to 142 GeV against 63 GeV true.
- A second bug in `rotations.ipynb` writes the random-rotation scalers into the `*_hep_halfpi.pkl`
  filenames.


## Layout

| Path | Purpose |
|---|---|
| `parton_recon/` | All real code. Importable package. |
| `configs/` | YAML experiment configs |
| `notebooks/` | Exploration only: imports from the package |
| `scripts/` | SLURM/sbatch wrappers for ACCRE |
| `tests/` | Invariance and correctness checks |
| `data/` | Generated data (gitignored). Currently JSON; HDF5 migration deferred, see `parton_recon/data/README.md` |
| `checkpoints/` | Trained models (gitignored) |

Note: `data/` at the top level holds generated datasets and is gitignored. `parton_recon/data/` is
the Python module that reads and writes them. Different things, similar names.

## Open questions for the professor


1. **Is regression the right framing?** Hadronization is stochastic, so the parton→hadron map is a
   distribution rather than a function. An MSE-trained regressor targets the conditional mean, which
   may not be a physically valid configuration.
2. **Should EMD be the training objective**, or stay an evaluation metric as it was previously?
3. **Is the 800–820 GeV jet window a deliberate control**, or should it widen for generalisation?
4. **What counts as success?** No target metric was recorded anywhere in the prior work.


