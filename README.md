# parton-recon

Reconstructing parton-level jet constituents from hadron-level constituents, using a graph neural
network.

Data comes from Pythia 8. See `parton_recon/data/generate/`.

**Oct 2026 updates (branch `rke/photon-veto-factorized`):** reproduction of the EMD/PCA plots on ACCRE,
the photon-veto fix in the generator, the factorized and attention models, full-sample training, and a
Sherpa hadronization cross-test. What changed, why, and step-by-step instructions are in
[`repro/README.md`](repro/README.md).

## Status

**Skeleton and a few scripts.** Directories exist with a stated purpose; few code, basic environment, few data for experimentation.

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

