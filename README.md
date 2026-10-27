# parton-recon

Reconstructing parton-level jet constituents from hadron-level constituents, using a graph neural network.

Data comes from Pythia 8. See `parton_recon/data/generate/`.

<<<<<<< Updated upstream
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


=======
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
=======
## Setup

Verified on macOS (Apple Silicon, MPS). Linux and ACCRE sections are written but not yet run on the hardware


### macOS (Apple Silicon)

```bash
brew install python@3.12
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
Training uses the GPU automatically via MPS (`torch.backends.mps.is_available()`).

### Linux with an NVIDIA GPU

Check the driver first: `nvidia-smi` shows a "CUDA Version" in the top right

```bash
python3.12 -m venv .venv
.venv/bin/pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/<tag>
.venv/bin/pip install -r requirements.txt
```
(`pip install -r requirements.txt` won't reinstall or downgrade the torch build you just chose — pip
matches `torch==2.14.0` regardless of its `+cu126`/`+cu130` local label.)

### Linux, CPU only

Same as above, but with `--index-url https://download.pytorch.org/whl/cpu` — this skips several GB of
`nvidia-*` wheels that a GPU build would otherwise pull in.



### Verify the install

```bash
.venv/bin/python -c "
import torch, torch_geometric, numpy
print('torch', torch.__version__, '| numpy', numpy.__version__)
print('MPS:', torch.backends.mps.is_available(), '| CUDA:', torch.cuda.is_available())
"
```

EMD toy check (also confirms the EMD backend actually works — the OpenMP build of `wasserstein` has
been observed to fail to load on macOS even with Homebrew's `libomp` installed; disabling OpenMP is a
one-line workaround, not a full fix — see `notebooks/03_emd_metric.md`). Going through
`parton_recon.metrics.emd` applies this automatically; the raw form below is only to verify the
environment directly:

```python
import wasserstein; wasserstein.without_openmp()
import numpy as np, energyflow
x = np.array([[100.0, 0.0, 0.0]])
y = np.array([[100.0, 0.0, 0.1]])
assert abs(energyflow.emd.emd(x, x, R=1.0)) < 1e-9
assert abs(energyflow.emd.emd(x, y, R=1.0) - 10.0) < 1e-6
```

### Running training

```bash
.venv/bin/python scripts/train.py
```
Paths inside the script are built from `__file__`, so it can be run from anywhere. Trains in
mini-batches (default `--batch-size 32`) so it scales past whatever fits in one GPU batch; run
`--help` for the full list of flags (data paths, `--max-events` for a quick local subset, epochs,
batch size, lr, val split, seed). Output goes to `checkpoints/gnn_chamfer_<timestamp>/`
(gitignored): `train_log.csv`, `checkpoint_best.pt`, `checkpoint_final.pt`, `config.json`.

Epoch-1 loss on the current 65-event dataset, seed 0, is `train ~5.50 val ~5.19` — no longer
bit-exact across runs, since training now shuffles batches every epoch (`--batch-size >= 52`
reproduces the old single-batch-per-epoch behavior but still won't match exactly, because shuffling
still runs). GPU runs may drift further after epoch 1 (scatter aggregation on CUDA is not
deterministic).

### Regenerating the data (optional, not portable yet)

Needs your own Pythia 8.317 build.
arm64 only, `generate_data.sh` hard-codes a local path to `pythia8-config`, generate with parton-recon/data/generate/generate_data.sh
Note: Make sure pythia is downloaded and generate_data.sh has the right address to the directory.



>>>>>>> Stashed changes
