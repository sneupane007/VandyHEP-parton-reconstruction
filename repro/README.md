# Reproduction on ACCRE and two fixes

Branch `rke/photon-veto-factorized`, Oct 2026. It has three parts:

1. A reproduction of `emd_distribution.png` and `pca_corner.png` from scratch.
2. Two problems in the data and in the metric that the reproduction showed up.
3. A model that predicts the jet's shape and its energy separately.

Your original `JetGNN`, `matched_mse_loss`, `train.py` and `train2.py` are unchanged.

## 1. Reproduction

Both PNGs at the repo root (Oct 13) came from the **Chamfer + LogPt** model, trained for 1000 epochs with batch size 32 and seed 0 on 6725 events. That gives 1345 validation events, which matches the plot title. The exact training and evaluation scripts from that commit (`6648361`) are kept here as `train_oct13.py` and `evaluate_oct13.py`.

I regenerated the data with `main214.cc` in legacy mode (`./generate 10000 <seed> <prefix> 0`), using seeds 1–100 and Pythia 8.315 from LCG_108. The first 6725 events form the training set (`data/repro6725_*`).

| | Original (Oct 13) | This reproduction |
|---|---|---|
| ln(EMD) peak | ~9.7, ~740 events | ~9.7, ~740 events |
| good / fair / bad | ~0 / ~1 / ~99 % | 0 / 1 / 99 % |
| PCA variance, PC1–PC5 | 50/21/17/5/4 % | 47/21/17/7/5 % |

The reproduced plots are `figures/emd_distribution_repro.png` and `figures/pca_corner_repro.png`.

## 2. What the reproduction showed

**About 55% of the "jets" were the prompt photon.** The process is `qg → qγ`, and the photon was clustered like any other particle, so in more than half the events the leading jet was the photon plus some soft activity. In those events a single particle carries more than 95% of the jet pT, at both parton and hadron level. About 3% of events also took their hadron and parton jets from opposite sides of the event.

The fix is in `main214.cc` and is now the default (`vetoAndMatch = 1`):

- The hard-process photon is dropped from both levels before jet finding. Photons from π⁰ decays and from shower radiation are kept.
- The hadron jet is now the one closest in ΔR (< 0.4) to the leading parton jet, rather than simply the leading hadron jet. For this, the hadron jet-finder threshold is 600 GeV, so a matched jet that hadronization pushes below 800 GeV is still found.
- Legacy mode (`0`) reproduces the old output exactly, event for event.

After the veto, no jets are photon-dominated and none come from the wrong side of the event. The median multiplicity is 15 partons and 51 hadrons, against 7 and 19 before. 14% of jets now have more than 24 partons.

**The bad EMD values were almost entirely an energy mismatch.** `energyflow.emd.emd` with `norm=False` adds |ΣpT_pred − ΣpT_true| in GeV. The Chamfer loss puts no constraint on the total, so several slots copy the ~800 GeV leading parton. The predicted ΣpT comes out at about 15,700 GeV against a true 811, and that difference is 100% of the EMD.

`energy_check.py` splits the EMD into its two parts:

- **shape-only EMD** (`norm=True`): both sets are normalised to Σ = 1. The result is the pT-weighted average ΔR that the jet's pT fractions have to move.
- **predicted vs true ΣpT**: how far off the total energy is.

Report these two numbers instead of ln(full EMD). The good ≤ 4 and fair ≤ 5.5 cuts depend on the GeV scale and on the jet pT.

### 2×2 comparison

All runs use 1345 validation events and the best checkpoint (`figures/compare_4runs.png`).

| Run | median ln(EMD) | pred ΣpT (true ≈ 812) | shape EMD |
|---|---|---|---|
| Chamfer, original data | 9.61 | 15739 GeV | 0.026 |
| Chamfer, photon veto | 7.16 | 2075 GeV | 0.051 |
| Matched, original data | 5.42 | 873 GeV | 0.133 |
| Matched, photon veto | 5.62 | 585 GeV | 0.090 |

What the table shows:

- The matched loss fixes most of the energy over-count.
- Chamfer gets the best shape score, but only because the shape EMD ignores how many particles there are and how much total energy they carry.
- Matched + veto under-predicts ΣpT by about 25%, probably because the regression is in log pT.
- Both matched runs overfit early: their best validation loss is at epochs 185 and 100, with 5380 training events.

## 3. Factorized model (`models/factorized.py`, `losses/matched_fraction.py`)

The model writes each parton's pT as pT_i = z_i × pT_total:

- **z_i:** a softmax over the slots, so the fractions sum to exactly 1. A slot that shouldn't exist goes to z ≈ 0, which replaces the 1 GeV padding target.
- **pT_total:** (Σ hadron pT) × exp(δ), where the network learns δ. In this sample the parton/hadron ΣpT ratio has a median of 1.000 with a 16–84% range of 0.996–1.004. So δ starts at 0 and only has to learn the small correction.

`forward()` still returns (log pT, η, φ) for each slot, so every evaluation script works on it unchanged. The loss uses `model.parts(batch)` instead.

The loss is `matched_mse_loss` applied to the shape: a Hungarian match on (√z, η, φ), with spare slots pushed to z = 0. The energy term is (δ − δ_true)².

- √z sits between z, where errors are dominated by the hardest parton, and log z, where they are dominated by the softest.
- Checkpoints now store `arch_config["model"]`, and the loaders use `build_model()`. Older checkpoints load as `JetGNN`.

Train it with `scripts/train_factorized.py --n-slots K`. The first runs (K = 24 and K = 32, 500 epochs, photon-vetoed data) are still running.

## Running it on ACCRE

```bash
# environment: Python 3.12 venv with requirements.txt (built from the LCG Python 3.12 on cvmfs)
bash parton_recon/data/generate/generate_data.sh         # build against LCG_108 Pythia
sbatch slurm/generate_veto_array.sbatch                  # 100 x 10k events -> data/raw_veto/
.venv/bin/python scripts/merge_generated.py data/raw_veto data/veto6725 --max-events 6725
sbatch -J fact_k24 slurm/train.sbatch scripts/train_factorized.py data/veto6725 --epochs 500
bash repro/make_plots.sh checkpoints/<run>/checkpoint_best.pt data/veto6725
.venv/bin/python repro/energy_check.py checkpoints/<run>/checkpoint_best.pt data/veto6725
```

`postprocess_runs.sh` polls `checkpoints/` and does the last two steps for every run that finishes. `compare_runs.py` overlays several runs on one figure.

## Other changes

- `metrics/emd.py`: `wasserstein.without_openmp()` now runs on macOS only. The Linux wheels don't ship the non-OpenMP build, so the call crashed.
- `main214.cc` takes `[nEvent] [seed] [outPrefix] [vetoAndMatch]` on the command line. Run with no arguments, it behaves as before, except that the veto is now on.
- The `generate` binary is still the old one in git. It needs rebuilding per machine with `generate_data.sh`.
