# Reproduction on ACCRE and model updates (Oct 2026)

Branch `rke/photon-veto-factorized`. Sujal: start with the summary and the step-by-step guide right below. Sections 1–5 have the details and the numbers behind each change.

Your original `JetGNN`, `matched_mse_loss`, `train.py` and `train2.py` are unchanged and still work. Everything new sits alongside them.

## What changed, in one list

1. **Reproduced your two plots** (`emd_distribution.png`, `pca_corner.png`) from scratch: the Chamfer + LogPt model. See §1.
2. **Generator fix in `main214.cc`.** About 55% of the "jets" were the prompt photon. The photon is now vetoed, and the hadron jet is matched to the parton jet in ΔR. See §2.
3. **The EMD was mostly an energy mismatch.** Report the shape-only EMD and ΣpT_pred / ΣpT_true instead of ln(full EMD) alone. See §2.
4. **New models:**
   - `FactorizedJetGNN` predicts pT fractions plus a total pT (§3).
   - `AttentionJetGNN` adds a cross-attention slot decoder and jet-axis-relative coordinates (§4). This is the current best model.
5. **Full sample:** a compact `.npz` data format (19 GB of JSON becomes about 40 MB), and a fixed held-out test set. See §4.
6. **Loss options:** `--mass-weight` (jet-mass penalty) and `--angle-weight pt`. See §4.
7. **Hadronization cross-test** with Sherpa: cluster vs Lund string with the same shower. The models, with or without the mass penalty, show no measurable hadronization dependence at this jet pT. See §5.

## Where things are

| Path | What it is |
|---|---|
| `parton_recon/data/generate/main214.cc` | Pythia generator. Arguments: `[nEvent] [seed] [outPrefix] [vetoAndMatch]` (§2) |
| `parton_recon/data/dataset.py` | `JetPairDataset`: reads your JSON, or the new `.npz` |
| `parton_recon/models/factorized.py`, `attention.py` | `FactorizedJetGNN`, `AttentionJetGNN`; `models/__init__.py` has `build_model()` and `predict_batched()` |
| `parton_recon/losses/matched_fraction.py` | Loss for both new models: shape + energy (+ mass) terms |
| `scripts/train_factorized.py` | Training for both new models. Run `--help` for all flags |
| `scripts/convert_to_arrays.py`, `scripts/merge_generated.py` | Per-seed JSON → `.npz`; per-seed JSON → one JSON |
| `repro/` | Oct-13 reproduction scripts, evaluation and plotting helpers, this README |
| `sherpa/` | Sherpa runcards, `sherpa_jets.cc` (HepMC3 → paired jets) and the job scripts (§5) |
| `slurm/` | SLURM job scripts. They use `--account=rke_group`; change it if you run under your own account |

## Step-by-step on ACCRE

Run everything from the repo root.

**0. Environment** (once). The Lmod `module load python` doesn't always work in batch shells, so build the venv from the LCG Python 3.12 on cvmfs:

```bash
P=/cvmfs/sft.cern.ch/lcg/releases/Python/3.12.11-531c6/x86_64-el9-gcc13-opt/bin/python3.12
env -i HOME=$HOME PATH=/usr/bin:/bin $P -m venv .venv
.venv/bin/pip install --upgrade pip && .venv/bin/pip install -r requirements.txt
```

Run Python as `env -i HOME=$HOME PATH=/usr/bin:/bin .venv/bin/python ...`. The `env -i` keeps the LCG and Lmod environments, which may be loaded in your shell for Pythia or Sherpa, out of the venv. The SLURM scripts already do this.

**1. Pythia data.**

```bash
bash parton_recon/data/generate/generate_data.sh   # builds 'generate' against LCG_108 Pythia 8.315
sbatch slurm/generate_veto_array.sbatch            # 100 jobs x 10k events, seeds 101-200 -> data/raw_veto/
sbatch slurm/convert.sbatch                        # -> data/veto_train.npz (seeds 101-190), data/veto_test.npz (191-200)
```

- About 6% of generated events pass the 800–820 GeV jet selection, so 1M events give about 59k jets.
- `slurm/generate_array.sbatch` makes the old (photon-included) sample for the reproduction.
- For a quick JSON subset, run `scripts/merge_generated.py data/raw_veto data/veto6725 --max-events 6725`.

**2. Train.** `slurm/train.sbatch <script> <data prefix or .npz> [flags]`:

```bash
# current best model, full sample, with the jet-mass penalty
sbatch -J full slurm/train.sbatch scripts/train_factorized.py data/veto_train.npz \
  --model attention --epochs 200 --patience 15 --workers 2 --mass-weight 0.03 --tag _full_m0.03
# quick run on 6725 jets
sbatch -J small slurm/train.sbatch scripts/train_factorized.py data/veto_train.npz \
  --model attention --max-events 6725 --epochs 300 --patience 40 --tag _small
```

- Each run writes `checkpoints/<name>_<timestamp>/`, holding `checkpoint_best.pt`, `checkpoint_last.pt`, `train_log.csv`, `run_args.json`, and `config.json` (written at the end).
- If a run hits the 48 h wall-clock limit, continue it with `--resume checkpoints/<run_dir>`. All other settings come from its `run_args.json`, and the continued run is bit-identical to an uninterrupted one.
- Speed on 8 CPUs: the attention model takes about 2 min/epoch on 6.7k jets and about 12 min/epoch on 53.9k. Our group has no GPU partition access.

**3. Evaluate on the held-out test set.**

```bash
.venv/bin/python repro/evaluate_test.py data/veto_test.npz results/my_table \
  "baseline=checkpoints/<run A>" "mass 0.03=checkpoints/<run B>"     # table + overlay plot
.venv/bin/python repro/jet_mass_plot.py checkpoints/<run>/checkpoint_best.pt data/veto_test.npz --all-events
```

- `evaluate_test.py` writes `<out>.txt`: median ln(EMD), fraction ≤ 4, shape EMD, ΣpT ratio, median Δm/m with its 16–84% band, and mass correlation. It also writes `<out>.png` with the four distributions.
- `repro/energy_check.py` prints the energy/shape split of the EMD for one checkpoint.
- `repro/make_plots.sh` makes your original EMD histogram and PCA corner plot.
- `repro/postprocess_runs.sh N` waits for runs to finish and does all of this on each run's validation split.

**4. Sherpa cross-hadronization samples** (§5):

```bash
bash sherpa/build.sh                        # builds sherpa/sherpa_jets
bash sherpa/integrate.sh                    # one-off matrix-element integration (~30 s)
sbatch sherpa/generate_array.sbatch         # 100 Ahadic + 100 Lund jobs x 10k events -> data/raw_sherpa/
bash sherpa/after_generation.sh             # .npz per model + sanity table + the 4 cross-test trainings
bash sherpa/evaluate_crosstest.sh           # score the 4 models on the Ahadic, Lund and Pythia test sets
```

## 1. Reproduction

Both PNGs at the repo root (Oct 13) came from the **Chamfer + LogPt** model: 1000 epochs, batch size 32, seed 0, on 6725 events. That gives 1345 validation events, which matches the plot title. The training and evaluation scripts from that commit (`6648361`) are kept as `repro/train_oct13.py` and `repro/evaluate_oct13.py`.

I regenerated the data with `main214.cc` in legacy mode (`./generate 10000 <seed> <prefix> 0`), seeds 1–100, Pythia 8.315 from LCG_108. The first 6725 events form the training set (`data/repro6725_*`).

| | Original (Oct 13) | This reproduction |
|---|---|---|
| ln(EMD) peak | ~9.7, ~740 events | ~9.7, ~740 events |
| good / fair / bad | ~0 / ~1 / ~99 % | 0 / 1 / 99 % |
| PCA variance, PC1–PC5 | 50/21/17/5/4 % | 47/21/17/7/5 % |

The reproduced plots are `figures/emd_distribution_repro.png` and `figures/pca_corner_repro.png`.

## 2. What the reproduction showed

**About 55% of the "jets" were the prompt photon.** The process is `qg → qγ`, and the photon was clustered like any other particle. So in more than half the events the leading jet was the photon plus some soft activity, and a single particle carried more than 95% of the jet pT at both levels. About 3% of events also took their hadron and parton jets from opposite sides of the event.

The fix in `main214.cc` is now the default (`vetoAndMatch = 1`):

- The hard-process photon is dropped from both levels before jet finding. Photons from π⁰ decays and from shower radiation are kept.
- The hadron jet is the one closest in ΔR (< 0.4) to the leading parton jet, rather than simply the leading hadron jet. The hadron jet-finder threshold is 600 GeV for this, so a matched jet that hadronization pushes below 800 GeV is still found.
- Legacy mode (`0`) reproduces the old output exactly, event for event.

After the veto, no jets are photon-dominated and none come from the wrong side of the event. The median multiplicity is 15 partons and 51 hadrons, against 7 and 19 before. 14% of jets now have more than 24 partons.

**The bad EMD values were almost entirely an energy mismatch.** `energyflow.emd.emd` with `norm=False` adds |ΣpT_pred − ΣpT_true| in GeV. The Chamfer loss puts no constraint on the total, so several slots copy the ~800 GeV leading parton. The predicted ΣpT comes out at about 15,700 GeV against a true 811, and that difference is 100% of the EMD.

`repro/energy_check.py` splits the EMD into two parts:

- **Shape-only EMD** (`norm=True`): both sets are normalised to Σ = 1. The result is the pT-weighted average ΔR that the jet's pT fractions have to move.
- **Predicted vs true ΣpT**: how far off the total energy is.

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

- **The matched loss fixes most of the energy over-count.**
- **Chamfer gets the best shape score**, but only because the shape EMD ignores how many particles there are and how much total energy they carry.
- **Matched + veto under-predicts ΣpT** by about 25%, probably because the regression is in log pT.
- **Both matched runs overfit early.** Their best validation loss is at epochs 185 and 100, with 5380 training events.

## 3. Factorized model (`models/factorized.py`, `losses/matched_fraction.py`)

The model writes each parton's pT as pT_i = z_i × pT_total:

- **z_i:** a softmax over the slots, so the fractions sum to exactly 1. A slot that shouldn't exist goes to z ≈ 0, which replaces the 1 GeV padding target.
- **pT_total:** (Σ hadron pT) × exp(δ), where the network learns δ. In this sample the parton/hadron ΣpT ratio has a median of 1.000 with a 16–84% range of 0.996–1.004. So δ starts at 0 and only has to learn a small correction.

`forward()` still returns (log pT, η, φ) for each slot, so every evaluation script works on it unchanged. The loss uses `model.parts(batch)` instead.

The loss is `matched_mse_loss` applied to the shape: a Hungarian match on (√z, η, φ), with spare slots pushed to z = 0. The energy term is (δ − δ_true)².

- √z sits between z, where errors are dominated by the hardest parton, and log z, where they are dominated by the softest.
- Checkpoints store `arch_config["model"]`, and the loaders use `build_model()`. Older checkpoints load as `JetGNN`.

## 4. Attention decoder, full sample, mass penalty

**Attention decoder** (`models/attention.py`, `--model attention`). FactorizedJetGNN builds every slot from one pooled vector and predicts an "average jet": its predicted mass has no tail above ~120 GeV, although the hadron jet mass correlates 0.98 with the parton one. `AttentionJetGNN` has 735k parameters:

- 3 multi-head GATv2 layers, 128 wide.
- 3 transformer-decoder layers whose slot queries cross-attend to the hadron nodes.
- With `relative=True` (the default), η and φ go in and come out as offsets from the pT-weighted hadron jet axis.

The output has the same fraction × total-pT form as FactorizedJetGNN. Evaluation goes through `predict_batched()`, because a single forward pass over thousands of jets runs out of memory on the login node.

**Full sample** (`scripts/convert_to_arrays.py`). The 100 vetoed seeds are 19 GB of JSON, almost all of it edge lists. The `.npz` keeps only the node features, about 40 MB, and `JetPairDataset` rebuilds the fully connected edges and their ΔR on access. These match the stored JSON edges to within 3e-7. Every script that takes `--hadron-json` accepts the `.npz`.

- `data/veto_train.npz`: seeds 101–190, 53.9k jets.
- `data/veto_test.npz`: seeds 191–200, 5985 jets. No run trains on these.

**Loss options** (`matched_fraction_loss`; the defaults reproduce the earlier loss):

- `--angle-weight pt` weights each parton's η/φ error by its pT fraction, as the EMD does.
- `--mass-weight λ` adds λ·(log m_pred − log m_true)².

**Test-set results** (5985 held-out jets, best checkpoints; tables in `results/test_*.txt`, mass plots in `results/jet_mass/`):

| Attention, jet-axis coords | train jets | ln(EMD) | good | shape EMD | median Δm/m (16–84%) | mass corr |
|---|---|---|---|---|---|---|
| current loss | 6.7k | 3.11 | 89% | 0.025 | −23% (−40, −6) | 0.952 |
| + pT-weighted angles | 6.7k | 3.12 | 93% | 0.025 | −37% (−60, −15) | 0.959 |
| + mass 0.03 | 6.7k | 3.36 | 87% | 0.033 | +0.1% (−15, +17) | 0.968 |
| + pT-weighted + mass 0.03 | 6.7k | 3.30 | 90% | 0.031 | −6% (−21, +11) | 0.962 |
| current loss | 53.9k | 3.14 | 95% | 0.024 | −18% (−42, −2) | 0.969 |

For comparison on the same test jets, FactorizedJetGNN K = 24 gets 4.15 / 42% / 0.074 / −33% / 0.71.

- **Predicted jets are too narrow.** Without the mass term, every model comes out too narrow. A squared-error fit puts a soft parton whose position is uncertain near its expected position, which is close to the jet axis.
- **Slot truncation is not the cause.** The 24 hardest partons carry more than 99.5% of the true mass.
- **More data doesn't fix it either.** Going from 6.7k to 53.9k training jets improves the shape but barely moves the mass.
- **The mass term removes the bias, at a small cost in per-jet EMD.** A deterministic per-jet prediction can't be optimal for both; a generative model that samples parton jets could be.
- **pT-weighting the angles makes the mass worse.**

## 5. Hadronization cross-test with Sherpa

**Why.** The mass penalty pulls predictions toward the generator's parton-level mass. Every supervised model inherits the generator's hadron → parton mapping, but the penalty might lean on it more. Since the goal is to apply this to data, we test it directly: train on one hadronization model and test on another.

**How.** Sherpa 3.0.1, with the same CSS shower and two hadronization models: `Ahadic` (cluster) and Pythia 8's Lund string (`FRAGMENTATION: Pythia8`). With the shower fixed, any difference comes from hadronization alone. Herwig vs Pythia would change the shower too, and with it the parton-level truth.

- **Runcards.** `sherpa/ahadic.yaml` and `sherpa/lund.yaml` differ only in `FRAGMENTATION`. Both run qg → qγ at 14 TeV with γ pT > 700 GeV and MPI on, as in `main214.cc`.
- **`sherpa/sherpa_jets.cc`** reads the HepMC3 events through a FIFO, so no event files are written.
  - **Parton level** is the set of particles entering the hadronization vertex: the vertex whose inputs are all partons and whose outputs include hadrons. That's the same content as Pythia's `isFinalPartonLevel()`: shower, MPI and beam-remnant partons. In Lund mode, Sherpa adds a second hadron → hadron bookkeeping vertex, and the code skips it.
  - **Selection** is the same as `main214.cc` with the photon veto and ΔR matching. The output is text, which `sherpa/to_npz.py` turns into the `.npz` format.
- **Samples.** Seeds 1–90 are for training and seeds 91–100 are held out for testing, per model. About 5.5% of events pass, so 1M events give about 55k jets.
- **Trainings** (`sherpa/after_generation.sh`): {Ahadic, Lund} × {current loss, mass 0.03}, each on 6725 jets. Each is tested on the Ahadic, Lund and Pythia test sets.
- **The key number** is how much each model degrades on the other hadronization model, with and without the mass penalty.

**Generation.**

- All 100 Ahadic jobs completed. 40 of the 100 Lund jobs crashed (34 killed, 2 segfaults, 4 fatal errors), roughly once per 20k events in the Sherpa → Pythia 8 string interface. Memory was normal (about 100 MB).
- Failed jobs are dropped whole, so the only events missing are the crash-triggering ones, about 5 in 100k.
- The Lund sample is 30.1k training jets (54 seeds) and 3359 test jets (6 seeds). The Ahadic sample is 49.6k training and 5527 test jets.

**Sanity check** (test sets; median multiplicity, median and mean jet mass in GeV, median leading-constituent pT fraction):

| sample | level | mult | mass med | mass mean | lead z |
|---|---|---|---|---|---|
| Sherpa Ahadic | parton | 8 | 75.0 | 92.1 | 0.635 |
| Sherpa Lund | parton | 8 | 73.7 | 92.9 | 0.627 |
| Sherpa Ahadic | hadron | 50 | 80.4 | 98.5 | 0.243 |
| Sherpa Lund | hadron | 47 | 77.8 | 98.7 | 0.250 |
| Pythia (main214) | parton | 15 | 73.3 | 91.2 | 0.518 |
| Pythia (main214) | hadron | 51 | 76.8 | 94.6 | 0.237 |

- **The two Sherpa samples agree at parton level** within statistics, as they should with one shower, and differ only at hadron level.
- **Sherpa's parton level has about half Pythia's multiplicity** and a harder leading parton, because of a different shower cutoff. So the Pythia test set below changes the parton-level definition, not just the hadronization.

**Results** (attention model, jet-axis coordinates, 6725 training jets; median ln(EMD) / shape EMD / median Δm/m with 16–84% band; full tables in `results/crosstest_*.txt`, mass plots in `results/crosstest_mass/`):

| trained on | loss | tested on Ahadic | tested on Lund | tested on Pythia |
|---|---|---|---|---|
| Ahadic | current | 3.31 / 0.030 / −40% (−61, −14) | 3.30 / 0.030 / −40% (−63, −15) | 3.29 / 0.030 / −45% (−64, −19) |
| Ahadic | + mass 0.03 | 3.90 / 0.057 / −9% (−28, +14) | 3.90 / 0.058 / −9% (−28, +15) | 3.87 / 0.056 / −12% (−31, +6) |
| Lund | current | 3.30 / 0.030 / −39% (−64, −15) | 3.29 / 0.030 / −41% (−65, −16) | 3.28 / 0.031 / −45% (−67, −20) |
| Lund | + mass 0.03 | 3.56 / 0.040 / +4% (−19, +30) | 3.54 / 0.039 / +4% (−18, +28) | 3.47 / 0.037 / +1% (−21, +20) |

- **No hadronization dependence at this pT.** Every model scores the same on the cluster and string test sets: Δ ln(EMD) ≤ 0.02 and Δ(Δm/m) ≤ 1%. A model trained on one hadronization model does just as well on the other, with or without the mass penalty. So the penalty does not tie the model to its training sample's hadronization.
- **This is expected at 800 GeV.** The hadronization shift of the jet mass is small here: the hadron-level masses differ by about 3%, about 2.6 GeV. The cross-test should be repeated at lower jet pT, where hadronization corrections grow.
- **The two mass-penalty models differ from each other** (−9% vs +4%), but each one is the same on both test sets. So the difference is run-to-run training variation, not hadronization. The mass-penalty runs stopped early (epochs 66 and 38) and are less stable than the current-loss runs (epochs 93 and 150). Several seeds per setting would pin down that spread.
- **On the Pythia test set**, the mass bias moves by 3–5 points for the current loss. That's mostly the different parton-level definition (shower cutoff), not hadronization.

**Full sample + mass penalty** (Pythia, 53.9k training jets, Pythia test set; `results/test_full_mass.txt`, `results/jet_mass/full_54k_mass003.png`):

| model | ln(EMD) | good | shape EMD | Δm/m (16–84%) | mass corr |
|---|---|---|---|---|---|
| 53.9k, current loss | 3.14 | 95% | 0.024 | −18% (−42, −2) | 0.969 |
| 53.9k + mass 0.03 | 3.24 | 91% | 0.029 | −1.5% (−15, +14) | 0.971 |

The full sample shrinks the cost of the mass penalty: shape EMD rises from 0.024 to 0.029 here, against 0.025 → 0.033 on 6.7k jets. That makes it the current best model overall.

## Other changes

- `metrics/emd.py`: `wasserstein.without_openmp()` now runs on macOS only. The Linux wheels don't ship the non-OpenMP build, so the call crashed.
- The `generate` binary in git is still your old one. Rebuild it per machine with `generate_data.sh`; the LCG build only runs where cvmfs is mounted.
