# data

Generated datasets. **Gitignored** — these are large and reproducible from the generator plus a
seed, so they do not belong in version control.

Expected contents: `events_*.h5` written by `scripts/generate.sh`.

Naming should encode what varies between runs, for example
`events_photon-incl_pt800-820_n100k_seed42.h5`.

Every file must carry its full generation config in its `meta/` group — Pythia settings, seed, jet
radius, pT window, whether the photon was included, whether the hadron-level cut was applied. A
dataset whose provenance is not recorded in the file itself cannot be trusted six months later.

Size is not a concern at this scale: roughly 4 fields × ~35 particles × 2 levels × 100k events comes
to about 100 MB.

Keep a small subset here (~2k events) for local development so the M4 can run the pipeline without
going through the ACCRE queue.
