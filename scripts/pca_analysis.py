"""PCA over per-event summary statistics of the hadron- and parton-level jets.

Each event becomes one row per level: multiplicity, pT mean/std, eta mean/std,
circular std of phi, and vector-summed jet pT. Features are standardized (zero
mean, unit variance) then projected onto the top principal components to see
whether hadron and parton jets separate in shape.

Run from the repo root:  .jupyter_venv/bin/python3 scripts/pca_analysis.py

generated with claude sonnet 5 high
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATED = REPO_ROOT / "parton_recon" / "data" / "generate"
OUT_PNG = Path(__file__).resolve().parent / "pca_analysis.png"

FEATURE_NAMES = [
    "multiplicity", "pt_mean", "pt_std", "eta_mean", "eta_std", "phi_circular_std", "jet_pt",
]


def summarize_event(node_features):
    arr = np.asarray(node_features, dtype=float)
    pt, eta, phi = arr[:, 0], arr[:, 1], arr[:, 2]
    px, py = pt * np.cos(phi), pt * np.sin(phi)
    jet_pt = np.hypot(px.sum(), py.sum())
    resultant = np.hypot(np.cos(phi).mean(), np.sin(phi).mean())
    phi_circular_std = np.sqrt(-2 * np.log(resultant)) if resultant > 0 else 0.0
    return [len(pt), pt.mean(), pt.std(), eta.mean(), eta.std(), phi_circular_std, jet_pt]


def load_summaries(path):
    events = json.loads(Path(path).read_text())
    return np.array([summarize_event(e["node_features"]) for e in events])


def main():
    hadron = load_summaries(GENERATED / "model1_hadron_level_graphs_.json")
    parton = load_summaries(GENERATED / "model1_parton_level_graphs_.json")
    if len(hadron) != len(parton):
        raise ValueError(
            f"event count mismatch: {len(hadron)} hadron rows vs {len(parton)} parton rows — "
            "regenerate the data before running PCA"
        )

    X = np.vstack([hadron, parton])
    labels = np.array(["hadron"] * len(hadron) + ["parton"] * len(parton))

    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA()
    components = pca.fit_transform(X_scaled)

    print("Explained variance ratio:")
    for i, ratio in enumerate(pca.explained_variance_ratio_, start=1):
        print(f"  PC{i}: {ratio:.3f}")

    print("\nPC1 / PC2 loadings:")
    for name, pc1, pc2 in zip(FEATURE_NAMES, pca.components_[0], pca.components_[1]):
        print(f"  {name:>16}: PC1={pc1:+.3f}  PC2={pc2:+.3f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    for level, marker in [("hadron", "o"), ("parton", "^")]:
        mask = labels == level
        ax.scatter(components[mask, 0], components[mask, 1], s=8, alpha=0.4, label=level, marker=marker)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nSaved scatter plot to {OUT_PNG}")


if __name__ == "__main__":
    main()
