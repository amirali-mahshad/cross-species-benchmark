# benchmark/intra_metrics.py

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from benchmark.metrics import pcr_variance_explained


def _valid_mask(X_pre, X_post, *arrays):
    mask = np.all(np.isfinite(X_pre), axis=1)
    mask &= np.all(np.isfinite(X_post), axis=1)

    for arr in arrays:
        s = pd.Series(arr)
        mask &= ~s.isna().to_numpy()
        mask &= s.astype(str).to_numpy() != "nan"

    return mask


def _knn_edge_set(X: np.ndarray, n_neighbors: int = 20) -> set[tuple[int, int]]:
    X = np.asarray(X)

    if X.shape[0] <= 1:
        return set()

    k = min(n_neighbors, X.shape[0] - 1)

    nn = NearestNeighbors(
        n_neighbors=k + 1,
        metric="euclidean",
        n_jobs=1,
    )

    nn.fit(X)
    idx = nn.kneighbors(X, return_distance=False)

    edges = set()

    for i, row in enumerate(idx):
        for j in row:
            if i == j:
                continue

            a = min(i, int(j))
            b = max(i, int(j))
            edges.add((a, b))

    return edges


def jaccard_knn_edge_overlap(
    X_pre: np.ndarray,
    X_post: np.ndarray,
    n_neighbors: int = 20,
) -> float:
    edges_pre = _knn_edge_set(X_pre, n_neighbors=n_neighbors)
    edges_post = _knn_edge_set(X_post, n_neighbors=n_neighbors)

    if len(edges_pre) == 0 and len(edges_post) == 0:
        return float("nan")

    union = edges_pre | edges_post
    inter = edges_pre & edges_post

    if len(union) == 0:
        return float("nan")

    return float(len(inter) / len(union))


def compute_intra_celltype_jaccard(
    X_pre: np.ndarray,
    X_post: np.ndarray,
    species: np.ndarray,
    cell_type: np.ndarray,
    n_neighbors: int = 20,
    min_cells_per_group: int = 30,
):
    """
    Jaccard kNN preservation inside species × cell_type groups.

    This is the important intra-cell-type metric:
        pre-integration local structure
        vs
        post-integration local structure
    """

    species = np.asarray(species).astype(str)
    cell_type = np.asarray(cell_type).astype(str)

    mask = _valid_mask(X_pre, X_post, species, cell_type)

    X_pre = X_pre[mask]
    X_post = X_post[mask]
    species = species[mask]
    cell_type = cell_type[mask]

    details = []

    for sp in sorted(np.unique(species)):
        for ct in sorted(np.unique(cell_type)):
            idx = (species == sp) & (cell_type == ct)

            n = int(idx.sum())

            if n < min_cells_per_group:
                continue

            k_eff = min(n_neighbors, n - 1)

            if k_eff < 2:
                continue

            try:
                score = jaccard_knn_edge_overlap(
                    X_pre=X_pre[idx],
                    X_post=X_post[idx],
                    n_neighbors=k_eff,
                )

                details.append(
                    {
                        "species": sp,
                        "cell_type": ct,
                        "n_cells": n,
                        "n_neighbors_used": k_eff,
                        "Jaccard_intra_celltype": score,
                    }
                )

            except Exception as e:
                warnings.warn(f"Jaccard failed for {sp} / {ct}: {e}")

    detail_df = pd.DataFrame(details)

    if detail_df.empty:
        summary = {
            "Jaccard_intra_celltype_mean": float("nan"),
            "Jaccard_intra_celltype_weighted_mean": float("nan"),
            "Jaccard_intra_celltype_std": float("nan"),
            "Jaccard_intra_celltype_n_groups": 0,
        }

        return summary, detail_df

    values = detail_df["Jaccard_intra_celltype"].astype(float)
    weights = detail_df["n_cells"].astype(float)

    summary = {
        "Jaccard_intra_celltype_mean": float(values.mean()),
        "Jaccard_intra_celltype_weighted_mean": float(np.average(values, weights=weights)),
        "Jaccard_intra_celltype_std": float(values.std(ddof=0)),
        "Jaccard_intra_celltype_n_groups": int(detail_df.shape[0]),
    }

    return summary, detail_df


def compute_pcr_comparison(
    X_pre: np.ndarray,
    X_post: np.ndarray,
    covariate: np.ndarray,
    covariate_name: str,
    n_components: int = 50,
    random_state: int = 0,
):
    """
    PCR before-vs-after comparison.

    Outputs:
        raw variance explained before integration
        raw variance explained after integration
        removed fraction = (pre - post) / pre
        retained fraction = post / pre

    For species in cross-species data:
        do not interpret removed_fraction as automatically good.
    """

    covariate = np.asarray(covariate).astype(str)

    mask = _valid_mask(X_pre, X_post, covariate)

    X_pre = X_pre[mask]
    X_post = X_post[mask]
    covariate = covariate[mask]

    pre = pcr_variance_explained(
        X=X_pre,
        covariate=covariate,
        n_components=n_components,
        random_state=random_state,
    )

    post = pcr_variance_explained(
        X=X_post,
        covariate=covariate,
        n_components=n_components,
        random_state=random_state,
    )

    if pd.isna(pre) or pre <= 1e-12 or pd.isna(post):
        removed = float("nan")
        retained = float("nan")
    else:
        removed = float((pre - post) / pre)
        retained = float(post / pre)

    return {
        f"PCR_comparison_{covariate_name}_pre_raw": float(pre),
        f"PCR_comparison_{covariate_name}_post_raw": float(post),
        f"PCR_comparison_{covariate_name}_removed_fraction": removed,
        f"PCR_comparison_{covariate_name}_retained_fraction": retained,
    }


def compute_intra_metrics(
    X_pre: np.ndarray,
    X_post: np.ndarray,
    species: np.ndarray,
    cell_type: np.ndarray,
    sample: np.ndarray | None = None,
    n_neighbors: int = 20,
    min_cells_per_group: int = 30,
    n_pcr_components: int = 50,
    random_state: int = 0,
):
    """
    Main function for post-hoc intra-cell-type metrics.
    """

    metrics = {}

    jaccard_summary, jaccard_detail = compute_intra_celltype_jaccard(
        X_pre=X_pre,
        X_post=X_post,
        species=species,
        cell_type=cell_type,
        n_neighbors=n_neighbors,
        min_cells_per_group=min_cells_per_group,
    )

    metrics.update(jaccard_summary)

    metrics.update(
        compute_pcr_comparison(
            X_pre=X_pre,
            X_post=X_post,
            covariate=species,
            covariate_name="species",
            n_components=n_pcr_components,
            random_state=random_state,
        )
    )

    if sample is not None:
        metrics.update(
            compute_pcr_comparison(
                X_pre=X_pre,
                X_post=X_post,
                covariate=sample,
                covariate_name="sample",
                n_components=n_pcr_components,
                random_state=random_state,
            )
        )

    # Optional placeholder. True PCR-comparison-cell is not essential for your first benchmark.
    metrics["PCR_comparison_cell_optional"] = float("nan")

    return pd.DataFrame([metrics]), jaccard_detail
