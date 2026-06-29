# benchmark/metrics.py

import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    silhouette_samples,
)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import chisquare


def _to_numpy_labels(x):
    return np.asarray(pd.Series(x).astype(str))


def _valid_mask(*arrays):
    mask = np.ones(len(arrays[0]), dtype=bool)

    for arr in arrays:
        s = pd.Series(arr)
        mask &= s.notna().values
        mask &= ~s.astype(str).isin(["nan", "None", "NA", "NaN", "unknown"]).values

    return mask


def _pcr_score(X, batch):
    """
    PCR-like score.

    Measures how much embedding variance is explained by batch/species.
    Higher score = less batch/species effect.
    """

    batch = _to_numpy_labels(batch)
    mask = _valid_mask(batch)

    X = np.asarray(X)[mask]
    batch = batch[mask]

    if len(np.unique(batch)) < 2:
        return np.nan

    design = pd.get_dummies(batch, drop_first=True).values

    if design.shape[1] == 0:
        return np.nan

    X_centered = X - X.mean(axis=0, keepdims=True)
    dim_var = X_centered.var(axis=0)

    if dim_var.sum() == 0:
        return np.nan

    weights = dim_var / dim_var.sum()

    r2_values = []

    for j in range(X_centered.shape[1]):
        y = X_centered[:, j]

        if np.var(y) == 0:
            r2_values.append(0.0)
            continue

        reg = LinearRegression()
        reg.fit(design, y)
        r2 = reg.score(design, y)
        r2_values.append(max(0.0, r2))

    weighted_r2 = np.sum(np.asarray(r2_values) * weights)

    return float(1.0 - np.clip(weighted_r2, 0.0, 1.0))


def _batch_asw_score(X, batch, cell_type):
    """
    Batch ASW inside cell types.

    Higher score = better batch/species mixing inside each cell type.
    """

    X = np.asarray(X)
    batch = _to_numpy_labels(batch)
    cell_type = _to_numpy_labels(cell_type)

    mask = _valid_mask(batch, cell_type)

    X = X[mask]
    batch = batch[mask]
    cell_type = cell_type[mask]

    scores = []

    for ct in np.unique(cell_type):
        idx = cell_type == ct

        if idx.sum() < 3:
            continue

        batch_sub = batch[idx]

        if len(np.unique(batch_sub)) < 2:
            continue

        try:
            sil = silhouette_samples(X[idx], batch_sub, metric="euclidean")
            score = np.mean(1.0 - np.abs(sil))
            scores.append(score)
        except Exception:
            continue

    if len(scores) == 0:
        return np.nan

    return float(np.mean(scores))


def _cell_type_asw_score(X, cell_type):
    """
    Cell-type ASW.

    Higher score = better biological separation.
    """

    X = np.asarray(X)
    cell_type = _to_numpy_labels(cell_type)

    mask = _valid_mask(cell_type)

    X = X[mask]
    cell_type = cell_type[mask]

    if len(np.unique(cell_type)) < 2:
        return np.nan

    try:
        sil = silhouette_score(X, cell_type, metric="euclidean")
        return float((sil + 1.0) / 2.0)
    except Exception:
        return np.nan


def _graph_connectivity_score(X, cell_type, n_neighbors=20):
    """
    Graph connectivity by cell type.

    Higher score = same cell type forms connected regions.
    """

    X = np.asarray(X)
    cell_type = _to_numpy_labels(cell_type)

    mask = _valid_mask(cell_type)

    X = X[mask]
    cell_type = cell_type[mask]

    n_cells = X.shape[0]

    if n_cells <= 2:
        return np.nan

    k_eff = min(n_neighbors + 1, n_cells)

    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn.fit(X)

    indices = nn.kneighbors(X, return_distance=False)[:, 1:]

    rows = np.repeat(np.arange(n_cells), indices.shape[1])
    cols = indices.reshape(-1)
    data = np.ones(len(rows), dtype=np.float32)

    graph = csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
    graph = graph.maximum(graph.T)

    scores = []

    for ct in np.unique(cell_type):
        idx = np.where(cell_type == ct)[0]

        if len(idx) < 2:
            continue

        subgraph = graph[idx][:, idx]

        _, labels = connected_components(
            subgraph,
            directed=False,
            connection="weak",
        )

        largest_component = np.bincount(labels).max()
        scores.append(largest_component / len(idx))

    if len(scores) == 0:
        return np.nan

    return float(np.mean(scores))


def _kbet_like_score(
    X,
    batch,
    cell_type,
    n_neighbors=20,
    alpha=0.05,
    max_cells=5000,
    random_state=0,
):
    """
    Simplified kBET-like metric.

    Higher score = local neighborhoods have expected batch composition.

    This is not an exact scIB kBET implementation.
    Use it for internal comparison, not as a claim of exact kBET replication.
    """

    rng = np.random.default_rng(random_state)

    X = np.asarray(X)
    batch = _to_numpy_labels(batch)
    cell_type = _to_numpy_labels(cell_type)

    mask = _valid_mask(batch, cell_type)

    X = X[mask]
    batch = batch[mask]
    cell_type = cell_type[mask]

    n_cells = X.shape[0]

    if n_cells <= n_neighbors + 1:
        return np.nan

    if len(np.unique(batch)) < 2:
        return np.nan

    k_eff = min(n_neighbors + 1, n_cells)

    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn.fit(X)

    indices = nn.kneighbors(X, return_distance=False)[:, 1:]

    eval_indices = np.arange(n_cells)

    if n_cells > max_cells:
        eval_indices = rng.choice(eval_indices, size=max_cells, replace=False)

    all_batches = np.unique(batch)
    accepted = []

    for i in eval_indices:
        ct = cell_type[i]

        group_mask = cell_type == ct
        group_batches = batch[group_mask]

        if len(np.unique(group_batches)) < 2:
            continue

        expected_counts = np.array(
            [np.sum(group_batches == b) for b in all_batches],
            dtype=float,
        )

        expected_probs = expected_counts / expected_counts.sum()

        neighbor_batches = batch[indices[i]]

        observed = np.array(
            [np.sum(neighbor_batches == b) for b in all_batches],
            dtype=float,
        )

        expected = expected_probs * observed.sum()

        keep = expected > 0

        if keep.sum() < 2:
            continue

        observed_keep = observed[keep]
        expected_keep = expected[keep]

        expected_keep = expected_keep * observed_keep.sum() / expected_keep.sum()

        try:
            p_value = chisquare(
                f_obs=observed_keep,
                f_exp=expected_keep,
            ).pvalue

            accepted.append(p_value >= alpha)
        except Exception:
            continue

    if len(accepted) == 0:
        return np.nan

    return float(np.mean(accepted))


def _alignment_score(X, species, n_neighbors=20):
    """
    Cross-species alignment score.

    Higher score = more nearest neighbors from the opposite species.
    """

    X = np.asarray(X)
    species = _to_numpy_labels(species)

    mask = _valid_mask(species)

    X = X[mask]
    species = species[mask]

    n_cells = X.shape[0]

    if len(np.unique(species)) < 2:
        return np.nan

    k_eff = min(n_neighbors + 1, n_cells)

    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn.fit(X)

    indices = nn.kneighbors(X, return_distance=False)[:, 1:]

    scores = []

    for i in range(n_cells):
        other_species_neighbors = np.sum(species[indices[i]] != species[i])
        max_possible = min(n_neighbors, np.sum(species != species[i]))

        if max_possible == 0:
            continue

        scores.append(other_species_neighbors / max_possible)

    if len(scores) == 0:
        return np.nan

    return float(np.mean(scores))


def compute_integration_metrics(
    adata,
    embedding_key: str,
    cell_type_key: str,
    species_key: str,
    sample_key: str | None = None,
    n_neighbors: int = 20,
    leiden_resolution: float = 1.0,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Compute benchmark metrics for one embedding.

    Evaluation is always centered on species via species_key.
    If sample_key is provided, sample metrics are also computed.
    """

    if embedding_key not in adata.obsm:
        raise ValueError(f"{embedding_key} not found in adata.obsm.")

    for key in [cell_type_key, species_key]:
        if key not in adata.obs.columns:
            raise ValueError(f"{key} not found in adata.obs.")

    if sample_key is not None and sample_key not in adata.obs.columns:
        raise ValueError(f"{sample_key} not found in adata.obs.")

    X = np.asarray(adata.obsm[embedding_key])

    safe_key = embedding_key.replace("/", "_").replace(" ", "_")

    neighbors_key = f"neighbors_{safe_key}"
    leiden_key = f"leiden_{safe_key}"

    sc.pp.neighbors(
        adata,
        use_rep=embedding_key,
        n_neighbors=n_neighbors,
        key_added=neighbors_key,
        random_state=random_state,
    )

    sc.tl.leiden(
        adata,
        neighbors_key=neighbors_key,
        key_added=leiden_key,
        resolution=leiden_resolution,
        random_state=random_state,
    )

    cell_type = adata.obs[cell_type_key]
    species = adata.obs[species_key]
    clusters = adata.obs[leiden_key]

    valid = _valid_mask(cell_type, clusters)

    nmi = normalized_mutual_info_score(
        _to_numpy_labels(cell_type)[valid],
        _to_numpy_labels(clusters)[valid],
    )

    ari = adjusted_rand_score(
        _to_numpy_labels(cell_type)[valid],
        _to_numpy_labels(clusters)[valid],
    )

    metrics = {
        "embedding_key": embedding_key,
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),

        # species mixing
        "PCR_species": _pcr_score(X, species),
        "bASW_species": _batch_asw_score(X, species, cell_type),
        "kBET_species": _kbet_like_score(
            X,
            batch=species,
            cell_type=cell_type,
            n_neighbors=n_neighbors,
            random_state=random_state,
        ),
        "alignment_score_species": _alignment_score(
            X,
            species=species,
            n_neighbors=n_neighbors,
        ),

        # biology conservation
        "GC_cell_type": _graph_connectivity_score(
            X,
            cell_type=cell_type,
            n_neighbors=n_neighbors,
        ),
        "cASW_cell_type": _cell_type_asw_score(X, cell_type),
        "NMI_cell_type": nmi,
        "ARI_cell_type": ari,
    }

    if sample_key is not None:
        sample = adata.obs[sample_key]

        metrics.update(
            {
                "PCR_sample": _pcr_score(X, sample),
                "bASW_sample": _batch_asw_score(X, sample, cell_type),
                "kBET_sample": _kbet_like_score(
                    X,
                    batch=sample,
                    cell_type=cell_type,
                    n_neighbors=n_neighbors,
                    random_state=random_state,
                ),
            }
        )

    return pd.DataFrame([metrics])