# benchmark/metrics.py

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.sparse import csgraph
from scipy.stats import chisquare
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors


# ============================================================
# General helpers
# ============================================================

def _nan() -> float:
    return float("nan")


def _nanmean_safe(values) -> float:
    values = [v for v in values if v is not None and not pd.isna(v)]
    if len(values) == 0:
        return _nan()
    return float(np.mean(values))


def _get_embedding(adata, embedding_key: str) -> np.ndarray:
    if embedding_key not in adata.obsm:
        raise ValueError(f"Missing adata.obsm['{embedding_key}'].")

    X = np.asarray(adata.obsm[embedding_key])

    if X.ndim != 2:
        raise ValueError(f"adata.obsm['{embedding_key}'] must be 2D.")

    if not np.all(np.isfinite(X)):
        warnings.warn(
            f"Embedding {embedding_key} contains non-finite values. "
            "Rows with non-finite values will be removed per metric."
        )

    return X.astype(np.float32)


def _obs_array(adata, key: str) -> np.ndarray:
    if key is None:
        raise ValueError("obs key is None.")

    if key not in adata.obs.columns:
        raise ValueError(f"Missing adata.obs['{key}'].")

    return adata.obs[key].astype(str).to_numpy()


def _valid_mask(X: np.ndarray, *arrays: np.ndarray) -> np.ndarray:
    mask = np.all(np.isfinite(X), axis=1)

    for arr in arrays:
        s = pd.Series(arr)
        mask &= ~s.isna().to_numpy()
        mask &= s.astype(str).to_numpy() != "nan"

    return mask


def _subset_valid(X: np.ndarray, *arrays: np.ndarray):
    mask = _valid_mask(X, *arrays)
    X2 = X[mask]
    arrays2 = [np.asarray(a)[mask] for a in arrays]
    return X2, arrays2


def _categorical_codes(labels: np.ndarray):
    cat = pd.Categorical(labels)
    return cat.codes.astype(int), list(cat.categories)


def _has_minimum_label_structure(labels: np.ndarray) -> bool:
    labels = np.asarray(labels)
    n = len(labels)
    n_labels = len(np.unique(labels))

    if n < 3:
        return False

    if n_labels < 2:
        return False

    if n_labels >= n:
        return False

    return True


# ============================================================
# kNN helpers
# ============================================================

def _knn_indices(X: np.ndarray, n_neighbors: int = 20) -> np.ndarray:
    """
    Return kNN indices excluding self.

    Output shape:
        n_cells x k_effective
    """

    X = np.asarray(X)

    if X.shape[0] <= 1:
        return np.empty((X.shape[0], 0), dtype=int)

    k = min(n_neighbors, X.shape[0] - 1)

    nn = NearestNeighbors(
        n_neighbors=k + 1,
        metric="euclidean",
        n_jobs=1,
    )

    nn.fit(X)
    raw = nn.kneighbors(X, return_distance=False)

    clean = []

    for i, row in enumerate(raw):
        row = row[row != i]
        row = row[:k]
        clean.append(row)

    return np.asarray(clean, dtype=int)


def _build_undirected_adjacency_from_knn(
    X: np.ndarray,
    n_neighbors: int = 20,
) -> sparse.csr_matrix:
    idx = _knn_indices(X, n_neighbors=n_neighbors)

    n = X.shape[0]

    rows = []
    cols = []

    for i in range(n):
        for j in idx[i]:
            rows.append(i)
            cols.append(j)
            rows.append(j)
            cols.append(i)

    data = np.ones(len(rows), dtype=np.float32)

    adj = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(n, n),
    )

    adj.data[:] = 1.0
    adj.eliminate_zeros()

    return adj


# ============================================================
# Silhouette metrics
# ============================================================

def cell_type_asw(
    X: np.ndarray,
    cell_type: np.ndarray,
) -> float:
    """
    Cell-type ASW.

    Higher = better cell-type separation.
    Scaled to [0, 1] as:
        (silhouette + 1) / 2
    """

    X, (cell_type,) = _subset_valid(X, cell_type)

    if not _has_minimum_label_structure(cell_type):
        return _nan()

    try:
        score = silhouette_score(X, cell_type, metric="euclidean")
        score = (score + 1.0) / 2.0
        return float(np.clip(score, 0.0, 1.0))
    except Exception as e:
        warnings.warn(f"cell_type_asw failed: {e}")
        return _nan()


def batch_asw_within_groups(
    X: np.ndarray,
    batch: np.ndarray,
    group: np.ndarray,
    min_cells_per_group: int = 10,
) -> float:
    """
    Batch/species/sample ASW calculated within biological groups.

    For each group, compute silhouette by batch.
    Then transform:
        1 - abs(silhouette)

    Higher = better batch mixing inside each group.
    """

    X, (batch, group) = _subset_valid(X, batch, group)

    scores = []

    for g in sorted(np.unique(group)):
        idx = group == g

        if idx.sum() < min_cells_per_group:
            continue

        b = batch[idx]

        if not _has_minimum_label_structure(b):
            continue

        try:
            s = silhouette_samples(X[idx], b, metric="euclidean")
            score = np.mean(1.0 - np.abs(s))
            scores.append(float(np.clip(score, 0.0, 1.0)))
        except Exception:
            continue

    return _nanmean_safe(scores)


def batch_asw_within_composite_groups(
    X: np.ndarray,
    batch: np.ndarray,
    group_a: np.ndarray,
    group_b: np.ndarray,
    min_cells_per_group: int = 10,
) -> float:
    """
    Batch ASW inside composite groups, e.g.

        species × cell_type

    Useful for sample mixing within each species and cell type.
    """

    composite = (
        pd.Series(group_a).astype(str)
        + "__"
        + pd.Series(group_b).astype(str)
    ).to_numpy()

    return batch_asw_within_groups(
        X=X,
        batch=batch,
        group=composite,
        min_cells_per_group=min_cells_per_group,
    )


# ============================================================
# LISI metrics
# ============================================================

def _lisi_score_from_neighbors(
    labels: np.ndarray,
    knn_idx: np.ndarray,
) -> np.ndarray:
    labels = np.asarray(labels)
    codes, categories = _categorical_codes(labels)

    n_categories = len(categories)

    if n_categories < 2:
        return np.full(labels.shape[0], np.nan)

    lisi_values = []

    for row in knn_idx:
        if len(row) == 0:
            lisi_values.append(np.nan)
            continue

        local_codes = codes[row]
        counts = np.bincount(local_codes, minlength=n_categories).astype(float)

        if counts.sum() == 0:
            lisi_values.append(np.nan)
            continue

        p = counts / counts.sum()
        denom = np.sum(p ** 2)

        if denom <= 0:
            lisi_values.append(np.nan)
        else:
            lisi_values.append(1.0 / denom)

    return np.asarray(lisi_values, dtype=float)


def ilisi(
    X: np.ndarray,
    batch: np.ndarray,
    n_neighbors: int = 20,
) -> float:
    """
    Integrated LISI for batch/species/sample mixing.

    Higher = better mixing.
    Normalized to [0, 1].
    """

    X, (batch,) = _subset_valid(X, batch)

    n_labels = len(np.unique(batch))

    if X.shape[0] <= n_neighbors or n_labels < 2:
        return _nan()

    try:
        knn_idx = _knn_indices(X, n_neighbors=n_neighbors)
        lisi_values = _lisi_score_from_neighbors(batch, knn_idx)

        normalized = (lisi_values - 1.0) / (n_labels - 1.0)
        normalized = np.clip(normalized, 0.0, 1.0)

        return float(np.nanmedian(normalized))
    except Exception as e:
        warnings.warn(f"ilisi failed: {e}")
        return _nan()


def clisi(
    X: np.ndarray,
    cell_type: np.ndarray,
    n_neighbors: int = 20,
) -> float:
    """
    Cell-type LISI.

    Here lower raw LISI means purer neighborhoods.
    We invert and normalize so higher = better cell-type conservation.
    """

    X, (cell_type,) = _subset_valid(X, cell_type)

    n_labels = len(np.unique(cell_type))

    if X.shape[0] <= n_neighbors or n_labels < 2:
        return _nan()

    try:
        knn_idx = _knn_indices(X, n_neighbors=n_neighbors)
        lisi_values = _lisi_score_from_neighbors(cell_type, knn_idx)

        normalized = (n_labels - lisi_values) / (n_labels - 1.0)
        normalized = np.clip(normalized, 0.0, 1.0)

        return float(np.nanmedian(normalized))
    except Exception as e:
        warnings.warn(f"clisi failed: {e}")
        return _nan()


def ilisi_within_groups(
    X: np.ndarray,
    batch: np.ndarray,
    group: np.ndarray,
    n_neighbors: int = 20,
    min_cells_per_group: int = 30,
) -> float:
    """
    iLISI calculated separately inside groups and averaged.

    Example:
        species iLISI inside each cell type.
    """

    X, (batch, group) = _subset_valid(X, batch, group)

    scores = []

    for g in sorted(np.unique(group)):
        idx = group == g

        if idx.sum() < min_cells_per_group:
            continue

        if len(np.unique(batch[idx])) < 2:
            continue

        score = ilisi(
            X=X[idx],
            batch=batch[idx],
            n_neighbors=min(n_neighbors, idx.sum() - 1),
        )

        scores.append(score)

    return _nanmean_safe(scores)


def ilisi_within_composite_groups(
    X: np.ndarray,
    batch: np.ndarray,
    group_a: np.ndarray,
    group_b: np.ndarray,
    n_neighbors: int = 20,
    min_cells_per_group: int = 30,
) -> float:
    composite = (
        pd.Series(group_a).astype(str)
        + "__"
        + pd.Series(group_b).astype(str)
    ).to_numpy()

    return ilisi_within_groups(
        X=X,
        batch=batch,
        group=composite,
        n_neighbors=n_neighbors,
        min_cells_per_group=min_cells_per_group,
    )


# ============================================================
# kBET approximation
# ============================================================

def kbet_score(
    X: np.ndarray,
    batch: np.ndarray,
    n_neighbors: int = 20,
    alpha: float = 0.05,
) -> float:
    """
    Lightweight kBET-like score.

    For every cell, compare local batch composition to global batch composition
    using a chi-square goodness-of-fit test.

    Higher = better mixing.
    """

    X, (batch,) = _subset_valid(X, batch)

    codes, categories = _categorical_codes(batch)
    n_categories = len(categories)

    if X.shape[0] <= n_neighbors or n_categories < 2:
        return _nan()

    try:
        knn_idx = _knn_indices(X, n_neighbors=n_neighbors)

        global_counts = np.bincount(codes, minlength=n_categories).astype(float)
        global_props = global_counts / global_counts.sum()

        accepted = []

        for row in knn_idx:
            local_codes = codes[row]
            obs = np.bincount(local_codes, minlength=n_categories).astype(float)

            if obs.sum() == 0:
                continue

            exp = global_props * obs.sum()

            mask = exp > 0

            if mask.sum() < 2:
                continue

            obs_m = obs[mask]
            exp_m = exp[mask]

            exp_m = exp_m * (obs_m.sum() / exp_m.sum())

            try:
                _, p_value = chisquare(f_obs=obs_m, f_exp=exp_m)
                accepted.append(float(p_value >= alpha))
            except Exception:
                continue

        return _nanmean_safe(accepted)

    except Exception as e:
        warnings.warn(f"kbet_score failed: {e}")
        return _nan()


def kbet_within_groups(
    X: np.ndarray,
    batch: np.ndarray,
    group: np.ndarray,
    n_neighbors: int = 20,
    alpha: float = 0.05,
    min_cells_per_group: int = 30,
) -> float:
    X, (batch, group) = _subset_valid(X, batch, group)

    scores = []

    for g in sorted(np.unique(group)):
        idx = group == g

        if idx.sum() < min_cells_per_group:
            continue

        if len(np.unique(batch[idx])) < 2:
            continue

        score = kbet_score(
            X=X[idx],
            batch=batch[idx],
            n_neighbors=min(n_neighbors, idx.sum() - 1),
            alpha=alpha,
        )

        scores.append(score)

    return _nanmean_safe(scores)


def kbet_within_composite_groups(
    X: np.ndarray,
    batch: np.ndarray,
    group_a: np.ndarray,
    group_b: np.ndarray,
    n_neighbors: int = 20,
    alpha: float = 0.05,
    min_cells_per_group: int = 30,
) -> float:
    composite = (
        pd.Series(group_a).astype(str)
        + "__"
        + pd.Series(group_b).astype(str)
    ).to_numpy()

    return kbet_within_groups(
        X=X,
        batch=batch,
        group=composite,
        n_neighbors=n_neighbors,
        alpha=alpha,
        min_cells_per_group=min_cells_per_group,
    )


# ============================================================
# PCR
# ============================================================

def pcr_variance_explained(
    X: np.ndarray,
    covariate: np.ndarray,
    n_components: int = 50,
    random_state: int = 0,
) -> float:
    """
    Variance in the embedding explained by a categorical covariate.

    Lower raw value = less covariate effect.
    """

    X, (covariate,) = _subset_valid(X, covariate)

    if X.shape[0] < 5 or len(np.unique(covariate)) < 2:
        return _nan()

    try:
        n_components = min(n_components, X.shape[0] - 1, X.shape[1])

        if n_components < 1:
            return _nan()

        pca = PCA(
            n_components=n_components,
            random_state=random_state,
        )

        PCs = pca.fit_transform(X)
        variance_ratio = pca.explained_variance_ratio_

        cov_df = pd.get_dummies(pd.Series(covariate), drop_first=False)

        if cov_df.shape[1] < 2:
            return _nan()

        C = cov_df.to_numpy(dtype=float)

        r2_values = []

        for i in range(PCs.shape[1]):
            y = PCs[:, i]
            model = LinearRegression()
            model.fit(C, y)
            r2 = model.score(C, y)
            r2 = np.clip(r2, 0.0, 1.0)
            r2_values.append(r2)

        r2_values = np.asarray(r2_values)

        explained = float(np.sum(variance_ratio * r2_values))
        explained = float(np.clip(explained, 0.0, 1.0))

        return explained

    except Exception as e:
        warnings.warn(f"pcr_variance_explained failed: {e}")
        return _nan()


def pcr_correction_score(
    X: np.ndarray,
    covariate: np.ndarray,
    n_components: int = 50,
    random_state: int = 0,
) -> float:
    """
    PCR correction score.

    Higher = less variance explained by the covariate.
    """

    raw = pcr_variance_explained(
        X=X,
        covariate=covariate,
        n_components=n_components,
        random_state=random_state,
    )

    if pd.isna(raw):
        return _nan()

    return float(1.0 - raw)


# ============================================================
# Graph connectivity
# ============================================================

def graph_connectivity(
    X: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 20,
    label_subset: set[str] | None = None,
) -> float:
    """
    Graph connectivity of cells with the same label.

    For each label:
        size of largest connected component / total cells with label

    Higher = better.
    """

    X, (labels,) = _subset_valid(X, labels)

    labels = labels.astype(str)

    if X.shape[0] <= 1:
        return _nan()

    try:
        adj = _build_undirected_adjacency_from_knn(
            X,
            n_neighbors=n_neighbors,
        )

        scores = []

        for lab in sorted(np.unique(labels)):
            if label_subset is not None and lab not in label_subset:
                continue

            idx = np.where(labels == lab)[0]

            if len(idx) == 0:
                continue

            if len(idx) == 1:
                scores.append(1.0)
                continue

            sub_adj = adj[idx][:, idx]

            n_components, comp_labels = csgraph.connected_components(
                sub_adj,
                directed=False,
                return_labels=True,
            )

            if n_components < 1:
                continue

            largest = np.bincount(comp_labels).max()
            score = largest / len(idx)

            scores.append(float(score))

        return _nanmean_safe(scores)

    except Exception as e:
        warnings.warn(f"graph_connectivity failed: {e}")
        return _nan()


# ============================================================
# Clustering-based metrics
# ============================================================

def cluster_embedding(
    X: np.ndarray,
    n_neighbors: int = 20,
    leiden_resolution: float = 1.0,
    random_state: int = 0,
    fallback_n_clusters: int | None = None,
) -> np.ndarray:
    """
    Cluster embedding with Leiden.
    If Leiden fails, use KMeans fallback.
    """

    try:
        temp = sc.AnnData(X=np.zeros((X.shape[0], 1), dtype=np.float32))
        temp.obsm["X_emb"] = X

        sc.pp.neighbors(
            temp,
            use_rep="X_emb",
            n_neighbors=n_neighbors,
            key_added="neighbors_X_emb",
        )

        sc.tl.leiden(
            temp,
            neighbors_key="neighbors_X_emb",
            resolution=leiden_resolution,
            key_added="benchmark_leiden",
            random_state=random_state,
        )

        return temp.obs["benchmark_leiden"].astype(str).to_numpy()

    except Exception as e:
        warnings.warn(f"Leiden clustering failed, using KMeans fallback. Error: {e}")

        if fallback_n_clusters is None:
            fallback_n_clusters = 10

        n_clusters = min(max(2, fallback_n_clusters), X.shape[0] - 1)

        km = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
        )

        return km.fit_predict(X).astype(str)


def ari_nmi_against_labels(
    clusters: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    labels = np.asarray(labels).astype(str)
    clusters = np.asarray(clusters).astype(str)

    mask = ~pd.Series(labels).isna().to_numpy()
    mask &= ~pd.Series(clusters).isna().to_numpy()

    labels = labels[mask]
    clusters = clusters[mask]

    if len(np.unique(labels)) < 2 or len(np.unique(clusters)) < 2:
        return _nan(), _nan()

    try:
        ari = adjusted_rand_score(labels, clusters)
        nmi = normalized_mutual_info_score(labels, clusters)

        ari = float(np.clip(ari, 0.0, 1.0))
        nmi = float(np.clip(nmi, 0.0, 1.0))

        return ari, nmi

    except Exception as e:
        warnings.warn(f"ari_nmi_against_labels failed: {e}")
        return _nan(), _nan()


# ============================================================
# Cross-species alignment score
# ============================================================

def cross_species_neighbor_fraction(
    X: np.ndarray,
    species: np.ndarray,
    n_neighbors: int = 20,
) -> float:
    """
    Fraction of each cell's neighbors that come from another species.

    Higher = stronger cross-species local mixing.
    """

    X, (species,) = _subset_valid(X, species)

    if X.shape[0] <= n_neighbors or len(np.unique(species)) < 2:
        return _nan()

    try:
        idx = _knn_indices(X, n_neighbors=n_neighbors)

        fractions = []

        for i, row in enumerate(idx):
            if len(row) == 0:
                continue

            frac = np.mean(species[row] != species[i])
            fractions.append(float(frac))

        return _nanmean_safe(fractions)

    except Exception as e:
        warnings.warn(f"cross_species_neighbor_fraction failed: {e}")
        return _nan()


def cross_species_neighbor_fraction_within_cell_type(
    X: np.ndarray,
    species: np.ndarray,
    cell_type: np.ndarray,
    n_neighbors: int = 20,
    min_cells_per_group: int = 30,
) -> float:
    """
    Cross-species neighbor fraction calculated inside each cell type.

    This is usually better than the global version for cross-species benchmarks.
    """

    X, (species, cell_type) = _subset_valid(X, species, cell_type)

    scores = []

    for ct in sorted(np.unique(cell_type)):
        idx = cell_type == ct

        if idx.sum() < min_cells_per_group:
            continue

        if len(np.unique(species[idx])) < 2:
            continue

        score = cross_species_neighbor_fraction(
            X=X[idx],
            species=species[idx],
            n_neighbors=min(n_neighbors, idx.sum() - 1),
        )

        scores.append(score)

    return _nanmean_safe(scores)


# ============================================================
# Species-specific / isolated labels
# ============================================================

def find_species_specific_cell_types(
    cell_type: np.ndarray,
    species: np.ndarray,
) -> set[str]:
    df = pd.DataFrame(
        {
            "cell_type": pd.Series(cell_type).astype(str),
            "species": pd.Series(species).astype(str),
        }
    )

    counts = df.groupby("cell_type")["species"].nunique()

    return set(counts[counts == 1].index.astype(str))


def isolated_label_asw(
    X: np.ndarray,
    cell_type: np.ndarray,
    isolated_labels: set[str],
) -> float:
    """
    ASW score for isolated/species-specific labels.

    For each isolated label:
        binary silhouette: label vs all other cells
    """

    X, (cell_type,) = _subset_valid(X, cell_type)

    cell_type = cell_type.astype(str)

    scores = []

    for lab in sorted(isolated_labels):
        binary = np.where(cell_type == lab, lab, "other")

        if np.sum(cell_type == lab) < 2:
            continue

        if not _has_minimum_label_structure(binary):
            continue

        try:
            s = silhouette_samples(X, binary, metric="euclidean")
            target_score = np.mean(s[cell_type == lab])
            target_score = (target_score + 1.0) / 2.0
            scores.append(float(np.clip(target_score, 0.0, 1.0)))
        except Exception:
            continue

    return _nanmean_safe(scores)


# ============================================================
# Main public function used by all runners
# ============================================================

def compute_integration_metrics(
    adata,
    embedding_key: str = "X_emb",
    cell_type_key: str = "cell_type_eval",
    species_key: str = "species",
    sample_key: str | None = "sample",
    n_neighbors: int = 20,
    leiden_resolution: float = 1.0,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Compute species-correction and biological-conservation metrics for one
    integrated AnnData object.

    ``sample_key`` remains accepted for compatibility with the existing
    runners, but no sample-correction metrics are calculated.

    Heavy/post-hoc metrics such as:
        ALCS
        Jaccard kNN preservation
        PCR comparison before-vs-after

    should be implemented in separate modules because they require
    reference/unintegrated embeddings or classifier training.
    """

    X = _get_embedding(adata, embedding_key)

    cell_type = _obs_array(adata, cell_type_key)
    species = _obs_array(adata, species_key)

    # ``sample_key`` is retained in the function signature so the existing
    # benchmark runners do not need to change. Sample-correction metrics are
    # intentionally not computed in this species-only benchmark.
    _ = sample_key

    metrics = {}

    metrics["embedding_key"] = embedding_key
    metrics["n_cells"] = int(adata.n_obs)
    metrics["n_genes"] = int(adata.n_vars)
    metrics["n_neighbors"] = int(n_neighbors)
    metrics["leiden_resolution"] = float(leiden_resolution)

    # ------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------

    fallback_n_clusters = len(np.unique(cell_type))

    clusters = cluster_embedding(
        X=X,
        n_neighbors=n_neighbors,
        leiden_resolution=leiden_resolution,
        random_state=random_state,
        fallback_n_clusters=fallback_n_clusters,
    )

    # ------------------------------------------------------------
    # Block A: Cross-species alignment / species mixing
    # ------------------------------------------------------------

    metrics["PCR_species"] = pcr_correction_score(
        X=X,
        covariate=species,
        random_state=random_state,
    )

    metrics["PCR_species_raw_variance_explained"] = pcr_variance_explained(
        X=X,
        covariate=species,
        random_state=random_state,
    )

    metrics["bASW_species"] = batch_asw_within_groups(
        X=X,
        batch=species,
        group=cell_type,
    )

    metrics["bASW_species_within_cell_type"] = metrics["bASW_species"]

    metrics["iLISI_species"] = ilisi(
        X=X,
        batch=species,
        n_neighbors=n_neighbors,
    )

    metrics["iLISI_species_within_cell_type"] = ilisi_within_groups(
        X=X,
        batch=species,
        group=cell_type,
        n_neighbors=n_neighbors,
    )

    metrics["kBET_species"] = kbet_score(
        X=X,
        batch=species,
        n_neighbors=n_neighbors,
    )

    metrics["kBET_species_within_cell_type"] = kbet_within_groups(
        X=X,
        batch=species,
        group=cell_type,
        n_neighbors=n_neighbors,
    )

    metrics["alignment_score_species"] = cross_species_neighbor_fraction(
        X=X,
        species=species,
        n_neighbors=n_neighbors,
    )

    metrics["alignment_score_species_within_cell_type"] = (
        cross_species_neighbor_fraction_within_cell_type(
            X=X,
            species=species,
            cell_type=cell_type,
            n_neighbors=n_neighbors,
        )
    )

    ari_species, nmi_species = ari_nmi_against_labels(
        clusters=clusters,
        labels=species,
    )

    metrics["ARI_species"] = ari_species
    metrics["NMI_species"] = nmi_species

    if pd.isna(ari_species):
        metrics["ARI_species_reversed"] = _nan()
    else:
        metrics["ARI_species_reversed"] = float(1.0 - ari_species)

    if pd.isna(nmi_species):
        metrics["NMI_species_reversed"] = _nan()
    else:
        metrics["NMI_species_reversed"] = float(1.0 - nmi_species)

    # A compact species mixing summary using available species metrics.
    metrics["species_mixing_score_unscaled"] = _nanmean_safe(
        [
            metrics["PCR_species"],
            metrics["bASW_species"],
            metrics["iLISI_species_within_cell_type"],
            metrics["kBET_species_within_cell_type"],
            metrics["alignment_score_species_within_cell_type"],
            metrics["ARI_species_reversed"],
            metrics["NMI_species_reversed"],
        ]
    )

    # Sample-correction metrics were intentionally removed.
    # This benchmark evaluates species correction and biological conservation.

    # ------------------------------------------------------------
    # Block C: Inter-cell-type biological conservation
    # ------------------------------------------------------------

    ari_cell_type, nmi_cell_type = ari_nmi_against_labels(
        clusters=clusters,
        labels=cell_type,
    )

    metrics["ARI_cell_type"] = ari_cell_type
    metrics["NMI_cell_type"] = nmi_cell_type

    metrics["cASW_cell_type"] = cell_type_asw(
        X=X,
        cell_type=cell_type,
    )

    metrics["cLISI_cell_type"] = clisi(
        X=X,
        cell_type=cell_type,
        n_neighbors=n_neighbors,
    )

    metrics["GC_cell_type"] = graph_connectivity(
        X=X,
        labels=cell_type,
        n_neighbors=n_neighbors,
    )

    metrics["inter_cell_type_bio_score_unscaled"] = _nanmean_safe(
        [
            metrics["ARI_cell_type"],
            metrics["NMI_cell_type"],
            metrics["cASW_cell_type"],
            metrics["cLISI_cell_type"],
            metrics["GC_cell_type"],
        ]
    )

    # ------------------------------------------------------------
    # Block D: species-specific / isolated cell-type preservation
    # ------------------------------------------------------------

    species_specific_labels = find_species_specific_cell_types(
        cell_type=cell_type,
        species=species,
    )

    metrics["n_species_specific_cell_types"] = int(len(species_specific_labels))

    metrics["species_specific_ASW_cell_type"] = isolated_label_asw(
        X=X,
        cell_type=cell_type,
        isolated_labels=species_specific_labels,
    )

    metrics["isolated_label_score"] = metrics["species_specific_ASW_cell_type"]

    metrics["species_specific_GC_cell_type"] = graph_connectivity(
        X=X,
        labels=cell_type,
        n_neighbors=n_neighbors,
        label_subset=species_specific_labels,
    )

    metrics["species_specific_bio_score_unscaled"] = _nanmean_safe(
        [
            metrics["species_specific_ASW_cell_type"],
            metrics["species_specific_GC_cell_type"],
        ]
    )

    # ------------------------------------------------------------
    # Optional compact global summaries
    # ------------------------------------------------------------

    metrics["bio_conservation_score_unscaled"] = _nanmean_safe(
        [
            metrics["inter_cell_type_bio_score_unscaled"],
            metrics["species_specific_bio_score_unscaled"],
        ]
    )

    # Species-only diagnostic summary. Sample-correction metrics are not part
    # of this score.
    metrics["simple_overall_score_unscaled"] = _nanmean_safe(
        [
            metrics["species_mixing_score_unscaled"],
            metrics["bio_conservation_score_unscaled"],
        ]
    )

    return pd.DataFrame([metrics])
