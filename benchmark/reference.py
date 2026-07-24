# benchmark/reference.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


def build_reference_name(
    dataset_name: str,
    n_top_genes: int,
    hvg_batch_key: str | None,
    n_pcs: int,
) -> str:
    if n_top_genes is None or n_top_genes <= 0:
        gene_part = "allgenes"
    else:
        gene_part = f"{n_top_genes}hvg"

    if hvg_batch_key is None or hvg_batch_key == "none":
        hvg_part = "hvg_by_none"
    else:
        hvg_part = f"hvg_by_{hvg_batch_key}"

    return f"{dataset_name}__reference_pca__{gene_part}__{hvg_part}__{n_pcs}pcs"


def create_reference_embedding(
    adata_path: str,
    counts_layer: str,
    species_key: str,
    cell_type_key: str,
    technical_key: str | None = None,
    n_top_genes: int = 1200,
    hvg_batch_key: str | None = "species",
    n_pcs: int = 20,
    random_state: int = 0,
):
    """
    Create deterministic pre-integration PCA reference.

    This should be used as the pre-integration space for:
        - Jaccard kNN preservation
        - PCR comparison before/after
        - ALCS reference classifier
    """

    adata = sc.read_h5ad(adata_path)

    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    required_obs = [species_key, cell_type_key]

    if technical_key is not None:
        required_obs.append(technical_key)

    if hvg_batch_key not in [None, "none"]:
        required_obs.append(hvg_batch_key)

    required_obs = sorted(set(required_obs))

    for key in required_obs:
        if key not in adata.obs.columns:
            raise ValueError(f"Missing adata.obs['{key}'].")

    if counts_layer not in adata.layers:
        raise ValueError(f"Missing adata.layers['{counts_layer}'].")

    for key in required_obs:
        adata.obs[key] = adata.obs[key].astype("category")

    if n_top_genes is not None and n_top_genes > 0:
        batch_key = None if hvg_batch_key in [None, "none"] else hvg_batch_key

        print(f"Selecting reference HVGs: {n_top_genes}")
        print(f"Reference HVG batch key: {batch_key}")

        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor="seurat_v3",
            layer=counts_layer,
            batch_key=batch_key,
            subset=True,
        )
    else:
        print("Reference uses all genes.")

    adata.X = adata.layers[counts_layer].copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)

    n_comps = min(n_pcs, adata.n_obs - 1, adata.n_vars - 1)

    if n_comps < 1:
        raise ValueError("Not enough cells/genes for PCA.")

    sc.tl.pca(
        adata,
        n_comps=n_comps,
        svd_solver="arpack",
        random_state=random_state,
    )

    adata.obsm["X_ref"] = adata.obsm["X_pca"].astype(np.float32)

    return adata


def load_or_create_reference(
    adata_path: str,
    result_root: str,
    dataset_name: str,
    counts_layer: str,
    species_key: str,
    cell_type_key: str,
    technical_key: str | None = None,
    n_top_genes: int = 1200,
    hvg_batch_key: str | None = "species",
    n_pcs: int = 20,
    random_state: int = 0,
    overwrite: bool = False,
):
    result_root = Path(result_root)

    ref_name = build_reference_name(
        dataset_name=dataset_name,
        n_top_genes=n_top_genes,
        hvg_batch_key=hvg_batch_key,
        n_pcs=n_pcs,
    )

    ref_dir = result_root / "_reference" / ref_name
    ref_dir.mkdir(parents=True, exist_ok=True)

    ref_h5ad = ref_dir / "reference_adata.h5ad"
    ref_emb = ref_dir / "reference_X_ref.npy"
    ref_obs = ref_dir / "reference_obs_names.csv"
    ref_var = ref_dir / "reference_var_names.csv"

    if ref_h5ad.exists() and ref_emb.exists() and not overwrite:
        print(f"Loading existing reference: {ref_dir}")
        adata_ref = sc.read_h5ad(ref_h5ad)

        if "X_ref" not in adata_ref.obsm:
            adata_ref.obsm["X_ref"] = np.load(ref_emb)

        return adata_ref, ref_dir

    print(f"Creating new reference: {ref_dir}")

    adata_ref = create_reference_embedding(
        adata_path=adata_path,
        counts_layer=counts_layer,
        species_key=species_key,
        cell_type_key=cell_type_key,
        technical_key=technical_key,
        n_top_genes=n_top_genes,
        hvg_batch_key=hvg_batch_key,
        n_pcs=n_pcs,
        random_state=random_state,
    )

    np.save(ref_emb, adata_ref.obsm["X_ref"])

    pd.Series(adata_ref.obs_names.astype(str)).to_csv(
        ref_obs,
        index=False,
        header=False,
    )

    pd.Series(adata_ref.var_names.astype(str)).to_csv(
        ref_var,
        index=False,
        header=False,
    )

    adata_ref.write_h5ad(ref_h5ad)

    print(f"Saved reference to: {ref_dir}")

    return adata_ref, ref_dir
