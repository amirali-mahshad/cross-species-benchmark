# runners/baseline_runner.py

import argparse

import scanpy as sc

from benchmark import (
    create_run_directory,
    compute_integration_metrics,
    save_run_outputs,
)


def resolve_hvg_batch_key(
    hvg_batch_key: str,
    species_key: str,
    technical_key: str | None,
):
    """
    Decide which obs column to use for batch-aware HVG selection.

    hvg_batch_key='auto':
        use technical_key if available, otherwise species_key

    hvg_batch_key='none':
        do not use batch-aware HVG selection

    otherwise:
        use the provided column name
    """

    if hvg_batch_key == "auto":
        if technical_key is not None:
            return technical_key
        return species_key

    if hvg_batch_key == "none":
        return None

    return hvg_batch_key


def optionally_select_hvgs(
    adata,
    counts_layer: str,
    n_top_genes: int,
    hvg_batch_key: str | None = None,
):
    """
    Select highly variable genes before PCA.

    If n_top_genes > 0:
        select n_top_genes HVGs using seurat_v3 on raw counts.

    If n_top_genes <= 0:
        use all genes.
    """

    if n_top_genes is None or n_top_genes <= 0:
        print("Using all genes for baseline PCA.")
        return adata

    if counts_layer not in adata.layers:
        raise ValueError(f"Missing adata.layers['{counts_layer}'].")

    if hvg_batch_key is not None and hvg_batch_key not in adata.obs.columns:
        raise ValueError(f"Missing adata.obs['{hvg_batch_key}'] for HVG selection.")

    print(f"Selecting {n_top_genes} HVGs for baseline PCA.")
    print(f"HVG layer: {counts_layer}")
    print(f"HVG batch key: {hvg_batch_key}")

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor="seurat_v3",
        layer=counts_layer,
        batch_key=hvg_batch_key,
        subset=True,
    )

    return adata


def prepare_baseline_pca(
    adata,
    counts_layer: str,
    n_pcs: int,
    scale_max_value: float = 10.0,
):
    """
    Create unintegrated PCA embedding from raw counts.

    Steps:
        raw counts
        normalize_total
        log1p
        scale
        PCA

    The final baseline embedding is stored in:
        adata.obsm['X_emb']
    """

    if counts_layer not in adata.layers:
        raise ValueError(f"Missing adata.layers['{counts_layer}'].")

    print("Preparing unintegrated PCA baseline.")

    # Use raw counts as the starting expression matrix
    adata.X = adata.layers[counts_layer].copy()

    # Standard Scanpy preprocessing
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.scale(
        adata,
        max_value=scale_max_value,
    )

    sc.tl.pca(
        adata,
        n_comps=n_pcs,
        svd_solver="arpack",
    )

    adata.obsm["X_emb"] = adata.obsm["X_pca"].copy()

    return adata


def build_model_name(
    n_top_genes: int,
    hvg_batch_key: str | None,
    n_pcs: int,
):
    """
    Build standardized baseline model name.
    """

    if n_top_genes is None or n_top_genes <= 0:
        gene_part = "allgenes"
    else:
        gene_part = f"{n_top_genes}hvg"

    if hvg_batch_key is None:
        hvg_part = "hvg_nobatch"
    else:
        hvg_part = f"hvg_by_{hvg_batch_key}"

    return f"baseline_pca_{gene_part}_{hvg_part}_{n_pcs}pcs"


def main(args):
    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------

    adata = sc.read_h5ad(args.adata_path)

    # ------------------------------------------------------------
    # Resolve HVG batch key
    # ------------------------------------------------------------

    hvg_batch_key_resolved = resolve_hvg_batch_key(
        hvg_batch_key=args.hvg_batch_key,
        species_key=args.species_key,
        technical_key=args.technical_key,
    )

    # ------------------------------------------------------------
    # Validate required columns/layers
    # ------------------------------------------------------------

    required_obs = [args.species_key, args.cell_type_key]

    if args.technical_key is not None:
        required_obs.append(args.technical_key)

    if hvg_batch_key_resolved is not None:
        required_obs.append(hvg_batch_key_resolved)

    required_obs = sorted(set(required_obs))

    for key in required_obs:
        if key not in adata.obs.columns:
            raise ValueError(f"Missing adata.obs['{key}'].")

    if args.counts_layer not in adata.layers:
        raise ValueError(f"Missing adata.layers['{args.counts_layer}'].")

    for key in required_obs:
        adata.obs[key] = adata.obs[key].astype("category")

    # ------------------------------------------------------------
    # HVG selection
    # ------------------------------------------------------------

    adata = optionally_select_hvgs(
        adata=adata,
        counts_layer=args.counts_layer,
        n_top_genes=args.n_top_genes,
        hvg_batch_key=hvg_batch_key_resolved,
    )

    # ------------------------------------------------------------
    # PCA baseline
    # ------------------------------------------------------------

    adata = prepare_baseline_pca(
        adata=adata,
        counts_layer=args.counts_layer,
        n_pcs=args.n_pcs,
        scale_max_value=args.scale_max_value,
    )

    embedding_key = "X_emb"

    # ------------------------------------------------------------
    # Model name and result directory
    # ------------------------------------------------------------

    model_name = build_model_name(
        n_top_genes=args.n_top_genes,
        hvg_batch_key=hvg_batch_key_resolved,
        n_pcs=args.n_pcs,
    )

    result_dir = create_run_directory(
        result_root=args.result_root,
        dataset_name=args.dataset_name,
        model_name=model_name,
        seed=args.seed,
        overwrite=args.overwrite,
    )

    # ------------------------------------------------------------
    # Config
    # ------------------------------------------------------------

    config = {
        "runner": "baseline_runner",
        "dataset_name": args.dataset_name,
        "adata_path": args.adata_path,
        "model_name": model_name,
        "correction_mode": "unintegrated",

        "counts_layer": args.counts_layer,
        "species_key": args.species_key,
        "technical_key": args.technical_key,
        "cell_type_key": args.cell_type_key,

        "n_top_genes": args.n_top_genes,
        "hvg_batch_key_requested": args.hvg_batch_key,
        "hvg_batch_key_resolved": hvg_batch_key_resolved,

        "n_pcs": args.n_pcs,
        "scale_max_value": args.scale_max_value,

        "n_neighbors": args.n_neighbors,
        "leiden_resolution": args.leiden_resolution,

        "seed": args.seed,
        "n_cells_after_hvg": int(adata.n_obs),
        "n_genes_after_hvg": int(adata.n_vars),
    }

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    metrics_df = compute_integration_metrics(
        adata=adata,
        embedding_key=embedding_key,
        cell_type_key=args.cell_type_key,
        species_key=args.species_key,
        sample_key=args.technical_key,
        n_neighbors=args.n_neighbors,
        leiden_resolution=args.leiden_resolution,
        random_state=args.seed,
    )

    metrics_df.insert(0, "dataset", args.dataset_name)
    metrics_df.insert(1, "model", model_name)
    metrics_df.insert(2, "correction_mode", "unintegrated")
    metrics_df.insert(3, "n_top_genes", args.n_top_genes)
    metrics_df.insert(4, "hvg_batch_key", hvg_batch_key_resolved)

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------

    save_run_outputs(
        adata=adata,
        result_dir=result_dir,
        config=config,
        metrics_df=metrics_df,
        embedding_key=embedding_key,
        model=None,
        save_adata=args.save_adata,
        save_embedding=True,
        save_model=False,
    )

    print(metrics_df)
    print(f"\nSaved baseline run to:\n{result_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--adata_path", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--result_root", default="results")

    parser.add_argument("--counts_layer", default="counts")
    parser.add_argument("--species_key", default="species")
    parser.add_argument("--technical_key", default=None)
    parser.add_argument("--cell_type_key", default="cell_type_eval")

    parser.add_argument(
        "--n_top_genes",
        type=int,
        default=1200,
        help=(
            "Number of highly variable genes for baseline PCA. "
            "Use 0 to run PCA on all genes."
        ),
    )

    parser.add_argument(
        "--hvg_batch_key",
        default="auto",
        help=(
            "Batch key for HVG selection. "
            "'auto' uses technical_key if available, otherwise species_key. "
            "'none' disables batch-aware HVG selection."
        ),
    )

    parser.add_argument(
        "--n_pcs",
        type=int,
        default=20,
        help="Number of principal components used as the baseline embedding.",
    )

    parser.add_argument(
        "--scale_max_value",
        type=float,
        default=10.0,
        help="Maximum value used in sc.pp.scale.",
    )

    parser.add_argument("--n_neighbors", type=int, default=20)
    parser.add_argument("--leiden_resolution", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--save_adata",
        action="store_true",
        help=(
            "Save full adata_with_embedding.h5ad. "
            "By default this is False to save disk space."
        ),
    )

    args = parser.parse_args()
    main(args)