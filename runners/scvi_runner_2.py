# runners/scvi_runner.py

import argparse

import scanpy as sc
import scvi

from benchmark import (
    create_run_directory,
    compute_integration_metrics,
    save_run_outputs,
)


def build_scvi_setup_args(
    correction_mode: str,
    species_key: str,
    technical_key: str | None,
):
    """
    Decide which covariates scVI should use for correction.

    correction_mode='species':
        batch_key = species_key

    correction_mode='technical':
        batch_key = technical_key, e.g. sample/lab

    correction_mode='both':
        batch_key = technical_key
        categorical_covariate_keys = [species_key]
    """

    if correction_mode == "species":
        return {
            "batch_key": species_key,
            "categorical_covariate_keys": None,
        }

    if correction_mode == "technical":
        if technical_key is None:
            raise ValueError(
                "technical_key must be provided when correction_mode='technical'."
            )

        return {
            "batch_key": technical_key,
            "categorical_covariate_keys": None,
        }

    if correction_mode == "both":
        if technical_key is None:
            raise ValueError(
                "technical_key must be provided when correction_mode='both'."
            )

        return {
            "batch_key": technical_key,
            "categorical_covariate_keys": [species_key],
        }

    raise ValueError(
        "correction_mode must be one of: species, technical, both."
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


def optionally_select_hvgs_for_scvi(
    adata,
    counts_layer: str,
    n_top_genes: int,
    hvg_batch_key: str | None = None,
):
    """
    Select highly variable genes before training scVI.

    If n_top_genes > 0:
        select n_top_genes HVGs using seurat_v3 on raw counts.

    If n_top_genes <= 0:
        use all genes.
    """

    if n_top_genes is None or n_top_genes <= 0:
        print("Using all genes for scVI.")
        return adata

    if counts_layer not in adata.layers:
        raise ValueError(f"Missing adata.layers['{counts_layer}'].")

    if hvg_batch_key is not None and hvg_batch_key not in adata.obs.columns:
        raise ValueError(f"Missing adata.obs['{hvg_batch_key}'] for HVG selection.")

    print(f"Selecting {n_top_genes} HVGs for scVI.")
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


def build_model_name(
    correction_mode: str,
    species_key: str,
    technical_key: str | None,
    n_top_genes: int,
):
    """
    Build standardized model name including correction mode and HVG setting.
    """

    if correction_mode == "species":
        base = f"scvi_correct_{species_key}"

    elif correction_mode == "technical":
        if technical_key is None:
            raise ValueError(
                "technical_key must be provided when correction_mode='technical'."
            )
        base = f"scvi_correct_{technical_key}"

    elif correction_mode == "both":
        if technical_key is None:
            raise ValueError(
                "technical_key must be provided when correction_mode='both'."
            )
        base = f"scvi_correct_{technical_key}_plus_{species_key}"

    else:
        raise ValueError("Invalid correction_mode.")

    if n_top_genes is None or n_top_genes <= 0:
        return f"{base}_allgenes"

    return f"{base}_{n_top_genes}hvg"


def main(args):
    scvi.settings.seed = args.seed

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------

    adata = sc.read_h5ad(args.adata_path)

    # ------------------------------------------------------------
    # Resolve setup choices
    # ------------------------------------------------------------

    setup_args = build_scvi_setup_args(
        correction_mode=args.correction_mode,
        species_key=args.species_key,
        technical_key=args.technical_key,
    )

    hvg_batch_key_resolved = resolve_hvg_batch_key(
        hvg_batch_key=args.hvg_batch_key,
        species_key=args.species_key,
        technical_key=args.technical_key,
    )

    model_name = build_model_name(
        correction_mode=args.correction_mode,
        species_key=args.species_key,
        technical_key=args.technical_key,
        n_top_genes=args.n_top_genes,
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

    # Make important obs columns categorical
    for key in required_obs:
        adata.obs[key] = adata.obs[key].astype("category")

    # ------------------------------------------------------------
    # HVG selection
    # ------------------------------------------------------------

    adata = optionally_select_hvgs_for_scvi(
        adata=adata,
        counts_layer=args.counts_layer,
        n_top_genes=args.n_top_genes,
        hvg_batch_key=hvg_batch_key_resolved,
    )

    # ------------------------------------------------------------
    # Create result directory
    # ------------------------------------------------------------

    result_dir = create_run_directory(
        result_root=args.result_root,
        dataset_name=args.dataset_name,
        model_name=model_name,
        seed=args.seed,
        overwrite=args.overwrite,
    )

    # ------------------------------------------------------------
    # Save config
    # ------------------------------------------------------------

    config = {
        "runner": "scvi_runner",
        "dataset_name": args.dataset_name,
        "adata_path": args.adata_path,
        "model_name": model_name,
        "correction_mode": args.correction_mode,

        "counts_layer": args.counts_layer,
        "species_key": args.species_key,
        "technical_key": args.technical_key,
        "cell_type_key": args.cell_type_key,

        "batch_key_used_by_scvi": setup_args["batch_key"],
        "categorical_covariate_keys_used_by_scvi": setup_args[
            "categorical_covariate_keys"
        ],

        "n_top_genes": args.n_top_genes,
        "hvg_batch_key_requested": args.hvg_batch_key,
        "hvg_batch_key_resolved": hvg_batch_key_resolved,

        "n_latent": args.n_latent,
        "gene_likelihood": args.gene_likelihood,
        "max_epochs": args.max_epochs,

        "n_neighbors": args.n_neighbors,
        "leiden_resolution": args.leiden_resolution,

        "seed": args.seed,
        "n_cells_after_hvg": int(adata.n_obs),
        "n_genes_after_hvg": int(adata.n_vars),
    }

    # ------------------------------------------------------------
    # scVI setup
    # ------------------------------------------------------------

    scvi.model.SCVI.setup_anndata(
        adata,
        layer=args.counts_layer,
        batch_key=setup_args["batch_key"],
        categorical_covariate_keys=setup_args["categorical_covariate_keys"],
    )

    model = scvi.model.SCVI(
        adata,
        n_latent=args.n_latent,
        gene_likelihood=args.gene_likelihood,
    )

    model.train(
        max_epochs=args.max_epochs,
        early_stopping=True,
    )

    # ------------------------------------------------------------
    # Standardized embedding output
    # ------------------------------------------------------------

    embedding_key = "X_emb"
    adata.obsm[embedding_key] = model.get_latent_representation()

    # ------------------------------------------------------------
    # Metrics
    # Evaluation is always centered on species_key.
    # sample/technical metrics are computed if technical_key is provided.
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
    metrics_df.insert(2, "correction_mode", args.correction_mode)
    metrics_df.insert(3, "n_top_genes", args.n_top_genes)
    metrics_df.insert(4, "hvg_batch_key", hvg_batch_key_resolved)

    # ------------------------------------------------------------
    # Save everything
    # ------------------------------------------------------------

    save_run_outputs(
        adata=adata,
        result_dir=result_dir,
        config=config,
        metrics_df=metrics_df,
        embedding_key=embedding_key,
        model=model,
        save_adata=args.save_adata,
        save_embedding=True,
        save_model=True,
    )

    print(metrics_df)
    print(f"\nSaved run to:\n{result_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--adata_path", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--result_root", default="results")

    parser.add_argument(
        "--correction_mode",
        default="species",
        choices=["species", "technical", "both"],
        help=(
            "species: correct species effect; "
            "technical: correct technical_key such as sample/lab; "
            "both: correct technical_key and condition on species."
        ),
    )

    parser.add_argument("--counts_layer", default="counts")
    parser.add_argument("--species_key", default="species")
    parser.add_argument("--technical_key", default=None)
    parser.add_argument("--cell_type_key", default="cell_type_eval")

    parser.add_argument(
        "--n_top_genes",
        type=int,
        default=1200,
        help=(
            "Number of highly variable genes for scVI. "
            "Use 0 to train on all genes."
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

    parser.add_argument("--n_latent", type=int, default=20)
    parser.add_argument(
        "--gene_likelihood",
        default="nb",
        choices=["nb", "zinb", "poisson"],
    )
    parser.add_argument("--max_epochs", type=int, default=None)

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