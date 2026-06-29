# runners/scvi_runner.py

import argparse
from pathlib import Path

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
        batch_key = species

    correction_mode='technical':
        batch_key = sample/lab/etc.

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


def build_model_name(
    correction_mode: str,
    species_key: str,
    technical_key: str | None,
):
    if correction_mode == "species":
        return f"scvi_correct_{species_key}"

    if correction_mode == "technical":
        return f"scvi_correct_{technical_key}"

    if correction_mode == "both":
        return f"scvi_correct_{technical_key}_plus_{species_key}"

    raise ValueError("Invalid correction_mode.")


def main(args):
    scvi.settings.seed = args.seed

    adata = sc.read_h5ad(args.adata_path)

    model_name = build_model_name(
        correction_mode=args.correction_mode,
        species_key=args.species_key,
        technical_key=args.technical_key,
    )

    setup_args = build_scvi_setup_args(
        correction_mode=args.correction_mode,
        species_key=args.species_key,
        technical_key=args.technical_key,
    )

    result_dir = create_run_directory(
        result_root=args.result_root,
        dataset_name=args.dataset_name,
        model_name=model_name,
        seed=args.seed,
        overwrite=args.overwrite,
    )

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
        "n_latent": args.n_latent,
        "gene_likelihood": args.gene_likelihood,
        "max_epochs": args.max_epochs,
        "n_neighbors": args.n_neighbors,
        "leiden_resolution": args.leiden_resolution,
        "seed": args.seed,
    }

    # ------------------------------------------------------------
    # Validate required columns
    # ------------------------------------------------------------

    required_obs = [args.species_key, args.cell_type_key]

    if args.technical_key is not None:
        required_obs.append(args.technical_key)

    for key in required_obs:
        if key not in adata.obs.columns:
            raise ValueError(f"Missing adata.obs['{key}'].")

    if args.counts_layer not in adata.layers:
        raise ValueError(f"Missing adata.layers['{args.counts_layer}'].")

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
    # sample/technical metrics are computed only if technical_key exists.
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
        save_adata=True,
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

    parser.add_argument("--n_latent", type=int, default=20)
    parser.add_argument("--gene_likelihood", default="nb", choices=["nb", "zinb", "poisson"])
    parser.add_argument("--max_epochs", type=int, default=None)

    parser.add_argument("--n_neighbors", type=int, default=20)
    parser.add_argument("--leiden_resolution", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    main(args)