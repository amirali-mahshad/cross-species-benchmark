# runners/harmony_runner.py

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import scanpy as sc
import scanpy.external as sce

from benchmark import (
    create_run_directory,
    compute_integration_metrics,
    save_run_outputs,
)


def build_harmony_keys(
    correction_mode: str,
    species_key: str,
    technical_key: str | None,
):
    """
    Decide which covariates Harmony should correct.

    correction_mode='species':
        key = species

    correction_mode='technical':
        key = technical_key, e.g. sample or lab

    correction_mode='both':
        key = [technical_key, species_key]
    """

    if correction_mode == "species":
        return species_key

    if correction_mode == "technical":
        if technical_key is None:
            raise ValueError(
                "technical_key must be provided when correction_mode='technical'."
            )
        return technical_key

    if correction_mode == "both":
        if technical_key is None:
            raise ValueError(
                "technical_key must be provided when correction_mode='both'."
            )
        return [technical_key, species_key]

    raise ValueError("correction_mode must be one of: species, technical, both.")


def build_model_name(
    correction_mode: str,
    species_key: str,
    technical_key: str | None,
    n_top_genes: int,
):
    """
    Build model name consistent with the resume-safe bash script.

    Examples:
        harmony_correct_species_1200hvg
        harmony_correct_sample_1200hvg
        harmony_correct_sample_plus_species_1200hvg
        harmony_correct_species_allgenes
    """

    if correction_mode == "species":
        correction_part = f"correct_{species_key}"

    elif correction_mode == "technical":
        if technical_key is None:
            raise ValueError("technical_key is required for correction_mode='technical'.")
        correction_part = f"correct_{technical_key}"

    elif correction_mode == "both":
        if technical_key is None:
            raise ValueError("technical_key is required for correction_mode='both'.")
        correction_part = f"correct_{technical_key}_plus_{species_key}"

    else:
        raise ValueError("Invalid correction_mode.")

    if n_top_genes is None or n_top_genes <= 0:
        gene_part = "allgenes"
    else:
        gene_part = f"{n_top_genes}hvg"

    return f"harmony_{correction_part}_{gene_part}"


def resolve_hvg_batch_key(hvg_batch_key: str | None):
    """
    Convert CLI hvg_batch_key to Scanpy-compatible value.
    """

    if hvg_batch_key is None:
        return None

    if hvg_batch_key == "none":
        return None

    return hvg_batch_key


def prepare_pca_for_harmony(
    adata,
    counts_layer: str,
    n_pcs: int,
    n_top_genes: int | None = None,
    hvg_batch_key: str | None = None,
    random_state: int = 0,
):
    """
    Prepare PCA input for Harmony.

    Harmony corrects PCA embeddings, not raw counts.

    Steps:
        1. copy raw counts into X
        2. optionally select HVGs
        3. normalize_total
        4. log1p
        5. scale
        6. PCA
    """

    if counts_layer not in adata.layers:
        raise ValueError(f"Missing adata.layers['{counts_layer}'].")

    if n_top_genes is not None and n_top_genes > 0:
        if hvg_batch_key is not None and hvg_batch_key not in adata.obs.columns:
            raise ValueError(
                f"Missing adata.obs['{hvg_batch_key}'] for HVG selection."
            )

        print(f"Selecting {n_top_genes} HVGs for Harmony.")
        print(f"HVG batch key: {hvg_batch_key}")

        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor="seurat_v3",
            layer=counts_layer,
            batch_key=hvg_batch_key,
            subset=True,
        )

    else:
        print("Using all genes for Harmony.")

    adata.X = adata.layers[counts_layer].copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Harmony usually runs on PCA of scaled log-normalized expression.
    sc.pp.scale(adata, max_value=10)

    n_comps = min(n_pcs, adata.n_obs - 1, adata.n_vars - 1)

    if n_comps < 2:
        raise ValueError(
            f"Not enough cells/genes for PCA. Computed n_comps={n_comps}."
        )

    sc.tl.pca(
        adata,
        n_comps=n_comps,
        svd_solver="arpack",
        random_state=random_state,
    )

    return adata


def save_harmony_model_artifact(
    result_dir,
    harmony_config: dict,
):
    """
    Harmony does not have neural-network weights like scVI.

    So we save the correction configuration as the model artifact.
    The actual corrected embedding is saved by save_run_outputs().
    """

    model_dir = Path(result_dir) / "model"
    model_dir.mkdir(exist_ok=True)

    with open(model_dir / "harmony_config.pkl", "wb") as f:
        pickle.dump(harmony_config, f)


def main(args):
    print("Reading AnnData:")
    print(args.adata_path)

    adata = sc.read_h5ad(args.adata_path)

    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    hvg_batch_key_resolved = resolve_hvg_batch_key(args.hvg_batch_key)

    model_name = build_model_name(
        correction_mode=args.correction_mode,
        species_key=args.species_key,
        technical_key=args.technical_key,
        n_top_genes=args.n_top_genes,
    )

    harmony_key = build_harmony_keys(
        correction_mode=args.correction_mode,
        species_key=args.species_key,
        technical_key=args.technical_key,
    )

    # ------------------------------------------------------------
    # Validate required columns
    # ------------------------------------------------------------

    required_obs = [args.species_key, args.cell_type_key]

    if args.technical_key is not None:
        required_obs.append(args.technical_key)

    if isinstance(harmony_key, list):
        required_obs.extend(harmony_key)
    else:
        required_obs.append(harmony_key)

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

    result_dir = create_run_directory(
        result_root=args.result_root,
        dataset_name=args.dataset_name,
        model_name=model_name,
        seed=args.seed,
        overwrite=args.overwrite,
    )

    config = {
        "runner": "harmony_runner",
        "dataset_name": args.dataset_name,
        "adata_path": args.adata_path,
        "model_name": model_name,
        "correction_mode": args.correction_mode,
        "counts_layer": args.counts_layer,
        "species_key": args.species_key,
        "technical_key": args.technical_key,
        "cell_type_key": args.cell_type_key,
        "harmony_key_used": harmony_key,
        "n_pcs": args.n_pcs,
        "n_top_genes": args.n_top_genes,
        "hvg_batch_key_requested": args.hvg_batch_key,
        "hvg_batch_key_resolved": hvg_batch_key_resolved,
        "n_neighbors": args.n_neighbors,
        "leiden_resolution": args.leiden_resolution,
        "theta": args.theta,
        "lambda_value": args.lambda_value,
        "max_iter_harmony": args.max_iter_harmony,
        "seed": args.seed,
        "save_adata": args.save_adata,
    }

    # ------------------------------------------------------------
    # PCA preparation
    # ------------------------------------------------------------

    adata = prepare_pca_for_harmony(
        adata=adata,
        counts_layer=args.counts_layer,
        n_pcs=args.n_pcs,
        n_top_genes=args.n_top_genes,
        hvg_batch_key=hvg_batch_key_resolved,
        random_state=args.seed,
    )

    # ------------------------------------------------------------
    # Harmony correction
    # ------------------------------------------------------------

    harmony_basis = "X_pca_harmony"

    harmony_kwargs = {
        "key": harmony_key,
        "basis": "X_pca",
        "adjusted_basis": harmony_basis,
        "max_iter_harmony": args.max_iter_harmony,
        "random_state": args.seed,
    }

    if args.theta is not None:
        harmony_kwargs["theta"] = args.theta

    if args.lambda_value is not None:
        # harmonypy uses 'lamb', but scanpy passes kwargs through.
        harmony_kwargs["lamb"] = args.lambda_value

    print("Running Harmony.")
    print(f"harmony_key: {harmony_key}")
    print(f"n_pcs: {args.n_pcs}")
    print(f"max_iter_harmony: {args.max_iter_harmony}")

    sce.pp.harmony_integrate(
        adata,
        **harmony_kwargs,
    )

    # ------------------------------------------------------------
    # Standardized embedding output
    # ------------------------------------------------------------

    embedding_key = "X_emb"

    X_harmony = adata.obsm[harmony_basis]

    # Safety check because harmonypy/scanpy versions may differ in orientation.
    # We want cells x PCs.
    if X_harmony.shape[0] != adata.n_obs and X_harmony.shape[1] == adata.n_obs:
        X_harmony = X_harmony.T

    if X_harmony.shape[0] != adata.n_obs:
        raise ValueError(
            "Harmony output has incompatible shape. "
            f"Expected first dimension = {adata.n_obs} cells, "
            f"got {X_harmony.shape}."
        )

    adata.obsm[embedding_key] = np.asarray(X_harmony, dtype=np.float32)

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
    metrics_df.insert(2, "correction_mode", args.correction_mode)
    metrics_df.insert(3, "n_top_genes", args.n_top_genes)
    metrics_df.insert(4, "hvg_batch_key", hvg_batch_key_resolved)

    # ------------------------------------------------------------
    # Save everything
    # ------------------------------------------------------------

    save_harmony_model_artifact(
        result_dir=result_dir,
        harmony_config=config,
    )

    save_run_outputs(
        adata=adata,
        result_dir=str(result_dir),
        config=config,
        metrics_df=metrics_df,
        embedding_key=embedding_key,
        model=None,
        save_adata=args.save_adata,
        save_embedding=True,
        save_model=False,
    )

    print(metrics_df)
    print("")
    print(f"Saved Harmony run to: {result_dir}")


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
            "both: correct technical_key and species together."
        ),
    )

    parser.add_argument("--counts_layer", default="counts")
    parser.add_argument("--species_key", default="species")
    parser.add_argument("--technical_key", default="sample")
    parser.add_argument("--cell_type_key", default="cell_type_eval")

    parser.add_argument("--n_pcs", type=int, default=20)
    parser.add_argument(
        "--n_top_genes",
        type=int,
        default=1200,
        help="Number of HVGs. Use 0 for all genes.",
    )
    parser.add_argument(
        "--hvg_batch_key",
        default="species",
        help="'none' or an obs column name such as species/sample.",
    )

    parser.add_argument("--n_neighbors", type=int, default=20)
    parser.add_argument("--leiden_resolution", type=float, default=1.0)

    parser.add_argument("--theta", type=float, default=None)
    parser.add_argument("--lambda_value", type=float, default=None)
    parser.add_argument("--max_iter_harmony", type=int, default=10)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_adata", action="store_true")

    args = parser.parse_args()
    main(args)