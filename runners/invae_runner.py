from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import scanpy as sc

from benchmark.metrics import compute_integration_metrics
from benchmark.paths import create_run_directory
from benchmark.io import save_run_outputs


def select_hvg(
    adata,
    counts_layer,
    n_top_genes,
    hvg_batch_key,
):
    if n_top_genes is None or n_top_genes <= 0:
        return adata

    print(f"Selecting {n_top_genes} HVGs for inVAE")
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


def prepare_invae_covariates(
    adata,
    correction_mode,
    species_key,
    technical_key,
):
    if correction_mode == "species":
        spur = {
            "cont": [],
            "cat": [species_key],
        }

    elif correction_mode == "technical":
        spur = {
            "cont": [],
            "cat": [technical_key],
        }

    elif correction_mode == "both":
        spur = {
            "cont": [],
            "cat": [
                species_key,
                technical_key,
            ],
        }

    else:
        raise ValueError(f"Unknown correction mode {correction_mode}")

    for col in spur["cat"]:
        adata.obs[col] = adata.obs[col].astype("category")

    return spur


def prepare_invariant_covariate(
    adata,
    invariant_key,
):
    if invariant_key is None:
        return None

    if str(invariant_key).lower() in {"none", "null", ""}:
        return None

    if invariant_key not in adata.obs.columns:
        raise ValueError(
            f"invariant_key={invariant_key!r} not found in adata.obs. "
            f"Available columns are: {list(adata.obs.columns)}"
        )

    adata.obs[invariant_key] = adata.obs[invariant_key].astype("category")

    print(f"Using invariant covariate for inVAE: {invariant_key}")
    print(adata.obs[invariant_key].value_counts(dropna=False))

    return invariant_key


def make_model_name(args):
    warmup_tag = (
        f"_wu{args.warm_up_epochs}"
        if args.warm_up_epochs > 0
        else ""
    )

    es_tag = (
        f"_es{args.early_stopping_patience}"
        if args.early_stopping
        else ""
    )

    inv_tag = (
        f"_inv{args.invariant_key}"
        if args.invariant_key is not None
        and str(args.invariant_key).lower() not in {"none", "null", ""}
        else "_invnone"
    )

    tc_tag = f"_tc{args.tc_beta:g}"
    kl_tag = f"_kl{args.kl_rate:g}"

    model_name = (
        f"invae_correct_{args.correction_mode}_"
        f"{args.n_top_genes}hvg_"
        f"zinv{args.latent_dim_inv}_"
        f"zspur{args.latent_dim_spur}_"
        f"{args.n_epochs}epochs"
        f"{tc_tag}"
        f"{kl_tag}"
        f"{warmup_tag}"
        f"{es_tag}"
        f"{inv_tag}"
    )

    return model_name


def add_run_metadata(
    metrics,
    args,
    model_name,
    run_dir,
    embedding_key,
):
    metrics = metrics.copy()

    run_folder = Path(run_dir).name

    metrics["dataset"] = args.dataset_name
    metrics["model"] = model_name
    metrics["correction_mode"] = args.correction_mode
    metrics["n_top_genes"] = args.n_top_genes
    metrics["hvg_batch_key"] = args.hvg_batch_key
    metrics["embedding_key"] = embedding_key
    metrics["seed"] = args.seed
    metrics["run_folder"] = run_folder

    metrics["latent_dim_inv"] = args.latent_dim_inv
    metrics["latent_dim_spur"] = args.latent_dim_spur
    metrics["tc_beta"] = args.tc_beta
    metrics["kl_rate"] = args.kl_rate
    metrics["n_epochs"] = args.n_epochs
    metrics["warm_up_epochs"] = args.warm_up_epochs
    metrics["early_stopping"] = args.early_stopping
    metrics["early_stopping_patience"] = (
        args.early_stopping_patience
        if args.early_stopping
        else 0
    )

    metrics["invariant_key"] = (
        args.invariant_key
        if args.invariant_key is not None
        else "none"
    )

    front_cols = [
        "dataset",
        "model",
        "correction_mode",
        "n_top_genes",
        "hvg_batch_key",
        "embedding_key",
        "seed",
        "run_folder",
        "latent_dim_inv",
        "latent_dim_spur",
        "tc_beta",
        "kl_rate",
        "n_epochs",
        "warm_up_epochs",
        "early_stopping",
        "early_stopping_patience",
        "invariant_key",
    ]

    metrics = metrics[
        front_cols
        + [
            col
            for col in metrics.columns
            if col not in front_cols
        ]
    ]

    return metrics


def run_invae_worker(
    input_h5ad,
    output_npy,
    invae_env,
    repo_path,
    correction_mode,
    species_key,
    technical_key,
    n_epochs,
    seed,
    latent_dim_inv,
    latent_dim_spur,
    tc_beta,
    kl_rate,
    warm_up_epochs,
    early_stopping,
    early_stopping_patience,
    invariant_key,
):
    worker_cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        invae_env,
        "python",
        "scripts/run_invae_training.py",
        "--input_h5ad",
        str(input_h5ad),
        "--output_npy",
        str(output_npy),
        "--repo_path",
        str(repo_path),
        "--correction_mode",
        correction_mode,
        "--species_key",
        species_key,
        "--technical_key",
        technical_key,
        "--n_epochs",
        str(n_epochs),
        "--seed",
        str(seed),
        "--latent_dim_inv",
        str(latent_dim_inv),
        "--latent_dim_spur",
        str(latent_dim_spur),
        "--tc_beta",
        str(tc_beta),
        "--kl_rate",
        str(kl_rate),
        "--warm_up_epochs",
        str(warm_up_epochs),
    ]

    if invariant_key is not None and str(invariant_key).lower() not in {"none", "null", ""}:
        worker_cmd.extend(
            [
                "--invariant_key",
                invariant_key,
            ]
        )

    if early_stopping:
        worker_cmd.extend(
            [
                "--early_stopping",
                "--early_stopping_patience",
                str(early_stopping_patience),
            ]
        )

    print("Running inVAE worker:")
    print(" ".join(worker_cmd))

    subprocess.run(
        worker_cmd,
        check=True,
    )


def main(args):
    embedding_key = "X_emb"

    adata = sc.read_h5ad(args.adata_path)

    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    adata = select_hvg(
        adata,
        counts_layer=args.counts_layer,
        n_top_genes=args.n_top_genes,
        hvg_batch_key=args.hvg_batch_key,
    )

    invariant_key = prepare_invariant_covariate(
        adata=adata,
        invariant_key=args.invariant_key,
    )

    model_name = make_model_name(args)

    run_dir = create_run_directory(
        result_root=args.result_root,
        dataset_name=args.dataset_name,
        model_name=model_name,
        seed=args.seed,
    )

    tmp_dir = Path(run_dir) / "invae_input"
    tmp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_h5ad = tmp_dir / "input.h5ad"

    if args.counts_layer not in adata.layers:
        raise ValueError(
            f"Missing adata.layers['{args.counts_layer}']."
        )

    adata.X = adata.layers[args.counts_layer].copy()

    prepare_invae_covariates(
        adata,
        correction_mode=args.correction_mode,
        species_key=args.species_key,
        technical_key=args.technical_key,
    )

    adata.write_h5ad(input_h5ad)

    output_npy = (
        Path(run_dir)
        / "embeddings"
        / f"{embedding_key}.npy"
    )

    output_npy.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_invae_worker(
        input_h5ad=input_h5ad,
        output_npy=output_npy,
        invae_env=args.invae_env,
        repo_path=args.invae_repo,
        correction_mode=args.correction_mode,
        species_key=args.species_key,
        technical_key=args.technical_key,
        n_epochs=args.n_epochs,
        seed=args.seed,
        latent_dim_inv=args.latent_dim_inv,
        latent_dim_spur=args.latent_dim_spur,
        tc_beta=args.tc_beta,
        kl_rate=args.kl_rate,
        warm_up_epochs=args.warm_up_epochs,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        invariant_key=invariant_key,
    )

    embedding = np.load(output_npy)
    adata.obsm[embedding_key] = embedding

    metrics = compute_integration_metrics(
        adata=adata,
        embedding_key=embedding_key,
        cell_type_key=args.cell_type_key,
        species_key=args.species_key,
        sample_key=args.technical_key,
        n_neighbors=args.n_neighbors,
        leiden_resolution=args.leiden_resolution,
        random_state=args.seed,
    )

    metrics = add_run_metadata(
        metrics=metrics,
        args=args,
        model_name=model_name,
        run_dir=run_dir,
        embedding_key=embedding_key,
    )

    save_run_outputs(
        adata=adata,
        result_dir=run_dir,
        config=vars(args),
        metrics_df=metrics,
        embedding_key=embedding_key,
        model=None,
        save_adata=args.save_adata,
        save_embedding=True,
        save_model=False,
    )

    print(f"inVAE finished:\n{run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--adata_path", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--result_root", required=True)

    parser.add_argument(
        "--correction_mode",
        choices=[
            "species",
            "technical",
            "both",
        ],
        required=True,
    )

    parser.add_argument("--counts_layer", default="counts")
    parser.add_argument("--species_key", default="species")
    parser.add_argument("--technical_key", default="sample")
    parser.add_argument("--cell_type_key", default="cell_type_eval")

    parser.add_argument("--n_top_genes", type=int, default=1200)
    parser.add_argument("--hvg_batch_key", default="species")

    parser.add_argument("--n_epochs", type=int, default=500)
    parser.add_argument("--n_neighbors", type=int, default=20)
    parser.add_argument("--leiden_resolution", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--latent_dim_inv", type=int, default=9)
    parser.add_argument("--latent_dim_spur", type=int, default=1)

    parser.add_argument("--tc_beta", type=float, default=0.0)
    parser.add_argument("--kl_rate", type=float, default=1.0)

    parser.add_argument("--warm_up_epochs", type=int, default=0)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=50)

    parser.add_argument("--invariant_key", default=None)

    parser.add_argument("--invae_env", default="invae_env")
    parser.add_argument("--invae_repo", default="models/inVAE")

    parser.add_argument("--save_adata", action="store_true")

    args = parser.parse_args()
    main(args)