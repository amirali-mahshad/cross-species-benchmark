# runners/scetm_runner.py

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

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
    if n_top_genes is None or n_top_genes <= 0:
        print("Using all genes for scETM.")
        return adata

    if counts_layer not in adata.layers:
        raise ValueError(f"Missing adata.layers['{counts_layer}'].")

    if hvg_batch_key is not None and hvg_batch_key not in adata.obs.columns:
        raise ValueError(f"Missing adata.obs['{hvg_batch_key}'] for HVG selection.")

    print(f"Selecting {n_top_genes} HVGs for scETM.")
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


def build_batch_values_for_scetm(
    adata,
    correction_mode: str,
    species_key: str,
    technical_key: str | None,
):
    """
    scETM uses one batch column: adata.obs['batch_indices'].

    correction_mode='species':
        batch_indices = species

    correction_mode='technical':
        batch_indices = sample/lab

    correction_mode='both':
        batch_indices = composite(sample + species)

    Important:
        'both' is not statistically identical to scVI's
        batch_key + categorical_covariate_keys design.
    """

    if correction_mode == "species":
        return adata.obs[species_key].astype(str)

    if correction_mode == "technical":
        if technical_key is None:
            raise ValueError(
                "technical_key must be provided when correction_mode='technical'."
            )
        return adata.obs[technical_key].astype(str)

    if correction_mode == "both":
        if technical_key is None:
            raise ValueError(
                "technical_key must be provided when correction_mode='both'."
            )

        return (
            adata.obs[technical_key].astype(str)
            + "__"
            + adata.obs[species_key].astype(str)
        )

    raise ValueError("correction_mode must be one of: species, technical, both.")


def prepare_scetm_input_adata(
    adata,
    counts_layer: str,
    correction_mode: str,
    species_key: str,
    technical_key: str | None,
    cell_type_key: str,
):
    """
    Prepare AnnData for scETM.

    scETM expects:
        adata.X                      cells x genes count matrix
        adata.obs['batch_indices']   batch labels
        adata.obs['cell_types']      cell type labels
    """

    if counts_layer not in adata.layers:
        raise ValueError(f"Missing adata.layers['{counts_layer}'].")

    batch_values = build_batch_values_for_scetm(
        adata=adata,
        correction_mode=correction_mode,
        species_key=species_key,
        technical_key=technical_key,
    )

    batch_codes = pd.Categorical(batch_values).codes.astype(int)

    adata.obs["batch_indices_original"] = batch_values.values
    adata.obs["batch_indices"] = batch_codes
    adata.obs["cell_types"] = adata.obs[cell_type_key].astype(str).values

    counts = adata.layers[counts_layer]

    if sparse.issparse(counts):
        adata.X = counts.copy()
    else:
        adata.X = np.asarray(counts).copy()

    return adata


def build_model_name(
    correction_mode: str,
    species_key: str,
    technical_key: str | None,
    n_top_genes: int,
    embedding_output: str,
):
    if correction_mode == "species":
        correction_part = f"correct_{species_key}"

    elif correction_mode == "technical":
        if technical_key is None:
            raise ValueError(
                "technical_key must be provided when correction_mode='technical'."
            )
        correction_part = f"correct_{technical_key}"

    elif correction_mode == "both":
        if technical_key is None:
            raise ValueError(
                "technical_key must be provided when correction_mode='both'."
            )
        correction_part = f"correct_{technical_key}_plus_{species_key}"

    else:
        raise ValueError("Invalid correction_mode.")

    if n_top_genes is None or n_top_genes <= 0:
        gene_part = "allgenes"
    else:
        gene_part = f"{n_top_genes}hvg"

    return f"scetm_{correction_part}_{gene_part}_{embedding_output}"


def run_scetm_worker(
    scetm_env: str,
    worker_script: str,
    scetm_repo: str,
    input_h5ad: str,
    output_embedding_npy: str,
    output_h5ad: str,
    ckpt_dir: str,
    train_instance_name: str,
    embedding_output: str,
    n_epochs: int,
    eval_every: int,
    n_samplers: int,
    seed: int,
    save_model_ckpt: bool,
):
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        scetm_env,
        "python",
        worker_script,
        "--input_h5ad",
        input_h5ad,
        "--output_embedding_npy",
        output_embedding_npy,
        "--output_h5ad",
        output_h5ad,
        "--repo_path",
        scetm_repo,
        "--ckpt_dir",
        ckpt_dir,
        "--train_instance_name",
        train_instance_name,
        "--embedding_key",
        embedding_output,
        "--n_epochs",
        str(n_epochs),
        "--eval_every",
        str(eval_every),
        "--n_samplers",
        str(n_samplers),
        "--seed",
        str(seed),
    ]

    if save_model_ckpt:
        cmd.append("--save_model_ckpt")

    print("Running scETM worker command:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)


def main(args):
    # ------------------------------------------------------------
    # Load data in proj3
    # ------------------------------------------------------------

    adata = sc.read_h5ad(args.adata_path)

    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    # ------------------------------------------------------------
    # Resolve HVG batch key
    # ------------------------------------------------------------

    hvg_batch_key_resolved = resolve_hvg_batch_key(
        hvg_batch_key=args.hvg_batch_key,
        species_key=args.species_key,
        technical_key=args.technical_key,
    )

    # ------------------------------------------------------------
    # Validate
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
    # Prepare scETM input columns
    # ------------------------------------------------------------

    adata = prepare_scetm_input_adata(
        adata=adata,
        counts_layer=args.counts_layer,
        correction_mode=args.correction_mode,
        species_key=args.species_key,
        technical_key=args.technical_key,
        cell_type_key=args.cell_type_key,
    )

    # ------------------------------------------------------------
    # Model name and output directory
    # ------------------------------------------------------------

    model_name = build_model_name(
        correction_mode=args.correction_mode,
        species_key=args.species_key,
        technical_key=args.technical_key,
        n_top_genes=args.n_top_genes,
        embedding_output=args.embedding_output,
    )

    result_dir = create_run_directory(
        result_root=args.result_root,
        dataset_name=args.dataset_name,
        model_name=model_name,
        seed=args.seed,
        overwrite=args.overwrite,
    )

    result_dir = Path(result_dir)

    scetm_input_dir = result_dir / "scetm_input"
    scetm_input_dir.mkdir(parents=True, exist_ok=True)

    model_dir = result_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    embeddings_dir = result_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    input_h5ad = scetm_input_dir / "scetm_input.h5ad"
    output_embedding_npy = embeddings_dir / f"scetm_{args.embedding_output}.npy"
    output_h5ad = scetm_input_dir / "scetm_worker_output.h5ad"

    print(f"Writing scETM input AnnData to: {input_h5ad}")
    adata.write_h5ad(input_h5ad)

    # ------------------------------------------------------------
    # Run scETM in scETM conda environment
    # ------------------------------------------------------------

    train_instance_name = f"{model_name}_seed{args.seed}"

    run_scetm_worker(
        scetm_env=args.scetm_env,
        worker_script=args.worker_script,
        scetm_repo=args.scetm_repo,
        input_h5ad=str(input_h5ad),
        output_embedding_npy=str(output_embedding_npy),
        output_h5ad=str(output_h5ad),
        ckpt_dir=str(model_dir),
        train_instance_name=train_instance_name,
        embedding_output=args.embedding_output,
        n_epochs=args.n_epochs,
        eval_every=args.eval_every,
        n_samplers=args.n_samplers,
        seed=args.seed,
        save_model_ckpt=args.save_model_ckpt,
    )

    # ------------------------------------------------------------
    # Read embedding back into benchmark AnnData
    # ------------------------------------------------------------

    embedding = np.load(output_embedding_npy)

    if embedding.shape[0] != adata.n_obs:
        raise ValueError(
            f"Embedding has {embedding.shape[0]} cells but adata has {adata.n_obs}."
        )

    embedding_key = "X_emb"
    adata.obsm[embedding_key] = embedding.astype(np.float32)

    # ------------------------------------------------------------
    # Config
    # ------------------------------------------------------------

    config = {
        "runner": "scetm_runner",
        "dataset_name": args.dataset_name,
        "adata_path": args.adata_path,
        "model_name": model_name,
        "correction_mode": args.correction_mode,

        "counts_layer": args.counts_layer,
        "species_key": args.species_key,
        "technical_key": args.technical_key,
        "cell_type_key": args.cell_type_key,

        "batch_column_used_by_scetm": "batch_indices",
        "batch_indices_original_column": "batch_indices_original",
        "cell_type_column_used_by_scetm": "cell_types",

        "n_top_genes": args.n_top_genes,
        "hvg_batch_key_requested": args.hvg_batch_key,
        "hvg_batch_key_resolved": hvg_batch_key_resolved,

        "embedding_output": args.embedding_output,
        "n_epochs": args.n_epochs,
        "eval_every": args.eval_every,
        "n_samplers": args.n_samplers,

        "scetm_env": args.scetm_env,
        "scetm_repo": args.scetm_repo,
        "worker_script": args.worker_script,

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
    metrics_df.insert(2, "correction_mode", args.correction_mode)
    metrics_df.insert(3, "n_top_genes", args.n_top_genes)
    metrics_df.insert(4, "hvg_batch_key", hvg_batch_key_resolved)

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------

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
    print(f"\nSaved scETM run to:\n{result_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--adata_path", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--result_root", default="results")

    parser.add_argument(
        "--correction_mode",
        default="species",
        choices=["species", "technical", "both"],
    )

    parser.add_argument("--counts_layer", default="counts")
    parser.add_argument("--species_key", default="species")
    parser.add_argument("--technical_key", default="sample")
    parser.add_argument("--cell_type_key", default="cell_type_eval")

    parser.add_argument(
        "--n_top_genes",
        type=int,
        default=1200,
        help="Number of HVGs. Use 0 for all genes.",
    )

    parser.add_argument(
        "--hvg_batch_key",
        default="species",
        help="'auto', 'none', or an obs column name such as species/sample.",
    )

    parser.add_argument(
        "--embedding_output",
        default="delta",
        choices=["delta", "theta"],
        help="Which scETM embedding to benchmark.",
    )

    parser.add_argument("--n_epochs", type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--n_samplers", type=int, default=4)

    parser.add_argument("--n_neighbors", type=int, default=20)
    parser.add_argument("--leiden_resolution", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_adata", action="store_true")

    parser.add_argument(
        "--scetm_env",
        default="scETM",
        help="Conda environment containing scETM.",
    )

    parser.add_argument(
        "--scetm_repo",
        default="models/scETM",
        help="Path to cloned scETM repo.",
    )

    parser.add_argument(
        "--worker_script",
        default="scripts/run_scetm_training.py",
        help="Worker script executed inside the scETM environment.",
    )

    parser.add_argument(
        "--save_model_ckpt",
        action="store_true",
        help="Ask scETM trainer to save model checkpoint if supported.",
    )

    args = parser.parse_args()
    main(args)