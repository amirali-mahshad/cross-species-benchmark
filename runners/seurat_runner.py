# runners/seurat_runner.py

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.io import mmwrite

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
        print("Using all genes for Seurat.")
        return adata

    if counts_layer not in adata.layers:
        raise ValueError(f"Missing adata.layers['{counts_layer}'].")

    if hvg_batch_key is not None and hvg_batch_key not in adata.obs.columns:
        raise ValueError(f"Missing adata.obs['{hvg_batch_key}'] for HVG selection.")

    print(f"Selecting {n_top_genes} HVGs for Seurat.")
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
    method: str,
    correction_mode: str,
    species_key: str,
    technical_key: str | None,
    n_top_genes: int,
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
        raise ValueError("correction_mode must be one of: species, technical, both.")

    if n_top_genes is None or n_top_genes <= 0:
        gene_part = "allgenes"
    else:
        gene_part = f"{n_top_genes}hvg"

    return f"seurat_{method}_{correction_part}_{gene_part}"


def write_seurat_input_files(
    adata,
    outdir: str | Path,
    counts_layer: str,
):
    """
    Write AnnData into simple files that R/Seurat can read robustly.

    Files:
        counts.mtx  : genes x cells sparse matrix
        obs.csv     : cell metadata with cell_id column
        genes.tsv   : gene names
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    counts = adata.layers[counts_layer]

    if not sparse.issparse(counts):
        counts = sparse.csr_matrix(counts)

    # AnnData is cells x genes.
    # Seurat wants genes x cells.
    counts_gene_by_cell = counts.T.tocoo()

    counts_mtx = outdir / "counts.mtx"
    obs_csv = outdir / "obs.csv"
    genes_tsv = outdir / "genes.tsv"

    print(f"Writing counts matrix to: {counts_mtx}")
    mmwrite(str(counts_mtx), counts_gene_by_cell)

    obs = adata.obs.copy()
    obs.insert(0, "cell_id", adata.obs_names.astype(str))
    obs.to_csv(obs_csv, index=False)

    pd.Series(adata.var_names.astype(str)).to_csv(
        genes_tsv,
        index=False,
        header=False,
    )

    return {
        "counts_mtx": str(counts_mtx),
        "obs_csv": str(obs_csv),
        "genes_tsv": str(genes_tsv),
    }


def run_seurat_r_script(
    seurat_env: str,
    r_script: str,
    input_files: dict,
    output_csv: str,
    method: str,
    correction_mode: str,
    species_key: str,
    technical_key: str,
    n_pcs: int,
    seed: int,
):
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        seurat_env,
        "Rscript",
        r_script,
        "--counts_mtx",
        input_files["counts_mtx"],
        "--obs_csv",
        input_files["obs_csv"],
        "--genes_tsv",
        input_files["genes_tsv"],
        "--output_csv",
        output_csv,
        "--method",
        method,
        "--correction_mode",
        correction_mode,
        "--species_key",
        species_key,
        "--technical_key",
        technical_key,
        "--n_pcs",
        str(n_pcs),
        "--seed",
        str(seed),
    ]

    print("Running command:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)


def read_seurat_embedding(
    embedding_csv: str,
    cell_order,
):
    emb = pd.read_csv(embedding_csv)

    if "cell_id" not in emb.columns:
        raise ValueError("Seurat embedding output must contain a cell_id column.")

    emb = emb.set_index("cell_id")

    missing = set(cell_order) - set(emb.index)
    extra = set(emb.index) - set(cell_order)

    if len(missing) > 0:
        raise ValueError(f"Missing cells in Seurat embedding: {len(missing)}")

    if len(extra) > 0:
        print(f"Warning: extra cells in Seurat embedding: {len(extra)}")

    emb = emb.loc[cell_order]

    return emb.to_numpy(dtype=np.float32)


def main(args):
    # ------------------------------------------------------------
    # Load data
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
    # Validate obs/layers
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
    # Model name and output directory
    # ------------------------------------------------------------

    model_name = build_model_name(
        method=args.method,
        correction_mode=args.correction_mode,
        species_key=args.species_key,
        technical_key=args.technical_key,
        n_top_genes=args.n_top_genes,
    )

    result_dir = create_run_directory(
        result_root=args.result_root,
        dataset_name=args.dataset_name,
        model_name=model_name,
        seed=args.seed,
        overwrite=args.overwrite,
    )

    result_dir = Path(result_dir)

    seurat_input_dir = result_dir / "seurat_input"
    seurat_output_csv = result_dir / "embeddings" / "seurat_embedding.csv"
    seurat_output_csv.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Write files for R/Seurat
    # ------------------------------------------------------------

    input_files = write_seurat_input_files(
        adata=adata,
        outdir=seurat_input_dir,
        counts_layer=args.counts_layer,
    )

    # ------------------------------------------------------------
    # Run Seurat in R
    # ------------------------------------------------------------

    run_seurat_r_script(
        seurat_env=args.seurat_env,
        r_script=args.r_script,
        input_files=input_files,
        output_csv=str(seurat_output_csv),
        method=args.method,
        correction_mode=args.correction_mode,
        species_key=args.species_key,
        technical_key=args.technical_key,
        n_pcs=args.n_pcs,
        seed=args.seed,
    )

    # ------------------------------------------------------------
    # Read Seurat embedding back into AnnData
    # ------------------------------------------------------------

    embedding_key = "X_emb"

    adata.obsm[embedding_key] = read_seurat_embedding(
        embedding_csv=str(seurat_output_csv),
        cell_order=adata.obs_names.astype(str),
    )

    # ------------------------------------------------------------
    # Config
    # ------------------------------------------------------------

    config = {
        "runner": "seurat_runner",
        "dataset_name": args.dataset_name,
        "adata_path": args.adata_path,
        "model_name": model_name,
        "method": args.method,
        "correction_mode": args.correction_mode,

        "counts_layer": args.counts_layer,
        "species_key": args.species_key,
        "technical_key": args.technical_key,
        "cell_type_key": args.cell_type_key,

        "n_top_genes": args.n_top_genes,
        "hvg_batch_key_requested": args.hvg_batch_key,
        "hvg_batch_key_resolved": hvg_batch_key_resolved,

        "n_pcs": args.n_pcs,
        "n_neighbors": args.n_neighbors,
        "leiden_resolution": args.leiden_resolution,

        "seurat_env": args.seurat_env,
        "r_script": args.r_script,

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
    print(f"\nSaved Seurat run to:\n{result_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--adata_path", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--result_root", default="results")

    parser.add_argument(
        "--method",
        default="cca",
        choices=["cca", "rpca"],
        help="Seurat integration method: cca or rpca.",
    )

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

    parser.add_argument("--n_pcs", type=int, default=20)
    parser.add_argument("--n_neighbors", type=int, default=20)
    parser.add_argument("--leiden_resolution", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_adata", action="store_true")

    parser.add_argument(
        "--seurat_env",
        default="seurat_env",
        help="Conda environment containing R and Seurat.",
    )

    parser.add_argument(
        "--r_script",
        default="scripts/run_seurat_integration.R",
        help="Path to the R script that runs Seurat.",
    )

    args = parser.parse_args()
    main(args)
