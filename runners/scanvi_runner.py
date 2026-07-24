
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch

from benchmark.metrics import compute_integration_metrics
from benchmark.paths import create_run_directory
from benchmark.io import save_run_outputs


def sanitize_tag(x):
    x = str(x)
    x = re.sub(r"[^A-Za-z0-9_.-]+", "-", x)
    return x.strip("-")


def select_hvg(
    adata,
    counts_layer,
    n_top_genes,
    hvg_batch_key,
):
    if n_top_genes is None or n_top_genes <= 0:
        return adata

    print(f"Selecting {n_top_genes} HVGs for scANVI")
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


def prepare_labels(
    adata,
    labels_key,
    unlabeled_category,
):
    if labels_key not in adata.obs.columns:
        raise ValueError(
            f"labels_key={labels_key!r} not found in adata.obs. "
            f"Available obs columns are: {list(adata.obs.columns)}"
        )

    labels = (
        adata.obs[labels_key]
        .astype("object")
        .where(~adata.obs[labels_key].isna(), unlabeled_category)
        .astype(str)
    )

    labels = pd.Categorical(labels)

    if unlabeled_category not in labels.categories:
        labels = labels.add_categories([unlabeled_category])

    adata.obs[labels_key] = labels

    print(f"Using scANVI labels_key: {labels_key}")
    print(f"Unlabeled category: {unlabeled_category}")
    print(adata.obs[labels_key].value_counts(dropna=False))

    return adata


def prepare_correction_covariates(
    adata,
    correction_mode,
    species_key,
    technical_key,
):
    if correction_mode == "species":
        batch_key = species_key
        categorical_covariate_keys = None

    elif correction_mode == "technical":
        batch_key = technical_key
        categorical_covariate_keys = None

    elif correction_mode == "both":
        # Use sample as primary batch and species as additional nuisance covariate.
        batch_key = technical_key
        categorical_covariate_keys = [species_key]

    else:
        raise ValueError(f"Unknown correction_mode={correction_mode!r}")

    used_keys = [batch_key]
    if categorical_covariate_keys is not None:
        used_keys.extend(categorical_covariate_keys)

    for key in used_keys:
        if key not in adata.obs.columns:
            raise ValueError(
                f"Covariate key {key!r} not found in adata.obs. "
                f"Available obs columns are: {list(adata.obs.columns)}"
            )
        adata.obs[key] = adata.obs[key].astype("category")

    print("scANVI correction setup:")
    print(f"  batch_key: {batch_key}")
    print(f"  categorical_covariate_keys: {categorical_covariate_keys}")

    return batch_key, categorical_covariate_keys


def make_model_name(args):
    label_tag = sanitize_tag(args.labels_key)

    model_name = (
        f"scanvi_correct_{args.correction_mode}_"
        f"{args.n_top_genes}hvg_"
        f"label{label_tag}_"
        f"{args.n_latent}latent_"
        f"scvi{args.scvi_epochs}epochs_"
        f"scanvi{args.scanvi_epochs}epochs"
    )

    return model_name


def add_run_metadata(
    metrics,
    args,
    model_name,
    run_dir,
    embedding_key,
    batch_key,
    categorical_covariate_keys,
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

    metrics["labels_key"] = args.labels_key
    metrics["unlabeled_category"] = args.unlabeled_category
    metrics["n_latent"] = args.n_latent
    metrics["n_layers"] = args.n_layers
    metrics["n_hidden"] = args.n_hidden
    metrics["gene_likelihood"] = args.gene_likelihood
    metrics["scvi_epochs"] = args.scvi_epochs
    metrics["scanvi_epochs"] = args.scanvi_epochs
    metrics["batch_key"] = batch_key
    metrics["categorical_covariate_keys"] = (
        ",".join(categorical_covariate_keys)
        if categorical_covariate_keys is not None
        else ""
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
        "labels_key",
        "unlabeled_category",
        "n_latent",
        "n_layers",
        "n_hidden",
        "gene_likelihood",
        "scvi_epochs",
        "scanvi_epochs",
        "batch_key",
        "categorical_covariate_keys",
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


def train_scanvi(
    adata,
    args,
    batch_key,
    categorical_covariate_keys,
):
    import scvi

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    scvi.settings.seed = args.seed
    scvi.settings.dl_num_workers = 0

    # Register AnnData for scVI first.
    # labels_key is included so SCANVI.from_scvi_model can reuse it.
    scvi.model.SCVI.setup_anndata(
        adata,
        layer=args.counts_layer,
        batch_key=batch_key,
        labels_key=args.labels_key,
        categorical_covariate_keys=categorical_covariate_keys,
    )

    scvi_model = scvi.model.SCVI(
        adata,
        n_latent=args.n_latent,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        gene_likelihood=args.gene_likelihood,
    )

    print("Training base scVI model before scANVI")
    scvi_model.train(
        max_epochs=args.scvi_epochs,
        train_size=args.train_size,
        validation_size=args.validation_size,
        batch_size=args.batch_size,
        early_stopping=args.early_stopping,
        accelerator=args.accelerator,
        devices=args.devices,
    )

    print("Initializing scANVI from pretrained scVI")
    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        scvi_model,
        unlabeled_category=args.unlabeled_category,
        labels_key=args.labels_key,
    )

    print("Training scANVI model")
    scanvi_model.train(
        max_epochs=args.scanvi_epochs,
        train_size=args.train_size,
        validation_size=args.validation_size,
        batch_size=args.batch_size,
        n_samples_per_label=args.n_samples_per_label,
        accelerator=args.accelerator,
        devices=args.devices,
    )

    return scanvi_model


def main(args):
    embedding_key = "X_emb"

    adata = sc.read_h5ad(args.adata_path)

    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    if args.counts_layer not in adata.layers:
        raise ValueError(
            f"Missing adata.layers[{args.counts_layer!r}]. "
            f"Available layers are: {list(adata.layers.keys())}"
        )

    adata = select_hvg(
        adata=adata,
        counts_layer=args.counts_layer,
        n_top_genes=args.n_top_genes,
        hvg_batch_key=args.hvg_batch_key,
    )

    prepare_labels(
        adata=adata,
        labels_key=args.labels_key,
        unlabeled_category=args.unlabeled_category,
    )

    batch_key, categorical_covariate_keys = prepare_correction_covariates(
        adata=adata,
        correction_mode=args.correction_mode,
        species_key=args.species_key,
        technical_key=args.technical_key,
    )

    model_name = make_model_name(args)

    run_dir = create_run_directory(
        result_root=args.result_root,
        dataset_name=args.dataset_name,
        model_name=model_name,
        seed=args.seed,
    )

    scanvi_model = train_scanvi(
        adata=adata,
        args=args,
        batch_key=batch_key,
        categorical_covariate_keys=categorical_covariate_keys,
    )

    latent = scanvi_model.get_latent_representation()
    adata.obsm[embedding_key] = latent

    try:
        adata.obs["scanvi_pred_label"] = scanvi_model.predict()
    except Exception as e:
        print(f"WARNING: Could not add scanvi predictions: {e}")

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
        batch_key=batch_key,
        categorical_covariate_keys=categorical_covariate_keys,
    )

    if args.save_model:
        model_dir = Path(run_dir) / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        scanvi_model.save(
            model_dir,
            overwrite=True,
            save_anndata=False,
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

    print(f"scANVI finished:\n{run_dir}")


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

    parser.add_argument(
        "--labels_key",
        default="cell_type_broad_inv",
        help="Broad labels used by scANVI, e.g. cell_type_broad_inv.",
    )

    parser.add_argument(
        "--unlabeled_category",
        default="Unknown",
        help="Category name used by scANVI for unlabeled cells.",
    )

    parser.add_argument("--n_top_genes", type=int, default=1200)
    parser.add_argument("--hvg_batch_key", default="species")

    parser.add_argument("--n_latent", type=int, default=20)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_hidden", type=int, default=128)
    parser.add_argument(
        "--gene_likelihood",
        choices=[
            "nb",
            "zinb",
            "poisson",
        ],
        default="nb",
    )

    parser.add_argument("--scvi_epochs", type=int, default=400)
    parser.add_argument("--scanvi_epochs", type=int, default=200)

    parser.add_argument("--train_size", type=float, default=0.9)
    parser.add_argument("--validation_size", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_samples_per_label", type=int, default=None)

    parser.add_argument("--early_stopping", action="store_true")

    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")

    parser.add_argument("--n_neighbors", type=int, default=20)
    parser.add_argument("--leiden_resolution", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--save_adata", action="store_true")
    parser.add_argument("--save_model", action="store_true")

    args = parser.parse_args()
    main(args)
