# benchmark/io.py

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


def save_run_outputs(
    adata,
    result_dir,
    config: dict,
    metrics_df: pd.DataFrame,
    embedding_key: str = "X_emb",
    model=None,
    save_adata: bool = True,
    save_embedding: bool = True,
    save_model: bool = True,
) -> None:
    """
    Save benchmark outputs for one model run.

    Saves:
        - config.json
        - run_info.json
        - metrics.csv
        - embedding.npy
        - adata_with_embedding.h5ad
        - model/ if model is provided
    """

    result_dir = Path(result_dir)

    if embedding_key not in adata.obsm:
        raise ValueError(f"{embedding_key} not found in adata.obsm.")

    # Save model
    if save_model and model is not None:
        model.save(
            result_dir / "model",
            overwrite=True,
        )

    # Save embedding
    if save_embedding:
        np.save(
            result_dir / "embeddings" / f"{embedding_key}.npy",
            adata.obsm[embedding_key],
        )

    # Save AnnData
    if save_adata:
        adata.write_h5ad(
            result_dir / "adata_with_embedding.h5ad"
        )

    # Save metrics
    metrics_df.to_csv(
        result_dir / "tables" / "metrics.csv",
        index=False,
    )

    # Save config
    with open(result_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # Save run metadata
    run_info = {
        "finished_at": datetime.now().isoformat(),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "embedding_key": embedding_key,
        "result_dir": str(result_dir),
    }

    with open(result_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=4)