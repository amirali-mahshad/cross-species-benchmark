# scripts/run_scetm_training.py

import argparse
import random
import sys
from pathlib import Path

import anndata as ad
import numpy as np


def add_scetm_repo_to_path(repo_path: str | None):
    """
    Add cloned scETM repo to Python path.

    Expected clone location in your project:
        models/scETM

    The repo contains:
        src/scETM
    """

    if repo_path is None:
        return

    repo = Path(repo_path).resolve()

    if not repo.exists():
        raise FileNotFoundError(f"scETM repo_path does not exist: {repo}")

    src = repo / "src"

    sys.path.insert(0, str(src))
    sys.path.insert(0, str(repo))

    print(f"Added scETM repo to sys.path: {repo}")
    print(f"Added scETM src to sys.path: {src}")


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def train_scetm(args):
    add_scetm_repo_to_path(args.repo_path)
    set_all_seeds(args.seed)

    from scETM import scETM as ScETMModel
    from scETM import UnsupervisedTrainer

    try:
        from scETM import set_seed

        set_seed(args.seed)
    except Exception:
        pass

    print("Reading input AnnData:")
    print(args.input_h5ad)

    adata = ad.read_h5ad(args.input_h5ad)

    if "batch_indices" not in adata.obs.columns:
        raise ValueError("scETM input must contain adata.obs['batch_indices'].")

    if "cell_types" not in adata.obs.columns:
        raise ValueError("scETM input must contain adata.obs['cell_types'].")

    n_genes = adata.n_vars
    n_batches = int(adata.obs["batch_indices"].nunique())

    print("AnnData:")
    print(adata)
    print(f"n_genes: {n_genes}")
    print(f"n_batches: {n_batches}")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing scETM model.")

    model = ScETMModel(
        n_genes,
        n_batches,
        enable_batch_bias=True,
    )

    trainer = UnsupervisedTrainer(
        model,
        adata,
        train_instance_name=args.train_instance_name,
        ckpt_dir=str(ckpt_dir),
    )

    print("Training scETM.")
    print(f"n_epochs: {args.n_epochs}")
    print(f"eval_every: {args.eval_every}")
    print(f"n_samplers: {args.n_samplers}")

    # scETM versions can differ slightly in train() arguments.
    # Try the richer call first, then fall back if needed.
    try:
        trainer.train(
            n_epochs=args.n_epochs,
            eval_every=args.eval_every,
            n_samplers=args.n_samplers,
            eval_kwargs={"cell_type_col": "cell_types"},
            save_model_ckpt=args.save_model_ckpt,
        )
    except TypeError:
        try:
            trainer.train(
                n_epochs=args.n_epochs,
                eval_every=args.eval_every,
                n_samplers=args.n_samplers,
                eval_kwargs={"cell_type_col": "cell_types"},
            )
        except TypeError:
            trainer.train(
                n_epochs=args.n_epochs,
                eval_every=args.eval_every,
                n_samplers=args.n_samplers,
            )

    print("Extracting scETM embeddings.")

    model.get_all_embeddings_and_nll(adata)

    if args.embedding_key not in adata.obsm.keys():
        raise ValueError(
            f"Embedding key '{args.embedding_key}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )

    embedding = np.asarray(adata.obsm[args.embedding_key], dtype=np.float32)

    output_embedding_npy = Path(args.output_embedding_npy)
    output_embedding_npy.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_embedding_npy, embedding)

    print(f"Saved scETM embedding to: {output_embedding_npy}")
    print(f"Embedding shape: {embedding.shape}")

    if args.output_h5ad is not None:
        output_h5ad = Path(args.output_h5ad)
        output_h5ad.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(output_h5ad)
        print(f"Saved scETM output AnnData to: {output_h5ad}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_h5ad", required=True)
    parser.add_argument("--output_embedding_npy", required=True)
    parser.add_argument("--output_h5ad", default=None)

    parser.add_argument(
        "--repo_path",
        default="models/scETM",
        help="Path to cloned scETM repo. Example: models/scETM",
    )

    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--train_instance_name", default="scETM_run")

    parser.add_argument(
        "--embedding_key",
        default="delta",
        choices=["delta", "theta"],
        help="scETM embedding to export. README tutorial evaluates delta.",
    )

    parser.add_argument("--n_epochs", type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--n_samplers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--save_model_ckpt",
        action="store_true",
        help="Ask scETM trainer to save model checkpoints if supported.",
    )

    args = parser.parse_args()
    train_scetm(args)
