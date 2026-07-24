from __future__ import annotations

import argparse
import random
import sys

import anndata as ad
import numpy as np
import torch


def build_spurious_covariates(args):
    if args.correction_mode == "species":
        return {
            "cont": [],
            "cat": [args.species_key],
        }

    elif args.correction_mode == "technical":
        return {
            "cont": [],
            "cat": [args.technical_key],
        }

    elif args.correction_mode == "both":
        return {
            "cont": [],
            "cat": [
                args.species_key,
                args.technical_key,
            ],
        }

    else:
        raise ValueError(
            f"Unknown correction_mode: {args.correction_mode}"
        )


def build_invariant_covariates(data, args):
    if args.invariant_key is None:
        return {
            "cont": [],
            "cat": [],
        }

    if str(args.invariant_key).lower() in {"none", "null", ""}:
        return {
            "cont": [],
            "cat": [],
        }

    if args.invariant_key not in data.obs.columns:
        raise ValueError(
            f"invariant_key={args.invariant_key!r} not found in data.obs. "
            f"Available obs columns are: {list(data.obs.columns)}"
        )

    data.obs[args.invariant_key] = (
        data.obs[args.invariant_key]
        .astype("category")
    )

    return {
        "cont": [],
        "cat": [args.invariant_key],
    }


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    sys.path.insert(0, args.repo_path)

    from inVAE import FinVAE

    data = ad.read_h5ad(args.input_h5ad)

    for key in [args.species_key, args.technical_key]:
        if key in data.obs.columns:
            data.obs[key] = data.obs[key].astype("category")

    spur = build_spurious_covariates(args)
    inv = build_invariant_covariates(data, args)

    print("inVAE invariant covariates:")
    print(inv)

    print("inVAE spurious covariates:")
    print(spur)

    print(f"latent_dim_inv: {args.latent_dim_inv}")
    print(f"latent_dim_spur: {args.latent_dim_spur}")
    print(f"tc_beta: {args.tc_beta}")
    print(f"kl_rate: {args.kl_rate}")

    model = FinVAE(
        adata=data,
        layer="counts",
        inv_covar_keys=inv,
        spur_covar_keys=spur,
        latent_dim_inv=args.latent_dim_inv,
        latent_dim_spur=args.latent_dim_spur,
        tc_beta=args.tc_beta,
        kl_rate=args.kl_rate,
    )

    early_stopping_patience = (
        args.early_stopping_patience
        if args.early_stopping
        else 0
    )

    model.train(
        n_epochs=args.n_epochs,
        lr_train=1e-3,
        weight_decay=1e-6,
        early_stopping=args.early_stopping,
        early_stopping_patience=early_stopping_patience,
        warm_up_epochs=args.warm_up_epochs,
    )

    latent = model.get_latent_representation(
        data,
        latent_type="invariant",
    )

    np.save(
        args.output_npy,
        latent,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_h5ad", required=True)
    parser.add_argument("--output_npy", required=True)
    parser.add_argument("--repo_path", required=True)

    parser.add_argument(
        "--correction_mode",
        choices=[
            "species",
            "technical",
            "both",
        ],
        required=True,
    )

    parser.add_argument("--species_key", default="species")
    parser.add_argument("--technical_key", default="sample")

    parser.add_argument("--n_epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--latent_dim_inv", type=int, default=9)
    parser.add_argument("--latent_dim_spur", type=int, default=1)

    parser.add_argument("--tc_beta", type=float, default=0.0)
    parser.add_argument("--kl_rate", type=float, default=1.0)

    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=50)
    parser.add_argument("--warm_up_epochs", type=int, default=0)

    parser.add_argument("--invariant_key", default=None)

    args = parser.parse_args()
    main(args)
