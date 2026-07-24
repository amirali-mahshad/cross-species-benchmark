# scripts/run_posthoc_metrics.py

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from benchmark.alcs import compute_alcs
from benchmark.intra_metrics import compute_intra_metrics
from benchmark.reference import load_or_create_reference


def find_result_folders(result_root: Path):
    folders = []

    for folder in sorted(result_root.iterdir()):
        if not folder.is_dir():
            continue

        if folder.name.startswith("_"):
            continue

        metrics_path = folder / "tables" / "metrics.csv"
        adata_path = folder / "adata_with_embedding.h5ad"
        emb_dir = folder / "embeddings"

        has_embedding = False

        if adata_path.exists():
            has_embedding = True

        if emb_dir.exists() and len(list(emb_dir.glob("*.npy"))) > 0:
            has_embedding = True

        if metrics_path.exists() and has_embedding:
            folders.append(folder)

    return folders


def find_embedding_npy(folder: Path, embedding_key: str = "X_emb"):
    emb_dir = folder / "embeddings"

    if not emb_dir.exists():
        return None

    candidates = [
        emb_dir / f"{embedding_key}.npy",
        emb_dir / "X_emb.npy",
        emb_dir / "embedding.npy",
    ]

    for path in candidates:
        if path.exists():
            return path

    all_npys = sorted(emb_dir.glob("*.npy"))

    if len(all_npys) == 1:
        return all_npys[0]

    if len(all_npys) > 1:
        for p in all_npys:
            if embedding_key in p.name:
                return p

        return all_npys[0]

    return None


def load_run_embedding(
    folder: Path,
    reference_obs_names,
    original_adata_path: str,
    embedding_key: str = "X_emb",
):
    """
    Load X_emb for a result folder.

    Preferred:
        adata_with_embedding.h5ad

    Fallback:
        embeddings/*.npy
    """

    adata_path = folder / "adata_with_embedding.h5ad"

    if adata_path.exists():
        adata = sc.read_h5ad(adata_path)

        if embedding_key not in adata.obsm:
            raise ValueError(
                f"{folder.name}: missing adata.obsm['{embedding_key}']."
            )

        # Reorder if possible
        if list(adata.obs_names.astype(str)) != list(reference_obs_names):
            if set(reference_obs_names).issubset(set(adata.obs_names.astype(str))):
                adata = adata[reference_obs_names].copy()
            else:
                raise ValueError(
                    f"{folder.name}: obs_names do not match reference."
                )

        return np.asarray(adata.obsm[embedding_key], dtype=np.float32)

    emb_path = find_embedding_npy(folder, embedding_key=embedding_key)

    if emb_path is None:
        raise FileNotFoundError(f"{folder.name}: no embedding npy found.")

    emb = np.load(emb_path).astype(np.float32)

    if emb.shape[0] != len(reference_obs_names):
        raise ValueError(
            f"{folder.name}: embedding has {emb.shape[0]} cells, "
            f"reference has {len(reference_obs_names)}."
        )

    return emb


def safe_concat_metrics(base_df, posthoc_df):
    base_df = base_df.reset_index(drop=True)
    posthoc_df = posthoc_df.reset_index(drop=True)

    duplicated = [c for c in posthoc_df.columns if c in base_df.columns]

    if len(duplicated) > 0:
        posthoc_df = posthoc_df.drop(columns=duplicated)

    return pd.concat([base_df, posthoc_df], axis=1)


def main(args):
    result_root = Path(args.result_root)
    result_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 1. Reference embedding
    # ------------------------------------------------------------

    adata_ref, ref_dir = load_or_create_reference(
        adata_path=args.adata_path,
        result_root=args.result_root,
        dataset_name=args.dataset_name,
        counts_layer=args.counts_layer,
        species_key=args.species_key,
        cell_type_key=args.cell_type_key,
        technical_key=args.technical_key,
        n_top_genes=args.n_top_genes,
        hvg_batch_key=args.hvg_batch_key,
        n_pcs=args.n_pcs,
        random_state=args.seed,
        overwrite=args.overwrite_reference,
    )

    reference_obs_names = adata_ref.obs_names.astype(str).to_list()

    X_pre = np.asarray(adata_ref.obsm["X_ref"], dtype=np.float32)
    species = adata_ref.obs[args.species_key].astype(str).to_numpy()
    cell_type = adata_ref.obs[args.cell_type_key].astype(str).to_numpy()

    if args.technical_key is not None and args.technical_key in adata_ref.obs.columns:
        sample = adata_ref.obs[args.technical_key].astype(str).to_numpy()
    else:
        sample = None

    # ------------------------------------------------------------
    # 2. Find result folders
    # ------------------------------------------------------------

    folders = find_result_folders(result_root)

    print(f"Reference folder: {ref_dir}")
    print(f"Found {len(folders)} result folders.")

    combined_rows = []
    failed_rows = []

    for folder in folders:
        print("")
        print("============================================================")
        print(f"Post-hoc metrics for: {folder.name}")
        print("============================================================")

        tables_dir = folder / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        posthoc_path = tables_dir / "posthoc_metrics.csv"
        plus_path = tables_dir / "metrics_plus_posthoc.csv"
        jaccard_detail_path = tables_dir / "jaccard_intra_celltype_detail.csv"
        alcs_detail_path = tables_dir / "alcs_detail.csv"

        if posthoc_path.exists() and not args.overwrite_posthoc:
            print(f"Already exists, skipping: {posthoc_path}")

            try:
                base_df = pd.read_csv(tables_dir / "metrics.csv")
                posthoc_df = pd.read_csv(posthoc_path)
                plus_df = safe_concat_metrics(base_df, posthoc_df)
                plus_df["run_folder"] = folder.name
                combined_rows.append(plus_df.iloc[0])
            except Exception as e:
                failed_rows.append(
                    {
                        "run_folder": folder.name,
                        "error": f"failed reading existing posthoc: {e}",
                    }
                )

            continue

        try:
            X_post = load_run_embedding(
                folder=folder,
                reference_obs_names=reference_obs_names,
                original_adata_path=args.adata_path,
                embedding_key=args.embedding_key,
            )

            if X_post.shape[0] != X_pre.shape[0]:
                raise ValueError(
                    f"X_post has {X_post.shape[0]} cells but X_pre has {X_pre.shape[0]}."
                )

            # ------------------------------------------------------------
            # Intra-cell-type metrics
            # ------------------------------------------------------------

            intra_df, jaccard_detail_df = compute_intra_metrics(
                X_pre=X_pre,
                X_post=X_post,
                species=species,
                cell_type=cell_type,
                sample=sample,
                n_neighbors=args.n_neighbors,
                min_cells_per_group=args.min_cells_per_group,
                n_pcr_components=args.n_pcr_components,
                random_state=args.seed,
            )

            # ------------------------------------------------------------
            # ALCS
            # ------------------------------------------------------------

            alcs_df, alcs_detail_df = compute_alcs(
                X_pre=X_pre,
                X_post=X_post,
                species=species,
                cell_type=cell_type,
                test_size=args.alcs_test_size,
                random_state=args.seed,
                min_cells_per_label=args.alcs_min_cells_per_label,
                max_iter=args.alcs_max_iter,
            )

            posthoc_df = safe_concat_metrics(intra_df, alcs_df)

            posthoc_df.insert(0, "run_folder", folder.name)
            posthoc_df.insert(1, "reference_folder", str(ref_dir))

            posthoc_df.to_csv(posthoc_path, index=False)
            jaccard_detail_df.to_csv(jaccard_detail_path, index=False)
            alcs_detail_df.to_csv(alcs_detail_path, index=False)

            base_df = pd.read_csv(tables_dir / "metrics.csv")
            plus_df = safe_concat_metrics(base_df, posthoc_df)
            plus_df.to_csv(plus_path, index=False)

            combined_rows.append(plus_df.iloc[0])

            print(f"Saved: {posthoc_path}")
            print(f"Saved: {plus_path}")
            print(f"Saved: {jaccard_detail_path}")
            print(f"Saved: {alcs_detail_path}")

        except Exception as e:
            print(f"FAILED: {folder.name}")
            print(e)

            failed_rows.append(
                {
                    "run_folder": folder.name,
                    "error": str(e),
                }
            )

    # ------------------------------------------------------------
    # 3. Save combined outputs
    # ------------------------------------------------------------

    if len(combined_rows) > 0:
        combined_df = pd.DataFrame(combined_rows)
        combined_path = result_root / "_metrics_plus_posthoc_all_runs.csv"
        combined_df.to_csv(combined_path, index=False)
        print("")
        print(f"Saved combined table: {combined_path}")
    else:
        combined_df = pd.DataFrame()

    if len(failed_rows) > 0:
        failed_df = pd.DataFrame(failed_rows)
        failed_path = result_root / "_posthoc_failed_runs.csv"
        failed_df.to_csv(failed_path, index=False)
        print(f"Saved failed runs table: {failed_path}")

    print("")
    print("Done.")

    return combined_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--adata_path", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--result_root", required=True)

    parser.add_argument("--counts_layer", default="counts")
    parser.add_argument("--species_key", default="species")
    parser.add_argument("--technical_key", default="sample")
    parser.add_argument("--cell_type_key", default="cell_type_eval")

    parser.add_argument("--embedding_key", default="X_emb")

    parser.add_argument("--n_top_genes", type=int, default=1200)
    parser.add_argument("--hvg_batch_key", default="species")
    parser.add_argument("--n_pcs", type=int, default=20)

    parser.add_argument("--n_neighbors", type=int, default=20)
    parser.add_argument("--min_cells_per_group", type=int, default=30)
    parser.add_argument("--n_pcr_components", type=int, default=20)

    parser.add_argument("--alcs_test_size", type=float, default=0.3)
    parser.add_argument("--alcs_min_cells_per_label", type=int, default=10)
    parser.add_argument("--alcs_max_iter", type=int, default=5000)

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--overwrite_reference", action="store_true")
    parser.add_argument("--overwrite_posthoc", action="store_true")

    args = parser.parse_args()
    main(args)
