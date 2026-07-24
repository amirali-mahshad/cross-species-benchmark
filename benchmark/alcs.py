# benchmark/alcs.py

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _filter_labels_by_min_count(X, y, min_cells_per_label: int):
    y = np.asarray(y).astype(str)

    counts = pd.Series(y).value_counts()
    keep_labels = set(counts[counts >= min_cells_per_label].index.astype(str))

    mask = np.array([lab in keep_labels for lab in y], dtype=bool)

    return X[mask], y[mask]


def _self_projection_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 0,
    min_cells_per_label: int = 10,
    max_iter: int = 5000,
):
    """
    Train/test classifier accuracy on one representation.
    """

    X = np.asarray(X)
    y = np.asarray(y).astype(str)

    finite = np.all(np.isfinite(X), axis=1)
    X = X[finite]
    y = y[finite]

    X, y = _filter_labels_by_min_count(
        X=X,
        y=y,
        min_cells_per_label=min_cells_per_label,
    )

    unique_labels = np.unique(y)

    if X.shape[0] < 20 or len(unique_labels) < 2:
        return {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "n_cells_used": int(X.shape[0]),
            "n_labels_used": int(len(unique_labels)),
        }

    label_counts = pd.Series(y).value_counts()

    if label_counts.min() < 2:
        return {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "n_cells_used": int(X.shape[0]),
            "n_labels_used": int(len(unique_labels)),
        }

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=max_iter,
                solver="lbfgs",
                multi_class="auto",
                n_jobs=1,
            ),
        )

        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        return {
            "accuracy": float(accuracy_score(y_test, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "n_cells_used": int(X.shape[0]),
            "n_labels_used": int(len(unique_labels)),
        }

    except Exception as e:
        warnings.warn(f"Self-projection classifier failed: {e}")

        return {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "n_cells_used": int(X.shape[0]),
            "n_labels_used": int(len(unique_labels)),
        }


def compute_alcs(
    X_pre: np.ndarray,
    X_post: np.ndarray,
    species: np.ndarray,
    cell_type: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 0,
    min_cells_per_label: int = 10,
    max_iter: int = 5000,
):
    """
    ALCS = loss of cell-type distinguishability after integration.

    For each species:
        classifier accuracy on pre-integration reference embedding
        minus
        classifier accuracy on integrated embedding

    High ALCS = bad, suggests overcorrection.
    """

    X_pre = np.asarray(X_pre)
    X_post = np.asarray(X_post)
    species = np.asarray(species).astype(str)
    cell_type = np.asarray(cell_type).astype(str)

    mask = np.all(np.isfinite(X_pre), axis=1)
    mask &= np.all(np.isfinite(X_post), axis=1)

    X_pre = X_pre[mask]
    X_post = X_post[mask]
    species = species[mask]
    cell_type = cell_type[mask]

    detail_rows = []

    for sp in sorted(np.unique(species)):
        idx = species == sp

        if idx.sum() < 20:
            continue

        pre_result = _self_projection_accuracy(
            X=X_pre[idx],
            y=cell_type[idx],
            test_size=test_size,
            random_state=random_state,
            min_cells_per_label=min_cells_per_label,
            max_iter=max_iter,
        )

        post_result = _self_projection_accuracy(
            X=X_post[idx],
            y=cell_type[idx],
            test_size=test_size,
            random_state=random_state,
            min_cells_per_label=min_cells_per_label,
            max_iter=max_iter,
        )

        alcs_acc = pre_result["accuracy"] - post_result["accuracy"]
        alcs_bal = pre_result["balanced_accuracy"] - post_result["balanced_accuracy"]

        detail_rows.append(
            {
                "species": sp,
                "n_cells_species": int(idx.sum()),

                "reference_accuracy": pre_result["accuracy"],
                "integrated_accuracy": post_result["accuracy"],
                "ALCS_accuracy": alcs_acc,
                "ALCS_accuracy_clipped": max(0.0, alcs_acc)
                if not pd.isna(alcs_acc)
                else float("nan"),

                "reference_balanced_accuracy": pre_result["balanced_accuracy"],
                "integrated_balanced_accuracy": post_result["balanced_accuracy"],
                "ALCS_balanced_accuracy": alcs_bal,
                "ALCS_balanced_accuracy_clipped": max(0.0, alcs_bal)
                if not pd.isna(alcs_bal)
                else float("nan"),

                "n_cells_used_reference": pre_result["n_cells_used"],
                "n_labels_used_reference": pre_result["n_labels_used"],
                "n_cells_used_integrated": post_result["n_cells_used"],
                "n_labels_used_integrated": post_result["n_labels_used"],
            }
        )

    detail_df = pd.DataFrame(detail_rows)

    if detail_df.empty:
        summary = {
            "ALCS_accuracy_mean": float("nan"),
            "ALCS_accuracy_clipped_mean": float("nan"),
            "ALCS_balanced_accuracy_mean": float("nan"),
            "ALCS_balanced_accuracy_clipped_mean": float("nan"),
            "ALCS_n_species": 0,
            "reference_self_projection_accuracy_mean": float("nan"),
            "integrated_self_projection_accuracy_mean": float("nan"),
        }

        return pd.DataFrame([summary]), detail_df

    summary = {
        "ALCS_accuracy_mean": float(detail_df["ALCS_accuracy"].mean()),
        "ALCS_accuracy_clipped_mean": float(detail_df["ALCS_accuracy_clipped"].mean()),
        "ALCS_balanced_accuracy_mean": float(detail_df["ALCS_balanced_accuracy"].mean()),
        "ALCS_balanced_accuracy_clipped_mean": float(
            detail_df["ALCS_balanced_accuracy_clipped"].mean()
        ),
        "ALCS_n_species": int(detail_df.shape[0]),
        "reference_self_projection_accuracy_mean": float(
            detail_df["reference_accuracy"].mean()
        ),
        "integrated_self_projection_accuracy_mean": float(
            detail_df["integrated_accuracy"].mean()
        ),
    }

    return pd.DataFrame([summary]), detail_df
