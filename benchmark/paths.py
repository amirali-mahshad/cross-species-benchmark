# benchmark/paths.py

from pathlib import Path
from datetime import datetime


def create_run_directory(
    result_root: str,
    dataset_name: str,
    model_name: str,
    seed: int = 0,
    overwrite: bool = False,
    extra_tag: str | None = None,
) -> Path:
    """
    Create a result directory for one benchmark run.

    Example output:
        results/GSE84133__scvi_correct_species__seed0/
    """

    result_root = Path(result_root)

    run_name = f"{dataset_name}__{model_name}__seed{seed}"

    if extra_tag is not None:
        run_name = f"{run_name}__{extra_tag}"

    run_dir = result_root / run_name

    if run_dir.exists() and not overwrite:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = result_root / f"{run_name}__{timestamp}"

    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "model").mkdir(exist_ok=True)
    (run_dir / "tables").mkdir(exist_ok=True)
    (run_dir / "embeddings").mkdir(exist_ok=True)

    return run_dir