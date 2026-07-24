#!/usr/bin/env bash
set -euo pipefail

# Reduce CUDA allocator fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# ============================================================
# Environments
# ============================================================

PY_ENV="proj3"
SEURAT_ENV="seurat_env"
SCETM_ENV="scETM"
INVAE_ENV="invae_env"

# ============================================================
# Dataset
# ============================================================

ADATA_PATH="/share_large/lbcg/mahshad/data/adipose/GSE176171/adata.h5ad"
DATASET_NAME="GSE176171"
RESULT_ROOT="/share_large/lbcg/mahshad/results"

# ============================================================
# AnnData keys
# ============================================================

COUNTS_LAYER="counts"
SPECIES_KEY="species"
TECHNICAL_KEY="sample"
CELL_TYPE_KEY="cell_type_eval"

# ============================================================
# Shared settings
# ============================================================

SEEDS=(0 1 2 3)
CORRECTION_MODES=("species" "technical" "both")
SEURAT_METHODS=("cca" "rpca")

N_TOP_GENES=1200
HVG_BATCH_KEY="species"

N_PCS=20
N_LATENT=20
N_NEIGHBORS=20
LEIDEN_RESOLUTION=1.0

GENE_LIKELIHOOD="nb"

# ============================================================
# scETM settings
# ============================================================

SCETM_REPO="models/scETM"
SCETM_WORKER_SCRIPT="scripts/run_scetm_training.py"
SCETM_EMBEDDING_OUTPUT="delta"
SCETM_EPOCHS=2000
SCETM_EVAL_EVERY=0
SCETM_N_SAMPLERS=4

# ============================================================
# inVAE settings
# ============================================================

INVAE_REPO="models/inVAE"
INVAE_EPOCHS=500

SAVE_ADATA=true

SAVE_ADATA_FLAG=()
if [ "${SAVE_ADATA}" = true ]; then
    SAVE_ADATA_FLAG=(--save_adata)
fi

# ============================================================
# Logging
# ============================================================

mkdir -p logs

LOG_FILE="logs/run_gse176171_benchmark_resume_$(date +%Y%m%d_%H%M%S).log"

echo "Logging to: ${LOG_FILE}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "============================================================"
echo "Resuming ${DATASET_NAME} benchmark"
echo "Date: $(date)"
echo "ADATA_PATH: ${ADATA_PATH}"
echo "RESULT_ROOT: ${RESULT_ROOT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"
echo "============================================================"

echo ""
echo "GPU state before resume:"
nvidia-smi

# ============================================================
# 1. Resume scVI: seed 3 only
# ============================================================

SCVI_SEED=3

for MODE in "${CORRECTION_MODES[@]}"; do

    echo ""
    echo "============================================================"
    echo "Running scVI | seed=${SCVI_SEED} | correction_mode=${MODE}"
    echo "============================================================"

    nvidia-smi \
        --query-gpu=index,name,memory.total,memory.used,memory.free \
        --format=csv

    conda run --no-capture-output -n "${PY_ENV}" \
    python -m runners.scvi_runner \
        --adata_path "${ADATA_PATH}" \
        --dataset_name "${DATASET_NAME}" \
        --result_root "${RESULT_ROOT}" \
        --correction_mode "${MODE}" \
        --counts_layer "${COUNTS_LAYER}" \
        --species_key "${SPECIES_KEY}" \
        --technical_key "${TECHNICAL_KEY}" \
        --cell_type_key "${CELL_TYPE_KEY}" \
        --n_top_genes "${N_TOP_GENES}" \
        --hvg_batch_key "${HVG_BATCH_KEY}" \
        --n_latent "${N_LATENT}" \
        --gene_likelihood "${GENE_LIKELIHOOD}" \
        --n_neighbors "${N_NEIGHBORS}" \
        --leiden_resolution "${LEIDEN_RESOLUTION}" \
        --seed "${SCVI_SEED}" \
        "${SAVE_ADATA_FLAG[@]}"

done

# ============================================================
# 2. Harmony: all seeds and modes
# ============================================================

for SEED in "${SEEDS[@]}"; do
    for MODE in "${CORRECTION_MODES[@]}"; do

        echo ""
        echo "============================================================"
        echo "Running Harmony | seed=${SEED} | correction_mode=${MODE}"
        echo "============================================================"

        conda run --no-capture-output -n "${PY_ENV}" \
        python -m runners.harmony_runner \
            --adata_path "${ADATA_PATH}" \
            --dataset_name "${DATASET_NAME}" \
            --result_root "${RESULT_ROOT}" \
            --correction_mode "${MODE}" \
            --counts_layer "${COUNTS_LAYER}" \
            --species_key "${SPECIES_KEY}" \
            --technical_key "${TECHNICAL_KEY}" \
            --cell_type_key "${CELL_TYPE_KEY}" \
            --n_top_genes "${N_TOP_GENES}" \
            --hvg_batch_key "${HVG_BATCH_KEY}" \
            --n_pcs "${N_PCS}" \
            --n_neighbors "${N_NEIGHBORS}" \
            --leiden_resolution "${LEIDEN_RESOLUTION}" \
            --seed "${SEED}" \
            "${SAVE_ADATA_FLAG[@]}"

    done
done

# ============================================================
# 3. Seurat CCA and RPCA: all seeds and modes
# ============================================================

for SEED in "${SEEDS[@]}"; do
    for MODE in "${CORRECTION_MODES[@]}"; do
        for SEURAT_METHOD in "${SEURAT_METHODS[@]}"; do

            echo ""
            echo "============================================================"
            echo "Running Seurat"
            echo "method=${SEURAT_METHOD}"
            echo "seed=${SEED}"
            echo "correction_mode=${MODE}"
            echo "============================================================"

            conda run --no-capture-output -n "${PY_ENV}" \
            python -m runners.seurat_runner \
                --adata_path "${ADATA_PATH}" \
                --dataset_name "${DATASET_NAME}" \
                --result_root "${RESULT_ROOT}" \
                --method "${SEURAT_METHOD}" \
                --correction_mode "${MODE}" \
                --counts_layer "${COUNTS_LAYER}" \
                --species_key "${SPECIES_KEY}" \
                --technical_key "${TECHNICAL_KEY}" \
                --cell_type_key "${CELL_TYPE_KEY}" \
                --n_top_genes "${N_TOP_GENES}" \
                --hvg_batch_key "${HVG_BATCH_KEY}" \
                --n_pcs "${N_PCS}" \
                --n_neighbors "${N_NEIGHBORS}" \
                --leiden_resolution "${LEIDEN_RESOLUTION}" \
                --seed "${SEED}" \
                --seurat_env "${SEURAT_ENV}" \
                "${SAVE_ADATA_FLAG[@]}"

        done
    done
done

# ============================================================
# 4. scETM: all seeds and modes
# ============================================================

for SEED in "${SEEDS[@]}"; do
    for MODE in "${CORRECTION_MODES[@]}"; do

        echo ""
        echo "============================================================"
        echo "Running scETM | seed=${SEED} | correction_mode=${MODE}"
        echo "============================================================"

        nvidia-smi \
            --query-gpu=index,name,memory.total,memory.used,memory.free \
            --format=csv

        conda run --no-capture-output -n "${PY_ENV}" \
        python -m runners.scetm_runner \
            --adata_path "${ADATA_PATH}" \
            --dataset_name "${DATASET_NAME}" \
            --result_root "${RESULT_ROOT}" \
            --correction_mode "${MODE}" \
            --counts_layer "${COUNTS_LAYER}" \
            --species_key "${SPECIES_KEY}" \
            --technical_key "${TECHNICAL_KEY}" \
            --cell_type_key "${CELL_TYPE_KEY}" \
            --n_top_genes "${N_TOP_GENES}" \
            --hvg_batch_key "${HVG_BATCH_KEY}" \
            --embedding_output "${SCETM_EMBEDDING_OUTPUT}" \
            --n_epochs "${SCETM_EPOCHS}" \
            --eval_every "${SCETM_EVAL_EVERY}" \
            --n_samplers "${SCETM_N_SAMPLERS}" \
            --n_neighbors "${N_NEIGHBORS}" \
            --leiden_resolution "${LEIDEN_RESOLUTION}" \
            --seed "${SEED}" \
            --scetm_env "${SCETM_ENV}" \
            --scetm_repo "${SCETM_REPO}" \
            --worker_script "${SCETM_WORKER_SCRIPT}" \
            "${SAVE_ADATA_FLAG[@]}"

    done
done

# ============================================================
# 5. inVAE: all seeds and modes
# ============================================================

for SEED in "${SEEDS[@]}"; do
    for MODE in "${CORRECTION_MODES[@]}"; do

        echo ""
        echo "============================================================"
        echo "Running inVAE | seed=${SEED} | correction_mode=${MODE}"
        echo "============================================================"

        nvidia-smi \
            --query-gpu=index,name,memory.total,memory.used,memory.free \
            --format=csv

        conda run --no-capture-output -n "${PY_ENV}" \
        python -m runners.invae_runner \
            --adata_path "${ADATA_PATH}" \
            --dataset_name "${DATASET_NAME}" \
            --result_root "${RESULT_ROOT}" \
            --correction_mode "${MODE}" \
            --counts_layer "${COUNTS_LAYER}" \
            --species_key "${SPECIES_KEY}" \
            --technical_key "${TECHNICAL_KEY}" \
            --cell_type_key "${CELL_TYPE_KEY}" \
            --n_top_genes "${N_TOP_GENES}" \
            --hvg_batch_key "${HVG_BATCH_KEY}" \
            --n_epochs "${INVAE_EPOCHS}" \
            --n_neighbors "${N_NEIGHBORS}" \
            --leiden_resolution "${LEIDEN_RESOLUTION}" \
            --seed "${SEED}" \
            --invae_env "${INVAE_ENV}" \
            --invae_repo "${INVAE_REPO}" \
            "${SAVE_ADATA_FLAG[@]}"

    done
done

echo ""
echo "============================================================"
echo "GSE176171 benchmark resume completed"
echo "Date: $(date)"
echo "Results: ${RESULT_ROOT}"
echo "Log: ${LOG_FILE}"
echo "============================================================"
