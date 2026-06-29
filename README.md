# Cross-species scRNA-seq integration benchmark

This project benchmarks cross-species integration methods on the GSE84133 human-mouse pancreas dataset.

## Methods

- Baseline PCA
- scVI
- Harmony
- Seurat CCA/RPCA
- scETM

## Dataset

The dataset used is GSE84133. Large data files are not included in this repository.

Expected local path:

```text
data/panc/GSE84133/adata.h5ad
