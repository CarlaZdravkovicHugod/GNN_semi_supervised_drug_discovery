# GNN_semi_supervised_drug_discovery

This repository contains code for semi-supervised molecular property prediction using Graph Neural Networks (GCN and GINE) on the QM9 dataset. Developed for the DTU course 02456 Deep Learning.

## Running Models Locally

Train a model using Hydra:

```
python src/run.py model=gcn5
```

Replace `gcn5` with any model defined in `configs/model/` (e.g., `gine5`, `gcn5`).


## Submitting Jobs on DTU HPC

Submit a job via LSF:

```
bsub < configs/bsubscripts/gcnX.sh
```

Example:

```
bsub < configs/bsubscripts/gine5.sh
```


## Repository Structure

```
configs/        - Hydra configs and HPC scripts
src/            - Models, trainer, utils, run.py
outputs/        - Saved runs
logs/           - HPC log files
requirements.txt
```

## Dataset

The project uses the QM9 dataset (PyTorch Geometric). Splits simulate a low-label regime:

* 72% unlabeled
* 8% labeled
* 10% validation
* 10% test


## Summary

* GINE outperforms GCN due to edge-aware message passing.
* Mean Teacher provides a modest semi-supervised improvement.
* Architecture and optimization choices dominate performance.

Install dependencies:

```
pip install -r requirements.txt
```
