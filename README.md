# StructBehavior

This repository contains data preparation and training code for StructBehavior on five datasets: BAF, Crime, InstaFake, TwiBot-20, and WeiboBot.

## Project layout

```text
Data/
  runme.py                 Data preparation entry point
  BAF/                     Dataset-specific preparation scripts
  Crime/
  InstaFake/
  TwiBot20/
  WeiboBot/
attribute_tree/sep_g.py    Atomic-view encoder
backbone/rgcn.py           Material-view encoder
backbone/fusion.py         Granularity fusion modules
backbone/relation.py       Node-shared molecular relation coefficients
backbone/training.py       Training mini-batch index/label selection
backbone/utils_experiment.py
main_BAF.py                Training entry points
main_Crime.py
main_InstaFake.py
main_Twibot20.py
main_WeiboBot.py
```

Each dataset directory contains `raw/`, `preprocessed_data/`, `material_data/`, `attribute_pool/`, and `attribute_tree/`. Generated data and model weights are excluded from version control.

## Environment

Use a Python environment with PyTorch, PyTorch Geometric (including `torch-sparse`), DGL, NumPy, pandas, and scikit-learn. Data preparation also uses the libraries imported by each dataset script, including Transformers for semantic embeddings. Install compatible versions of PyTorch, PyTorch Geometric, and DGL for your CUDA or CPU environment.

Download `bert-base-uncased` and `bert-base-chinese` into `bert/` as required by the data preparation scripts. The model weights are not included in this repository.

## Prepare data

Place the raw datasets in their corresponding `Data/<dataset>/raw/` directories. The source datasets are [TwiBot-20](https://github.com/BunsenFeng/TwiBot-20), [InstaFake](https://github.com/fcakyon/instafake-dataset/tree/master/data/automated-v1.0), [BAF](https://github.com/feedzai/bank-account-fraud/), [WeiboBot](https://github.com/BunsenFeng/Botection/tree/master/dataset), and [Crime](https://catalog.data.gov/dataset/crime-data-from-2020-to-present). Use `automated-v1.0` for InstaFake and `Base.csv` for BAF.

Run the preparation pipeline from the `Data` directory:

```bash
cd Data
python runme.py --dataset TwiBot20
```

Replace `TwiBot20` with `BAF`, `Crime`, `InstaFake`, or `WeiboBot` for another dataset. The pipeline runs preprocessing, material-view preparation, molecular-view preparation, and atomic-view preparation in that order. Some datasets require source files or pretrained representations described in their `raw/README.md` files.

## Train

Run training from the repository root after preparing the selected dataset:

```bash
python main_Twibot20.py
```

The five `main_*.py` scripts expose dataset-specific arguments through `--help`. `--fusion` selects `concat`, `adaptive`, `global_learnable`, or `hybrid`; the default is `hybrid`. `--batch_size` controls the training-instance batches used by CE, FSA, and RA (default: 256). Shared graph propagation retains the unlabeled context permitted by the transductive setting, while contrastive positives and negatives come only from the selected training batch. FSA and RA each use two independent endpoint projection heads.

Molecular and material layers use learned relation coefficients normalized by a softmax across relation types and shared across nodes. Training selects model parameters using the validation primary metric (macro-F1 for Crime, AUC for BAF, and F1 for the other datasets), restores that checkpoint, and evaluates the test split once. `--save_top_k` retains validation-ranking summaries; it does not trigger additional test evaluations.

## Revision status

The October 10, 2026 code update aligns the implementation with the revised manuscript's training-batch scope, relation coefficients, projection heads, and validation-selected final testing. This update has not rerun the dataset experiments and does not establish that the manuscript's reported numerical results were produced by this code revision. Default hyperparameters are starting configurations and have not been revalidated for every dataset in this repository. Checkpoints from earlier versions require retraining because the relation parameters and projection-head architecture have changed.
