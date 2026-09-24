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

The five `main_*.py` scripts expose dataset-specific arguments through `--help`. `--fusion` selects `concat`, `adaptive`, `global_learnable`, or `hybrid`; the default is `hybrid`. The training scripts correspond to the revised implementation. Their default hyperparameters are starting configurations and have not been revalidated for every dataset in this repository.
