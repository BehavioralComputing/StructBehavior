# StructBehavior

The code for paper *Structure-Aware Multi-Level Behavior Representation with Structural Information Principles*.

## Abstract

> Understanding complex behavioral patterns is essential in domains such as social platforms, financial systems, and public security. Existing behavior modeling approaches often treat behaviors as flat feature vectors or isolated events, lacking the capacity to represent their hierarchical and structured nature. To address this limitation, we propose StructBehavior, a novel framework that models each behavior instance as a multi-level structure comprising atomic (attribute-level), molecular (intra-behavioral), and material (inter-behavioral) semantics. By integrating structural entropy–guided encoding trees, graph enhancement modules, and cross-view contrastive learning, StructBehavior generates expressive, robust, and interpretable behavior representations. Experiments on diverse real-world datasets confirm its effectiveness and generalizability. StructBehavior provides a principled and unified solution for structured behavior representation, enabling reliable behavioral analysis across complex and heterogeneous environments.

## File structure

Training scripts for 5 datasets mentioned in the paper are located at the root directory, named as `main_foo.py`. `attribute_tree/` and `backbone/` store scripts needed to run training scripts.

`bert/` stores two bert models we used to generate semantic embeddings. They are `bert-base-uncased` and `bert-base-chinese`. More details are included in the *Generate semantic embeddings* chapter.

Data required by the training scripts are stored under `Data/` folder, where each dataset has an exclusive folder.

File structure under a single dataset's data directory(i.e. `~/Data/foo`) is listed below:

```txt
- foo
|---- attribute_pool  # Data for molecular layer
|---- attribute_tree  # Data for atomic layer
|---- material_data   # Data for material layer
|---- preprocessed_data  # Preprocessed raw data
|---- raw  # Raw data
```

## Generate semantic embeddings

We use `bert-base-uncased` for all datasets except `WeiboBot` to generate semantic embeddings of nodes. For `WeiboBot` dataset, we use `bert-base-chinese` instead because of its Chinese content.

Two BERT models can be found here:

* https://huggingface.co/google-bert/bert-base-uncased
* https://huggingface.co/google-bert/bert-base-chinese

The file structure under each bert folder should looks like this(take `bert-base-uncased` as an example):

```txt
- bert-base-uncased
|---- config.json
|---- pytorch_model.bin
|---- tokenizer_config.json
|---- tokenizer.json
|---- vocab.txt
```

In other words, you should download at least the above files from the two links mentioned before for uncased bert and Chinese bert.

## Prepare data for training

To prepare data required by the running scripts, the following steps are needed:

1. Download raw data and put them in `raw/` folder, whose link will be given below;
2. Download BERT from the links above and organize them in right structure;
3. Preprocess raw data by running `runme.py --dataset foo` under `Data/` folder. Parameters following `--dataset` should be same as dataset folder name under `Data/` folder(e.g. `--dataset InstaFake`).

**PS:** We recommend using `runme.py` to run all the scripts as a whole, because running scripts separately may suffer from path issues.

### Raw data links

* **Twibot-20**: https://github.com/BunsenFeng/TwiBot-20;
* **InstaFake**: https://github.com/fcakyon/instafake-dataset/tree/master/data/automated-v1.0;
* **BAF**: https://github.com/feedzai/bank-account-fraud/;
* **WeiboBot**: https://github.com/BunsenFeng/Botection/tree/master/dataset;
* **Crime**: https://catalog.data.gov/dataset/crime-data-from-2020-to-present.

**PS:** For **InstaFake** dataset, you should download raw data under `automated-v1.0/` folder, **NOT** under `fake-v-1.0`. For **BAF** dataset, you should download `Base.csv` among the csvs provided.

## Cite our paper as

`#TODO`
