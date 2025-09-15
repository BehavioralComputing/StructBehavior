# StructBehavior

The code for paper *Granularity-Tunable  Structural Representation  for Behaviors  of  Uncertain Discriminability*.

## Abstract

> Accurate behavior discrimination is essential in domains such as fraud detection, social platforms and cybersecurity, where adversaries deliberately mimic normal activity to evade detection. A key challenge lies in the \emph{uncertain discriminability} of behaviors, namely informative cues may reside in fine-grained attributes or only emerge through coarser relational structures. Existing methods typically operate at a fixed granularity or capture only partial structural aspects, leaving them unable to adapt to this ambiguity.
In this work, we propose \textsc{StructBehavior}, a granularity-tunable structural representation framework. The framework organizes behaviors into a graph-based hierarchy anchored at three semantic levels: intra-behavior attributes (atomic level), intra-behavior structures (molecular level), and inter-behavior associations (material level). In particular, within our \textsc{StructBehavior}, a structural entropy minimization objective is incorporated to uncover latent hierarchies and strengthen semantic coherence, while a cross-level contrastive learning scheme is employed to align multiple abstraction levels as complementary semantic views. Tunable contributions across different levels enable dynamic adaptation to the most discriminative granularity, implementing the principle of \emph{``fit is best''}. Extensive experiments on real-world datasets that span imbalanced, multi-class, and adversarial scenarios demonstrate that our \textsc{StructBehavior} consistently outperforms competitive baselines, delivering robust and discriminative representations.

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
