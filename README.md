# EvoDen

This README documents the **static EvoDen pipeline** provided by the three entry scripts below. It performs sequence extraction, representation learning, and downstream classification on a fixed transaction graph.

The code is organized into three stages:

1. `1_extract_sequences.py`: risk-potential guided transaction sequence extraction.
2. `2_train_transformer.py`: reconstruction-based Transformer sequence learning.
3. `3_train_classifier.py`: downstream MLP-based money laundering detection.

## Directory Structure

```text
├── 1_extract_sequences.py
├── 2_train_transformer.py
├── 3_train_classifier.py
└── README.md
```


The scripts create the output directories `sequence/`, `representations/`, and `checkpoints/` when run.

## Requirements

Install the third-party packages in an environment that also provides the external project modules listed above:

```bash
pip install torch numpy scikit-learn
```


## Stage 1: Sequence Extraction

```bash
python 1_extract_sequences.py \
  --data_name elliptic_bitcoin_dataset \
  --edge_list ./data/elliptic_bitcoin_dataset/my_edge_list.txt \
  --potential ./data/elliptic_bitcoin_dataset/potential_scores.pt \
  --output_dir ./sequence \
  --seq_len 10 \
  --restart_prob 0.15
```

This stage saves:

```text
sequence/dec_seq_input.pt
sequence/features.pt
sequence/labels.pt
sequence/distance.pt
```

## Stage 2: Reconstruction-based Transformer Learning

```bash
python 2_train_transformer.py \
  --sequence_dir ./sequence \
  --output_dir ./representations \
  --checkpoint_dir ./checkpoints \
  --epochs 1000 \
  --batch_size 512 \
  --n_layers 6
```

This stage trains the Transformer encoder-decoder with the reconstruction objective and saves transaction-level representations:

```text
representations/representation*.pt
representations/true_labels*.pt
checkpoints/transformer_reconstruction.pt
```

## Stage 3: Downstream Classification

```bash
python 3_train_classifier.py \
  --representation_dir ./representations \
  --checkpoint_dir ./checkpoints \
  --mode both \
  --epochs 100
```


