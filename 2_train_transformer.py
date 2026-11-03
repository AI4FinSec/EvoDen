"""Stage 2: reconstruction-based Transformer sequence learning for EvoDen."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from aggregate import aggregate
from get_batch_more import get_batch_more
from reconstruction import rec_train
from transformer_1 import Transformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tensor(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return torch.load(path, map_location="cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EvoDen reconstruction Transformer.")
    parser.add_argument("--sequence_dir", type=str, default="./sequence")
    parser.add_argument("--output_dir", type=str, default="./representations")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--d_model", type=int, default=0, help="Use 0 to match input feature dimension.")
    parser.add_argument("--d_ff", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=3047)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    sequence_dir = Path(args.sequence_dir)
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dec_seq_input = load_tensor(sequence_dir / "dec_seq_input.pt")
    features = load_tensor(sequence_dir / "features.pt")
    labels = load_tensor(sequence_dir / "labels.pt")
    distance = load_tensor(sequence_dir / "distance.pt")

    features = torch.as_tensor(features, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.float32) - 1
    distance = torch.as_tensor(distance, dtype=torch.float32)

    enc_feature_size = features.size(2)
    dec_feature_size = features.size(2)
    d_model = enc_feature_size if args.d_model <= 0 else args.d_model

    model = Transformer(
        enc_feature_size,
        dec_feature_size,
        args.n_layers,
        args.num_heads,
        d_model,
        args.d_ff,
    )

    device = torch.device(args.device)
    model = model.to(device)
    features = features.to(device)
    labels = labels.to(device)
    distance = distance.to(device)
    dec_seq_input = dec_seq_input.to(device)

    print("------------Start reconstruction------------")
    rec_train(args.epochs, features, dec_seq_input, d_model, model, distance, labels)
    print("------------End reconstruction------------")

    torch.save(model.state_dict(), checkpoint_dir / "transformer_reconstruction.pt")

    model.eval()
    loader = get_batch_more(
        batch_size=args.batch_size,
        x=features,
        y=dec_seq_input,
        u=distance,
        v=labels,
    )

    with torch.no_grad():
        for step, (features_batch, dec_seq_input_batch, distance_batch, labels_batch) in enumerate(loader):
            _, enc_outputs, _ = model(
                dec_seq_input_batch,
                dec_seq_input_batch,
                features_batch,
                features_batch,
            )
            representation = aggregate(enc_outputs, distance_batch)
            true_labels = labels_batch[:, 0]
            torch.save(representation.detach().cpu(), output_dir / f"representation{step}.pt")
            torch.save(true_labels.detach().cpu(), output_dir / f"true_labels{step}.pt")


if __name__ == "__main__":
    main()
