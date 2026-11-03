"""Stage 3: downstream money laundering classifier for EvoDen."""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import List

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from classifier import classifier_test, classifier_train
from MLP_ReLU import MLP_Classfier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def numeric_suffix(path: Path) -> int:
    match = re.search(r"(\d+)(?:\.pt)?$", path.name)
    return int(match.group(1)) if match else -1


def sorted_tensor_files(directory: Path, prefix: str) -> List[Path]:
    files = sorted(directory.glob(f"{prefix}*.pt"), key=numeric_suffix)
    if not files:
        # Support the legacy filenames without .pt suffix.
        files = sorted(directory.glob(f"{prefix}*"), key=numeric_suffix)
    return files


def load_representations(representation_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    rep_files = sorted_tensor_files(representation_dir, "representation")
    label_files = sorted_tensor_files(representation_dir, "true_labels")
    if not rep_files or not label_files:
        raise FileNotFoundError(
            f"Cannot find representation*/true_labels* files in {representation_dir}."
        )
    if len(rep_files) != len(label_files):
        raise ValueError("The number of representation files and label files does not match.")

    representations = []
    labels = []
    for rep_path, label_path in zip(rep_files, label_files):
        representations.append(torch.load(rep_path, map_location="cpu"))
        labels.append(torch.load(label_path, map_location="cpu"))

    representation = torch.vstack(representations)
    true_labels = torch.hstack(labels)
    return representation, true_labels


def normalize_labels(true_labels: torch.Tensor) -> torch.Tensor:
    """Map Elliptic labels to binary labels: illicit=1, licit=0, unknown removed."""
    labels = true_labels.clone()
    labels = labels[labels != 2]
    labels[labels == 1] = -1
    labels = labels + 1
    return labels.long()


def filter_unknown_labels(
    representation: torch.Tensor,
    true_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = true_labels != 2
    representation = representation[mask]
    labels = true_labels[mask].clone()
    labels[labels == 1] = -1
    labels = labels + 1
    return representation, labels.long()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate EvoDen classifier.")
    parser.add_argument("--representation_dir", type=str, default="./representations")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint_name", type=str, default="classifier_best.pt")
    parser.add_argument("--mode", choices=["train", "test", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--random_state", type=int, default=3047)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=3047)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    representation_dir = Path(args.representation_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / args.checkpoint_name

    representation, raw_labels = load_representations(representation_dir)
    representation, labels = filter_unknown_labels(representation, raw_labels)

    x_train, x_test, y_train, y_test = train_test_split(
        representation,
        labels,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels.cpu().numpy(),
    )

    device = torch.device(args.device)
    d_model = representation.size(1)
    model = MLP_Classfier(d_model=d_model, num_class=2).to(device)

    if args.mode in {"train", "both"}:
        print("------------Start classifier training------------")
        classifier_train(epochs=args.epochs, x_train=x_train, y_train=y_train, model=model)
        print("------------End classifier training------------")
        torch.save(model.state_dict(), checkpoint_path)

    if args.mode == "test":
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. Run with --mode train first."
            )
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    print("------------Start classifier testing------------")
    classifier_test(x_test=x_test, y_test=y_test, model=model)
    print("------------End classifier testing------------")


if __name__ == "__main__":
    main()
