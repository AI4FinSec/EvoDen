

from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA

from data_load_elliptic import DataHandler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_edge_list(edge_path: Path, num_nodes: int) -> Dict[int, List[int]]:
    """Load a directed edge list into an adjacency dictionary."""
    adjacency: Dict[int, List[int]] = defaultdict(list)
    with edge_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().replace(",", " ").split()
            if len(parts) < 2:
                continue
            src, dst = int(parts[0]), int(parts[1])
            if 0 <= src < num_nodes and 0 <= dst < num_nodes:
                adjacency[src].append(dst)
    return adjacency


def load_potential_scores(potential_path: Path, num_nodes: int) -> np.ndarray:
    """Load precomputed node potential scores from a required external file."""
    if not potential_path.is_file():
        raise FileNotFoundError(f"Risk-potential score file not found: {potential_path}")

    if potential_path.suffix in {".pt", ".pth"}:
        potential = torch.load(potential_path, map_location="cpu")
        potential = torch.as_tensor(potential, dtype=torch.float64).numpy()
    elif potential_path.suffix == ".npy":
        potential = np.load(potential_path).astype(np.float64)
    else:
        potential = np.loadtxt(potential_path, dtype=np.float64)

    potential = np.asarray(potential, dtype=np.float64).reshape(-1)
    if potential.shape[0] != num_nodes:
        raise ValueError(
            f"Potential length {potential.shape[0]} does not match num_nodes {num_nodes}."
        )
    potential = np.maximum(potential, 0.0)
    if float(potential.sum()) == 0.0:
        raise ValueError("All potential scores are zero. Please check the potential file.")
    return potential


def sample_neighbor_by_potential(
    neighbors: List[int], potential: np.ndarray, rng: np.random.Generator
) -> int:
    """Sample one neighbor using normalized potential scores as weights."""
    weights = potential[np.asarray(neighbors, dtype=np.int64)]
    total = float(weights.sum())
    if total <= 0:
        probs = None
    else:
        probs = weights / total
    return int(rng.choice(neighbors, p=probs))


def potential_guided_walk(
    start_node: int,
    adjacency: Dict[int, List[int]],
    potential: np.ndarray,
    seq_len: int,
    restart_prob: float,
    rng: np.random.Generator,
) -> Tuple[List[int], List[float]]:
    """Run a risk-potential guided walk from one start node."""
    sequence = [start_node]
    distance = [1.0]
    current = start_node
    current_distance = 1.0

    while len(sequence) < seq_len:
        if rng.random() < restart_prob:
            current = start_node
            current_distance = 1.0
        else:
            neighbors = adjacency.get(current, [])
            if not neighbors:
                current = start_node
                current_distance = 1.0
            else:
                current = sample_neighbor_by_potential(neighbors, potential, rng)
                current_distance += 1.0

        sequence.append(current)
        distance.append(current_distance)

    return sequence, distance


def maybe_apply_pca(features: np.ndarray, pca_dim: int) -> np.ndarray:
    """Optionally reduce feature dimension with PCA."""
    if pca_dim <= 0:
        return features
    pca = PCA(n_components=pca_dim)
    reduced = pca.fit_transform(features)
    print(f"PCA explained variance ratio: {float(sum(pca.explained_variance_ratio_)):.6f}")
    return reduced


def extract_sequences(args: argparse.Namespace) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = DataHandler()
    data.load(data_name=args.data_name)

    raw_features = np.asarray(data.features)
    labels = torch.as_tensor(data.labels)
    num_nodes = raw_features.shape[0]

    features_np = maybe_apply_pca(raw_features, args.pca_dim)
    features = torch.as_tensor(features_np, dtype=torch.float32)

    adjacency = load_edge_list(Path(args.edge_list), num_nodes)
    potential = load_potential_scores(
        Path(args.potential),
        num_nodes=num_nodes,
    )

    rng = np.random.default_rng(args.seed)
    sequences = torch.zeros((num_nodes, args.seq_len), dtype=torch.long)
    distances = torch.ones((num_nodes, args.seq_len), dtype=torch.float32)

    for node_id in range(num_nodes):
        seq, dist = potential_guided_walk(
            start_node=node_id,
            adjacency=adjacency,
            potential=potential,
            seq_len=args.seq_len,
            restart_prob=args.restart_prob,
            rng=rng,
        )
        sequences[node_id] = torch.as_tensor(seq, dtype=torch.long)
        distances[node_id] = torch.as_tensor(dist, dtype=torch.float32)

    seq_features = torch.zeros((num_nodes, args.seq_len, features.size(1)), dtype=torch.float32)
    seq_labels = torch.zeros((num_nodes, args.seq_len), dtype=torch.long)
    for node_id in range(num_nodes):
        seq_features[node_id] = features[sequences[node_id]]
        seq_labels[node_id] = labels[sequences[node_id]]

    return sequences, seq_features, seq_labels, distances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract EvoDen transaction sequences.")
    parser.add_argument("--data_name", type=str, default="elliptic_bitcoin_dataset")
    parser.add_argument("--edge_list", type=str, default="./data/elliptic_bitcoin_dataset/my_edge_list.txt")
    parser.add_argument(
        "--potential",
        type=str,
        required=True,
        help="Required external .pt/.pth/.npy/.txt file containing precomputed node potential scores.",
    )
    parser.add_argument("--output_dir", type=str, default="./sequence")
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--restart_prob", type=float, default=0.15)
    parser.add_argument("--pca_dim", type=int, default=0, help="Use 0 to keep original feature dimension.")
    parser.add_argument("--seed", type=int, default=3047)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    sequences, seq_features, seq_labels, distances = extract_sequences(args)
    elapsed = time.time() - start
    print(f"Total sequence extraction time: {elapsed:.2f}s")

    torch.save(sequences, output_dir / "dec_seq_input.pt")
    torch.save(seq_features, output_dir / "features.pt")
    torch.save(seq_labels, output_dir / "labels.pt")
    torch.save(distances, output_dir / "distance.pt")


if __name__ == "__main__":
    main()
