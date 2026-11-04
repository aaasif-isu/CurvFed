#!/usr/bin/env python3
"""Pure Federated Learning with Weighted EMD + ORC aggregation.

Every selected client trains a complete CIFAR-10 ResNet18 locally. No model
split, smashed-activation transmission, or split server is used. The server:

1. extracts a bounded support set from each client's local feature distribution;
2. computes exact discrete Earth Mover's Distance (EMD) between active clients;
3. builds a k-nearest-neighbor client graph;
4. computes Ollivier-Ricci curvature (ORC) on graph edges;
5. clusters clients using EMD similarity adjusted by ORC;
6. performs sample-weighted FedAvg inside each cluster; and
7. combines cluster models using sample mass and EMD/ORC cluster quality.

Dependencies:
    pip install torch torchvision numpy pandas openpyxl scipy scikit-learn POT

Run from the directory containing this file and FL_Weighted_EMD_ORC_config.py:
    python FL_Weighted_EMD_ORC_config.py
    python FL_ResNet18_CIFAR10_Weighted_EMD_ORC.py
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import linprog
from scipy.sparse.csgraph import shortest_path
from sklearn.cluster import SpectralClustering
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18

try:
    import ot  # Python Optimal Transport (POT)
except ImportError as exc:
    raise ImportError(
        "Missing dependency 'POT'. Install it with: pip install POT"
    ) from exc

from FL_Weighted_EMD_ORC_config import CONFIG, ExperimentConfig


StateDict = Mapping[str, torch.Tensor]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_resnet18(num_classes: int) -> nn.Module:
    """CIFAR-10 ResNet18 with a 3x3 first convolution and no max pooling."""
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    return model


class FeatureResNet18(nn.Module):
    """Read penultimate embeddings without changing the trained state dict."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.model
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)
        x = m.avgpool(x)
        return torch.flatten(x, 1)


def load_cifar10(cfg: ExperimentConfig) -> Tuple[Dataset, Dataset, np.ndarray]:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    # Feature distributions must not contain random augmentation noise.
    deterministic_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    test_transform = deterministic_transform

    root = Path(cfg.data_root) / "CIFAR10"
    train = datasets.CIFAR10(
        root, train=True, download=cfg.download_data, transform=train_transform
    )
    feature_train = datasets.CIFAR10(
        root, train=True, download=False, transform=deterministic_transform
    )
    test = datasets.CIFAR10(
        root, train=False, download=cfg.download_data, transform=test_transform
    )
    labels = np.asarray(train.targets, dtype=np.int64)
    return train, feature_train, test, labels


def iid_partition(num_samples: int, num_users: int, seed: int) -> Dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    return {
        client_id: np.asarray(split, dtype=np.int64)
        for client_id, split in enumerate(np.array_split(indices, num_users))
    }


def dirichlet_partition(
    labels: np.ndarray,
    num_users: int,
    alpha: float,
    seed: int,
    min_samples: int = 10,
    max_attempts: int = 500,
) -> Dict[int, np.ndarray]:
    """Class-wise Dirichlet partition with deterministic retry seeds."""
    num_classes = int(labels.max()) + 1
    for attempt in range(max_attempts):
        rng = np.random.default_rng(seed + attempt)
        client_indices: List[List[int]] = [[] for _ in range(num_users)]

        for class_id in range(num_classes):
            class_indices = np.flatnonzero(labels == class_id)
            rng.shuffle(class_indices)
            proportions = rng.dirichlet(np.full(num_users, alpha))

            # Light balancing avoids unusably empty clients without making the
            # partition IID. This follows the common federated benchmark rule.
            current_sizes = np.asarray([len(x) for x in client_indices])
            proportions *= current_sizes < (len(labels) / num_users)
            if proportions.sum() == 0:
                proportions = np.ones(num_users, dtype=np.float64)
            proportions /= proportions.sum()

            cuts = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
            for client_id, chunk in enumerate(np.split(class_indices, cuts)):
                client_indices[client_id].extend(chunk.tolist())

        sizes = [len(x) for x in client_indices]
        if min(sizes) >= min_samples:
            result = {}
            for client_id, values in enumerate(client_indices):
                values_array = np.asarray(values, dtype=np.int64)
                rng.shuffle(values_array)
                result[client_id] = values_array
            return result

    raise RuntimeError(
        f"Could not build a Dirichlet partition with at least {min_samples} "
        f"samples per client after {max_attempts} attempts."
    )


def make_partitions(labels: np.ndarray, cfg: ExperimentConfig) -> Dict[int, np.ndarray]:
    if cfg.partition == "iid":
        return iid_partition(len(labels), cfg.num_users, cfg.seed)
    return dirichlet_partition(labels, cfg.num_users, cfg.alpha, cfg.seed)


def make_loader(
    dataset: Dataset,
    indices: Sequence[int] | None,
    batch_size: int,
    shuffle: bool,
    cfg: ExperimentConfig,
    seed: int,
) -> DataLoader:
    source = dataset if indices is None else Subset(dataset, list(map(int, indices)))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        source,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and torch.cuda.is_available(),
        generator=generator,
        persistent_workers=cfg.num_workers > 0,
    )


def make_optimizer(model: nn.Module, cfg: ExperimentConfig):
    if cfg.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
    return torch.optim.SGD(
        model.parameters(),
        lr=cfg.learning_rate,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )


def train_client(
    global_state: StateDict,
    train_dataset: Dataset,
    indices: Sequence[int],
    client_id: int,
    round_number: int,
    cfg: ExperimentConfig,
    device: torch.device,
) -> Tuple[OrderedDict, float, float]:
    model = make_resnet18(cfg.num_classes).to(device)
    model.load_state_dict(global_state, strict=True)
    model.train()
    optimizer = make_optimizer(model, cfg)
    criterion = nn.CrossEntropyLoss()
    loader = make_loader(
        train_dataset,
        indices,
        cfg.batch_size,
        True,
        cfg,
        seed=cfg.seed + round_number * 10_000 + client_id,
    )

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for _ in range(cfg.local_epochs):
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(1) == targets).sum().item())
            total_examples += batch_size

    state = OrderedDict(
        (name, tensor.detach().cpu().clone())
        for name, tensor in model.state_dict().items()
    )
    del model, optimizer, loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (
        state,
        100.0 * total_correct / max(1, total_examples),
        total_loss / max(1, total_examples),
    )


@torch.no_grad()
def extract_feature_support(
    client_state: StateDict,
    feature_dataset: Dataset,
    indices: Sequence[int],
    client_id: int,
    round_number: int,
    cfg: ExperimentConfig,
    device: torch.device,
) -> np.ndarray:
    """Extract a deterministic, bounded support set of normalized embeddings."""
    rng = np.random.default_rng(cfg.seed + 1_000_003 * round_number + client_id)
    indices_array = np.asarray(indices, dtype=np.int64)
    take = min(cfg.feature_samples_per_client, len(indices_array))
    chosen = rng.choice(indices_array, size=take, replace=False)

    model = make_resnet18(cfg.num_classes).to(device)
    model.load_state_dict(client_state, strict=True)
    feature_model = FeatureResNet18(model).to(device).eval()
    loader = make_loader(
        feature_dataset,
        chosen,
        min(cfg.batch_size, 256),
        False,
        cfg,
        seed=cfg.seed + round_number + client_id,
    )

    chunks = []
    for images, _ in loader:
        embeddings = feature_model(images.to(device, non_blocking=True))
        embeddings = F.normalize(embeddings, p=2, dim=1)
        chunks.append(embeddings.cpu().numpy().astype(np.float64))
    features = np.concatenate(chunks, axis=0)

    # Deterministic farthest-point sampling preserves distributional coverage
    # better than taking the first support_size examples.
    support_size = min(cfg.emd_support_size, len(features))
    center = features.mean(axis=0, keepdims=True)
    first = int(np.argmax(np.linalg.norm(features - center, axis=1)))
    selected = [first]
    min_dist = np.linalg.norm(features - features[first], axis=1)
    for _ in range(1, support_size):
        next_index = int(np.argmax(min_dist))
        selected.append(next_index)
        min_dist = np.minimum(
            min_dist, np.linalg.norm(features - features[next_index], axis=1)
        )

    del feature_model, model, loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return features[selected]


def emd_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Exact discrete EMD with uniform masses and Euclidean ground cost."""
    a = np.full(len(x), 1.0 / len(x), dtype=np.float64)
    b = np.full(len(y), 1.0 / len(y), dtype=np.float64)
    cost = ot.dist(x, y, metric="euclidean")
    return float(ot.emd2(a, b, cost, numItermax=200_000))


def pairwise_emd(supports: Sequence[np.ndarray]) -> np.ndarray:
    count = len(supports)
    distances = np.zeros((count, count), dtype=np.float64)
    for i in range(count):
        for j in range(i + 1, count):
            value = emd_distance(supports[i], supports[j])
            distances[i, j] = value
            distances[j, i] = value
    return distances


def build_knn_graph(distances: np.ndarray, k: int) -> np.ndarray:
    """Return a connected, symmetric weighted kNN graph."""
    n = len(distances)
    positive = distances[distances > 0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    similarities = np.exp(-distances / max(scale, 1e-12))
    np.fill_diagonal(similarities, 0.0)

    adjacency = np.zeros_like(similarities)
    for i in range(n):
        neighbors = np.argsort(distances[i])[1 : k + 1]
        adjacency[i, neighbors] = similarities[i, neighbors]
    adjacency = np.maximum(adjacency, adjacency.T)

    # Join disconnected components using their closest cross-component pair.
    while True:
        components = graph_components(adjacency)
        if len(components) <= 1:
            break
        best = None
        for a_idx in range(len(components)):
            for b_idx in range(a_idx + 1, len(components)):
                for i in components[a_idx]:
                    for j in components[b_idx]:
                        candidate = (distances[i, j], i, j)
                        if best is None or candidate < best:
                            best = candidate
        _, i, j = best
        adjacency[i, j] = adjacency[j, i] = similarities[i, j]
    return adjacency


def graph_components(adjacency: np.ndarray) -> List[List[int]]:
    unseen = set(range(len(adjacency)))
    components = []
    while unseen:
        root = unseen.pop()
        stack = [root]
        component = [root]
        while stack:
            node = stack.pop()
            neighbors = np.flatnonzero(adjacency[node] > 0).tolist()
            for neighbor in neighbors:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
                    component.append(neighbor)
        components.append(sorted(component))
    return components


def node_measure(adjacency: np.ndarray, node: int, idleness: float) -> np.ndarray:
    measure = np.zeros(len(adjacency), dtype=np.float64)
    measure[node] = idleness
    weights = adjacency[node].copy()
    total = weights.sum()
    if total > 0:
        measure += (1.0 - idleness) * weights / total
    else:
        measure[node] = 1.0
    return measure


def transport_lp(a: np.ndarray, b: np.ndarray, cost: np.ndarray) -> float:
    """Wasserstein-1 distance between graph probability measures."""
    n = len(a)
    objective = cost.reshape(-1)
    a_eq = []
    b_eq = []
    for i in range(n):
        row = np.zeros((n, n), dtype=np.float64)
        row[i, :] = 1.0
        a_eq.append(row.reshape(-1))
        b_eq.append(a[i])
    # Last column constraint is redundant once all row constraints are present.
    for j in range(n - 1):
        column = np.zeros((n, n), dtype=np.float64)
        column[:, j] = 1.0
        a_eq.append(column.reshape(-1))
        b_eq.append(b[j])
    result = linprog(
        objective,
        A_eq=np.asarray(a_eq),
        b_eq=np.asarray(b_eq),
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"ORC transport optimization failed: {result.message}")
    return float(result.fun)


def ollivier_ricci_curvature(
    adjacency: np.ndarray, idleness: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute exact discrete ORC for every undirected graph edge."""
    n = len(adjacency)
    # Strong similarity corresponds to short graph length.
    graph_lengths = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(graph_lengths, 0.0)
    edges = np.argwhere(np.triu(adjacency > 0, 1))
    for i, j in edges:
        length = 1.0 / max(adjacency[i, j], 1e-12)
        graph_lengths[i, j] = graph_lengths[j, i] = length
    geodesic = shortest_path(graph_lengths, directed=False)

    curvature = np.zeros_like(adjacency)
    measures = [node_measure(adjacency, i, idleness) for i in range(n)]
    for i, j in edges:
        wasserstein = transport_lp(measures[i], measures[j], geodesic)
        edge_distance = geodesic[i, j]
        kappa = 1.0 - wasserstein / max(edge_distance, 1e-12)
        curvature[i, j] = curvature[j, i] = kappa
    return curvature, geodesic


def cluster_active_clients(
    distances: np.ndarray,
    adjacency: np.ndarray,
    curvature: np.ndarray,
    cfg: ExperimentConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Spectral clustering from EMD similarity strengthened by positive ORC."""
    positive = distances[distances > 0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    emd_similarity = np.exp(-distances / max(scale, 1e-12))

    # ORC lies at most at 1, but can be negative. exp(kappa) is positive and
    # naturally suppresses bridge-like negative-curvature edges.
    affinity = np.zeros_like(adjacency)
    mask = adjacency > 0
    affinity[mask] = emd_similarity[mask] * np.exp(curvature[mask])
    affinity = np.maximum(affinity, affinity.T)
    np.fill_diagonal(affinity, 1.0)

    if cfg.n_clusters == 1:
        labels = np.zeros(len(distances), dtype=np.int64)
    else:
        labels = SpectralClustering(
            n_clusters=cfg.n_clusters,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=cfg.seed,
            n_init=20,
        ).fit_predict(affinity)
    return labels.astype(np.int64), affinity


def weighted_average_states(
    states: Sequence[StateDict], weights: Sequence[float]
) -> OrderedDict:
    if len(states) != len(weights) or not states:
        raise ValueError("States and weights must have the same nonzero length.")
    normalized = np.asarray(weights, dtype=np.float64)
    if np.any(normalized < 0) or normalized.sum() <= 0:
        raise ValueError("Aggregation weights must be nonnegative with positive sum.")
    normalized /= normalized.sum()

    output = OrderedDict()
    for name in states[0].keys():
        reference = states[0][name]
        if torch.is_floating_point(reference):
            accumulator = torch.zeros_like(reference, dtype=torch.float64)
            for state, weight in zip(states, normalized):
                accumulator.add_(state[name].to(torch.float64), alpha=float(weight))
            output[name] = accumulator.to(reference.dtype)
        else:
            # Integer buffers such as num_batches_tracked are not meaningfully
            # averaged. Retain the value from the highest-weight contributor.
            output[name] = states[int(np.argmax(normalized))][name].clone()
    return output


def cluster_quality(
    members: Sequence[int],
    distances: np.ndarray,
    curvature: np.ndarray,
    adjacency: np.ndarray,
    cfg: ExperimentConfig,
) -> float:
    """Higher means internally compact and positively curved."""
    if len(members) == 1:
        # A singleton carries no evidence of internal cohesion.
        return cfg.min_quality

    pairs = [(i, j) for p, i in enumerate(members) for j in members[p + 1 :]]
    intra_distances = np.asarray([distances[i, j] for i, j in pairs])
    all_positive = distances[distances > 0]
    distance_scale = float(np.median(all_positive)) if len(all_positive) else 1.0
    compactness = math.exp(-float(intra_distances.mean()) / max(distance_scale, 1e-12))

    internal_edges = [
        curvature[i, j] for i, j in pairs if adjacency[i, j] > 0
    ]
    mean_curvature = float(np.mean(internal_edges)) if internal_edges else -1.0
    curvature_score = 1.0 / (1.0 + math.exp(-2.0 * mean_curvature))

    # Multiplication requires both distributional compactness and graph quality.
    quality = 2.0 * compactness * curvature_score
    return float(np.clip(quality, cfg.min_quality, cfg.max_quality))


def hierarchical_weighted_aggregation(
    client_states: Sequence[StateDict],
    sample_counts: Sequence[int],
    cluster_labels: np.ndarray,
    distances: np.ndarray,
    curvature: np.ndarray,
    adjacency: np.ndarray,
    cfg: ExperimentConfig,
) -> Tuple[OrderedDict, Dict[int, dict]]:
    cluster_models = []
    cluster_scores = []
    diagnostics: Dict[int, dict] = {}
    total_samples = float(sum(sample_counts))

    for cluster_id in sorted(np.unique(cluster_labels).tolist()):
        members = np.flatnonzero(cluster_labels == cluster_id).tolist()
        member_counts = [sample_counts[i] for i in members]
        model = weighted_average_states(
            [client_states[i] for i in members], member_counts
        )
        samples = int(sum(member_counts))
        quality = cluster_quality(
            members, distances, curvature, adjacency, cfg
        )
        sample_fraction = samples / total_samples
        score = (
            sample_fraction ** cfg.sample_weight_power
            * quality ** cfg.quality_weight_power
        )
        cluster_models.append(model)
        cluster_scores.append(score)
        diagnostics[int(cluster_id)] = {
            "active_positions": members,
            "samples": samples,
            "sample_fraction": sample_fraction,
            "quality": quality,
            "unnormalized_score": score,
        }

    normalized = np.asarray(cluster_scores, dtype=np.float64)
    normalized /= normalized.sum()
    for cluster_id, weight in zip(sorted(diagnostics), normalized):
        diagnostics[cluster_id]["weight"] = float(weight)
    global_state = weighted_average_states(cluster_models, normalized)
    return global_state, diagnostics


@torch.no_grad()
def evaluate(
    state: StateDict,
    dataset: Dataset,
    cfg: ExperimentConfig,
    device: torch.device,
) -> Tuple[float, float]:
    model = make_resnet18(cfg.num_classes).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    loader = make_loader(
        dataset, None, cfg.batch_size, False, cfg, seed=cfg.seed
    )
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        total_loss += float(criterion(logits, targets).item())
        total_correct += int((logits.argmax(1) == targets).sum().item())
        total_examples += targets.size(0)
    del model, loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (
        100.0 * total_correct / max(1, total_examples),
        total_loss / max(1, total_examples),
    )


def save_partition_summary(
    output_dir: Path,
    partitions: Mapping[int, np.ndarray],
    labels: np.ndarray,
) -> None:
    rows = []
    for client_id, indices in partitions.items():
        counts = np.bincount(labels[indices], minlength=10)
        row = {"client": client_id, "num_samples": len(indices)}
        row.update({f"class_{class_id}": int(value) for class_id, value in enumerate(counts)})
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "client_partitions.csv", index=False)


def json_ready_diagnostics(
    diagnostics: Dict[int, dict], active_clients: Sequence[int]
) -> str:
    converted = {}
    for cluster_id, info in diagnostics.items():
        item = dict(info)
        positions = item.pop("active_positions")
        item["client_ids"] = [int(active_clients[position]) for position in positions]
        converted[str(cluster_id)] = item
    return json.dumps(converted, sort_keys=True)


def main() -> None:
    cfg = CONFIG
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg.results_root) / cfg.experiment_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("PURE FL + WEIGHTED EMD + ORC")
    print("=" * 78)
    print(f"Device:                 {device}")
    print(f"Experiment:             {cfg.experiment_tag}")
    print(f"Active clients/round:   {cfg.num_active_clients}")
    print(f"Output:                 {output_dir}")
    print("=" * 78)

    train_dataset, feature_dataset, test_dataset, train_labels = load_cifar10(cfg)
    partitions = make_partitions(train_labels, cfg)
    save_partition_summary(output_dir, partitions, train_labels)

    global_model = make_resnet18(cfg.num_classes)
    global_state = OrderedDict(
        (name, tensor.detach().cpu().clone())
        for name, tensor in global_model.state_dict().items()
    )
    del global_model

    selection_rng = np.random.default_rng(cfg.seed)
    history = []
    cumulative_time = 0.0

    for round_number in range(1, cfg.global_rounds + 1):
        round_start = time.perf_counter()
        active_clients = selection_rng.choice(
            cfg.num_users, size=cfg.num_active_clients, replace=False
        ).astype(int).tolist()
        print(f"\n========== Global Round {round_number}/{cfg.global_rounds} ==========")
        print(f"Active clients: {active_clients}")

        client_states = []
        client_counts = []
        client_accuracies = []
        client_losses = []
        supports = []

        for client_id in active_clients:
            indices = partitions[client_id]
            state, accuracy, loss = train_client(
                global_state,
                train_dataset,
                indices,
                client_id,
                round_number,
                cfg,
                device,
            )
            support = extract_feature_support(
                state,
                feature_dataset,
                indices,
                client_id,
                round_number,
                cfg,
                device,
            )
            client_states.append(state)
            client_counts.append(len(indices))
            client_accuracies.append(accuracy)
            client_losses.append(loss)
            supports.append(support)
            print(
                f"Client {client_id:3d} | samples={len(indices):5d} | "
                f"Train Accuracy={accuracy:7.3f}% | Train Loss={loss:.4f}"
            )

        distances = pairwise_emd(supports)
        adjacency = build_knn_graph(distances, cfg.graph_k_neighbors)
        curvature, _ = ollivier_ricci_curvature(adjacency, cfg.orc_idleness)
        cluster_labels, _ = cluster_active_clients(
            distances, adjacency, curvature, cfg
        )
        global_state, diagnostics = hierarchical_weighted_aggregation(
            client_states,
            client_counts,
            cluster_labels,
            distances,
            curvature,
            adjacency,
            cfg,
        )

        cluster_groups = defaultdict(list)
        for position, cluster_id in enumerate(cluster_labels.tolist()):
            cluster_groups[int(cluster_id)].append(active_clients[position])
        print(f"Cluster groups: {dict(cluster_groups)}")
        print("----- Cluster aggregation weights -----")
        for cluster_id in sorted(diagnostics):
            info = diagnostics[cluster_id]
            print(
                f"Cluster {cluster_id}: samples={info['samples']}, "
                f"quality={info['quality']:.4f}, weight={info['weight']:.4f}"
            )
        print(
            "Weight sum: "
            f"{sum(info['weight'] for info in diagnostics.values()):.6f}"
        )

        test_accuracy, test_loss = evaluate(
            global_state, test_dataset, cfg, device
        )
        total_selected_samples = sum(client_counts)
        train_accuracy = float(
            np.average(client_accuracies, weights=client_counts)
        )
        train_loss = float(np.average(client_losses, weights=client_counts))
        round_time = time.perf_counter() - round_start
        cumulative_time += round_time

        row = {
            "round": round_number,
            "acc_train": train_accuracy,
            "loss_train": train_loss,
            "acc_test": test_accuracy,
            "loss_test": test_loss,
            "round_time_sec": round_time,
            "cumulative_time_sec": cumulative_time,
            "selected_clients": json.dumps(active_clients),
            "selected_samples": total_selected_samples,
            "cluster_details": json_ready_diagnostics(diagnostics, active_clients),
            "mean_pairwise_emd": float(
                distances[np.triu_indices_from(distances, 1)].mean()
            ),
            "mean_edge_orc": float(curvature[adjacency > 0].mean()),
        }
        history.append(row)

        # Write after every round so an interrupted Nova job retains progress.
        pd.DataFrame(history).to_excel(
            output_dir / "round_metrics.xlsx", index=False
        )
        pd.DataFrame(history).to_csv(
            output_dir / "round_metrics.csv", index=False
        )
        if cfg.save_round_checkpoints:
            torch.save(
                global_state,
                output_dir / f"global_model_round_{round_number:03d}.pt",
            )

        print("-" * 78)
        print(
            f"Round {round_number:3d} | Train Accuracy: {train_accuracy:7.3f}% "
            f"| Train Loss: {train_loss:.4f}"
        )
        print(
            f"Round {round_number:3d} | Test Accuracy:  {test_accuracy:7.3f}% "
            f"| Test Loss:  {test_loss:.4f}"
        )
        print(
            f"Round time: {round_time:.2f} sec | "
            f"Cumulative time: {cumulative_time:.2f} sec"
        )

    torch.save(global_state, output_dir / "final_global_model.pt")
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(cfg.__dict__, handle, indent=2)

    print("\n" + "=" * 78)
    print("PURE FL + WEIGHTED EMD + ORC TRAINING COMPLETED")
    print("=" * 78)
    print(f"Final test accuracy: {history[-1]['acc_test']:.3f}%")
    print(f"Final test loss:     {history[-1]['loss_test']:.4f}")
    print(f"Total time:          {cumulative_time:.2f} sec")
    print(f"Results directory:   {output_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
