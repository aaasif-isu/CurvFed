import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def dataset_iid(dataset_or_len, num_users: int, seed: int = 42, max_samples_per_client=None):
    """Partition a dataset into `num_users` disjoint IID shards covering the dataset."""
    if isinstance(dataset_or_len, int):
        num_samples = dataset_or_len
    else:
        num_samples = len(dataset_or_len)

    if num_users <= 0:
        raise ValueError("num_users must be positive.")
    if num_samples < num_users:
        raise ValueError(f"Dataset size ({num_samples}) must be >= num_users ({num_users}).")

    rng = np.random.default_rng(seed)
    all_idxs = np.arange(num_samples, dtype=np.int64)
    rng.shuffle(all_idxs)
    splits = np.array_split(all_idxs, num_users)

    dict_users = {}
    for i in range(num_users):
        shard = splits[i].tolist()
        if max_samples_per_client is not None and len(shard) > max_samples_per_client:
            shard = shard[:max_samples_per_client]
        dict_users[i] = shard
    return dict_users


def dataset_noniid_dirichlet(
    labels,
    num_users,
    alpha,
    seed=42,
    min_samples_per_client=10,
    max_attempts=1000,
    max_samples_per_client=None,
):
    labels = np.asarray(labels, dtype=np.int64)
    class_ids = np.unique(labels)
    rng = np.random.default_rng(seed)

    for attempt in range(max_attempts):
        client_idcs = {i: [] for i in range(num_users)}
        for c in class_ids:
            class_indices = np.where(labels == c)[0]
            rng.shuffle(class_indices)

            proportions = rng.dirichlet(
                np.full(
                    num_users,
                    alpha,
                    dtype=np.float64,
                )
            )

            split_points = (
                np.cumsum(proportions)[:-1]
                * len(class_indices)
            ).astype(int)

            splits = np.split(class_indices, split_points)
            for i in range(num_users):
                client_idcs[i].extend(splits[i].tolist())

        lengths = [len(client_idcs[i]) for i in range(num_users)]
        if min(lengths) >= min_samples_per_client:
            for i in range(num_users):
                rng.shuffle(client_idcs[i])
                if max_samples_per_client is not None and len(client_idcs[i]) > max_samples_per_client:
                    client_idcs[i] = client_idcs[i][:max_samples_per_client]
            return client_idcs

    raise RuntimeError(
        f"Could not satisfy min_samples_per_client={min_samples_per_client} "
        f"after {max_attempts} attempts."
    )


def partition_client_indices(config, dataset_train, train_labels):
    """Partition training indices according to config (IID vs Dirichlet Non-IID)."""
    if config.iid_like:
        return dataset_iid(
            len(dataset_train),
            num_users=config.num_users,
            seed=config.seed,
            max_samples_per_client=config.max_samples_per_client,
        )
    return dataset_noniid_dirichlet(
        labels=train_labels,
        num_users=config.num_users,
        alpha=config.alpha,
        seed=config.seed,
        min_samples_per_client=min(10, max(1, len(dataset_train) // (config.num_users * 2))),
        max_samples_per_client=config.max_samples_per_client,
    )


def compute_label_histograms_and_tv(dict_users, train_labels, num_users: int, num_classes: int):
    """Compute per-client label counts, normalized histograms, and pairwise Total Variation matrix (0.5 * L1)."""
    train_labels = np.asarray(train_labels, dtype=np.int64)
    counts = np.zeros((num_users, num_classes), dtype=np.int64)
    histograms = np.zeros((num_users, num_classes), dtype=np.float64)

    for i in range(num_users):
        idxs = list(dict_users[i])
        if len(idxs) > 0:
            c = np.bincount(train_labels[idxs], minlength=num_classes)
            counts[i] = c
            histograms[i] = c.astype(np.float64) / float(c.sum())

    tv_matrix = np.zeros((num_users, num_users), dtype=np.float64)
    for i in range(num_users):
        for j in range(i + 1, num_users):
            tv = 0.5 * float(np.sum(np.abs(histograms[i] - histograms[j])))
            tv_matrix[i, j] = tv_matrix[j, i] = tv

    return counts, histograms, tv_matrix


def partition_personalized_test_indices(
    dict_users,
    train_labels,
    test_labels,
    num_users: int,
    num_classes: int,
    seed: int = 42,
):
    """Partition test indices so each client's test set matches its local training class distribution."""
    counts, _, _ = compute_label_histograms_and_tv(dict_users, train_labels, num_users, num_classes)
    test_labels = np.asarray(test_labels, dtype=np.int64)
    rng = np.random.default_rng(seed + 500)

    client_test_idcs = {i: [] for i in range(num_users)}
    for c in range(num_classes):
        class_test_indices = np.where(test_labels == c)[0]
        if len(class_test_indices) == 0:
            continue
        rng.shuffle(class_test_indices)

        col = counts[:, c].astype(np.float64)
        col_sum = float(col.sum())
        if col_sum > 0:
            proportions = col / col_sum
        else:
            proportions = np.full(num_users, 1.0 / num_users, dtype=np.float64)

        split_points = (np.cumsum(proportions)[:-1] * len(class_test_indices)).astype(int)
        splits = np.split(class_test_indices, split_points)
        for i in range(num_users):
            client_test_idcs[i].extend(splits[i].tolist())

    all_test = np.arange(len(test_labels), dtype=np.int64)
    for i in range(num_users):
        if len(client_test_idcs[i]) == 0:
            client_test_idcs[i] = rng.choice(
                all_test, size=min(64, len(all_test)), replace=False
            ).tolist()

    return client_test_idcs

