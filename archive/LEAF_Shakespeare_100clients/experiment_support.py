"""Shared datasets and split CharLSTM for configured SplitFed experiments."""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets as torchvision_datasets, transforms


# Canonical 80-character vocabulary used by LEAF Shakespeare experiments.
SHAKESPEARE_VOCAB = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"


class EncodedShakespeareDataset(Dataset):
    """Compact encoded LEAF samples plus their natural-client index map."""

    def __init__(self, inputs, targets, client_indices, client_names):
        self.inputs = torch.as_tensor(inputs, dtype=torch.uint8)
        self._targets = torch.as_tensor(targets, dtype=torch.long)
        self.client_indices = client_indices
        self.client_names = client_names

    def __len__(self):
        return len(self._targets)

    def __getitem__(self, index):
        return self.inputs[index].long(), self._targets[index]

    @property
    def targets(self):
        return self._targets.numpy()


class CharLSTMClient(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)

    def forward(self, tokens):
        embedded = self.embedding(tokens.long())
        sequence, _ = self.lstm(embedded)
        return sequence


class CharLSTMServer(nn.Module):
    def __init__(self, hidden_size, num_layers, vocab_size):
        super().__init__()
        server_layers = max(1, num_layers - 1)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=server_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, vocab_size)

    def forward(self, smashed):
        sequence, _ = self.lstm(smashed)
        return self.classifier(sequence[:, -1, :])


def build_charlstm(config):
    client = CharLSTMClient(
        config.num_classes,
        config.charlstm_embed_dim,
        config.charlstm_hidden_size,
    )
    server = CharLSTMServer(
        config.charlstm_hidden_size,
        config.charlstm_num_layers,
        config.num_classes,
    )
    return client, server


def _normalise_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "".join(value)
    raise TypeError(f"Expected Shakespeare text as str/list, got {type(value).__name__}.")


def _encode_sequence(sequence, char_to_id, expected_length):
    sequence = _normalise_text(sequence)
    if len(sequence) != expected_length:
        raise ValueError(
            f"LEAF Shakespeare sequence has length {len(sequence)}; "
            f"expected {expected_length}."
        )
    unsupported = sorted(set(sequence) - set(char_to_id))
    if unsupported:
        raise ValueError(f"Unsupported Shakespeare input characters: {unsupported}")
    return np.fromiter(
        (char_to_id[char] for char in sequence),
        dtype=np.uint8,
        count=expected_length,
    )


def _encode_target(target, char_to_id):
    target = _normalise_text(target)
    if len(target) != 1:
        raise ValueError(f"Expected one next-character label, got {target!r}.")
    if target not in char_to_id:
        raise ValueError(f"Unsupported Shakespeare target character: {target!r}")
    return char_to_id[target]


def _build_encoded_dataset(client_blocks, client_names, sequence_length):
    input_blocks = []
    target_blocks = []
    client_indices = {}
    offset = 0

    for local_client_id in range(len(client_names)):
        inputs, targets = client_blocks[local_client_id]
        input_blocks.append(inputs)
        target_blocks.append(targets)
        next_offset = offset + len(targets)
        client_indices[local_client_id] = range(offset, next_offset)
        offset = next_offset

    if not input_blocks:
        inputs = np.empty((0, sequence_length), dtype=np.uint8)
        targets = np.empty((0,), dtype=np.int64)
    else:
        inputs = np.concatenate(input_blocks, axis=0)
        targets = np.concatenate(target_blocks, axis=0)

    return EncodedShakespeareDataset(
        inputs=inputs,
        targets=targets,
        client_indices=client_indices,
        client_names=client_names,
    )


def _load_leaf_shakespeare(config):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "LEAF Shakespeare requires the Hugging Face datasets package. "
            "Install it with: python -m pip install datasets"
        ) from exc

    if len(SHAKESPEARE_VOCAB) != config.num_classes:
        raise ValueError(
            f"Configured Shakespeare vocabulary size is {config.num_classes}, "
            f"but the canonical LEAF vocabulary has {len(SHAKESPEARE_VOCAB)} characters."
        )

    cache_dir = Path(config.data_root) / "shakespeare_leaf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading natural LEAF Shakespeare clients from {config.shakespeare_dataset_name}")
    raw = load_dataset(
        config.shakespeare_dataset_name,
        split="train",
        cache_dir=str(cache_dir),
    )

    required_columns = {"character_id", "x", "y"}
    missing = required_columns - set(raw.column_names)
    if missing:
        raise ValueError(f"LEAF Shakespeare is missing columns: {sorted(missing)}")

    counts = Counter(raw["character_id"])
    eligible_client_ids = sorted(
        client_id
        for client_id, count in counts.items()
        if count >= config.shakespeare_min_samples_per_client
    )
    if len(eligible_client_ids) < config.num_users:
        raise ValueError(
            f"Only {len(eligible_client_ids)} natural Shakespeare clients have at least "
            f"{config.shakespeare_min_samples_per_client} samples; "
            f"{config.num_users} are required."
        )

    # The same seed makes the 10-client set a subset of the 100-client set.
    rng = np.random.default_rng(config.seed)
    rng.shuffle(eligible_client_ids)
    selected_natural_ids = eligible_client_ids[:config.num_users]
    selected_set = set(selected_natural_ids)

    def keep_selected(batch):
        return [client_id in selected_set for client_id in batch["character_id"]]

    selected_rows = raw.filter(keep_selected, batched=True)
    samples_by_client = defaultdict(lambda: ([], []))
    for client_id, sequence, target in zip(
        selected_rows["character_id"],
        selected_rows["x"],
        selected_rows["y"],
    ):
        samples_by_client[client_id][0].append(sequence)
        samples_by_client[client_id][1].append(target)

    char_to_id = {char: index for index, char in enumerate(SHAKESPEARE_VOCAB)}
    train_blocks = {}
    test_blocks = {}
    client_names = {}
    train_sizes = []
    test_sizes = []

    for local_client_id, natural_client_id in enumerate(selected_natural_ids):
        sequences, targets = samples_by_client[natural_client_id]
        sample_count = len(targets)
        split_index = int(sample_count * config.shakespeare_train_fraction)
        split_index = min(max(1, split_index), sample_count - 1)

        encoded_inputs = np.stack([
            _encode_sequence(
                sequence,
                char_to_id,
                config.shakespeare_sequence_length,
            )
            for sequence in sequences
        ])
        encoded_targets = np.asarray(
            [_encode_target(target, char_to_id) for target in targets],
            dtype=np.int64,
        )

        # Preserve the original per-client order: earlier samples train,
        # later samples test. This avoids random overlap leakage.
        train_blocks[local_client_id] = (
            encoded_inputs[:split_index],
            encoded_targets[:split_index],
        )
        test_blocks[local_client_id] = (
            encoded_inputs[split_index:],
            encoded_targets[split_index:],
        )
        client_names[local_client_id] = natural_client_id
        train_sizes.append(split_index)
        test_sizes.append(sample_count - split_index)

    train = _build_encoded_dataset(
        train_blocks,
        client_names,
        config.shakespeare_sequence_length,
    )
    test = _build_encoded_dataset(
        test_blocks,
        client_names,
        config.shakespeare_sequence_length,
    )

    print(
        f"Selected {config.num_users} natural clients using seed {config.seed}; "
        f"train samples={len(train)}, test samples={len(test)}"
    )
    print(
        "Per-client train size: "
        f"min={min(train_sizes)}, median={int(np.median(train_sizes))}, max={max(train_sizes)}"
    )
    print(
        "Per-client test size:  "
        f"min={min(test_sizes)}, median={int(np.median(test_sizes))}, max={max(test_sizes)}"
    )
    return train, test


def natural_client_indices(dataset_train, dataset_test):
    """Return local client maps created by the natural Shakespeare loader."""
    if not hasattr(dataset_train, "client_indices") or not hasattr(dataset_test, "client_indices"):
        raise ValueError("The supplied datasets do not contain natural client partitions.")
    return dataset_train.client_indices, dataset_test.client_indices


def load_configured_datasets(config):
    root = Path(config.data_root)
    if config.dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean, std)
        ])
        train = torchvision_datasets.CIFAR10(
            root / "CIFAR10", train=True, download=True, transform=train_transform
        )
        test = torchvision_datasets.CIFAR10(
            root / "CIFAR10", train=False, download=True, transform=test_transform
        )
    elif config.dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
        ])
        train = torchvision_datasets.MNIST(
            root / "MNIST", train=True, download=True, transform=transform
        )
        test = torchvision_datasets.MNIST(
            root / "MNIST", train=False, download=True, transform=transform
        )
    elif config.dataset == "shakespeare":
        train, test = _load_leaf_shakespeare(config)
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")

    train_labels = np.asarray(train.targets, dtype=np.int64)
    test_labels = np.asarray(test.targets, dtype=np.int64)
    return train, test, train_labels, test_labels
