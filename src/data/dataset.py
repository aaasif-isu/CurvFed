"""Shared datasets for configured SplitFed experiments."""

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


# Canonical 80-character vocabulary used by LEAF Shakespeare experiments.
SHAKESPEARE_VOCAB = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"


class ShakespeareSequenceDataset(Dataset):
    def __init__(self, encoded, sequence_length):
        self.encoded = torch.as_tensor(encoded, dtype=torch.long)
        self.sequence_length = sequence_length

    def __len__(self):
        return max(0, len(self.encoded) - self.sequence_length)

    def __getitem__(self, index):
        x = self.encoded[index:index + self.sequence_length]
        y = self.encoded[index + self.sequence_length]
        return x, y

    @property
    def targets(self):
        return self.encoded[self.sequence_length:].numpy()


def _load_shakespeare(config):
    path = Path(config.shakespeare_text_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Tiny Shakespeare to {path}")
        urlretrieve(config.shakespeare_download_url, path)

    text = path.read_text(encoding="utf-8")
    if len(SHAKESPEARE_VOCAB) != config.num_classes:
        raise ValueError(
            f"Configured Shakespeare vocabulary size is {config.num_classes}, "
            f"but the canonical vocabulary has {len(SHAKESPEARE_VOCAB)} characters."
        )
    unsupported = sorted(set(text) - set(SHAKESPEARE_VOCAB))
    if unsupported:
        print(f"Removing unsupported Shakespeare characters: {unsupported}")
        text = "".join(
            char for char in text
            if char in SHAKESPEARE_VOCAB
        )
    char_to_id = {char: index for index, char in enumerate(SHAKESPEARE_VOCAB)}
    encoded = np.asarray([char_to_id[char] for char in text], dtype=np.int64)
    split = int(len(encoded) * config.shakespeare_train_fraction)
    sequence_length = config.shakespeare_sequence_length
    train = ShakespeareSequenceDataset(encoded[:split], sequence_length)
    test = ShakespeareSequenceDataset(encoded[split:], sequence_length)
    return train, test


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
        train = datasets.CIFAR10(root / "CIFAR10", train=True, download=True, transform=train_transform)
        test = datasets.CIFAR10(root / "CIFAR10", train=False, download=True, transform=test_transform)
    elif config.dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
        ])
        train = datasets.MNIST(root / "MNIST", train=True, download=True, transform=transform)
        test = datasets.MNIST(root / "MNIST", train=False, download=True, transform=transform)
    elif config.dataset == "shakespeare":
        train, test = _load_shakespeare(config)
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")

    train_labels = np.asarray(train.targets, dtype=np.int64)
    test_labels = np.asarray(test.targets, dtype=np.int64)
    return train, test, train_labels, test_labels
