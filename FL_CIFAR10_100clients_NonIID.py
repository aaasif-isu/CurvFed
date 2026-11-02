"""Pure Federated Learning (FedAvg) on CIFAR-10 with ResNet-18.

This is a pure FL baseline:
  * every selected client receives the complete ResNet-18 model;
  * every client trains the complete model on its local data;
  * clients return model parameters, not smashed activations;
  * the server performs sample-weighted FedAvg;
  * there is no split server, EMD, ORC, FedWaD, or client clustering.

Change only the ACTIVE EXPERIMENT section for normal runs.
"""

from __future__ import annotations

import copy
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18


# =====================================================================
# ACTIVE EXPERIMENT — CHANGE ONLY THESE VALUES
# =====================================================================

NUM_USERS = 100                 # 10 or 100
ALPHA = 0.5                    # 999999 for IID; otherwise 0.1, 0.5, or 1.0
FRAC = 0.1                     # 1.0 for 10 users; 0.1 for 100 users

SEED = 42
GLOBAL_ROUNDS = 50
LOCAL_EPOCHS = 1
LEARNING_RATE = 0.0001
BATCH_SIZE = 256

# Adam matches the 1e-4 learning rate used in the SplitFed experiments.
# If the SplitFed baseline uses a different optimizer, change this too.
OPTIMIZER = "adam"             # "adam" or "sgd"


# =====================================================================
# FIXED SETTINGS — NORMALLY DO NOT CHANGE
# =====================================================================

DATA_ROOT = Path("data")
OUTPUT_ROOT = Path("FL_Results")
NUM_CLASSES = 10
MIN_SAMPLES_PER_CLIENT = 10
NUM_WORKERS = 4


def validate_config() -> None:
    """Validate the experiment settings before downloading or training."""

    if NUM_USERS not in (10, 100):
        raise ValueError("NUM_USERS must be 10 or 100.")

    if ALPHA not in (999999, 0.1, 0.5, 1.0):
        raise ValueError("ALPHA must be 999999, 0.1, 0.5, or 1.0.")

    if not 0 < FRAC <= 1:
        raise ValueError("FRAC must satisfy 0 < FRAC <= 1.")

    active_clients = max(1, int(NUM_USERS * FRAC))

    if NUM_USERS == 10 and active_clients != 10:
        raise ValueError("For 10 users, use FRAC=1.0.")

    if NUM_USERS == 100 and active_clients != 10:
        raise ValueError("For 100 users, use FRAC=0.1.")

    if min(GLOBAL_ROUNDS, LOCAL_EPOCHS, BATCH_SIZE) <= 0:
        raise ValueError("Rounds, local epochs, and batch size must be positive.")

    if LEARNING_RATE <= 0:
        raise ValueError("LEARNING_RATE must be positive.")

    if OPTIMIZER not in ("adam", "sgd"):
        raise ValueError("OPTIMIZER must be 'adam' or 'sgd'.")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # These settings improve repeatability.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_worker_init_fn(base_seed: int):
    """Return a deterministic DataLoader worker initializer."""

    def seed_worker(worker_id: int) -> None:
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    return seed_worker


def build_datasets():
    """Download and construct CIFAR-10 training and test datasets."""

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

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=DATA_ROOT / "CIFAR10",
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=DATA_ROOT / "CIFAR10",
        train=False,
        download=True,
        transform=test_transform,
    )

    labels = np.asarray(train_dataset.targets, dtype=np.int64)
    return train_dataset, test_dataset, labels


def alpha_tag() -> str:
    """Return a filename-safe partition label."""

    if ALPHA >= 999999:
        return "iid"
    return str(ALPHA).replace(".", "p")


def partition_file_path() -> Path:
    """Return the saved client-partition filename."""

    partition_dir = DATA_ROOT / "partitions"
    partition_dir.mkdir(parents=True, exist_ok=True)

    return partition_dir / (
        f"cifar10_clients{NUM_USERS}_alpha{alpha_tag()}_seed{SEED}.npz"
    )


def iid_partition(num_samples: int) -> dict[int, np.ndarray]:
    """Split shuffled training indices approximately equally across clients."""

    rng = np.random.default_rng(SEED)
    indices = rng.permutation(num_samples)
    splits = np.array_split(indices, NUM_USERS)

    return {
        client_id: np.asarray(split, dtype=np.int64)
        for client_id, split in enumerate(splits)
    }


def dirichlet_partition(labels: np.ndarray) -> dict[int, np.ndarray]:
    """Create a class-wise Dirichlet non-IID partition.

    The partition is regenerated until every client has at least
    MIN_SAMPLES_PER_CLIENT samples.
    """

    attempt = 0

    while True:
        rng = np.random.default_rng(SEED + attempt)
        client_indices: list[list[int]] = [[] for _ in range(NUM_USERS)]

        for class_id in range(NUM_CLASSES):
            class_indices = np.where(labels == class_id)[0]
            rng.shuffle(class_indices)

            proportions = rng.dirichlet(
                np.full(NUM_USERS, ALPHA, dtype=np.float64)
            )

            # Avoid sending still more samples to already-large clients.
            average_size = len(labels) / NUM_USERS
            capacity_mask = np.asarray(
                [len(indices) < average_size for indices in client_indices],
                dtype=np.float64,
            )
            proportions = proportions * capacity_mask

            if proportions.sum() == 0:
                proportions = np.full(NUM_USERS, 1.0 / NUM_USERS)
            else:
                proportions = proportions / proportions.sum()

            split_points = (
                np.cumsum(proportions)[:-1] * len(class_indices)
            ).astype(int)

            class_splits = np.split(class_indices, split_points)

            for client_id, split in enumerate(class_splits):
                client_indices[client_id].extend(split.tolist())

        sizes = [len(indices) for indices in client_indices]

        if min(sizes) >= MIN_SAMPLES_PER_CLIENT:
            result = {}

            for client_id, indices in enumerate(client_indices):
                indices_array = np.asarray(indices, dtype=np.int64)
                rng.shuffle(indices_array)
                result[client_id] = indices_array

            return result

        attempt += 1

        if attempt >= 1000:
            raise RuntimeError(
                "Unable to create a Dirichlet partition with at least "
                f"{MIN_SAMPLES_PER_CLIENT} samples per client."
            )


def save_partition(
    path: Path,
    client_indices: dict[int, np.ndarray],
) -> None:
    """Save all client index arrays to one compressed NumPy archive."""

    np.savez_compressed(
        path,
        **{
            f"client_{client_id}": indices
            for client_id, indices in client_indices.items()
        },
    )


def load_partition(path: Path) -> dict[int, np.ndarray]:
    """Load a previously saved client partition."""

    archive = np.load(path, allow_pickle=False)

    client_indices = {
        client_id: np.asarray(
            archive[f"client_{client_id}"], dtype=np.int64
        )
        for client_id in range(NUM_USERS)
    }

    return client_indices


def get_or_create_partition(labels: np.ndarray) -> dict[int, np.ndarray]:
    """Reuse a saved partition, or create and save a new one."""

    path = partition_file_path()

    if path.exists():
        print(f"Loading saved client partition: {path}")
        client_indices = load_partition(path)
    else:
        if ALPHA >= 999999:
            print("Creating IID client partition...")
            client_indices = iid_partition(len(labels))
        else:
            print(f"Creating Dirichlet non-IID partition with alpha={ALPHA}...")
            client_indices = dirichlet_partition(labels)

        save_partition(path, client_indices)
        print(f"Saved client partition: {path}")

    all_indices = np.concatenate(list(client_indices.values()))

    if len(all_indices) != len(labels):
        raise RuntimeError(
            "Client partition does not contain exactly all training samples."
        )

    if len(np.unique(all_indices)) != len(labels):
        raise RuntimeError("Client partition contains duplicate sample indices.")

    return client_indices


def print_partition_summary(
    labels: np.ndarray,
    client_indices: dict[int, np.ndarray],
) -> None:
    """Print client sizes and class counts for verification."""

    print("\n========== CLIENT PARTITION SUMMARY ==========")

    for client_id in range(NUM_USERS):
        indices = client_indices[client_id]
        counts = np.bincount(labels[indices], minlength=NUM_CLASSES)

        print(
            f"Client {client_id:3d}: samples={len(indices):5d}, "
            f"class_counts={counts.tolist()}"
        )

    print("==============================================\n")


def build_model() -> nn.Module:
    """Build a CIFAR-10-adapted ResNet-18."""

    model = resnet18(weights=None, num_classes=NUM_CLASSES)

    # Standard ImageNet ResNet starts with a 7x7 stride-2 convolution and
    # max-pooling. CIFAR-10 uses 32x32 images, so use the common CIFAR variant.
    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()

    return model


def build_optimizer(model: nn.Module):
    """Create the configured local optimizer."""

    if OPTIMIZER == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
        )

    return torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=5e-4,
    )


def train_local_model(
    global_model: nn.Module,
    train_dataset,
    indices: np.ndarray,
    device: torch.device,
    round_number: int,
    client_id: int,
) -> tuple[OrderedDict, int, float, float]:
    """Train one complete local model and return its state and metrics."""

    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()

    optimizer = build_optimizer(local_model)
    criterion = nn.CrossEntropyLoss()

    generator = torch.Generator()
    generator.manual_seed(SEED + round_number * 1000 + client_id)

    loader = DataLoader(
        Subset(train_dataset, indices.tolist()),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=generator,
        worker_init_fn=make_worker_init_fn(
            SEED + round_number * 1000 + client_id
        ),
    )

    loss_sum = 0.0
    correct = 0
    seen = 0

    for _ in range(LOCAL_EPOCHS):
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = local_model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            loss_sum += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == targets).sum().item()
            seen += batch_size

    state = OrderedDict(
        (name, tensor.detach().cpu().clone())
        for name, tensor in local_model.state_dict().items()
    )

    average_loss = loss_sum / seen
    accuracy = 100.0 * correct / seen

    del local_model, optimizer, loader

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return state, len(indices), average_loss, accuracy


def weighted_fedavg(
    client_states: list[OrderedDict],
    client_sample_counts: list[int],
) -> OrderedDict:
    """Perform sample-count-weighted FedAvg over complete client models."""

    if not client_states:
        raise ValueError("FedAvg received no client states.")

    total_samples = sum(client_sample_counts)

    if total_samples <= 0:
        raise ValueError("FedAvg received no client samples.")

    averaged = OrderedDict()

    for key in client_states[0].keys():
        first_tensor = client_states[0][key]

        if torch.is_floating_point(first_tensor):
            value = torch.zeros_like(first_tensor)

            for state, sample_count in zip(
                client_states, client_sample_counts
            ):
                value += state[key] * (sample_count / total_samples)

            averaged[key] = value
        else:
            # BatchNorm's num_batches_tracked is integer-valued and should
            # not be averaged as a floating-point parameter.
            largest_client = int(np.argmax(client_sample_counts))
            averaged[key] = client_states[largest_client][key].clone()

    return averaged


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the current global model on the central test set."""

    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    loss_sum = 0.0
    correct = 0
    seen = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss_sum += criterion(logits, targets).item()
        correct += (logits.argmax(dim=1) == targets).sum().item()
        seen += targets.size(0)

    return loss_sum / seen, 100.0 * correct / seen


def experiment_name() -> str:
    """Return a descriptive result name."""

    partition = "IID" if ALPHA >= 999999 else f"NonIID_alpha{ALPHA}"

    return (
        f"FL_resnet18_cifar10_{partition}_clients{NUM_USERS}_"
        f"frac{FRAC}_seed{SEED}_rounds{GLOBAL_ROUNDS}"
    ).replace(".", "p")


def save_results(
    records: list[dict],
    output_dir: Path,
    base_name: str,
) -> None:
    """Save round-level results after every round."""

    frame = pd.DataFrame(records)
    csv_path = output_dir / f"{base_name}.csv"
    xlsx_path = output_dir / f"{base_name}.xlsx"

    frame.to_csv(csv_path, index=False)
    frame.to_excel(xlsx_path, index=False)


def main() -> None:
    """Run the complete pure-FL experiment."""

    validate_config()
    seed_everything(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    active_clients = max(1, int(NUM_USERS * FRAC))

    print("=" * 70)
    print("PURE FEDERATED LEARNING — FedAvg")
    print("=" * 70)
    print(f"Device:                 {device}")
    print("Dataset:                CIFAR-10")
    print("Model:                  Full ResNet-18 on every client")
    print(f"Total clients:          {NUM_USERS}")
    print(f"Active clients/round:   {active_clients}")
    print(f"Alpha:                  {ALPHA}")
    print(f"Global rounds:          {GLOBAL_ROUNDS}")
    print(f"Local epochs:           {LOCAL_EPOCHS}")
    print(f"Learning rate:          {LEARNING_RATE}")
    print(f"Optimizer:              {OPTIMIZER}")
    print(f"Batch size:             {BATCH_SIZE}")
    print(f"Seed:                   {SEED}")
    print("=" * 70)

    train_dataset, test_dataset, labels = build_datasets()
    client_indices = get_or_create_partition(labels)
    print_partition_summary(labels, client_indices)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        worker_init_fn=make_worker_init_fn(SEED),
    )

    global_model = build_model().to(device)

    output_dir = OUTPUT_ROOT / experiment_name()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = experiment_name()

    selection_rng = np.random.default_rng(SEED)
    records: list[dict] = []
    cumulative_time = 0.0

    initial_loss, initial_accuracy = evaluate(
        global_model, test_loader, device
    )
    print(
        f"Initial global model | Test Accuracy: {initial_accuracy:.3f}% "
        f"| Test Loss: {initial_loss:.4f}\n"
    )

    for round_number in range(1, GLOBAL_ROUNDS + 1):
        round_start = time.perf_counter()

        selected_clients = selection_rng.choice(
            NUM_USERS,
            size=active_clients,
            replace=False,
        ).tolist()

        print(
            f"\n========== Global Round "
            f"{round_number}/{GLOBAL_ROUNDS} =========="
        )
        print(f"Selected clients: {selected_clients}")

        client_states = []
        client_sample_counts = []
        weighted_train_loss_sum = 0.0
        weighted_train_accuracy_sum = 0.0
        participating_samples = 0

        # Keep the global model on CPU while each local model is trained.
        global_model_cpu = copy.deepcopy(global_model).cpu()

        for client_id in selected_clients:
            state, sample_count, train_loss, train_accuracy = train_local_model(
                global_model=global_model_cpu,
                train_dataset=train_dataset,
                indices=client_indices[client_id],
                device=device,
                round_number=round_number,
                client_id=client_id,
            )

            client_states.append(state)
            client_sample_counts.append(sample_count)
            participating_samples += sample_count
            weighted_train_loss_sum += train_loss * sample_count
            weighted_train_accuracy_sum += train_accuracy * sample_count

            print(
                f"Client {client_id:3d} | samples={sample_count:5d} "
                f"| Train Accuracy={train_accuracy:7.3f}% "
                f"| Train Loss={train_loss:.4f}"
            )

        del global_model_cpu

        averaged_state = weighted_fedavg(
            client_states,
            client_sample_counts,
        )
        global_model.load_state_dict(averaged_state, strict=True)
        global_model.to(device)

        train_loss = weighted_train_loss_sum / participating_samples
        train_accuracy = weighted_train_accuracy_sum / participating_samples
        test_loss, test_accuracy = evaluate(
            global_model, test_loader, device
        )

        round_time = time.perf_counter() - round_start
        cumulative_time += round_time

        print("-" * 70)
        print(
            f"Round {round_number:3d} | "
            f"Train Accuracy: {train_accuracy:.3f}% "
            f"| Train Loss: {train_loss:.4f}"
        )
        print(
            f"Round {round_number:3d} | "
            f"Test Accuracy:  {test_accuracy:.3f}% "
            f"| Test Loss:  {test_loss:.4f}"
        )
        print(
            f"Round time: {round_time:.2f} sec | "
            f"Cumulative time: {cumulative_time:.2f} sec"
        )
        print("-" * 70)

        records.append(
            {
                "round": round_number,
                "selected_clients": ",".join(
                    str(client_id) for client_id in selected_clients
                ),
                "participating_samples": participating_samples,
                "acc_train": train_accuracy,
                "loss_train": train_loss,
                "acc_test": test_accuracy,
                "loss_test": test_loss,
                "round_time_sec": round_time,
                "cumulative_time_sec": cumulative_time,
            }
        )

        save_results(records, output_dir, base_name)

        checkpoint_path = output_dir / "global_model_latest.pth"
        torch.save(
            {
                "round": round_number,
                "model_state_dict": global_model.state_dict(),
                "config": {
                    "num_users": NUM_USERS,
                    "alpha": ALPHA,
                    "frac": FRAC,
                    "seed": SEED,
                    "global_rounds": GLOBAL_ROUNDS,
                    "local_epochs": LOCAL_EPOCHS,
                    "learning_rate": LEARNING_RATE,
                    "batch_size": BATCH_SIZE,
                    "optimizer": OPTIMIZER,
                },
            },
            checkpoint_path,
        )

    final_model_path = output_dir / "global_model_final.pth"
    torch.save(global_model.state_dict(), final_model_path)

    print("\n" + "=" * 70)
    print("PURE FL TRAINING COMPLETED")
    print("=" * 70)
    print(f"Final test accuracy: {records[-1]['acc_test']:.3f}%")
    print(f"Final test loss:     {records[-1]['loss_test']:.4f}")
    print(f"Total time:          {cumulative_time:.2f} sec")
    print(f"Results directory:   {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
