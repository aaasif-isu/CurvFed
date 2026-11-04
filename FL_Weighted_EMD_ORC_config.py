"""Configuration for Pure FL + Weighted EMD + ORC experiments.

For a normal experiment, edit only the ACTIVE EXPERIMENT section.
The training program imports CONFIG from this file.
"""

from dataclasses import dataclass
from typing import Literal


PartitionName = Literal["iid", "dirichlet"]
OptimizerName = Literal["adam", "sgd"]


# ============================================================================
# ACTIVE EXPERIMENT -- EDIT THESE VALUES
# ============================================================================

NUM_USERS = 100                 # Total federated clients: 10 or 100
FRAC = 0.1                      # 1.0 for 10 users; 0.1 for 100 users
PARTITION: PartitionName = "dirichlet"
ALPHA = 0.5                     # Used only when PARTITION="dirichlet"

SEED = 42
GLOBAL_ROUNDS = 50
LOCAL_EPOCHS = 1
BATCH_SIZE = 256
LEARNING_RATE = 0.0001
OPTIMIZER: OptimizerName = "adam"
WEIGHT_DECAY = 0.0

# Client-distribution extraction and exact discrete EMD.
FEATURE_SAMPLES_PER_CLIENT = 256
EMD_SUPPORT_SIZE = 32

# Client graph and Ollivier-Ricci curvature.
N_CLUSTERS = 3
GRAPH_K_NEIGHBORS = 3
ORC_IDLENESS = 0.5

# Cluster-weight score:
#   score_k = sample_fraction_k ** SAMPLE_WEIGHT_POWER
#             * quality_k ** QUALITY_WEIGHT_POWER
# Setting QUALITY_WEIGHT_POWER=0 makes the second aggregation stage equivalent
# to sample weighting; use a positive value for EMD/ORC-aware aggregation.
SAMPLE_WEIGHT_POWER = 1.0
QUALITY_WEIGHT_POWER = 1.0
MIN_QUALITY = 0.10
MAX_QUALITY = 2.00


# ============================================================================
# FIXED SETTINGS -- NORMALLY DO NOT CHANGE
# ============================================================================

DATA_ROOT = "data"
RESULTS_ROOT = "FL_Weighted_EMD_ORC_Results"
NUM_CLASSES = 10
NUM_WORKERS = 2
PIN_MEMORY = True
DOWNLOAD_DATA = True
SAVE_ROUND_CHECKPOINTS = False


@dataclass(frozen=True)
class ExperimentConfig:
    num_users: int
    frac: float
    partition: PartitionName
    alpha: float
    seed: int
    global_rounds: int
    local_epochs: int
    batch_size: int
    learning_rate: float
    optimizer: OptimizerName
    weight_decay: float
    feature_samples_per_client: int
    emd_support_size: int
    n_clusters: int
    graph_k_neighbors: int
    orc_idleness: float
    sample_weight_power: float
    quality_weight_power: float
    min_quality: float
    max_quality: float
    data_root: str
    results_root: str
    num_classes: int
    num_workers: int
    pin_memory: bool
    download_data: bool
    save_round_checkpoints: bool

    @property
    def num_active_clients(self) -> int:
        return max(1, int(self.num_users * self.frac))

    @property
    def partition_tag(self) -> str:
        if self.partition == "iid":
            return "IID"
        return f"NonIID_alpha{str(self.alpha).replace('.', 'p')}"

    @property
    def experiment_tag(self) -> str:
        frac = str(self.frac).replace(".", "p")
        return (
            f"FL_Weighted_EMD_ORC_resnet18_cifar10_{self.partition_tag}_"
            f"clients{self.num_users}_frac{frac}_seed{self.seed}_"
            f"rounds{self.global_rounds}"
        )


def _validate(cfg: ExperimentConfig) -> None:
    if cfg.num_users not in (10, 100):
        raise ValueError("NUM_USERS must be 10 or 100.")
    if not 0 < cfg.frac <= 1:
        raise ValueError("FRAC must satisfy 0 < FRAC <= 1.")
    if cfg.num_users == 10 and cfg.num_active_clients != 10:
        raise ValueError("For 10 clients, use FRAC=1.0 so all 10 participate.")
    if cfg.num_users == 100 and cfg.num_active_clients != 10:
        raise ValueError("For 100 clients, use FRAC=0.1 so 10 participate.")
    if cfg.partition not in ("iid", "dirichlet"):
        raise ValueError("PARTITION must be 'iid' or 'dirichlet'.")
    if cfg.partition == "dirichlet" and cfg.alpha <= 0:
        raise ValueError("ALPHA must be positive for a Dirichlet partition.")
    if min(cfg.global_rounds, cfg.local_epochs, cfg.batch_size) <= 0:
        raise ValueError("Rounds, local epochs, and batch size must be positive.")
    if cfg.learning_rate <= 0 or cfg.weight_decay < 0:
        raise ValueError("Learning rate must be positive and weight decay nonnegative.")
    if cfg.optimizer not in ("adam", "sgd"):
        raise ValueError("OPTIMIZER must be 'adam' or 'sgd'.")
    if cfg.feature_samples_per_client <= 0 or cfg.emd_support_size <= 1:
        raise ValueError("Feature sample and EMD support sizes must be positive.")
    if cfg.emd_support_size > cfg.feature_samples_per_client:
        raise ValueError("EMD_SUPPORT_SIZE cannot exceed FEATURE_SAMPLES_PER_CLIENT.")
    if not 1 <= cfg.n_clusters <= cfg.num_active_clients:
        raise ValueError("N_CLUSTERS must be between 1 and active clients per round.")
    if not 1 <= cfg.graph_k_neighbors < cfg.num_active_clients:
        raise ValueError("GRAPH_K_NEIGHBORS must be between 1 and active_clients-1.")
    if not 0 <= cfg.orc_idleness < 1:
        raise ValueError("ORC_IDLENESS must satisfy 0 <= value < 1.")
    if cfg.sample_weight_power < 0 or cfg.quality_weight_power < 0:
        raise ValueError("Weight powers must be nonnegative.")
    if not 0 < cfg.min_quality <= cfg.max_quality:
        raise ValueError("Quality bounds must satisfy 0 < MIN <= MAX.")


CONFIG = ExperimentConfig(
    num_users=NUM_USERS,
    frac=FRAC,
    partition=PARTITION,
    alpha=ALPHA,
    seed=SEED,
    global_rounds=GLOBAL_ROUNDS,
    local_epochs=LOCAL_EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    optimizer=OPTIMIZER,
    weight_decay=WEIGHT_DECAY,
    feature_samples_per_client=FEATURE_SAMPLES_PER_CLIENT,
    emd_support_size=EMD_SUPPORT_SIZE,
    n_clusters=N_CLUSTERS,
    graph_k_neighbors=GRAPH_K_NEIGHBORS,
    orc_idleness=ORC_IDLENESS,
    sample_weight_power=SAMPLE_WEIGHT_POWER,
    quality_weight_power=QUALITY_WEIGHT_POWER,
    min_quality=MIN_QUALITY,
    max_quality=MAX_QUALITY,
    data_root=DATA_ROOT,
    results_root=RESULTS_ROOT,
    num_classes=NUM_CLASSES,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    download_data=DOWNLOAD_DATA,
    save_round_checkpoints=SAVE_ROUND_CHECKPOINTS,
)

_validate(CONFIG)


if __name__ == "__main__":
    print("========== PURE FL + WEIGHTED EMD + ORC CONFIG ==========")
    print(CONFIG)
    print(f"Active clients per round: {CONFIG.num_active_clients}")
    print(f"Partition:               {CONFIG.partition_tag}")
    print(f"Experiment tag:          {CONFIG.experiment_tag}")
    print("==========================================================")
