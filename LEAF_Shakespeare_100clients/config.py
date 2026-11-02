"""Central configuration for SplitFed experiments.

For normal experiments, edit only the ACTIVE EXPERIMENT section.
"""

from dataclasses import dataclass
from typing import Literal

DatasetName = Literal["cifar10", "mnist", "shakespeare"]
ModelName = Literal["resnet18", "charlstm"]


# =====================================================================
# ACTIVE EXPERIMENT — CHANGE ONLY THESE VALUES
# =====================================================================

DATASET: DatasetName = "shakespeare"
MODEL: ModelName = "charlstm"
NUM_USERS = 100                 # 10 or 100
ALPHA = 0.5                    # Ignored for natural LEAF Shakespeare

SEED = 42
GLOBAL_ROUNDS = 50
LOCAL_EPOCHS = 1
LEARNING_RATE = 0.0001
FRAC = 0.1                     # 1.0 for 10 users; 0.1 for 100 users
BATCH_SIZE = 32                # Safer for naturally small Shakespeare clients
N_CLUSTERS = 3

# FedWaD-only controls. They are ignored by other methods.
DISTANCE_METHOD = "fedwad"     # "emd" or "fedwad"
FEDWAD_N_SUPP = 10
FEDWAD_EPOCHS = 20
FEDWAD_T = 0.5
DISTANCE_POOL_SIZE = 1


# =====================================================================
# FIXED DATASET / MODEL SETTINGS — NORMALLY DO NOT CHANGE
# =====================================================================

DATA_ROOT = "data"

# LEAF Shakespeare as republished by Flower Labs. Each character_id is a
# natural federated client (a character/play pair). The source rows already
# contain an 80-character input sequence and its next-character label.
SHAKESPEARE_DATASET_NAME = "flwrlabs/shakespeare"
SHAKESPEARE_TRAIN_FRACTION = 0.8
SHAKESPEARE_SEQUENCE_LENGTH = 80
SHAKESPEARE_MIN_SAMPLES_PER_CLIENT = 20

CHARLSTM_EMBED_DIM = 8
CHARLSTM_HIDDEN_SIZE = 256
CHARLSTM_NUM_LAYERS = 2

DATASET_NUM_CLASSES = {"cifar10": 10, "mnist": 10, "shakespeare": 80}
DATASET_CHANNELS = {"cifar10": 3, "mnist": 1, "shakespeare": None}


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: DatasetName
    model: ModelName
    num_users: int
    alpha: float
    seed: int
    global_rounds: int
    local_epochs: int
    learning_rate: float
    frac: float
    batch_size: int
    n_clusters: int
    distance_method: str
    fedwad_n_supp: int
    fedwad_epochs: int
    fedwad_t: float
    distance_pool_size: int
    data_root: str
    shakespeare_dataset_name: str
    shakespeare_train_fraction: float
    shakespeare_sequence_length: int
    shakespeare_min_samples_per_client: int
    charlstm_embed_dim: int
    charlstm_hidden_size: int
    charlstm_num_layers: int

    @property
    def num_active_clients(self) -> int:
        return max(1, int(self.num_users * self.frac))

    @property
    def uses_natural_partition(self) -> bool:
        return self.dataset == "shakespeare"

    @property
    def iid_like(self) -> bool:
        return not self.uses_natural_partition and self.alpha >= 999999

    @property
    def partition_name(self) -> str:
        if self.uses_natural_partition:
            return "natural_noniid"
        return "iid" if self.iid_like else f"dirichlet_alpha_{self.alpha}"

    @property
    def num_classes(self) -> int:
        return DATASET_NUM_CLASSES[self.dataset]

    @property
    def input_channels(self):
        return DATASET_CHANNELS[self.dataset]

    @property
    def experiment_tag(self) -> str:
        partition_tag = self.partition_name.replace(".", "p")
        return (
            f"{self.dataset}_{self.model}_clients{self.num_users}_"
            f"frac{self.frac}_{partition_tag}_seed{self.seed}_"
            f"rounds{self.global_rounds}"
        )


def _validate(cfg: ExperimentConfig) -> None:
    valid_pairs = {
        ("cifar10", "resnet18"),
        ("mnist", "resnet18"),
        ("shakespeare", "charlstm"),
    }
    if (cfg.dataset, cfg.model) not in valid_pairs:
        raise ValueError(
            f"Unsupported dataset/model pair: {cfg.dataset}/{cfg.model}. "
            f"Valid pairs: {sorted(valid_pairs)}"
        )
    if cfg.num_users not in (10, 100):
        raise ValueError("NUM_USERS must be 10 or 100 for this experiment matrix.")
    if cfg.dataset != "shakespeare" and cfg.alpha not in (999999, 0.1, 0.5, 1.0):
        raise ValueError("ALPHA must be one of: 999999, 0.1, 0.5, 1.0.")
    if not 0 < cfg.frac <= 1:
        raise ValueError("FRAC must satisfy 0 < FRAC <= 1.")
    if cfg.num_users == 10 and cfg.num_active_clients != 10:
        raise ValueError("The 10-client setting must use FRAC=1.0 (10 active clients).")
    if cfg.num_users == 100 and cfg.num_active_clients != 10:
        raise ValueError("The 100-client setting must use FRAC=0.1 (10 active clients).")
    if min(cfg.global_rounds, cfg.local_epochs, cfg.batch_size) <= 0:
        raise ValueError("Rounds, local epochs, and batch size must be positive.")
    if cfg.learning_rate <= 0:
        raise ValueError("LEARNING_RATE must be positive.")
    if not 1 <= cfg.n_clusters <= cfg.num_active_clients:
        raise ValueError("N_CLUSTERS must be between 1 and active clients per round.")
    if cfg.distance_method not in ("emd", "fedwad"):
        raise ValueError("DISTANCE_METHOD must be 'emd' or 'fedwad'.")
    if cfg.fedwad_n_supp <= 0 or cfg.fedwad_epochs <= 0:
        raise ValueError("FedWaD support size and epochs must be positive.")
    if not 0 < cfg.fedwad_t < 1:
        raise ValueError("FEDWAD_T must be strictly between 0 and 1.")
    if cfg.distance_pool_size <= 0:
        raise ValueError("DISTANCE_POOL_SIZE must be positive.")
    if not 0 < cfg.shakespeare_train_fraction < 1:
        raise ValueError("SHAKESPEARE_TRAIN_FRACTION must be between 0 and 1.")
    if cfg.shakespeare_sequence_length != 80:
        raise ValueError("LEAF Shakespeare inputs must contain exactly 80 characters.")
    if cfg.shakespeare_min_samples_per_client < 2:
        raise ValueError("Each Shakespeare client must retain at least two samples.")


CONFIG = ExperimentConfig(
    dataset=DATASET,
    model=MODEL,
    num_users=NUM_USERS,
    alpha=ALPHA,
    seed=SEED,
    global_rounds=GLOBAL_ROUNDS,
    local_epochs=LOCAL_EPOCHS,
    learning_rate=LEARNING_RATE,
    frac=FRAC,
    batch_size=BATCH_SIZE,
    n_clusters=N_CLUSTERS,
    distance_method=DISTANCE_METHOD,
    fedwad_n_supp=FEDWAD_N_SUPP,
    fedwad_epochs=FEDWAD_EPOCHS,
    fedwad_t=FEDWAD_T,
    distance_pool_size=DISTANCE_POOL_SIZE,
    data_root=DATA_ROOT,
    shakespeare_dataset_name=SHAKESPEARE_DATASET_NAME,
    shakespeare_train_fraction=SHAKESPEARE_TRAIN_FRACTION,
    shakespeare_sequence_length=SHAKESPEARE_SEQUENCE_LENGTH,
    shakespeare_min_samples_per_client=SHAKESPEARE_MIN_SAMPLES_PER_CLIENT,
    charlstm_embed_dim=CHARLSTM_EMBED_DIM,
    charlstm_hidden_size=CHARLSTM_HIDDEN_SIZE,
    charlstm_num_layers=CHARLSTM_NUM_LAYERS,
)

_validate(CONFIG)

if __name__ == "__main__":
    print("========== ACTIVE EXPERIMENT CONFIG ==========")
    print(CONFIG)
    print(f"Active clients / round: {CONFIG.num_active_clients}")
    print(f"Partition:              {CONFIG.partition_name}")
    print(f"Experiment tag:         {CONFIG.experiment_tag}")
    print("===============================================")
