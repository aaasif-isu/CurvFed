"""Central configuration for SplitFed experiments.

Supports loading all parameters from `config.yaml` (or any YAML file via
`load_config_from_yaml(path)` / `CURVFED_CONFIG`), while keeping Python defaults
for backward compatibility.
"""

import ast
import os
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

try:
    import yaml
except ImportError:
    yaml = None

DatasetName = Literal["cifar10", "mnist", "shakespeare"]
ModelName = Literal["resnet18", "charlstm", "smallcnn"]
MethodName = Literal[
    "baseline",
    "fedavg",
    "emd_only",
    "emd_orc",
    "curvfed",
    "emd_orc_weighted",
    "fedwad",
    "tca",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YAML_PATH = REPO_ROOT / "config.yaml"


# =====================================================================
# FALLBACK PYTHON DEFAULTS (used if config.yaml is not present)
# =====================================================================

METHOD: MethodName = "baseline"
DATASET: DatasetName = "cifar10"
MODEL: ModelName = "resnet18"
NUM_USERS = 100                 # 10 or 100
ALPHA = 999999                  # 999999, 0.1, 0.5, or 1.0

SEED = 42
GLOBAL_ROUNDS = 50
LOCAL_EPOCHS = 1
LEARNING_RATE = 0.0001
FRAC = 0.1
BATCH_SIZE = 256
N_CLUSTERS = 3

# FedWaD-only controls. They are ignored by other methods.
DISTANCE_METHOD = "fedwad"     # "emd" or "fedwad"
FEDWAD_N_SUPP = 10
FEDWAD_EPOCHS = 20
FEDWAD_T = 0.5
DISTANCE_POOL_SIZE = 1

# Phase-1 ablation & consistency controls
CONSISTENT_SPLIT_WEIGHTS = os.getenv("CONSISTENT_SPLIT_WEIGHTS", "1").strip().lower() in ("1", "true", "yes")
CLUSTER_MODE = os.getenv("CLUSTER_MODE", "geometry").strip().lower()   # "geometry" or "random"
ZSCORE_MODE = os.getenv("ZSCORE_MODE", "per_client").strip().lower()   # "per_client", "pooled", or "none"

# Phase-2 TCA controls
TCA_MODE = os.getenv("TCA_MODE", "decoupled").strip().lower()          # "decoupled" or "feedback"
TCA_COST = os.getenv("TCA_COST", "update").strip().lower()             # "update" or "emd"
TCA_MARGINAL = os.getenv("TCA_MARGINAL", "hard").strip().lower()       # "hard", "unbalanced", or "none"
TCA_CALIBRATE = os.getenv("TCA_CALIBRATE", "0").strip().lower() in ("1", "true", "yes")
TCA_KAPPA_TARGET: Union[str, float] = os.getenv("TCA_KAPPA_TARGET", "auto").strip()
TCA_VAR_SCALE = float(os.getenv("TCA_VAR_SCALE", os.getenv("TCA_REG", "1.0")))
TCA_TAU = float(os.getenv("TCA_TAU", "1.0"))
TCA_REG = TCA_VAR_SCALE  # deprecated alias


# =====================================================================
# FIXED DATASET / MODEL SETTINGS — NORMALLY DO NOT CHANGE
# =====================================================================

DATA_ROOT = "data"
SHAKESPEARE_TEXT_PATH = "data/shakespeare/input.txt"
SHAKESPEARE_DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)
SHAKESPEARE_TRAIN_FRACTION = 0.9
SHAKESPEARE_SEQUENCE_LENGTH = 80
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
    method: str = METHOD
    distance_method: str = DISTANCE_METHOD
    fedwad_n_supp: int = FEDWAD_N_SUPP
    fedwad_epochs: int = FEDWAD_EPOCHS
    fedwad_t: float = FEDWAD_T
    distance_pool_size: int = DISTANCE_POOL_SIZE
    data_root: str = DATA_ROOT
    shakespeare_text_path: str = SHAKESPEARE_TEXT_PATH
    shakespeare_download_url: str = SHAKESPEARE_DOWNLOAD_URL
    shakespeare_train_fraction: float = SHAKESPEARE_TRAIN_FRACTION
    shakespeare_sequence_length: int = SHAKESPEARE_SEQUENCE_LENGTH
    charlstm_embed_dim: int = CHARLSTM_EMBED_DIM
    charlstm_hidden_size: int = CHARLSTM_HIDDEN_SIZE
    charlstm_num_layers: int = CHARLSTM_NUM_LAYERS
    max_samples_per_client: Optional[int] = None
    max_test_samples: Optional[int] = None
    smoke_test: bool = False
    consistent_split_weights: bool = CONSISTENT_SPLIT_WEIGHTS
    cluster_mode: str = CLUSTER_MODE
    zscore_mode: str = ZSCORE_MODE
    run_noise_floor: bool = True
    output_dir: Optional[str] = None
    save_excel: bool = True
    tca_mode: str = TCA_MODE
    tca_cost: str = TCA_COST
    tca_marginal: str = TCA_MARGINAL
    tca_calibrate: bool = TCA_CALIBRATE
    tca_kappa_target: Union[str, float] = TCA_KAPPA_TARGET
    tca_var_scale: float = TCA_VAR_SCALE
    tca_tau: float = TCA_TAU

    @property
    def tca_reg(self) -> float:
        return self.tca_var_scale

    @property
    def num_active_clients(self) -> int:
        return max(1, int(self.num_users * self.frac))

    @property
    def iid_like(self) -> bool:
        return self.alpha >= 999999

    @property
    def num_classes(self) -> int:
        return DATASET_NUM_CLASSES[self.dataset]

    @property
    def input_channels(self):
        return DATASET_CHANNELS[self.dataset]

    @property
    def experiment_tag(self) -> str:
        alpha_tag = "iid" if self.iid_like else str(self.alpha).replace(".", "p")
        smoke_prefix = "smoketest_" if self.smoke_test else ""
        return (
            f"{smoke_prefix}{self.dataset}_{self.model}_clients{self.num_users}_"
            f"frac{self.frac}_alpha{alpha_tag}_seed{self.seed}_"
            f"rounds{self.global_rounds}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _validate(cfg: ExperimentConfig) -> None:
    valid_methods = {
        "baseline",
        "fedavg",
        "emd_only",
        "emd_orc",
        "curvfed",
        "emd_orc_weighted",
        "fedwad",
        "tca",
    }
    if cfg.method.lower() not in valid_methods:
        raise ValueError(
            f"Unsupported method: {cfg.method}. Valid methods: {sorted(valid_methods)}"
        )
    valid_pairs = {
        ("cifar10", "resnet18"),
        ("mnist", "resnet18"),
        ("mnist", "smallcnn"),
        ("cifar10", "smallcnn"),
        ("shakespeare", "charlstm"),
    }
    if (cfg.dataset, cfg.model) not in valid_pairs:
        raise ValueError(
            f"Unsupported dataset/model pair: {cfg.dataset}/{cfg.model}. "
            f"Valid pairs: {sorted(valid_pairs)}"
        )
    if cfg.num_users < 2:
        raise ValueError("NUM_USERS must be at least 2.")
    if cfg.alpha <= 0:
        raise ValueError("ALPHA must be strictly positive (use 999999 for IID, or e.g. 0.1, 0.5, 1.0).")
    if not 0 < cfg.frac <= 1:
        raise ValueError("FRAC must satisfy 0 < FRAC <= 1.")
    if min(cfg.global_rounds, cfg.local_epochs, cfg.batch_size) <= 0:
        raise ValueError("Rounds, local epochs, and batch size must be positive.")
    if cfg.learning_rate <= 0:
        raise ValueError("LEARNING_RATE must be positive.")
    if not 1 <= cfg.n_clusters <= cfg.num_active_clients:
        raise ValueError(
            f"N_CLUSTERS ({cfg.n_clusters}) must be between 1 and active clients per round ({cfg.num_active_clients})."
        )
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
    if cfg.max_samples_per_client is not None and cfg.max_samples_per_client <= 0:
        raise ValueError("max_samples_per_client must be positive when provided.")
    if cfg.max_test_samples is not None and cfg.max_test_samples <= 0:
        raise ValueError("max_test_samples must be positive when provided.")
    if cfg.cluster_mode not in ("geometry", "random"):
        raise ValueError("cluster_mode must be 'geometry' or 'random'.")
    if cfg.zscore_mode not in ("per_client", "pooled", "none"):
        raise ValueError("zscore_mode must be 'per_client', 'pooled', or 'none'.")
    if cfg.tca_mode not in ("decoupled", "feedback"):
        raise ValueError("tca_mode must be 'decoupled' or 'feedback'.")
    if cfg.tca_cost not in ("update", "emd"):
        raise ValueError("tca_cost must be 'update' or 'emd'.")
    if cfg.tca_marginal not in ("hard", "unbalanced", "none"):
        raise ValueError("tca_marginal must be 'hard', 'unbalanced', or 'none'.")
    if cfg.tca_var_scale <= 0:
        raise ValueError("tca_var_scale must be strictly positive.")
    if cfg.tca_tau <= 0:
        raise ValueError("tca_tau must be strictly positive.")
    if isinstance(cfg.tca_kappa_target, str):
        if cfg.tca_kappa_target.strip().lower() != "auto":
            try:
                float(cfg.tca_kappa_target)
            except ValueError as exc:
                raise ValueError("tca_kappa_target must be 'auto' or a float.") from exc


def get_smoke_test_config() -> ExperimentConfig:
    """Return the CPU smoke-test configuration (MNIST, SmallCNN, 4 clients, alpha=0.1, 2 rounds)."""
    cfg = ExperimentConfig(
        dataset="mnist",
        model="smallcnn",
        num_users=4,
        alpha=0.1,
        seed=42,
        global_rounds=2,
        local_epochs=1,
        learning_rate=0.001,
        frac=1.0,
        batch_size=64,
        n_clusters=2,
        method="emd_orc",
        distance_method="emd",
        fedwad_n_supp=10,
        fedwad_epochs=5,
        fedwad_t=0.5,
        distance_pool_size=1,
        data_root=DATA_ROOT,
        shakespeare_text_path=SHAKESPEARE_TEXT_PATH,
        shakespeare_download_url=SHAKESPEARE_DOWNLOAD_URL,
        shakespeare_train_fraction=SHAKESPEARE_TRAIN_FRACTION,
        shakespeare_sequence_length=SHAKESPEARE_SEQUENCE_LENGTH,
        charlstm_embed_dim=CHARLSTM_EMBED_DIM,
        charlstm_hidden_size=CHARLSTM_HIDDEN_SIZE,
        charlstm_num_layers=CHARLSTM_NUM_LAYERS,
        max_samples_per_client=512,
        max_test_samples=1024,
        smoke_test=True,
    )
    _validate(cfg)
    return cfg


def _fallback_parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Fallback parser for flat key: value YAML files when PyYAML is unavailable."""
    result: Dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, val_part = line.split(":", 1)
        key = key.strip()
        in_single, in_double = False, False
        clean_chars = []
        for ch in val_part:
            if ch == "'" and not in_double:
                in_single = not in_single
            elif ch == '"' and not in_single:
                in_double = not in_double
            elif ch == "#" and not in_single and not in_double:
                break
            clean_chars.append(ch)
        val_str = "".join(clean_chars).strip()
        low = val_str.lower()
        if low in ("null", "none", "~", ""):
            result[key] = None
        elif low == "true":
            result[key] = True
        elif low == "false":
            result[key] = False
        else:
            try:
                result[key] = ast.literal_eval(val_str)
            except Exception:
                result[key] = val_str
    return result


_YAML_KEY_ALIASES = {
    "rounds": "global_rounds",
    "lr": "learning_rate",
    "noise_floor": "run_noise_floor",
    "consistent_split": "consistent_split_weights",
    "tca.mode": "tca_mode",
    "tca.cost": "tca_cost",
    "tca.marginal": "tca_marginal",
    "tca.calibrate": "tca_calibrate",
    "tca.kappa_target": "tca_kappa_target",
    "tca.var_scale": "tca_var_scale",
    "tca.reg": "tca_var_scale",
    "tca_reg": "tca_var_scale",
    "tca.tau": "tca_tau",
}


def load_config_from_yaml(yaml_path: Union[str, os.PathLike] = DEFAULT_YAML_PATH) -> ExperimentConfig:
    """Load an `ExperimentConfig` from a YAML file."""
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"YAML configuration file not found: {path}")

    text = path.read_text(encoding="utf-8")
    raw_data = (yaml.safe_load(text) or {}) if yaml is not None else _fallback_parse_simple_yaml(text)

    if not isinstance(raw_data, dict):
        raise ValueError(f"YAML config at {path} must be a top-level key-value mapping.")

    valid_field_names = {f.name for f in fields(ExperimentConfig)}
    normalized: Dict[str, Any] = {}
    for k, v in raw_data.items():
        raw_k = str(k).strip().lower()
        if raw_k == "tca" and isinstance(v, dict):
            for sub_k, sub_v in v.items():
                mapped_sub = _YAML_KEY_ALIASES.get(f"tca.{str(sub_k).strip().lower()}")
                if mapped_sub is None:
                    raise ValueError(f"Unknown TCA sub-parameter 'tca.{sub_k}' in {path}.")
                normalized[mapped_sub] = sub_v
            continue
        key = _YAML_KEY_ALIASES.get(raw_k, raw_k)
        if key not in valid_field_names:
            raise ValueError(
                f"Unknown parameter '{k}' in {path}. "
                f"Valid parameters are: {sorted(valid_field_names)}"
            )
        normalized[key] = v

    for str_key in (
        "method", "dataset", "model", "distance_method", "cluster_mode", "zscore_mode",
        "tca_mode", "tca_cost", "tca_marginal",
    ):
        if str_key in normalized and normalized[str_key] is not None:
            normalized[str_key] = str(normalized[str_key]).strip().lower()

    if "tca_kappa_target" in normalized and normalized["tca_kappa_target"] is not None:
        kt = normalized["tca_kappa_target"]
        if isinstance(kt, str):
            kt_s = kt.strip().lower()
            normalized["tca_kappa_target"] = "auto" if kt_s == "auto" else float(kt_s)
        else:
            normalized["tca_kappa_target"] = float(kt)

    for int_key in (
        "num_users", "seed", "global_rounds", "local_epochs", "batch_size", "n_clusters",
        "fedwad_n_supp", "fedwad_epochs", "distance_pool_size", "shakespeare_sequence_length",
        "charlstm_embed_dim", "charlstm_hidden_size", "charlstm_num_layers",
    ):
        if int_key in normalized and normalized[int_key] is not None:
            normalized[int_key] = int(normalized[int_key])

    for opt_int_key in ("max_samples_per_client", "max_test_samples"):
        if opt_int_key in normalized and normalized[opt_int_key] is not None:
            normalized[opt_int_key] = int(normalized[opt_int_key])

    for float_key in (
        "alpha",
        "learning_rate",
        "frac",
        "fedwad_t",
        "shakespeare_train_fraction",
        "tca_var_scale",
        "tca_tau",
    ):
        if float_key in normalized and normalized[float_key] is not None:
            normalized[float_key] = float(normalized[float_key])

    for bool_key in ("smoke_test", "consistent_split_weights", "run_noise_floor", "save_excel", "tca_calibrate"):
        if bool_key in normalized and normalized[bool_key] is not None:
            val = normalized[bool_key]
            normalized[bool_key] = (
                val.strip().lower() in ("1", "true", "yes") if isinstance(val, str) else bool(val)
            )

    if normalized.get("smoke_test", False) and (
        normalized.get("global_rounds", 50) == 50 and normalized.get("num_users", 100) == 100
    ):
        base_smoke = get_smoke_test_config()
        smoke_overrides = {
            k: normalized[k]
            for k in (
                "method", "cluster_mode", "zscore_mode", "consistent_split_weights",
                "run_noise_floor", "output_dir", "save_excel", "seed",
                "tca_mode", "tca_cost", "tca_marginal", "tca_calibrate",
                "tca_kappa_target", "tca_var_scale", "tca_tau",
            )
            if k in normalized
        }
        cfg = replace(base_smoke, **smoke_overrides)
        _validate(cfg)
        return cfg

    base_defaults = ExperimentConfig(
        dataset=DATASET, model=MODEL, num_users=NUM_USERS, alpha=ALPHA, seed=SEED,
        global_rounds=GLOBAL_ROUNDS, local_epochs=LOCAL_EPOCHS, learning_rate=LEARNING_RATE,
        frac=FRAC, batch_size=BATCH_SIZE, n_clusters=N_CLUSTERS, method=METHOD,
        distance_method=DISTANCE_METHOD, fedwad_n_supp=FEDWAD_N_SUPP, fedwad_epochs=FEDWAD_EPOCHS,
        fedwad_t=FEDWAD_T, distance_pool_size=DISTANCE_POOL_SIZE, data_root=DATA_ROOT,
        shakespeare_text_path=SHAKESPEARE_TEXT_PATH, shakespeare_download_url=SHAKESPEARE_DOWNLOAD_URL,
        shakespeare_train_fraction=SHAKESPEARE_TRAIN_FRACTION,
        shakespeare_sequence_length=SHAKESPEARE_SEQUENCE_LENGTH,
        charlstm_embed_dim=CHARLSTM_EMBED_DIM, charlstm_hidden_size=CHARLSTM_HIDDEN_SIZE,
        charlstm_num_layers=CHARLSTM_NUM_LAYERS,
        consistent_split_weights=CONSISTENT_SPLIT_WEIGHTS, cluster_mode=CLUSTER_MODE,
        zscore_mode=ZSCORE_MODE,
        tca_mode=TCA_MODE, tca_cost=TCA_COST, tca_marginal=TCA_MARGINAL,
        tca_calibrate=TCA_CALIBRATE, tca_kappa_target=TCA_KAPPA_TARGET,
        tca_var_scale=TCA_VAR_SCALE, tca_tau=TCA_TAU,
    )
    cfg = replace(base_defaults, **normalized)
    _validate(cfg)
    return cfg


def save_config_to_yaml(cfg: ExperimentConfig, yaml_path: Union[str, os.PathLike]) -> None:
    """Save an `ExperimentConfig` to a YAML file."""
    path = Path(yaml_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = cfg.to_dict()
    if yaml is not None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
    else:
        lines = [f"{k}: {repr(v) if v is not None else 'null'}" for k, v in data.items()]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


SMOKE_TEST_CONFIG = get_smoke_test_config()

_env_yaml = os.getenv("CURVFED_CONFIG", "").strip()
_active_yaml_path = Path(_env_yaml) if _env_yaml else DEFAULT_YAML_PATH

if os.getenv("CURVFED_SMOKE_TEST", "0").strip().lower() in ("1", "true", "yes"):
    CONFIG = SMOKE_TEST_CONFIG
elif _active_yaml_path.exists():
    CONFIG = load_config_from_yaml(_active_yaml_path)
    _env_overrides: Dict[str, Any] = {}
    if "CONSISTENT_SPLIT_WEIGHTS" in os.environ:
        _env_overrides["consistent_split_weights"] = CONSISTENT_SPLIT_WEIGHTS
    if "CLUSTER_MODE" in os.environ:
        _env_overrides["cluster_mode"] = CLUSTER_MODE
    if "ZSCORE_MODE" in os.environ:
        _env_overrides["zscore_mode"] = ZSCORE_MODE
    if _env_overrides:
        CONFIG = replace(CONFIG, **_env_overrides)
else:
    CONFIG = ExperimentConfig(
        dataset=DATASET, model=MODEL, num_users=NUM_USERS, alpha=ALPHA, seed=SEED,
        global_rounds=GLOBAL_ROUNDS, local_epochs=LOCAL_EPOCHS, learning_rate=LEARNING_RATE,
        frac=FRAC, batch_size=BATCH_SIZE, n_clusters=N_CLUSTERS, method=METHOD,
        distance_method=DISTANCE_METHOD, fedwad_n_supp=FEDWAD_N_SUPP, fedwad_epochs=FEDWAD_EPOCHS,
        fedwad_t=FEDWAD_T, distance_pool_size=DISTANCE_POOL_SIZE, data_root=DATA_ROOT,
        shakespeare_text_path=SHAKESPEARE_TEXT_PATH, shakespeare_download_url=SHAKESPEARE_DOWNLOAD_URL,
        shakespeare_train_fraction=SHAKESPEARE_TRAIN_FRACTION,
        shakespeare_sequence_length=SHAKESPEARE_SEQUENCE_LENGTH,
        charlstm_embed_dim=CHARLSTM_EMBED_DIM, charlstm_hidden_size=CHARLSTM_HIDDEN_SIZE,
        charlstm_num_layers=CHARLSTM_NUM_LAYERS,
    )

_validate(CONFIG)

if __name__ == "__main__":
    print("========== ACTIVE EXPERIMENT CONFIG ==========")
    print(f"Loaded from YAML:       {_active_yaml_path if _active_yaml_path.exists() else 'None (Python defaults)'}")
    print(CONFIG)
    print(f"Method:                 {CONFIG.method}")
    print(f"Active clients / round: {CONFIG.num_active_clients}")
    print(f"IID-like partition:     {CONFIG.iid_like}")
    print(f"Experiment tag:         {CONFIG.experiment_tag}")
    print("==============================================")
