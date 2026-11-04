"""Central configuration for Federated Learning experiments.

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

# TCA controls
TCA_MODE = os.getenv("TCA_MODE", "decoupled").strip().lower()          # "decoupled" or "feedback"
TCA_COST = os.getenv("TCA_COST", "update").strip().lower()             # "update"
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
    method: str = METHOD
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
    output_dir: Optional[str] = None
    save_excel: bool = True
    tca_mode: str = TCA_MODE
    tca_cost: str = TCA_COST
    tca_marginal: str = TCA_MARGINAL
    tca_calibrate: bool = TCA_CALIBRATE
    tca_kappa_target: Union[str, float] = TCA_KAPPA_TARGET
    tca_var_scale: float = TCA_VAR_SCALE
    tca_tau: float = TCA_TAU
    experiment: str = "none"



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
    if not 0 < cfg.shakespeare_train_fraction < 1:
        raise ValueError("SHAKESPEARE_TRAIN_FRACTION must be between 0 and 1.")
    if cfg.max_samples_per_client is not None and cfg.max_samples_per_client <= 0:
        raise ValueError("max_samples_per_client must be positive when provided.")
    if cfg.max_test_samples is not None and cfg.max_test_samples <= 0:
        raise ValueError("max_test_samples must be positive when provided.")
    if cfg.tca_mode not in ("decoupled", "feedback"):
        raise ValueError("tca_mode must be 'decoupled' or 'feedback'.")
    if cfg.tca_cost != "update":
        raise ValueError("tca_cost must be 'update'.")
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
        method="tca",
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




def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge `override` into a copy of `base` (merging nested dicts like `tca`)."""
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge_dicts(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config_from_yaml(yaml_path: Union[str, os.PathLike] = DEFAULT_YAML_PATH) -> ExperimentConfig:
    """Load an `ExperimentConfig` from a YAML file with preset resolution (`experiment` + `experiments`)."""
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"YAML configuration file not found: {path}")

    text = path.read_text(encoding="utf-8")
    raw_data = (yaml.safe_load(text) or {}) if yaml is not None else _fallback_parse_simple_yaml(text)

    if not isinstance(raw_data, dict):
        raise ValueError(f"YAML config at {path} must be a top-level key-value mapping.")

    raw_data = dict(raw_data)
    experiments_map = raw_data.pop("experiments", None) or {}
    if not isinstance(experiments_map, dict):
        raise ValueError(f"'experiments' in {path} must be a mapping of preset names to dicts.")

    raw_exp = raw_data.pop("experiment", None)
    selected_exp = "" if raw_exp is None else str(raw_exp).strip()

    if selected_exp and selected_exp.lower() not in ("none", "null"):
        if selected_exp not in experiments_map:
            valid_names = sorted(experiments_map.keys())
            raise ValueError(
                f"Unknown experiment preset '{selected_exp}' in {path}. "
                f"Valid preset names are: {valid_names} (or 'none' / '')."
            )
        preset_dict = experiments_map[selected_exp] or {}
        if not isinstance(preset_dict, dict):
            raise ValueError(f"Preset '{selected_exp}' in {path} must be a mapping.")
        merged_raw = _deep_merge_dicts(raw_data, preset_dict)
        resolved_exp_name = selected_exp
    else:
        merged_raw = raw_data
        resolved_exp_name = "none"

    valid_field_names = {f.name for f in fields(ExperimentConfig)}
    normalized: Dict[str, Any] = {"experiment": resolved_exp_name}
    for k, v in merged_raw.items():
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
        "method", "dataset", "model",
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
        "num_users", "seed", "global_rounds", "local_epochs", "batch_size",
        "shakespeare_sequence_length", "charlstm_embed_dim", "charlstm_hidden_size", "charlstm_num_layers",
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
        "shakespeare_train_fraction",
        "tca_var_scale",
        "tca_tau",
    ):
        if float_key in normalized and normalized[float_key] is not None:
            normalized[float_key] = float(normalized[float_key])

    for bool_key in ("smoke_test", "save_excel", "tca_calibrate"):
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
                "experiment", "method",
                "output_dir", "save_excel", "seed",
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
        frac=FRAC, batch_size=BATCH_SIZE, method=METHOD, data_root=DATA_ROOT,
        shakespeare_text_path=SHAKESPEARE_TEXT_PATH, shakespeare_download_url=SHAKESPEARE_DOWNLOAD_URL,
        shakespeare_train_fraction=SHAKESPEARE_TRAIN_FRACTION,
        shakespeare_sequence_length=SHAKESPEARE_SEQUENCE_LENGTH,
        charlstm_embed_dim=CHARLSTM_EMBED_DIM, charlstm_hidden_size=CHARLSTM_HIDDEN_SIZE,
        charlstm_num_layers=CHARLSTM_NUM_LAYERS,
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
else:
    CONFIG = ExperimentConfig(
        dataset=DATASET, model=MODEL, num_users=NUM_USERS, alpha=ALPHA, seed=SEED,
        global_rounds=GLOBAL_ROUNDS, local_epochs=LOCAL_EPOCHS, learning_rate=LEARNING_RATE,
        frac=FRAC, batch_size=BATCH_SIZE, method=METHOD, data_root=DATA_ROOT,
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
