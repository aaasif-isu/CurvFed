"""CPU unit tests for standard Federated Learning (FedAvg and TCA),
complete model training, dataset partitioning, and YAML configuration."""

import copy
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

torch.set_num_threads(1)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.default_config import ExperimentConfig
from src.core.client import Client
from src.core.server import Server
from src.core.train_utils import FedAvg, weighted_average
from src.data.partition import dataset_iid, partition_client_indices
from src.methods.tca import tca_aggregate_models
from src.models import build_model


def test_weighted_average_equals_fedavg_for_equal_weights():
    """1. Uniform sample-weighted aggregation matches FedAvg across floating-point and integer buffers."""
    torch.manual_seed(42)
    client_ids = [0, 1, 2, 3]
    state_list = [
        {
            "conv.weight": torch.randn(8, 4, 3, 3, dtype=torch.float32),
            "fc.bias": torch.randn(10, dtype=torch.float32),
            "bn.num_batches_tracked": torch.tensor(10 + cid, dtype=torch.int64),
        }
        for cid in client_ids
    ]
    equal_weights = [100, 100, 100, 100]

    expected_fedavg = FedAvg(state_list)
    weighted = weighted_average(state_list, equal_weights)

    for key in expected_fedavg:
        if torch.is_floating_point(expected_fedavg[key]):
            assert torch.allclose(weighted[key], expected_fedavg[key], atol=1e-6, rtol=1e-6)
        else:
            assert torch.equal(weighted[key], expected_fedavg[key])


def test_tca_aggregate_models_and_feedback_vs_decoupled_init():
    """2. tca_aggregate_models and Server.get_client_init_state distinguish feedback vs decoupled mode."""
    torch.manual_seed(42)
    cfg = ExperimentConfig(
        dataset="mnist",
        model="smallcnn",
        num_users=2,
        alpha=0.5,
        seed=42,
        global_rounds=1,
        local_epochs=1,
        learning_rate=0.01,
        frac=1.0,
        batch_size=8,
        smoke_test=True,
    )
    device = torch.device("cpu")
    base_model = build_model(cfg).to(device)
    server = Server(global_model=base_model, device=device, num_users=2)

    w0 = {k: torch.zeros_like(v) for k, v in base_model.state_dict().items()}
    w1 = {
        k: (torch.ones_like(v) if torch.is_floating_point(v) else v.clone())
        for k, v in base_model.state_dict().items()
    }
    A = np.array([[0.8, 0.2], [0.2, 0.8]], dtype=np.float64)
    rho = np.array([0.5, 0.5], dtype=np.float64)

    w_glob, pers_by_id = tca_aggregate_models([w0, w1], A, rho, active_users=[0, 1])
    server.update_from_tca(w_glob, pers_by_id)

    dec_init_0 = server.get_client_init_state(0, method="tca", tca_mode="decoupled")
    fb_init_0 = server.get_client_init_state(0, method="tca", tca_mode="feedback")

    for k in w_glob:
        if torch.is_floating_point(w_glob[k]):
            assert torch.allclose(dec_init_0[k], torch.full_like(w_glob[k], 0.5), atol=1e-6)
            assert torch.allclose(fb_init_0[k], torch.full_like(w_glob[k], 0.2), atol=1e-6)


def test_baseline_global_model_changes_after_one_round_when_frac_less_than_one():
    """3. The baseline global model changes after one round when FRAC < 1."""
    torch.manual_seed(42)
    np.random.seed(42)

    cfg = ExperimentConfig(
        dataset="mnist",
        model="smallcnn",
        num_users=4,
        alpha=0.5,
        seed=42,
        global_rounds=1,
        local_epochs=1,
        learning_rate=0.01,
        frac=0.5,  # FRAC < 1: only 2 of 4 clients active
        batch_size=8,
        smoke_test=True,
    )

    assert cfg.num_active_clients == 2 < cfg.num_users

    device = torch.device("cpu")
    images = torch.randn(64, 1, 28, 28)
    labels = torch.randint(0, 10, (64,))
    synthetic_dataset = TensorDataset(images, labels)
    dict_users = dataset_iid(len(synthetic_dataset), num_users=cfg.num_users, seed=cfg.seed)

    base_model = build_model(cfg).to(device)
    server = Server(
        global_model=base_model,
        device=device,
        num_users=cfg.num_users,
    )

    initial_global_state = copy.deepcopy(server.global_state)

    active_users = [0, 1]  # 2 active clients out of 4 (FRAC = 0.5 < 1.0)
    w_locals = []
    sample_counts = []

    for idx in active_users:
        local = Client(
            idx=idx,
            lr=cfg.learning_rate,
            device=device,
            batch_size=cfg.batch_size,
            local_ep=cfg.local_epochs,
            dataset_train=synthetic_dataset,
            dataset_test=synthetic_dataset,
            idxs=dict_users[idx],
            idxs_test=dict_users[idx],
        )
        net = copy.deepcopy(server.global_model).to(device)
        net.load_state_dict(server.get_client_init_state(idx, method="baseline"))
        w_client = local.train(net)
        w_locals.append(w_client)
        sample_counts.append(len(dict_users[idx]))

    updated_global_state = server.aggregate_fedavg(w_locals, sample_counts)

    changed_keys = [
        k
        for k in initial_global_state
        if not torch.allclose(initial_global_state[k], updated_global_state[k])
    ]
    assert len(changed_keys) > 0, "Baseline global model did not change when FRAC < 1"



def test_iid_partition_is_disjoint_and_covers_dataset():
    """4. The IID partition is disjoint and covers the dataset."""
    num_samples = 1000
    num_users = 10
    labels = np.arange(num_samples) % 10

    cfg = ExperimentConfig(
        dataset="mnist",
        model="smallcnn",
        num_users=num_users,
        alpha=999999,
        seed=42,
        global_rounds=1,
        local_epochs=1,
        learning_rate=0.001,
        frac=1.0,
        batch_size=32,
        smoke_test=True,
    )
    assert cfg.iid_like

    dummy_dataset = list(range(num_samples))
    dict_users = partition_client_indices(cfg, dummy_dataset, labels)

    assert len(dict_users) == num_users
    for i in range(num_users):
        for j in range(i + 1, num_users):
            overlap = set(dict_users[i]).intersection(set(dict_users[j]))
            assert len(overlap) == 0, f"Clients {i} and {j} share {len(overlap)} indices"

    all_assigned = [idx for idxs in dict_users.values() for idx in idxs]
    assert len(all_assigned) == num_samples
    assert set(all_assigned) == set(range(num_samples))


def test_load_and_save_yaml_config(tmp_path):
    """5. YAML configuration file loads all parameters accurately and validates keys."""
    from configs.default_config import (
        DEFAULT_YAML_PATH,
        load_config_from_yaml,
        save_config_to_yaml,
    )

    # Default repo config.yaml must load cleanly
    assert DEFAULT_YAML_PATH.exists()
    default_cfg = load_config_from_yaml(DEFAULT_YAML_PATH)
    assert default_cfg.method in ("baseline", "fedavg", "tca")

    # Custom YAML with specific parameters
    custom_yaml = tmp_path / "custom_exp.yaml"
    custom_yaml.write_text(
        """
method: "tca"
dataset: "mnist"
model: "smallcnn"
num_users: 10
frac: 0.5
alpha: 0.1
seed: 123
global_rounds: 3
local_epochs: 2
learning_rate: 0.005
batch_size: 32
save_excel: false
output_dir: "results/test_yaml_run"
tca:
  mode: "feedback"
  cost: "update"
  marginal: "unbalanced"
  tau: 2.0
  var_scale: 0.5
  calibrate: true
""",
        encoding="utf-8",
    )
    cfg = load_config_from_yaml(custom_yaml)
    assert cfg.method == "tca"
    assert cfg.dataset == "mnist"
    assert cfg.model == "smallcnn"
    assert cfg.num_users == 10
    assert cfg.frac == 0.5
    assert cfg.num_active_clients == 5
    assert cfg.alpha == 0.1
    assert cfg.seed == 123
    assert cfg.global_rounds == 3
    assert cfg.local_epochs == 2
    assert cfg.learning_rate == 0.005
    assert cfg.batch_size == 32
    assert cfg.save_excel is False
    assert cfg.output_dir == "results/test_yaml_run"
    assert cfg.tca_mode == "feedback"
    assert cfg.tca_cost == "update"
    assert cfg.tca_marginal == "unbalanced"
    assert cfg.tca_tau == 2.0
    assert cfg.tca_var_scale == 0.5
    assert cfg.tca_calibrate is True


    # Round-trip save and reload
    saved_yaml = tmp_path / "saved.yaml"
    save_config_to_yaml(cfg, saved_yaml)
    reloaded = load_config_from_yaml(saved_yaml)
    assert reloaded == cfg

    # Unknown key raises helpful ValueError
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("non_existent_param: 42\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown parameter"):
        load_config_from_yaml(bad_yaml)




