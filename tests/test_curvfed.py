"""CPU unit tests for CurvFed aggregation, ORC graph construction,
Ricci shortest-path distance metric, partial-participation baseline update,
and IID dataset partitioning."""

import copy
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import networkx as nx
import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

torch.set_num_threads(1)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.default_config import ExperimentConfig
from src.core.client import Client
from src.core.server import Server
from src.core.train_utils import FedAvg, clusterwise_fedavg, weighted_average
from src.data.partition import dataset_iid, partition_client_indices
from src.methods.curvfed_agg import (
    cluster_clients_from_ricci,
    compute_ricci_distance_matrix,
    compute_ricci_for_round,
)
from src.models import build_model


def test_sample_weighted_clusterwise_aggregation_equals_plain_weighted_fedavg():
    """1. Sample-weighted cluster-wise aggregation equals plain sample-weighted FedAvg
    (documents the algebraic cancellation of cluster sample totals N_c)."""
    torch.manual_seed(42)
    client_ids = [0, 1, 2, 3, 4]
    server_weights_by_id = {
        cid: {
            "conv.weight": torch.randn(8, 4, 3, 3, dtype=torch.float32),
            "fc.bias": torch.randn(10, dtype=torch.float32),
            "bn.num_batches_tracked": torch.tensor(10 + cid, dtype=torch.int64),
        }
        for cid in client_ids
    }
    dict_users = {
        0: list(range(50)),
        1: list(range(120)),
        2: list(range(30)),
        3: list(range(300)),
        4: list(range(80)),
    }

    state_list = [server_weights_by_id[cid] for cid in client_ids]
    sample_counts = [len(dict_users[cid]) for cid in client_ids]
    expected_fedavg = weighted_average(state_list, sample_counts)

    for cluster_groups in [
        {0: [0, 1], 1: [2, 3, 4]},
        {0: [0], 1: [1, 2], 2: [3, 4]},
        {0: [0, 2, 4], 1: [1], 2: [3]},
    ]:
        cluster_global, _ = clusterwise_fedavg(
            server_weights_by_id=server_weights_by_id,
            cluster_groups=cluster_groups,
            dict_users=dict_users,
        )
        for key in expected_fedavg:
            if torch.is_floating_point(expected_fedavg[key]):
                assert torch.allclose(
                    cluster_global[key], expected_fedavg[key], atol=1e-6, rtol=1e-6
                ), f"Mismatch on {key} for cluster_groups={cluster_groups}"


def test_orc_graph_uses_emd_over_tau_as_weight_attribute(tmp_path):
    """2. The ORC graph uses emd/tau as the 'weight' edge attribute."""
    emd = np.array(
        [
            [0.0, 2.0, 4.0, 8.0, 10.0],
            [2.0, 0.0, 3.0, 7.0, 9.0],
            [4.0, 3.0, 0.0, 6.0, 8.0],
            [8.0, 7.0, 6.0, 0.0, 2.5],
            [10.0, 9.0, 8.0, 2.5, 0.0],
        ],
        dtype=np.float64,
    )
    positives = emd[np.triu_indices(emd.shape[0], 1)]
    tau = float(np.median(positives[positives > 0]))

    Gk = compute_ricci_for_round(emd, out_dir=str(tmp_path))
    assert Gk.number_of_edges() > 0

    for u, v, data in Gk.edges(data=True):
        assert "weight" in data
        assert "ricciCurvature" in data
        expected_weight = float(emd[u, v] / tau)
        assert data["weight"] == pytest.approx(expected_weight, rel=1e-6, abs=1e-8)


def test_ricci_distance_matrix_symmetric_zero_diag_and_triangle_inequality(tmp_path):
    """3. The Ricci distance matrix is symmetric, zero on the diagonal, and
    satisfies the triangle inequality after shortest-path completion."""
    emd = np.array(
        [
            [0.0, 1.2, 1.5, 6.0, 6.5, 7.0],
            [1.2, 0.0, 1.1, 5.8, 6.2, 6.9],
            [1.5, 1.1, 0.0, 5.5, 6.0, 6.4],
            [6.0, 5.8, 5.5, 0.0, 1.3, 1.6],
            [6.5, 6.2, 6.0, 1.3, 0.0, 1.0],
            [7.0, 6.9, 6.4, 1.6, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    Gk = compute_ricci_for_round(emd, out_dir=str(tmp_path))
    edges = list(Gk.edges())
    u0, v0 = edges[0]
    Gk[u0][v0]["ricciCurvature"] = -0.75

    D = compute_ricci_distance_matrix(Gk)
    n = D.shape[0]

    assert D.shape == (n, n)
    assert np.all(np.isfinite(D))
    assert np.allclose(D, D.T, atol=1e-12), "Ricci distance matrix must be symmetric"
    assert np.allclose(np.diag(D), 0.0, atol=1e-12), "Diagonal must be zero"

    for i in range(n):
        for j in range(n):
            for k in range(n):
                assert D[i, k] <= D[i, j] + D[j, k] + 1e-10, (
                    f"Triangle inequality violated: D[{i},{k}]={D[i,k]} > "
                    f"D[{i},{j}]+D[{j},{k}]={D[i,j]+D[j,k]}"
                )

    labels, groups = cluster_clients_from_ricci(Gk, out_dir=str(tmp_path), n_clusters=2)
    assert len(labels) == n
    assert sum(len(v) for v in groups.values()) == n


def test_baseline_server_model_changes_after_one_round_when_frac_less_than_one():
    """4. The baseline server model changes after one round when FRAC < 1."""
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
        n_clusters=2,
        smoke_test=True,
    )
    assert cfg.num_active_clients == 2 < cfg.num_users

    device = torch.device("cpu")
    images = torch.randn(64, 1, 28, 28)
    labels = torch.randint(0, 10, (64,))
    synthetic_dataset = TensorDataset(images, labels)
    dict_users = dataset_iid(len(synthetic_dataset), num_users=cfg.num_users, seed=cfg.seed)

    client_model, server_model = build_model(cfg)
    client_model = client_model.to(device)
    server_model = server_model.to(device)

    server = Server(
        models={i: server_model for i in range(cfg.num_users)},
        device=device,
        lr=cfg.learning_rate,
        num_users=cfg.num_users,
        global_model=copy.deepcopy(server_model).to(device),
    )

    initial_server_state = copy.deepcopy(server.global_model.state_dict())
    w_glob_client = copy.deepcopy(client_model.state_dict())

    active_users = [0, 1]  # 2 active clients out of 4 (FRAC = 0.5 < 1.0)
    w_locals_client = []
    w_locals_server = []

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
        net = copy.deepcopy(client_model).to(device)
        net.load_state_dict(w_glob_client)
        w_client = local.train(net, server, iter_epoch=0)
        w_locals_client.append(copy.deepcopy(w_client))
        w_locals_server.append(copy.deepcopy(server.models[idx].state_dict()))

    w_glob_server = FedAvg(w_locals_server)
    server.global_model.load_state_dict(w_glob_server)
    updated_server_state = server.global_model.state_dict()

    changed_keys = [
        k
        for k in initial_server_state
        if not torch.allclose(initial_server_state[k], updated_server_state[k])
    ]
    assert len(changed_keys) > 0, "Baseline global server model did not change when FRAC < 1"


def test_iid_partition_is_disjoint_and_covers_dataset():
    """5. The IID partition is disjoint and covers the dataset."""
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
        n_clusters=2,
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

