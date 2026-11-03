import argparse
import copy
import os
import random
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.default_config import CONFIG, ExperimentConfig, get_smoke_test_config
from src.core.client import Client
from src.core.server import Server
from src.core.train_utils import FedAvg, clusterwise_fedavg
from src.data.dataset import load_configured_datasets
from src.data.partition import DatasetSplit, partition_client_indices
from src.methods.curvfed_agg import (
    build_distrib_for_round,
    cluster_clients_from_emd,
    cluster_clients_from_ricci,
    compute_emd_for_round,
    compute_emd_noise_floor,
    compute_ricci_for_round,
    make_round_dirs,
)
from src.models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def run_experiment(
    config: ExperimentConfig = CONFIG,
    method: str = "baseline",
    run_noise_floor: bool = False,
    save_excel: bool = True,
):
    method = method.lower()
    if method not in ("baseline", "emd_only", "emd_orc", "emd_orc_weighted"):
        raise ValueError(f"Unsupported method: {method}")

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Running {method.upper()} ({config.model} on {config.dataset})")

    dataset_train, dataset_test, train_labels, test_labels = load_configured_datasets(config)
    dict_users = partition_client_indices(config, dataset_train, train_labels)

    for cid in range(config.num_users):
        cls_counts = np.bincount(train_labels[list(dict_users[cid])], minlength=config.num_classes)
        print(f"Client {cid:2d} | samples={len(dict_users[cid]):4d} | class_counts={cls_counts.tolist()}")

    if config.max_test_samples is not None and len(dataset_test) > config.max_test_samples:
        test_rng = np.random.default_rng(config.seed + 777)
        test_indices = sorted(
            test_rng.choice(len(dataset_test), size=config.max_test_samples, replace=False).tolist()
        )
        eval_test_dataset = DatasetSplit(dataset_test, test_indices)
    else:
        test_indices = list(range(len(dataset_test)))
        eval_test_dataset = dataset_test

    dict_users_test = {i: test_indices for i in range(config.num_users)}

    client_model, server_model = build_model(config)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        client_model = torch.nn.DataParallel(client_model)
        server_model = torch.nn.DataParallel(server_model)
    client_model = client_model.to(device)
    server_model = server_model.to(device)

    server = Server(
        models={i: server_model for i in range(config.num_users)},
        device=device,
        lr=config.learning_rate,
        num_users=config.num_users,
        global_model=copy.deepcopy(server_model).to(device),
    )

    w_glob_client = copy.deepcopy(client_model.state_dict())
    w_glob_server = copy.deepcopy(server.global_model.state_dict())

    metrics = {
        "global_acc_test": [],
        "global_loss_test": [],
        "round_time_sec": [],
        "noise_floor": [],
    }
    exp_work_dir = os.path.join("results", f"work_{method}_{config.experiment_tag}")
    
    for round_idx in range(config.global_rounds):
        round_start_time = time.perf_counter()

        rng = np.random.default_rng(config.seed + round_idx)
        active_users = rng.choice(range(config.num_users), config.num_active_clients, replace=False)
        active_users = np.array(sorted(int(x) for x in active_users), dtype=np.int64)

        use_smashed = (method != "baseline") or run_noise_floor
        if use_smashed:
            smashed_dir, out_dir, distrib_dir = make_round_dirs(round_idx, experiment_root=exp_work_dir)
        else:
            smashed_dir, out_dir, distrib_dir = None, None, None

        w_locals_client = []
        w_locals_server = []
        server_weights_by_id = {}

        for idx in active_users:
            idx_int = int(idx)
            local = Client(
                idx=idx_int,
                lr=config.learning_rate,
                device=device,
                batch_size=config.batch_size,
                local_ep=config.local_epochs,
                dataset_train=dataset_train,
                dataset_test=dataset_test,
                idxs=dict_users[idx_int],
                idxs_test=dict_users_test[idx_int],
            )

            net = copy.deepcopy(client_model).to(device)
            net.load_state_dict(w_glob_client)
            w_client = local.train(
                net,
                server,
                iter_epoch=round_idx,
                smashed_dir=smashed_dir,
                model_name=config.model,
            )
            w_locals_client.append(copy.deepcopy(w_client))

            w_server = copy.deepcopy(server.models[idx_int].state_dict())
            w_locals_server.append(w_server)
            server_weights_by_id[idx_int] = w_server

        if use_smashed:
            matrix_client_ids = build_distrib_for_round(smashed_dir, distrib_dir, seed=config.seed)
            emd = compute_emd_for_round(distrib_dir, out_dir)
            if run_noise_floor or config.smoke_test:
                nf_stats = compute_emd_noise_floor(distrib_dir, emd_matrix=emd, seed=config.seed + round_idx)
                metrics["noise_floor"].append(nf_stats)

        if method == "baseline":
            w_glob_server = FedAvg(w_locals_server)
        elif method == "emd_only":
            _, cluster_groups = cluster_clients_from_emd(
                emd, out_dir=out_dir, n_clusters=config.n_clusters, client_ids=matrix_client_ids
            )
            print(f"Round {round_idx + 1} EMD cluster groups: {cluster_groups}")
            w_glob_server, _ = clusterwise_fedavg(server_weights_by_id, cluster_groups)
        elif method == "emd_orc":
            ricci_graph = compute_ricci_for_round(emd, out_dir=out_dir)
            _, cluster_groups = cluster_clients_from_ricci(
                ricci_graph, out_dir=out_dir, n_clusters=config.n_clusters, client_ids=matrix_client_ids
            )
            print(f"Round {round_idx + 1} EMD+ORC cluster groups: {cluster_groups}")
            w_glob_server, _ = clusterwise_fedavg(server_weights_by_id, cluster_groups)
        elif method == "emd_orc_weighted":
            ricci_graph = compute_ricci_for_round(emd, out_dir=out_dir)
            _, cluster_groups = cluster_clients_from_ricci(
                ricci_graph, out_dir=out_dir, n_clusters=config.n_clusters, client_ids=matrix_client_ids
            )
            print(f"Round {round_idx + 1} Weighted EMD+ORC cluster groups: {cluster_groups}")
            w_glob_server, _ = clusterwise_fedavg(
                server_weights_by_id, cluster_groups, dict_users=dict_users
            )

        w_glob_client = FedAvg(w_locals_client)

        client_model.load_state_dict(w_glob_client)
        server.global_model.load_state_dict(w_glob_server)
        
        for i in range(config.num_users):
            server.models[i].load_state_dict(w_glob_server)

        server.global_model.eval()
        client_model.eval()
        correct, total, total_loss = 0, 0, 0.0
        criterion = torch.nn.CrossEntropyLoss()
        test_loader = torch.utils.data.DataLoader(
            eval_test_dataset, batch_size=config.batch_size, shuffle=False
        )

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                fx_client = client_model(images)
                fx_server = server.global_model(fx_client)
                loss = criterion(fx_server, labels)

                total_loss += loss.item() * labels.size(0)
                preds = fx_server.max(1, keepdim=True)[1]
                correct += preds.eq(labels.view_as(preds)).sum().item()
                total += labels.size(0)

        global_acc = 100.0 * correct / total
        global_loss = total_loss / total
        round_time = time.perf_counter() - round_start_time

        print(
            f"Round {round_idx + 1}/{config.global_rounds} | "
            f"Method: {method.upper()} | Global Acc: {global_acc:.2f}% | "
            f"Loss: {global_loss:.4f} | Time: {round_time:.2f}s"
        )

        metrics["global_acc_test"].append(global_acc)
        metrics["global_loss_test"].append(global_loss)
        metrics["round_time_sec"].append(round_time)

    if save_excel:
        out_folder = os.path.join("results", method.upper())
        os.makedirs(out_folder, exist_ok=True)
        df = pd.DataFrame(
            {
                "round": list(range(1, config.global_rounds + 1)),
                "global_acc_test": metrics["global_acc_test"],
                "global_loss_test": metrics["global_loss_test"],
                "round_time_sec": metrics["round_time_sec"],
            }
        )
        df.to_excel(os.path.join(out_folder, f"metrics_{config.experiment_tag}.xlsx"), index=False)

    print("Training finished.")
    return metrics, client_model, server.global_model


def main():
    parser = argparse.ArgumentParser(description="Run CurvFed SplitFed experiment.")
    parser.add_argument(
        "--method",
        type=str,
        default="baseline",
        choices=["baseline", "emd_only", "emd_orc", "emd_orc_weighted"],
        help="Aggregation method to run.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use the CPU smoke-test configuration (MNIST, SmallCNN, 4 clients, 2 rounds).",
    )
    parser.add_argument(
        "--noise-floor",
        action="store_true",
        help="Run Sanity Check A (within-client vs cross-client EMD noise floor check).",
    )
    args = parser.parse_args()

    cfg = get_smoke_test_config() if args.smoke_test else CONFIG
    run_experiment(config=cfg, method=args.method, run_noise_floor=args.noise_floor or cfg.smoke_test)


if __name__ == "__main__":
    main()
