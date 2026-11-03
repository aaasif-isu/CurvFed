import argparse
import copy
import json
import os
import random
import shutil
import sys
import time
from dataclasses import replace
from typing import Dict, List, Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.default_config import (
    CONFIG,
    DEFAULT_YAML_PATH,
    ExperimentConfig,
    _validate,
    get_smoke_test_config,
    load_config_from_yaml,
    save_config_to_yaml,
)
from src.core.client import Client
from src.core.server import Server
from src.core.train_utils import FedAvg, clusterwise_fedavg
from src.data.dataset import load_configured_datasets
from src.data.partition import (
    DatasetSplit,
    compute_label_histograms_and_tv,
    partition_client_indices,
    partition_personalized_test_indices,
)
from src.methods.curvfed_agg import (
    apply_cluster_mode,
    build_distrib_for_round,
    cluster_clients_from_emd,
    cluster_clients_from_ricci,
    compute_cluster_ari,
    compute_emd_for_round,
    compute_emd_noise_floor,
    compute_fedwad_for_round,
    compute_ricci_for_round,
    make_round_dirs,
)
from src.methods.tca import (
    build_cost_matrix,
    build_variance_vector,
    estimate_q,
    flatten_split_params,
    lazy_calibrate,
    plan_curvature,
    solve_tca_plan,
    tca_aggregate_split_models,
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


def _save_label_histograms_and_tv(
    out_folder: str,
    dict_users: Dict[int, List[int]],
    train_labels: np.ndarray,
    num_users: int,
    num_classes: int,
) -> np.ndarray:
    os.makedirs(out_folder, exist_ok=True)
    counts, histograms, tv_matrix = compute_label_histograms_and_tv(
        dict_users, train_labels, num_users, num_classes
    )

    hist_rows = []
    for cid in range(num_users):
        row = {"client_id": cid, "total_samples": int(counts[cid].sum())}
        for c in range(num_classes):
            row[f"count_class_{c}"] = int(counts[cid, c])
        for c in range(num_classes):
            row[f"prob_class_{c}"] = float(histograms[cid, c])
        hist_rows.append(row)

    pd.DataFrame(hist_rows).to_csv(
        os.path.join(out_folder, "client_label_histograms.csv"), index=False
    )

    tv_df = pd.DataFrame(
        tv_matrix,
        index=[f"client_{i}" for i in range(num_users)],
        columns=[f"client_{j}" for j in range(num_users)],
    )
    tv_df.to_csv(os.path.join(out_folder, "pairwise_label_tv_matrix.csv"))
    np.save(os.path.join(out_folder, "pairwise_label_tv_matrix.npy"), tv_matrix)
    return tv_matrix


def _evaluate_split_pair(
    client_net: torch.nn.Module,
    server_net: torch.nn.Module,
    dataset,
    batch_size: int,
    device: torch.device,
):
    client_net.eval()
    server_net.eval()
    correct, total, total_loss = 0, 0, 0.0
    criterion = torch.nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            fx_client = client_net(images)
            fx_server = server_net(fx_client)
            loss = criterion(fx_server, labels)
            total_loss += loss.item() * labels.size(0)
            preds = fx_server.max(1, keepdim=True)[1]
            correct += preds.eq(labels.view_as(preds)).sum().item()
            total += labels.size(0)
    if total == 0:
        return 0.0, 0.0
    return 100.0 * correct / total, total_loss / total


def _canonicalize_method(method: str) -> str:
    m = method.strip().lower()
    if m == "fedavg":
        return "baseline"
    if m == "curvfed":
        return "emd_orc"
    return m


def run_experiment(
    config: ExperimentConfig = CONFIG,
    method: Optional[str] = None,
    run_noise_floor: Optional[bool] = None,
    save_excel: Optional[bool] = None,
    output_dir: Optional[str] = None,
):
    raw_method = (method or config.method or "baseline").strip().lower()
    canonical_method = _canonicalize_method(raw_method)
    if canonical_method not in ("baseline", "emd_only", "emd_orc", "emd_orc_weighted", "fedwad", "tca"):
        raise ValueError(f"Unsupported method: {raw_method}")
    method = canonical_method

    if run_noise_floor is None:
        run_noise_floor = config.run_noise_floor
    if save_excel is None:
        save_excel = config.save_excel
    resolved_output_dir = output_dir or config.output_dir

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if method == "tca":
        default_sub = (
            f"tca_{config.tca_mode}_{config.tca_cost}_{config.tca_marginal}_"
            f"cal{int(config.tca_calibrate)}_{config.experiment_tag}"
        )
    else:
        default_sub = f"{method}_{config.cluster_mode}_{config.zscore_mode}_{config.experiment_tag}"
    out_folder = resolved_output_dir or os.path.join("results", default_sub)
    rounds_csv_dir = os.path.join(out_folder, "rounds")
    os.makedirs(rounds_csv_dir, exist_ok=True)

    resolved_cfg = replace(
        config,
        method=method,
        run_noise_floor=bool(run_noise_floor),
        save_excel=bool(save_excel),
        output_dir=out_folder,
    )
    save_config_to_yaml(resolved_cfg, os.path.join(out_folder, "resolved_config.yaml"))

    print(
        f"Using device: {device} | Running {method.upper()} "
        f"(model={config.model}, dataset={config.dataset}, "
        f"clients={config.num_users}, frac={config.frac}, alpha={config.alpha}, "
        f"rounds={config.global_rounds}, cluster_mode={config.cluster_mode}, "
        f"zscore_mode={config.zscore_mode}, consistent_split={config.consistent_split_weights}, "
        f"tca_mode={config.tca_mode}, tca_cost={config.tca_cost}, "
        f"tca_marginal={config.tca_marginal}, tca_calibrate={config.tca_calibrate})"
    )

    dataset_train, dataset_test, train_labels, test_labels = load_configured_datasets(config)
    dict_users = partition_client_indices(config, dataset_train, train_labels)
    _save_label_histograms_and_tv(
        out_folder, dict_users, train_labels, config.num_users, config.num_classes
    )

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
    pers_test_idcs = partition_personalized_test_indices(
        dict_users,
        train_labels,
        test_labels[test_indices],
        config.num_users,
        config.num_classes,
        seed=config.seed,
    )
    pers_test_datasets = {
        i: DatasetSplit(eval_test_dataset, pers_test_idcs[i]) for i in range(config.num_users)
    }

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
    pers_client_states: Dict[int, dict] = {
        i: copy.deepcopy(w_glob_client) for i in range(config.num_users)
    }
    pers_server_states: Dict[int, dict] = {
        i: copy.deepcopy(w_glob_server) for i in range(config.num_users)
    }

    metrics = {
        "global_acc_test": [],
        "global_loss_test": [],
        "personalized_mean_acc": [],
        "personalized_min_acc": [],
        "ari_prev_round": [],
        "round_time_sec": [],
        "noise_floor": [],
        "round_rows": [],
    }
    exp_work_dir = os.path.join(out_folder, "work")
    prev_cluster_groups: Optional[Dict[int, List[int]]] = None
    
    for round_idx in range(config.global_rounds):
        round_start_time = time.perf_counter()

        rng = np.random.default_rng(config.seed + round_idx)
        active_users = rng.choice(range(config.num_users), config.num_active_clients, replace=False)
        active_users = np.array(sorted(int(x) for x in active_users), dtype=np.int64)

        need_emd = (
            method in ("emd_only", "emd_orc", "emd_orc_weighted", "fedwad")
            or (method == "tca" and config.tca_cost == "emd")
        )
        use_smashed = need_emd or run_noise_floor or config.smoke_test
        if use_smashed:
            smashed_dir, out_dir, distrib_dir = make_round_dirs(round_idx, experiment_root=exp_work_dir)
        else:
            smashed_dir, out_dir, distrib_dir = None, None, None

        w_locals_client = []
        w_locals_server = []
        client_weights_by_id = {}
        server_weights_by_id = {}
        active_sample_counts: List[int] = []
        active_deltas: List[np.ndarray] = []
        norm_v1_list: List[float] = []
        norm_v2_list: List[float] = []

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
            if method == "tca" and config.tca_mode == "feedback":
                net.load_state_dict(pers_client_states[idx_int])
                server.models[idx_int].load_state_dict(pers_server_states[idx_int])
            else:
                net.load_state_dict(w_glob_client)
                server.models[idx_int].load_state_dict(w_glob_server)

            start_vec = flatten_split_params(net, server.models[idx_int])
            w_client = local.train(
                net,
                server,
                iter_epoch=round_idx,
                smashed_dir=smashed_dir,
                model_name=config.model,
            )
            mid_vec = local.mid_param_vec if local.mid_param_vec is not None else start_vec
            end_vec = flatten_split_params(net, server.models[idx_int])

            v1 = mid_vec - start_vec
            v2 = end_vec - mid_vec
            norm_v1_list.append(float(np.linalg.norm(v1)))
            norm_v2_list.append(float(np.linalg.norm(v2)))
            active_deltas.append(end_vec - start_vec)
            active_sample_counts.append(len(dict_users[idx_int]))

            w_client_copy = copy.deepcopy(w_client)
            w_locals_client.append(w_client_copy)
            client_weights_by_id[idx_int] = w_client_copy

            w_server = copy.deepcopy(server.models[idx_int].state_dict())
            w_locals_server.append(w_server)
            server_weights_by_id[idx_int] = w_server

        q_arr, median_q = estimate_q(norm_v1_list, norm_v2_list)
        per_client_q = {int(cid): float(qv) for cid, qv in zip(active_users, q_arr)}

        nf_stats = None
        fedwad_dist = None
        emd = None
        if use_smashed:
            matrix_client_ids = build_distrib_for_round(
                smashed_dir, distrib_dir, seed=config.seed, cleanup_smashed=True
            )
            emd = compute_emd_for_round(distrib_dir, out_dir, zscore_mode=config.zscore_mode)
            if method == "fedwad" and config.distance_method == "fedwad":
                fedwad_dist = compute_fedwad_for_round(
                    distrib_dir,
                    out_dir=out_dir,
                    zscore_mode=config.zscore_mode,
                    seed=config.seed + round_idx,
                    n_supp=config.fedwad_n_supp,
                    n_epoch=config.fedwad_epochs,
                    t_val=config.fedwad_t,
                )
            if run_noise_floor or config.smoke_test:
                nf_stats = compute_emd_noise_floor(
                    distrib_dir,
                    emd_matrix=emd,
                    seed=config.seed + round_idx,
                    zscore_mode=config.zscore_mode,
                )
                metrics["noise_floor"].append(nf_stats)
            if distrib_dir and os.path.exists(distrib_dir):
                shutil.rmtree(distrib_dir, ignore_errors=True)

        cluster_groups: Dict[int, List[int]] = {0: [int(u) for u in active_users]}
        cluster_models_server = None
        cluster_models_client = None
        kappa_min = float("nan")
        kappa_mean = float("nan")
        sparsity = float("nan")
        inequality_violations = float("nan")
        beta_val = float("nan")

        if method == "baseline":
            w_glob_server = FedAvg(w_locals_server)
            w_glob_client = FedAvg(w_locals_client)
        elif method == "tca":
            sample_weights = np.asarray(active_sample_counts, dtype=np.float64)
            sample_weights = sample_weights / max(float(sample_weights.sum()), 1e-12)
            v_vec = build_variance_vector(active_sample_counts, var_scale=config.tca_var_scale)
            D_mat = build_cost_matrix(config.tca_cost, deltas=active_deltas, emd_matrix=emd)
            A_plan = solve_tca_plan(
                D_mat,
                v_vec,
                marginal=config.tca_marginal,
                tau=config.tca_tau,
            )
            curv_stats = plan_curvature(A_plan, D_mat)

            if config.tca_calibrate:
                A_plan, beta_val, _ = lazy_calibrate(
                    A_plan,
                    median_q=median_q,
                    kappa_target=config.tca_kappa_target,
                    D=D_mat,
                    curv_stats=curv_stats,
                )
                curv_stats = plan_curvature(A_plan, D_mat)

            kappa_min = float(curv_stats["kappa_min"])
            kappa_mean = float(curv_stats["kappa_mean"])
            sparsity = float(curv_stats["sparsity"])
            inequality_violations = int(curv_stats["inequality_violations"])

            np.save(os.path.join(rounds_csv_dir, f"A_round_{round_idx + 1:03d}.npy"), A_plan)

            (
                w_glob_client,
                w_glob_server,
                active_pers_client,
                active_pers_server,
            ) = tca_aggregate_split_models(
                w_locals_client,
                w_locals_server,
                A_plan,
                sample_weights,
                [int(u) for u in active_users],
            )
            for cid_int in [int(u) for u in active_users]:
                pers_client_states[cid_int] = copy.deepcopy(active_pers_client[cid_int])
                pers_server_states[cid_int] = copy.deepcopy(active_pers_server[cid_int])
        else:
            if method == "emd_only":
                _, raw_groups = cluster_clients_from_emd(
                    emd, out_dir=out_dir, n_clusters=config.n_clusters, client_ids=matrix_client_ids
                )
            elif method == "fedwad":
                dist_mat = fedwad_dist if fedwad_dist is not None else emd
                _, raw_groups = cluster_clients_from_emd(
                    dist_mat, out_dir=out_dir, n_clusters=config.n_clusters, client_ids=matrix_client_ids
                )
            else:
                ricci_graph = compute_ricci_for_round(emd, out_dir=out_dir)
                _, raw_groups = cluster_clients_from_ricci(
                    ricci_graph,
                    out_dir=out_dir,
                    n_clusters=config.n_clusters,
                    client_ids=matrix_client_ids,
                )

            cluster_groups = apply_cluster_mode(
                raw_groups,
                cluster_mode=config.cluster_mode,
                seed=config.seed + 10000 + round_idx,
            )
            print(
                f"Round {round_idx + 1} ({method.upper()}, mode={config.cluster_mode}) "
                f"cluster groups: {cluster_groups}"
            )
            weight_users = dict_users if method == "emd_orc_weighted" else None
            w_glob_server, cluster_models_server = clusterwise_fedavg(
                server_weights_by_id, cluster_groups, dict_users=weight_users
            )
            if config.consistent_split_weights:
                w_glob_client, cluster_models_client = clusterwise_fedavg(
                    client_weights_by_id, cluster_groups, dict_users=weight_users
                )
            else:
                w_glob_client = FedAvg(w_locals_client)

        ari_prev = compute_cluster_ari(prev_cluster_groups, cluster_groups)
        prev_cluster_groups = copy.deepcopy(cluster_groups)
        metrics["ari_prev_round"].append(ari_prev)

        client_model.load_state_dict(w_glob_client)
        server.global_model.load_state_dict(w_glob_server)
        for i in range(config.num_users):
            if method == "tca" and config.tca_mode == "feedback":
                server.models[i].load_state_dict(pers_server_states[i])
            else:
                server.models[i].load_state_dict(w_glob_server)

        global_acc, global_loss = _evaluate_split_pair(
            client_model, server.global_model, eval_test_dataset, config.batch_size, device
        )

        sorted_cluster_keys = sorted(k for k, v in cluster_groups.items() if len(v) > 0)
        client_to_cluster_pos = {}
        for pos, ck in enumerate(sorted_cluster_keys):
            for cid in cluster_groups[ck]:
                client_to_cluster_pos[int(cid)] = pos

        pers_client_accs = []
        eval_c_net = copy.deepcopy(client_model).to(device)
        eval_s_net = copy.deepcopy(server.global_model).to(device)
        eval_client_ids = (
            list(range(config.num_users))
            if method == "tca"
            else [int(u) for u in active_users]
        )
        for idx_int in eval_client_ids:
            if method == "tca":
                eval_c_net.load_state_dict(pers_client_states[idx_int])
                eval_s_net.load_state_dict(pers_server_states[idx_int])
            elif cluster_models_server is not None and idx_int in client_to_cluster_pos:
                pos = client_to_cluster_pos[idx_int]
                eval_s_net.load_state_dict(cluster_models_server[pos])
                if cluster_models_client is not None:
                    eval_c_net.load_state_dict(cluster_models_client[pos])
                else:
                    eval_c_net.load_state_dict(w_glob_client)
            else:
                eval_c_net.load_state_dict(w_glob_client)
                eval_s_net.load_state_dict(w_glob_server)

            c_acc, _ = _evaluate_split_pair(
                eval_c_net, eval_s_net, pers_test_datasets[idx_int], config.batch_size, device
            )
            pers_client_accs.append(float(c_acc))

        pers_mean_acc = float(np.mean(pers_client_accs)) if pers_client_accs else 0.0
        pers_min_acc = float(np.min(pers_client_accs)) if pers_client_accs else 0.0
        round_time = time.perf_counter() - round_start_time

        cluster_sizes = {int(k): len(v) for k, v in sorted(cluster_groups.items())}
        round_row = {
            "method": method,
            "config": config.experiment_tag,
            "zscore_mode": config.zscore_mode,
            "cluster_mode": config.cluster_mode,
            "consistent_split_weights": config.consistent_split_weights,
            "tca_mode": config.tca_mode if method == "tca" else "N/A",
            "tca_cost": config.tca_cost if method == "tca" else "N/A",
            "tca_marginal": config.tca_marginal if method == "tca" else "N/A",
            "tca_calibrate": bool(config.tca_calibrate) if method == "tca" else False,
            "round": round_idx + 1,
            "global_acc_test": global_acc,
            "global_loss_test": global_loss,
            "cluster_assignments": json.dumps(cluster_groups),
            "cluster_sizes": json.dumps(cluster_sizes),
            "ari_prev_round": ari_prev,
            "within_emd_mean": nf_stats["within_mean"] if nf_stats else float("nan"),
            "within_emd_std": nf_stats["within_std"] if nf_stats else float("nan"),
            "cross_emd_mean": nf_stats["cross_mean"] if nf_stats else float("nan"),
            "cross_emd_std": nf_stats["cross_std"] if nf_stats else float("nan"),
            "emd_snr": nf_stats["snr"] if nf_stats else float("nan"),
            "personalized_mean_acc": pers_mean_acc,
            "personalized_min_acc": pers_min_acc,
            "kappa_min": kappa_min,
            "kappa_mean": kappa_mean,
            "sparsity": sparsity,
            "inequality_violations": inequality_violations,
            "per_client_q": json.dumps(per_client_q),
            "median_q": median_q,
            "beta": beta_val,
            "round_time_sec": round_time,
        }

        pd.DataFrame([round_row]).to_csv(
            os.path.join(rounds_csv_dir, f"round_{round_idx + 1:03d}.csv"), index=False
        )
        metrics["round_rows"].append(round_row)
        metrics["global_acc_test"].append(global_acc)
        metrics["global_loss_test"].append(global_loss)
        metrics["personalized_mean_acc"].append(pers_mean_acc)
        metrics["personalized_min_acc"].append(pers_min_acc)
        metrics["round_time_sec"].append(round_time)

        tca_info = (
            f" | kappa(min/mean): {kappa_min:.3f}/{kappa_mean:.3f} | "
            f"sparsity: {sparsity:.3f} | q_med: {median_q:.3f} | beta: {beta_val:.3f}"
            if method == "tca"
            else ""
        )
        print(
            f"Round {round_idx + 1}/{config.global_rounds} | "
            f"Method: {method.upper()} | Global Acc: {global_acc:.2f}% | "
            f"Pers Mean/Min: {pers_mean_acc:.2f}%/{pers_min_acc:.2f}% | "
            f"ARI: {ari_prev:.3f}{tca_info} | Time: {round_time:.2f}s"
        )

    if os.path.exists(exp_work_dir):
        shutil.rmtree(exp_work_dir, ignore_errors=True)

    full_df = pd.DataFrame(metrics["round_rows"])
    full_df.to_csv(os.path.join(out_folder, "round_metrics.csv"), index=False)
    if save_excel:
        full_df.to_excel(
            os.path.join(out_folder, f"metrics_{config.experiment_tag}.xlsx"), index=False
        )

    print(f"Training finished. Results written to: {out_folder}")
    return metrics, client_model, server.global_model


def main():
    parser = argparse.ArgumentParser(
        description="Run CurvFed SplitFed experiment (reads config.yaml by default)."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help=f"Path to YAML configuration file (defaults to {DEFAULT_YAML_PATH}).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=[
            "baseline",
            "fedavg",
            "emd_only",
            "emd_orc",
            "curvfed",
            "emd_orc_weighted",
            "fedwad",
            "tca",
        ],
        help="Aggregation method to run (overrides 'method' in YAML).",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Use CPU smoke-test config.")
    parser.add_argument(
        "--noise-floor",
        dest="noise_floor",
        action="store_true",
        default=None,
        help="Enable Sanity Check A (within/cross EMD SNR).",
    )
    parser.add_argument(
        "--no-noise-floor",
        dest="noise_floor",
        action="store_false",
        help="Disable Sanity Check A.",
    )
    parser.add_argument("--cluster-mode", type=str, default=None, choices=["geometry", "random"])
    parser.add_argument("--zscore-mode", type=str, default=None, choices=["per_client", "pooled", "none"])
    parser.add_argument("--consistent-split-weights", type=int, default=None, choices=[0, 1])
    parser.add_argument("--tca-mode", type=str, default=None, choices=["decoupled", "feedback"])
    parser.add_argument("--tca-cost", type=str, default=None, choices=["update", "emd"])
    parser.add_argument("--tca-marginal", type=str, default=None, choices=["hard", "unbalanced", "none"])
    parser.add_argument("--tca-calibrate", type=int, default=None, choices=[0, 1])
    parser.add_argument("--tca-kappa-target", type=str, default=None)
    parser.add_argument("--tca-var-scale", type=float, default=None)
    parser.add_argument("--tca-tau", type=float, default=None)
    parser.add_argument("--tca-reg", type=float, default=None, help="Deprecated alias for --tca-var-scale.")
    parser.add_argument("--dataset", type=str, default=None, choices=["cifar10", "mnist", "shakespeare"])
    parser.add_argument("--model", type=str, default=None, choices=["resnet18", "smallcnn", "charlstm"])
    parser.add_argument("--num-users", type=int, default=None)
    parser.add_argument("--frac", type=float, default=None)
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.smoke_test:
        cfg = get_smoke_test_config()
    elif args.config is not None:
        cfg = load_config_from_yaml(args.config)
    else:
        cfg = CONFIG

    overrides = {}
    if args.method is not None:
        overrides["method"] = args.method
    if args.noise_floor is not None:
        overrides["run_noise_floor"] = bool(args.noise_floor)
    if args.cluster_mode is not None:
        overrides["cluster_mode"] = args.cluster_mode
    if args.zscore_mode is not None:
        overrides["zscore_mode"] = args.zscore_mode
    if args.consistent_split_weights is not None:
        overrides["consistent_split_weights"] = bool(args.consistent_split_weights)
    if args.tca_mode is not None:
        overrides["tca_mode"] = args.tca_mode
    if args.tca_cost is not None:
        overrides["tca_cost"] = args.tca_cost
    if args.tca_marginal is not None:
        overrides["tca_marginal"] = args.tca_marginal
    if args.tca_calibrate is not None:
        overrides["tca_calibrate"] = bool(args.tca_calibrate)
    if args.tca_kappa_target is not None:
        kt_str = args.tca_kappa_target.strip().lower()
        overrides["tca_kappa_target"] = "auto" if kt_str == "auto" else float(kt_str)
    if args.tca_reg is not None:
        overrides["tca_var_scale"] = float(args.tca_reg)
    if args.tca_var_scale is not None:
        overrides["tca_var_scale"] = float(args.tca_var_scale)
    if args.tca_tau is not None:
        overrides["tca_tau"] = float(args.tca_tau)
    if args.dataset is not None:
        overrides["dataset"] = args.dataset
    if args.model is not None:
        overrides["model"] = args.model
    if args.num_users is not None:
        overrides["num_users"] = args.num_users
    if args.frac is not None:
        overrides["frac"] = args.frac
    if args.n_clusters is not None:
        overrides["n_clusters"] = args.n_clusters
    if args.alpha is not None:
        overrides["alpha"] = args.alpha
    if args.rounds is not None:
        overrides["global_rounds"] = args.rounds
    if args.local_epochs is not None:
        overrides["local_epochs"] = args.local_epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir

    if overrides:
        cfg = replace(cfg, **overrides)
        _validate(cfg)

    run_experiment(
        config=cfg,
        method=cfg.method,
        run_noise_floor=cfg.run_noise_floor or cfg.smoke_test,
        save_excel=cfg.save_excel,
        output_dir=cfg.output_dir,
    )


if __name__ == "__main__":
    main()
