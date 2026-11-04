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
from src.core.train_utils import FedAvg, flatten_model_params, weighted_average
from src.data.dataset import load_configured_datasets
from src.data.partition import (
    DatasetSplit,
    compute_label_histograms_and_tv,
    partition_client_indices,
    partition_personalized_test_indices,
)
from src.methods.tca import (
    build_cost_matrix,
    build_variance_vector,
    estimate_q,
    lazy_calibrate,
    plan_curvature,
    solve_tca_plan,
    tca_aggregate_models,
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


def _module_param_l2_norm(module: torch.nn.Module) -> float:
    sq_sum = 0.0
    for p in module.parameters():
        if p.is_floating_point():
            v = p.detach().to(dtype=torch.float64)
            sq_sum += float(torch.sum(v * v).item())
    return float(np.sqrt(sq_sum))


def _evaluate_model(
    model: torch.nn.Module,
    dataset,
    batch_size: int,
    device: torch.device,
    num_classes: Optional[int] = None,
    return_pred_hist: bool = False,
):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    pred_list: List[np.ndarray] = []
    inferred_classes: Optional[int] = num_classes
    criterion = torch.nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            if inferred_classes is None:
                inferred_classes = int(logits.size(1))
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.max(1, keepdim=True)[1]
            correct += preds.eq(labels.view_as(preds)).sum().item()
            total += labels.size(0)
            if return_pred_hist:
                pred_list.append(preds.view(-1).detach().cpu().numpy())
    if total == 0:
        empty_hist = [0] * int(inferred_classes or 10)
        return (0.0, 0.0, empty_hist) if return_pred_hist else (0.0, 0.0)
    acc = 100.0 * correct / total
    avg_loss = total_loss / total
    if return_pred_hist:
        all_preds = np.concatenate(pred_list, axis=0) if pred_list else np.empty(0, dtype=np.int64)
        pred_hist = np.bincount(all_preds, minlength=int(inferred_classes or 10)).astype(int).tolist()
        return acc, avg_loss, pred_hist
    return acc, avg_loss


def _canonicalize_method(method: str) -> str:
    m = method.strip().lower()
    if m == "fedavg":
        return "baseline"
    return m


def run_experiment(
    config: ExperimentConfig = CONFIG,
    method: Optional[str] = None,
    save_excel: Optional[bool] = None,
    output_dir: Optional[str] = None,
):
    raw_method = (method or config.method or "baseline").strip().lower()
    canonical_method = _canonicalize_method(raw_method)
    if canonical_method not in ("baseline", "tca"):
        raise ValueError(f"Unsupported method: {raw_method}")
    method = canonical_method

    if save_excel is None:
        save_excel = config.save_excel
    resolved_output_dir = output_dir or config.output_dir

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = (
        config.experiment
        if (config.experiment and config.experiment.strip().lower() not in ("", "none", "null"))
        else raw_method
    )
    vs_tag = f"_vs{config.tca_var_scale:g}" if method == "tca" else ""
    default_sub = f"{exp_name}_{config.alpha}{vs_tag}_seed{config.seed}"
    out_folder = resolved_output_dir or os.path.join("results", default_sub)
    rounds_csv_dir = os.path.join(out_folder, "rounds")
    os.makedirs(rounds_csv_dir, exist_ok=True)

    resolved_cfg = replace(
        config,
        experiment=exp_name,
        method=method,
        save_excel=bool(save_excel),
        output_dir=out_folder,
    )
    save_config_to_yaml(resolved_cfg, os.path.join(out_folder, "resolved_config.yaml"))

    print(f"Experiment: {exp_name}")
    print(f"Final merged config: {json.dumps(resolved_cfg.to_dict(), indent=2)}")
    print(
        f"Using device: {device} | Running {method.upper()} "
        f"(model={config.model}, dataset={config.dataset}, "
        f"clients={config.num_users}, frac={config.frac}, alpha={config.alpha}, "
        f"rounds={config.global_rounds}, lr={config.learning_rate:g}, batch_size={config.batch_size}, "
        f"tca_mode={config.tca_mode}, tca_cost={config.tca_cost}, "
        f"tca_marginal={config.tca_marginal}, tca_var_scale={config.tca_var_scale:g}, "
        f"tca_calibrate={config.tca_calibrate})"
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
    eval_test_labels = test_labels[test_indices]
    pers_test_idcs = partition_personalized_test_indices(
        dict_users,
        train_labels,
        eval_test_labels,
        config.num_users,
        config.num_classes,
        seed=config.seed,
    )
    pers_test_datasets = {
        i: DatasetSplit(eval_test_dataset, pers_test_idcs[i]) for i in range(config.num_users)
    }
    for cid in range(config.num_users):
        test_cls_counts = np.bincount(
            eval_test_labels[list(pers_test_idcs[cid])], minlength=config.num_classes
        )
        print(
            f"Client {cid:2d} held-out test | samples={len(pers_test_idcs[cid]):4d} | "
            f"class_counts={test_cls_counts.tolist()}"
        )

    base_model = build_model(config)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        base_model = torch.nn.DataParallel(base_model)
    base_model = base_model.to(device)

    server = Server(
        global_model=base_model,
        device=device,
        num_users=config.num_users,
    )

    # --- Round 0 evaluation before any training ---
    r0_glob_acc, r0_glob_loss, r0_pred_hist = _evaluate_model(
        server.global_model,
        eval_test_dataset,
        config.batch_size,
        device,
        num_classes=config.num_classes,
        return_pred_hist=True,
    )
    r0_glob_norm = _module_param_l2_norm(server.global_model)
    r0_eval_net = copy.deepcopy(server.global_model).to(device)
    r0_pers_accs = []
    r0_pers_log_items = []
    r0_pers_norm_items = []
    for idx_int in range(config.num_users):
        r0_eval_net.load_state_dict(server.personalized_states[idx_int])
        split_ds = pers_test_datasets[idx_int]
        split_n = len(split_ds)
        c_acc, _ = _evaluate_model(
            r0_eval_net, split_ds, config.batch_size, device
        )
        p_norm = _module_param_l2_norm(r0_eval_net)
        r0_pers_accs.append(float(c_acc))
        r0_pers_log_items.append(f"c{idx_int}: {c_acc:.2f}% (n={split_n})")
        r0_pers_norm_items.append(f"c{idx_int}: {p_norm:.4f}")
    r0_pers_mean = float(np.mean(r0_pers_accs)) if r0_pers_accs else 0.0
    r0_pers_min = float(np.min(r0_pers_accs)) if r0_pers_accs else 0.0
    print(
        f"Round 0/{config.global_rounds} (Init) | Global Acc: {r0_glob_acc:.2f}% | "
        f"Global Loss: {r0_glob_loss:.4f} | Pers Mean/Min: {r0_pers_mean:.2f}%/{r0_pers_min:.2f}%"
    )
    print(f"  [Round 0] Global Param L2 Norm: total={r0_glob_norm:.4f}")
    print(f"  [Round 0] Global Test Pred Histogram ({len(r0_pred_hist)} classes): {r0_pred_hist}")
    print(f"  [Round 0] Per-client Pers Acc (held-out split): " + " | ".join(r0_pers_log_items))
    print(f"  [Round 0] Per-client Pers Param L2 Norm: " + " | ".join(r0_pers_norm_items))

    metrics = {
        "global_acc_test": [],
        "global_loss_test": [],
        "personalized_mean_acc": [],
        "personalized_min_acc": [],
        "round_time_sec": [],
        "round_rows": [],
    }


    for round_idx in range(config.global_rounds):
        round_start_time = time.perf_counter()

        rng = np.random.default_rng(config.seed + round_idx)
        active_users = rng.choice(range(config.num_users), config.num_active_clients, replace=False)
        active_users = np.array(sorted(int(x) for x in active_users), dtype=np.int64)

        w_locals: List[dict] = []
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

            net = copy.deepcopy(server.global_model).to(device)
            init_state = server.get_client_init_state(
                idx_int, method=method, tca_mode=config.tca_mode
            )
            net.load_state_dict(init_state)

            start_vec = flatten_model_params(net)
            w_client = local.train(net)
            mid_vec = local.mid_param_vec if local.mid_param_vec is not None else start_vec
            end_vec = flatten_model_params(net)

            v1 = mid_vec - start_vec
            v2 = end_vec - mid_vec
            norm_v1_list.append(float(np.linalg.norm(v1)))
            norm_v2_list.append(float(np.linalg.norm(v2)))
            active_deltas.append(end_vec - start_vec)
            active_sample_counts.append(len(dict_users[idx_int]))

            w_locals.append(copy.deepcopy(w_client))

        q_arr, median_q, q_half_raw_arr = estimate_q(norm_v1_list, norm_v2_list, return_raw=True)
        per_client_q = {int(cid): float(qv) for cid, qv in zip(active_users, q_arr)}
        q_half_raw = {int(cid): float(rv) for cid, rv in zip(active_users, q_half_raw_arr)}

        kappa_min = float("nan")
        kappa_mean = float("nan")
        sparsity = float("nan")
        inequality_violations = float("nan")
        metric_violations = float("nan")
        beta_val = float("nan")



        if method == "baseline":
            server.aggregate_fedavg(w_locals, active_sample_counts)
        elif method == "tca":
            rho = np.asarray(active_sample_counts, dtype=np.float64)
            rho = rho / max(float(rho.sum()), 1e-12)
            v_vec = build_variance_vector(active_sample_counts, var_scale=config.tca_var_scale)
            D_norm = build_cost_matrix(config.tca_cost, deltas=active_deltas)
            C = D_norm ** 2
            A_plan = solve_tca_plan(
                C,
                v_vec,
                rho=rho,
                marginal=config.tca_marginal,
                tau=config.tca_tau,
            )
            curv_stats = plan_curvature(A_plan, D_norm)

            if config.tca_calibrate:
                A_plan, beta_val, _ = lazy_calibrate(
                    A_plan,
                    median_q=median_q,
                    kappa_target=config.tca_kappa_target,
                    D=D_norm,
                    curv_stats=curv_stats,
                )
                curv_stats = plan_curvature(A_plan, D_norm)

            kappa_min = float(curv_stats["kappa_min"])
            kappa_mean = float(curv_stats["kappa_mean"])
            sparsity = float(curv_stats["sparsity"])
            inequality_violations = int(curv_stats["inequality_violations"])
            metric_violations = int(curv_stats["metric_violations"])

            np.save(os.path.join(rounds_csv_dir, f"A_round_{round_idx + 1:03d}.npy"), A_plan)
            if config.num_users <= 10:
                a_str = np.array2string(
                    np.round(A_plan, 3),
                    formatter={"float_kind": lambda x: f"{x:6.3f}"},
                    separator=" ",
                )
                print(f"  [Round {round_idx + 1}] Plan matrix A (shape={A_plan.shape}):\n{a_str}")

            w_glob, active_pers = tca_aggregate_models(
                w_locals,
                A_plan,
                rho,
                [int(u) for u in active_users],
            )
            server.update_from_tca(w_glob, active_pers)

        global_acc, global_loss, global_pred_hist = _evaluate_model(
            server.global_model,
            eval_test_dataset,
            config.batch_size,
            device,
            num_classes=config.num_classes,
            return_pred_hist=True,
        )
        glob_tot_norm = _module_param_l2_norm(server.global_model)


        pers_client_accs = []
        pers_client_log_items = []
        pers_client_norm_items = []
        per_client_pers_acc_map = {}
        per_client_test_size_map = {}
        eval_net = copy.deepcopy(server.global_model).to(device)
        eval_client_ids = (
            list(range(config.num_users))
            if method == "tca"
            else [int(u) for u in active_users]
        )
        for idx_int in eval_client_ids:
            if method == "tca":
                eval_net.load_state_dict(server.personalized_states[idx_int])
            else:
                eval_net.load_state_dict(server.global_state)

            split_ds = pers_test_datasets[idx_int]
            split_n = len(split_ds)
            c_acc, _ = _evaluate_model(
                eval_net, split_ds, config.batch_size, device
            )
            p_tot_norm = _module_param_l2_norm(eval_net)
            pers_client_accs.append(float(c_acc))
            per_client_pers_acc_map[idx_int] = float(c_acc)
            per_client_test_size_map[idx_int] = int(split_n)
            pers_client_log_items.append(f"c{idx_int}: {c_acc:.2f}% (n={split_n})")
            pers_client_norm_items.append(f"c{idx_int}: {p_tot_norm:.4f}")

        print(f"  [Round {round_idx + 1}] Global Param L2 Norm: total={glob_tot_norm:.4f}")
        print(
            f"  [Round {round_idx + 1}] Global Test Pred Histogram ({len(global_pred_hist)} classes): "
            f"{global_pred_hist}"
        )
        print(
            f"  [Round {round_idx + 1}] Per-client Pers Acc (held-out split): "
            + " | ".join(pers_client_log_items)
        )
        print(
            f"  [Round {round_idx + 1}] Per-client Pers Param L2 Norm: "
            + " | ".join(pers_client_norm_items)
        )

        pers_mean_acc = float(np.mean(pers_client_accs)) if pers_client_accs else 0.0
        pers_min_acc = float(np.min(pers_client_accs)) if pers_client_accs else 0.0
        round_time = time.perf_counter() - round_start_time

        round_row = {
            "method": method,
            "config": config.experiment_tag,
            "tca_mode": config.tca_mode if method == "tca" else "N/A",
            "tca_cost": config.tca_cost if method == "tca" else "N/A",
            "tca_marginal": config.tca_marginal if method == "tca" else "N/A",
            "tca_var_scale": float(config.tca_var_scale) if method == "tca" else float("nan"),
            "tca_calibrate": bool(config.tca_calibrate) if method == "tca" else False,
            "round": round_idx + 1,
            "global_acc_test": global_acc,
            "global_loss_test": global_loss,
            "personalized_mean_acc": pers_mean_acc,
            "personalized_min_acc": pers_min_acc,
            "per_client_pers_acc": json.dumps(per_client_pers_acc_map),
            "per_client_test_size": json.dumps(per_client_test_size_map),
            "kappa_min": kappa_min,
            "kappa_mean": kappa_mean,
            "sparsity": sparsity,
            "inequality_violations": inequality_violations,
            "metric_violations": metric_violations,
            "q_half_raw": json.dumps(q_half_raw),
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
            f"Pers Mean/Min: {pers_mean_acc:.2f}%/{pers_min_acc:.2f}%{tca_info} | "
            f"Time: {round_time:.2f}s"
        )

    full_df = pd.DataFrame(metrics["round_rows"])
    full_df.to_csv(os.path.join(out_folder, "round_metrics.csv"), index=False)
    if save_excel:
        full_df.to_excel(
            os.path.join(out_folder, f"metrics_{config.experiment_tag}.xlsx"), index=False
        )

    print(f"Training finished. Results written to: {out_folder}")
    return metrics, server.global_model



def main():
    parser = argparse.ArgumentParser(
        description="Run CurvFed Federated Learning experiment (reads config.yaml by default)."
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
            "tca",
        ],
        help="Aggregation method to run (overrides 'method' in YAML).",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Use CPU smoke-test config.")
    parser.add_argument("--tca-mode", type=str, default=None, choices=["decoupled", "feedback"])
    parser.add_argument("--tca-cost", type=str, default=None, choices=["update"])
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
        save_excel=cfg.save_excel,
        output_dir=cfg.output_dir,
    )



if __name__ == "__main__":
    main()
