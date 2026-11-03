import copy
import numpy as np
import torch


def flatten_split_params(
    client_net: torch.nn.Module, server_net: torch.nn.Module
) -> np.ndarray:
    """Flatten trainable floating-point parameters of both client-half and server-half models."""
    vecs = []
    for module in (client_net, server_net):
        for p in module.parameters():
            if p.is_floating_point():
                vecs.append(p.detach().cpu().reshape(-1).to(dtype=torch.float64).numpy())
    if not vecs:
        return np.zeros(1, dtype=np.float64)
    return np.concatenate(vecs, axis=0)


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        if torch.is_floating_point(w_avg[k]):
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
        else:
            w_avg[k] = w[0][k].clone()
    return w_avg


def weighted_average(state_dicts, weights):
    if len(state_dicts) == 0:
        raise ValueError("weighted_average received an empty list of state_dicts.")
    if len(state_dicts) != len(weights):
        raise ValueError(
            f"State dict count ({len(state_dicts)}) does not match weight count ({len(weights)})."
        )

    weights_array = np.asarray(weights, dtype=np.float64)
    if np.any(weights_array < 0):
        raise ValueError("Aggregation weights must be non-negative.")

    total_weight = float(np.sum(weights_array))
    if total_weight <= 0.0:
        raise ValueError("Sum of aggregation weights must be strictly positive.")

    normalized_weights = weights_array / total_weight
    avg_state = copy.deepcopy(state_dicts[0])

    for key in avg_state.keys():
        tensor = avg_state[key]
        if torch.is_floating_point(tensor):
            acc = torch.zeros_like(tensor, dtype=torch.float64)
            for state, weight in zip(state_dicts, normalized_weights):
                acc += state[key].to(dtype=torch.float64) * float(weight)
            avg_state[key] = acc.to(dtype=tensor.dtype)
        else:
            avg_state[key] = state_dicts[0][key].clone()

    return avg_state


def clusterwise_fedavg(server_weights_by_id, cluster_groups, dict_users=None):
    """Aggregate server weights within clusters, then across clusters.

    If `dict_users` is provided, uses sample-size weighting both within clusters
    and across clusters (which algebraically simplifies to plain sample-weighted
    FedAvg across all active clients). Otherwise uses uniform averaging within
    each cluster and across clusters.
    """
    if not cluster_groups:
        raise ValueError("cluster_groups is empty.")

    cluster_models = []
    cluster_sample_weights = []

    for cluster_id, client_ids in sorted(cluster_groups.items()):
        if len(client_ids) == 0:
            continue

        cluster_weights = [server_weights_by_id[int(cid)] for cid in client_ids]
        if dict_users is not None:
            client_sample_counts = [len(dict_users[int(cid)]) for cid in client_ids]
            cluster_total_samples = int(sum(client_sample_counts))
            cluster_model = weighted_average(cluster_weights, client_sample_counts)
            cluster_models.append(cluster_model)
            cluster_sample_weights.append(cluster_total_samples)
        else:
            cluster_model = FedAvg(cluster_weights)
            cluster_models.append(cluster_model)

    if len(cluster_models) == 0:
        raise RuntimeError("No cluster models were produced during clusterwise_fedavg.")

    if dict_users is not None:
        final_global_model = weighted_average(cluster_models, cluster_sample_weights)
    else:
        final_global_model = FedAvg(cluster_models)

    return final_global_model, cluster_models
