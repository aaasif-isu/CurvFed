import copy
import numpy as np
import torch


def flatten_model_params(model: torch.nn.Module) -> np.ndarray:
    """Flatten trainable floating-point parameters of a model into a 1D float64 vector."""
    vecs = []
    for p in model.parameters():
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

