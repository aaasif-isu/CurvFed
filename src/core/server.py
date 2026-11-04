import copy
from typing import Dict, List
import numpy as np
import torch
from torch import nn
from .train_utils import weighted_average


class Server:
    """Central Federated Learning coordinator that manages the global model and
    per-client personalized models and performs aggregation without participating
    in forward or backward propagation."""

    def __init__(self, global_model: nn.Module, device: torch.device, num_users: int):
        self.device = device
        self.num_users = int(num_users)
        self.global_model = copy.deepcopy(global_model).to(device)
        self.global_state: dict = copy.deepcopy(self.global_model.state_dict())
        self.personalized_states: Dict[int, dict] = {
            i: copy.deepcopy(self.global_state) for i in range(self.num_users)
        }

    def get_client_init_state(
        self,
        client_id: int,
        method: str = "baseline",
        tca_mode: str = "decoupled",
    ) -> dict:
        """Return the initial model state_dict for `client_id` at the start of a round.
        In TCA feedback mode, each client starts from its own personalized model;
        in decoupled mode or FedAvg baseline, clients start from the global model."""
        cid = int(client_id)
        if method == "tca" and tca_mode == "feedback":
            return copy.deepcopy(self.personalized_states[cid])
        return copy.deepcopy(self.global_state)

    def aggregate_fedavg(self, w_locals: List[dict], sample_counts: List[int]) -> dict:
        """Perform standard sample-weighted FedAvg aggregation across active client models."""
        rho = np.asarray(sample_counts, dtype=np.float64)
        rho = rho / max(float(rho.sum()), 1e-12)
        self.global_state = weighted_average(w_locals, rho)
        self.global_model.load_state_dict(self.global_state)
        for cid in range(self.num_users):
            self.personalized_states[cid] = copy.deepcopy(self.global_state)
        return copy.deepcopy(self.global_state)

    def update_from_tca(self, w_glob: dict, active_pers_states: Dict[int, dict]) -> None:
        """Update the global model and active personalized models after TCA aggregation."""
        self.global_state = copy.deepcopy(w_glob)
        self.global_model.load_state_dict(self.global_state)
        for cid, state in active_pers_states.items():
            self.personalized_states[int(cid)] = copy.deepcopy(state)

