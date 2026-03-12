# analysis_scripts/sfl_synthetic_from_smashed.py
# Runs a quick SplitFedV1-style loop on precomputed "smashed" tensors.
# Expects files like: analysis_scripts/smashed/client{0..4}_round{0..K}.pt
# Each tensor shape: [N, 64, 8, 8]

import os
import glob
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# -------------------------
# Setup
# -------------------------
torch.manual_seed(1234)
np.random.seed(1234)

HERE = os.path.dirname(__file__)
SMASHED_DIR = os.path.join(HERE, "smashed")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -------------------------
# Server-side model (head)
# -------------------------
class Baseblock(nn.Module):
    def __init__(self, inp, planes, stride=1, dim_change=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, planes, 3, stride=stride, padding=1)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(planes)
        self.dim_change = dim_change

    def forward(self, x):
        res = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.dim_change is not None:
            res = self.dim_change(res)
        return torch.relu(out + res)

class ResNet18Server(nn.Module):
    """Compatible with client smashed activations shaped [B, 64, 8, 8]."""
    def __init__(self, classes=7):
        super().__init__()
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
        )
        self.layer4 = self._layer(64, 128, stride=2)
        self.layer5 = self._layer(128, 256, stride=2)
        self.layer6 = self._layer(256, 512, stride=2)
        self.avg    = nn.AdaptiveAvgPool2d((1, 1))
        self.fc     = nn.Linear(512, classes)

    def _layer(self, inp, planes, stride=2):
        dim_change = nn.Sequential(
            nn.Conv2d(inp, planes, 1, stride=stride),
            nn.BatchNorm2d(planes),
        )
        return nn.Sequential(
            Baseblock(inp, planes, stride=stride, dim_change=dim_change),
            Baseblock(planes, planes)
        )

    def forward(self, x3):
        out2 = self.layer3(x3)
        x3   = torch.relu(out2 + x3)     # residual add
        x4   = self.layer4(x3)
        x5   = self.layer5(x4)
        x6   = self.layer6(x5)
        x7   = self.avg(x6).flatten(1)
        return self.fc(x7)

# -------------------------
# Hyperparameters
# -------------------------
num_clients = 5
rounds      = 3        # global rounds
batch_size  = 64
lr          = 1e-3
classes     = 7

# -------------------------
# Data loaders per client
# -------------------------
def load_client_dataset(cid: int) -> TensorDataset:
    pattern = os.path.join(SMASHED_DIR, f"client{cid}_round*.pt")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No smashed tensors for client {cid} at {pattern}")
    xs = [torch.load(f, map_location="cpu") for f in files]
    X = torch.cat(xs, dim=0)  # [N, 64, 8, 8]
    if X.dim() != 4 or X.size(1) != 64:
        raise ValueError(f"Expected [N,64,8,8] smashed features, got {tuple(X.shape)} for client {cid}")
    # Synthetic labels (for quick loop; not used for real accuracy claims)
    y = torch.randint(0, classes, (X.size(0),))
    return TensorDataset(X, y)

client_sets = [load_client_dataset(c) for c in range(num_clients)]

# -------------------------
# FedAvg
# -------------------------
def fed_avg(state_dicts):
    """Federated average for model state_dicts.
    - Floating-point tensors: arithmetic mean
    - Non-floating tensors (e.g., Long): keep from the first model
    """
    import copy, torch
    w = copy.deepcopy(state_dicts[0])
    for k in w.keys():
        if torch.is_floating_point(w[k]):
            acc = w[k].clone()
            for i in range(1, len(state_dicts)):
                acc += state_dicts[i][k].to(acc.dtype)
            w[k] = acc / float(len(state_dicts))
        else:
            # e.g., BatchNorm num_batches_tracked (int) — do not average
            # leave as w[k] from the first model
            pass
    return w


# -------------------------
# Initialize server models (SFLV1-style: per-client server then FedAvg)
# -------------------------
net_models = [ResNet18Server(classes).to(device) for _ in range(num_clients)]
criterion  = nn.CrossEntropyLoss()

# -------------------------
# Training
# -------------------------
for r in range(rounds):
    w_locals = []
    train_losses, train_accs = [], []

    for cid in range(num_clients):
        net = copy.deepcopy(net_models[cid]).to(device)
        net.train()
        opt = torch.optim.Adam(net.parameters(), lr=lr)

        loader = DataLoader(client_sets[cid], batch_size=batch_size, shuffle=True)

        batch_loss, batch_correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = net(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

            batch_loss   += loss.item() * yb.size(0)
            batch_correct += (logits.argmax(1) == yb).sum().item()
            total        += yb.size(0)

        train_losses.append(batch_loss / total)
        train_accs.append(100.0 * batch_correct / total)
        w_locals.append(copy.deepcopy(net.state_dict()))

    # FedAvg -> global
    w_glob = fed_avg(w_locals)

    # Broadcast global to all client-specific server models
    for cid in range(num_clients):
        net_models[cid].load_state_dict(w_glob)

    print(f"Round {r+1}/{rounds} | Train Acc avg: {np.mean(train_accs):.2f}% | "
          f"Loss avg: {np.mean(train_losses):.4f}")

# Optional: save the global server head
out_path = os.path.join(HERE, "server_head_global.pth")
torch.save(w_glob, out_path)
print(f"Finished. Saved global server head to: {out_path}")
