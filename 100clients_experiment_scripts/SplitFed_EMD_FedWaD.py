#============================================================================
# SplitfedV1 (SFLV1) learning: ResNet18 on HAM10000
# HAM10000 dataset: Tschandl, P.: The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions (2018), doi:10.7910/DVN/DBW86T

# We have three versions of our implementations
# Version1: without using socket and no DP+PixelDP
# Version2: with using socket but no DP+PixelDP
# Version3: without using socket but with DP+PixelDP

# This program is Version1: Single program simulation 
# ============================================================================
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import math
import os.path
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob
from pandas import DataFrame
import json
from sklearn.cluster import AgglomerativeClustering
import ot
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci

import random
import numpy as np
import os
import time
from config import CONFIG
from experiment_support import build_charlstm, load_configured_datasets


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import shutil
from FedWaD import FedOT


SEED = CONFIG.seed

# Run this file twice with the same config/seed to obtain the controlled
# comparison: "emd" is the existing baseline and "fedwad" is the paper method.
DISTANCE_METHOD = str(
    os.getenv(
        "SPLITFED_DISTANCE_METHOD",
        getattr(CONFIG, "distance_method", "fedwad"),
    )
).lower()
if DISTANCE_METHOD not in {"emd", "fedwad"}:
    raise ValueError("CONFIG.distance_method must be 'emd' or 'fedwad'.")

FEDWAD_N_SUPP = int(
    os.getenv("FEDWAD_N_SUPP", getattr(CONFIG, "fedwad_n_supp", 10))
)
FEDWAD_EPOCHS = int(
    os.getenv("FEDWAD_EPOCHS", getattr(CONFIG, "fedwad_epochs", 20))
)
FEDWAD_T = float(
    os.getenv("FEDWAD_T", getattr(CONFIG, "fedwad_t", 0.5))
)
DISTANCE_POOL_SIZE = int(
    os.getenv(
        "SPLITFED_DISTANCE_POOL_SIZE",
        getattr(CONFIG, "distance_pool_size", 1),
    )
)
if FEDWAD_N_SUPP < 1 or FEDWAD_EPOCHS < 1:
    raise ValueError("FedWaD support size and epochs must be positive.")
if not 0.0 < FEDWAD_T < 1.0:
    raise ValueError("CONFIG.fedwad_t must be strictly between 0 and 1.")
if DISTANCE_POOL_SIZE < 1:
    raise ValueError("CONFIG.distance_pool_size must be positive.")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    print(torch.cuda.get_device_name(0))  

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

  

#===================================================================
program = (
    f"SFLV1 {CONFIG.model} on {CONFIG.dataset} NonIID "
    f"{DISTANCE_METHOD.upper()} Only"
)
print(f"---------{program}----------")              # this is to identify the program in the slurm outputs files

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("CUDA available:", torch.cuda.is_available())
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("GPU Capability OK!")
else:
    print("WARNING: Not running on GPU")



# To print in color -------test/train of the client side
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))     

def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

#===================================================================
# ============================================================
# Experiment parameters from config.py
# ============================================================

num_users = CONFIG.num_users
epochs = CONFIG.global_rounds
frac = CONFIG.frac
lr = CONFIG.learning_rate
alpha = CONFIG.alpha
batch_size = CONFIG.batch_size
local_epochs = CONFIG.local_epochs
n_clusters = CONFIG.n_clusters


#=====================================================================================================
#                           Client-side Model definition
#=====================================================================================================
# Model at client side
class ResNet18_client_side(nn.Module):
    def __init__(self):
        super(ResNet18_client_side, self).__init__()
        self.layer1 = nn.Sequential (
                nn.Conv2d(CONFIG.input_channels, 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU (inplace = True),
                nn.MaxPool2d(kernel_size = 3, stride = 2, padding =1),
            )
        self.layer2 = nn.Sequential  (
                nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU (inplace = True),
                nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(64),              
            )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        
    def forward(self, x):
        resudial1 = F.relu(self.layer1(x))
        out1 = self.layer2(resudial1)
        out1 = out1 + resudial1 # adding the resudial inputs -- downsampling not required in this layer
        resudial2 = F.relu(out1)
        return resudial2
 
 
           

if CONFIG.model == "resnet18":
    net_glob_client = ResNet18_client_side()
else:
    net_glob_client, _charlstm_server = build_charlstm(CONFIG)
if torch.cuda.device_count() > 1:
    print("We use",torch.cuda.device_count(), "GPUs")
    net_glob_client = nn.DataParallel(net_glob_client)    

net_glob_client.to(device)
print(net_glob_client)     

#=====================================================================================================
#                           Server-side Model definition
#=====================================================================================================
# Model at server side
class Baseblock(nn.Module):
    expansion = 1
    def __init__(self, input_planes, planes, stride = 1, dim_change = None):
        super(Baseblock, self).__init__()
        self.conv1 = nn.Conv2d(input_planes, planes, stride =  stride, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, stride = 1, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dim_change = dim_change
        
    def forward(self, x):
        res = x
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        
        if self.dim_change is not None:
            res =self.dim_change(res)
            
        output += res
        output = F.relu(output)
        
        return output


class ResNet18_server_side(nn.Module):
    def __init__(self, block, num_layers, classes):
        super(ResNet18_server_side, self).__init__()
        self.input_planes = 64
        self.layer3 = nn.Sequential (
                nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(64),
                nn.ReLU (inplace = True),
                nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(64),       
                )   
        
        self.layer4 = self._layer(block, 128, num_layers[0], stride = 2)
        self.layer5 = self._layer(block, 256, num_layers[1], stride = 2)
        self.layer6 = self._layer(block, 512, num_layers[2], stride = 2)
        self.averagePool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        
    def _layer(self, block, planes, num_layers, stride = 2):
        dim_change = None
        if stride != 1 or planes != self.input_planes * block.expansion:
            dim_change = nn.Sequential(nn.Conv2d(self.input_planes, planes*block.expansion, kernel_size = 1, stride = stride),
                                       nn.BatchNorm2d(planes*block.expansion))
        netLayers = []
        netLayers.append(block(self.input_planes, planes, stride = stride, dim_change = dim_change))
        self.input_planes = planes * block.expansion
        for i in range(1, num_layers):
            netLayers.append(block(self.input_planes, planes))
            self.input_planes = planes * block.expansion
            
        return nn.Sequential(*netLayers)
        
    
    def forward(self, x):
        out2 = self.layer3(x)

        # Original ResNet residual connection
        out2 = out2 + x
        x3 = F.relu(out2)

        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)

        x7 = self.averagePool(x6)
        x8 = torch.flatten(x7, 1)

        y_hat = self.fc(x8)

        return y_hat

if CONFIG.model == "resnet18":
    net_glob_server = ResNet18_server_side(Baseblock,[2, 2, 2], CONFIG.num_classes)
else:
    net_glob_server = _charlstm_server
if torch.cuda.device_count() > 1:
    print("We use",torch.cuda.device_count(), "GPUs")
    net_glob_server = nn.DataParallel(net_glob_server)   # to use the multiple GPUs 

net_glob_server.to(device)
print(net_glob_server)      

#===================================================================================
# For Server Side Loss and Accuracy 
loss_train_collect = []
acc_train_collect = []

loss_test_collect = []
acc_test_collect = []

batch_acc_train = []
batch_loss_train = []


criterion = nn.CrossEntropyLoss()
count1 = 0
# count2 = 0
#====================================================================================================
#                                  Server Side Program
#====================================================================================================
# Federated averaging: FedAvg
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def make_round_dirs(global_round):
    round_name = f"round_{global_round + 1:03d}"

    method_details = (
        f"_supp{FEDWAD_N_SUPP}_fwepochs{FEDWAD_EPOCHS}_t{FEDWAD_T:g}"
        if DISTANCE_METHOD == "fedwad" else ""
    )
    experiment_root = (
        f"{CONFIG.dataset}_{CONFIG.model}_"
        f"{DISTANCE_METHOD}_only{method_details}_"
        f"pool{DISTANCE_POOL_SIZE}_"
        f"alpha{CONFIG.alpha}_"
        f"seed{CONFIG.seed}_"
        f"clients{CONFIG.num_users}"
    )

    smashed_dir = os.path.join(
        experiment_root,
        "smashed",
        round_name
    )

    out_dir = os.path.join(
        experiment_root,
        "outputs",
        round_name
    )

    distrib_dir = os.path.join(
        out_dir,
        "distrib"
    )

    # Remove artifacts from an earlier execution of this round
    if os.path.exists(smashed_dir):
        shutil.rmtree(smashed_dir)

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    os.makedirs(smashed_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(distrib_dir, exist_ok=True)

    return smashed_dir, out_dir, distrib_dir



def build_distrib_for_round(smashed_dir, distrib_dir):
    client_batches = {}

    for fname in sorted(os.listdir(smashed_dir)):
        if not (fname.startswith("client_") and fname.endswith(".pt")):
            continue

        parts = fname.replace(".pt", "").split("_")
        client_id = int(parts[1])

        payload = torch.load(
            os.path.join(smashed_dir, fname),
            map_location="cpu"
        )

        # New files contain both components needed by FedOTDD. Tensor-only
        # files remain readable for the original EMD baseline.
        if isinstance(payload, dict):
            x = payload["features"].float()
            y = payload["labels"].long()
        else:
            x = payload.float()
            y = None

        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        client_batches.setdefault(client_id, []).append((x, y))

    for client_id, batches in client_batches.items():
        x_all = torch.cat([x for x, _ in batches], dim=0)
        labels_available = all(y is not None for _, y in batches)
        y_all = (
            torch.cat([y for _, y in batches], dim=0)
            if labels_available else None
        )

        # Fixed-size representation for comparable EMD computation
        if x_all.size(0) > 256:
            generator = torch.Generator().manual_seed(SEED + client_id)
            selected = torch.randperm(
                x_all.size(0),
                generator=generator
            )[:256]
            x_all = x_all[selected]
            if y_all is not None:
                y_all = y_all[selected]

        torch.save(
            {"features": x_all, "labels": y_all},
            os.path.join(distrib_dir, f"client{client_id}.pt")
        )


def _load_client_distributions(distrib_dir, require_labels=False):
    def client_id(path):
        return int(
            os.path.basename(path).replace("client", "").replace(".pt", "")
        )

    files = sorted(
        glob(os.path.join(distrib_dir, "client*.pt")),
        key=client_id,
    )
    if not files:
        raise RuntimeError(
            f"No client distribution files found in {distrib_dir}"
        )

    distributions = []
    for path in files:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            x = payload["features"].float().numpy()
            labels = payload.get("labels")
            y = None if labels is None else labels.long().numpy()
        else:
            x = payload.float().numpy()
            y = None

        if require_labels and y is None:
            raise RuntimeError(
                "FedWaD/FedOTDD requires labels paired with smashed features. "
                f"Recreate the round artifacts; {path} is tensor-only."
            )

        # Keep preprocessing identical to the existing EMD experiment.
        x = (x - x.mean(0, keepdims=True)) / (
            x.std(0, keepdims=True) + 1e-8
        )
        distributions.append((client_id(path), x.astype(np.float64), y))
    return distributions


def compute_emd_for_round(distrib_dir, out_dir):
    distributions = _load_client_distributions(distrib_dir)
    client_ids = [client_id for client_id, _, _ in distributions]
    Xs = [x for _, x, _ in distributions]

    k = len(Xs)
    emd = np.zeros((k, k), dtype=np.float64)

    for i in range(k):
        Xi, wi = Xs[i], ot.unif(len(Xs[i]))
        for j in range(i + 1, k):
            Xj, wj = Xs[j], ot.unif(len(Xs[j]))
            M = ot.dist(Xi, Xj, metric="euclidean")
            emd[i, j] = emd[j, i] = float(ot.emd2(wi, wj, M))

    np.save(os.path.join(out_dir, "emd_matrix.npy"), emd)

    with open(os.path.join(out_dir, "emd_matrix_readable.txt"), "w") as f:
        f.write(f"Shape: {emd.shape}\n\n")
        for row in emd:
            f.write(" ".join([f"{v:.4f}" for v in row]) + "\n")

    return emd, client_ids


def _augment_for_fedotdd(x, y):
    """Paper-style diagonal class-conditional OTDD augmentation.

    Every smashed feature is concatenated with the mean and diagonal square
    root covariance of its local class. This supports HAM10000 clients that do
    not all contain the same subset of classes.
    """
    if x.shape[0] != len(y):
        raise ValueError("Feature and label counts do not match.")

    augmented = np.empty((x.shape[0], 3 * x.shape[1]), dtype=np.float64)
    for class_id in np.unique(y):
        mask = y == class_id
        class_x = x[mask]
        mean = class_x.mean(axis=0)
        # Diagonal sqrt covariance, matching use_diag=True in the paper code.
        if class_x.shape[0] > 1:
            variance = class_x.var(axis=0, ddof=1)
        else:
            variance = np.zeros(x.shape[1], dtype=np.float64)
        std = np.sqrt(np.maximum(variance, 0.0) + 1e-6)
        augmented[mask] = np.concatenate(
            [
                class_x,
                np.repeat(mean[None, :], class_x.shape[0], axis=0),
                np.repeat(std[None, :], class_x.shape[0], axis=0),
            ],
            axis=1,
        )
    return augmented


def compute_fedwad_for_round(distrib_dir, out_dir):
    distributions = _load_client_distributions(
        distrib_dir,
        require_labels=True,
    )
    client_ids = [client_id for client_id, _, _ in distributions]
    augmented = [
        _augment_for_fedotdd(x, y)
        for _, x, y in distributions
    ]

    k = len(augmented)
    distances = np.zeros((k, k), dtype=np.float64)
    numpy_state = np.random.get_state()
    try:
        for i in range(k):
            for j in range(i + 1, k):
                # Pair-specific deterministic initialization makes reruns and
                # parameter sweeps directly comparable.
                np.random.seed(SEED + i * k + j)
                fedwad = FedOT(
                    n_supp=FEDWAD_N_SUPP,
                    n_epoch=FEDWAD_EPOCHS,
                    t_val=FEDWAD_T,
                    verbose=False,
                    metric="sqeuclidean",
                )
                fedwad.fit(
                    augmented[i],
                    augmented[j],
                    approx_interp=True,
                )
                distance = float(fedwad.cost ** 2)
                if not np.isfinite(distance):
                    raise FloatingPointError(
                        f"Non-finite FedWaD distance for clients {i} and {j}."
                    )
                distances[i, j] = distances[j, i] = distance
    finally:
        np.random.set_state(numpy_state)

    stem = (
        f"fedwad_matrix_supp{FEDWAD_N_SUPP}_"
        f"epochs{FEDWAD_EPOCHS}"
    )
    np.save(os.path.join(out_dir, f"{stem}.npy"), distances)
    with open(os.path.join(out_dir, f"{stem}_readable.txt"), "w") as f:
        f.write(f"Shape: {distances.shape}\n")
        f.write(f"n_supp: {FEDWAD_N_SUPP}\n")
        f.write(f"epochs: {FEDWAD_EPOCHS}\n")
        f.write(f"t: {FEDWAD_T}\n\n")
        for row in distances:
            f.write(" ".join(f"{value:.6f}" for value in row) + "\n")
    return distances, client_ids


def cluster_clients_from_distance(
    distance_matrix,
    client_ids,
    out_dir,
    n_clusters=CONFIG.n_clusters
):
    """
    Cluster clients directly from a pairwise distance matrix.

    This is the EMD-only experiment:
        EMD -> Agglomerative Clustering

    No Ollivier-Ricci Curvature is used.
    """

    if not 1 <= n_clusters <= len(client_ids):
        raise ValueError(
            f"n_clusters={n_clusters} must be between 1 and the number "
            f"of selected clients ({len(client_ids)})."
        )
    if len(set(client_ids)) != len(client_ids):
        raise ValueError("Client IDs in the distance matrix must be unique.")

    # Safety checks
    expected_shape = (len(client_ids), len(client_ids))
    if distance_matrix.shape != expected_shape:
        raise RuntimeError(
            f"Expected distance matrix shape "
            f"{expected_shape}, "
            f"found {distance_matrix.shape}"
        )

    if not np.allclose(
        distance_matrix,
        distance_matrix.T,
        atol=1e-8
    ):
        raise RuntimeError(
            "Distance matrix is not symmetric."
        )

    if not np.allclose(
        np.diag(distance_matrix),
        0.0,
        atol=1e-8
    ):
        raise RuntimeError(
            "Distance matrix diagonal must be zero."
        )

    # Agglomerative clustering accepts the EMD
    # matrix directly as a precomputed distance matrix.
    try:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="average"
        )
    except TypeError:
        # Compatibility with older sklearn
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            linkage="average"
        )

    labels = model.fit_predict(
        distance_matrix
    )

    cluster_groups = {}

    for client_id, cluster_id in zip(client_ids, labels):
        cluster_groups.setdefault(
            int(cluster_id),
            []
        ).append(
            client_id
        )

    # Save readable cluster information
    with open(
        os.path.join(
            out_dir,
            "cluster_summary.txt"
        ),
        "w"
    ) as f:

        f.write(
            f"{DISTANCE_METHOD.upper()}-only client clustering\n\n"
        )

        for client_id, cluster_id in zip(client_ids, labels):
            f.write(
                f"Client {client_id} -> "
                f"Cluster {cluster_id}\n"
            )

        f.write(
            "\nCluster groups\n"
        )

        for cluster_id, clients in sorted(
            cluster_groups.items()
        ):
            f.write(
                f"Cluster {cluster_id}: "
                f"{clients}\n"
            )

    np.savetxt(
        os.path.join(
            out_dir,
            "client_clusters.txt"
        ),
        labels,
        fmt="%d"
    )
    np.savetxt(
        os.path.join(out_dir, "distance_client_ids.txt"),
        np.asarray(client_ids, dtype=np.int64),
        fmt="%d",
    )

    return labels, cluster_groups


def compute_ricci_for_round(emd, out_dir):
    n = emd.shape[0]
    positives = emd[np.triu_indices(n, 1)]
    tau = np.median(positives[positives > 0]) if np.any(positives > 0) else 1.0

    sim = np.exp(-emd / max(tau, 1e-8))
    np.fill_diagonal(sim, 0.0)

    G = nx.Graph()
    k = min(3, n - 1)

    for i in range(n):
        G.add_node(i)
        nbrs = np.argsort(sim[i])[::-1][:k]
        for j in nbrs:
            if i != j:
                G.add_edge(i, j, weight=float(sim[i, j]), distance=float(emd[i, j]))

    ricci = OllivierRicci(G, alpha=0.5, method="OTD", verbose="ERROR")
    Gk = ricci.compute_ricci_curvature()

    edge_kappa = {}
    for u, v in Gk.edges():
        edge_kappa[f"{u}-{v}"] = float(Gk[u][v].get("ricciCurvature", 0.0))

    with open(os.path.join(out_dir, "ricci_edges.json"), "w") as f:
        json.dump(edge_kappa, f, indent=2)
    return Gk


def cluster_clients_from_ricci(Gk, out_dir, n_clusters=CONFIG.n_clusters):
    n = Gk.number_of_nodes()

    ricci_distance = np.ones((n, n), dtype=np.float64)

    np.fill_diagonal(ricci_distance, 0.0)

    for u, v, data in Gk.edges(data=True):
        curvature = float(data.get("ricciCurvature", 0.0))

        # Higher curvature = stronger structural similarity.
        # Convert it to a non-negative distance.
        distance = max(0.0, 1.0 - curvature)

        ricci_distance[u, v] = distance
        ricci_distance[v, u] = distance

    try:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="average"
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            linkage="average"
        )

    labels = model.fit_predict(ricci_distance)

    cluster_groups = {}

    for client_id, cluster_id in enumerate(labels):
        cluster_groups.setdefault(
            int(cluster_id),
            []
        ).append(client_id)

    with open(
        os.path.join(out_dir, "cluster_summary.txt"),
        "w"
    ) as f:

        f.write("Ricci-based client clustering\n\n")

        for client_id, cluster_id in enumerate(labels):
            f.write(
                f"Client {client_id} -> "
                f"Cluster {cluster_id}\n"
            )

        f.write("\nCluster groups\n")

        for cluster_id, clients in sorted(
            cluster_groups.items()
        ):
            f.write(
                f"Cluster {cluster_id}: "
                f"{clients}\n"
            )

    np.savetxt(
        os.path.join(out_dir, "client_clusters.txt"),
        labels,
        fmt="%d"
    )

    return labels, cluster_groups


def clusterwise_fedavg(server_weights_by_id, cluster_groups):
    cluster_models = []

    for cluster_id, client_ids in sorted(cluster_groups.items()):

        cluster_weights = [
            server_weights_by_id[client_id]
            for client_id in client_ids
        ]

        cluster_model = FedAvg(cluster_weights)
        cluster_models.append(cluster_model)

    # Only after ALL clusters have been averaged
    final_global_model = FedAvg(cluster_models)

    return final_global_model, cluster_models


def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 *correct.float()/preds.shape[0]
    return acc

# to print train - test together in each round-- these are made global
acc_avg_all_user_train = 0
loss_avg_all_user_train = 0
loss_train_collect_user = []
acc_train_collect_user = []
# loss_test_collect_user = []
# acc_test_collect_user = []

w_glob_server = net_glob_server.state_dict()
server_weights_by_id =  {}

#client idx collector
idx_collect = []
# Initialization of net_model_server and net_server (server-side model)
net_model_server = [net_glob_server for i in range(num_users)]
net_server = copy.deepcopy(net_model_server[0]).to(device)
#optimizer_server = torch.optim.Adam(net_server.parameters(), lr = lr)

# Server-side function associated with Training 
def train_server(fx_client, y, l_epoch_count, l_epoch, idx, len_batch):
    global net_model_server, criterion, optimizer_server
    global device, batch_acc_train, batch_loss_train
    global loss_train_collect, acc_train_collect, count1
    global acc_avg_all_user_train, loss_avg_all_user_train
    global idx_collect, w_glob_server, net_server
    global loss_train_collect_user
    global acc_train_collect_user
    global lr
    global server_weights_by_id
    
    net_server = copy.deepcopy(net_model_server[idx]).to(device)
    net_server.train()
    optimizer_server = torch.optim.Adam(net_server.parameters(), lr = lr)

    
    # train and update
    optimizer_server.zero_grad()
    
    fx_client = fx_client.to(device)
    y = y.to(device)
    
    #---------forward prop-------------
    fx_server = net_server(fx_client)
    
    # calculate loss
    loss = criterion(fx_server, y)
    # calculate accuracy
    acc = calculate_accuracy(fx_server, y)
    
    #--------backward prop--------------
    loss.backward()
    dfx_client = fx_client.grad.clone().detach()
    optimizer_server.step()
    
    batch_loss_train.append(loss.item())
    batch_acc_train.append(acc.item())
    
    # Update the server-side model for the current batch
    net_model_server[idx] = copy.deepcopy(net_server)
    
    # count1: to track the completion of the local batch associated with one client
    count1 += 1
    if count1 == len_batch:
        acc_avg_train = sum(batch_acc_train)/len(batch_acc_train)           # it has accuracy for one batch
        loss_avg_train = sum(batch_loss_train)/len(batch_loss_train)
        
        batch_acc_train = []
        batch_loss_train = []
        count1 = 0
        
        prRed('Client{} Train => Local Epoch: {} \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, l_epoch_count, acc_avg_train, loss_avg_train))
        
        # copy the last trained model in the batch       
        w_server = net_server.state_dict()      
        
        # If one local epoch is completed, after this a new client will come
        if l_epoch_count == l_epoch-1:
            
            # l_epoch_check = True                # to evaluate_server function - to check local epoch has completed or not 
            # We store the state of the net_glob_server() 
            server_weights_by_id[idx] = copy.deepcopy(w_server)
            
            # we store the last accuracy in the last batch of the epoch and it is not the average of all local epochs
            # this is because we work on the last trained model and its accuracy (not earlier cases)
            
            #print("accuracy = ", acc_avg_train)
            acc_avg_train_all = acc_avg_train
            loss_avg_train_all = loss_avg_train
                        
            # accumulate accuracy and loss for each new user
            loss_train_collect_user.append(loss_avg_train_all)
            acc_train_collect_user.append(acc_avg_train_all)
            
            # collect the id of each new user                        
            if idx not in idx_collect:
                idx_collect.append(idx) 
                #print(idx_collect)
        
        # This is for federation process--------------------
        if len(idx_collect) == CONFIG.num_active_clients:
        
        # All clients have completed training for this global round.
        # Do NOT perform FedAvg here.
        # Aggregation will happen later after EMD-only clustering.

            acc_avg_all_user_train = (
                sum(acc_train_collect_user) /
                len(acc_train_collect_user)
            )
            loss_avg_all_user_train = (
                sum(loss_train_collect_user) /
                len(loss_train_collect_user)
            )
            loss_train_collect.append(loss_avg_all_user_train)
            acc_train_collect.append(acc_avg_all_user_train)
            
            acc_train_collect_user = []
            loss_train_collect_user = []
            
            idx_collect = []
            
    # send gradients to the client               
    return dfx_client


#==============================================================================================================
#                                       Clients-side Program
#==============================================================================================================
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

# Client-side functions associated with Training and Testing
class Client(object):
    def __init__(self, net_client_model, idx, lr, device, dataset_train = None, dataset_test = None, idxs = None, idxs_test = None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = CONFIG.local_epochs
        #self.selected_clients = []
        client_batch_size = min(CONFIG.batch_size, len(idxs))
        self.ldr_train = DataLoader(
            DatasetSplit(dataset_train, idxs),
            batch_size=client_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
    #     self.ldr_test = DataLoader(
    #     DatasetSplit(dataset_test, idxs_test),
    #     batch_size=128,
    #     shuffle=False,
    #     num_workers=8,
    #     pin_memory=True,
    #     persistent_workers=True
    # )
        

    
        

    def train(self, net, smashed_dir=None):
        net.train()
        optimizer_client = torch.optim.Adam(net.parameters(), lr = self.lr) 
        
        for iter in range(self.local_ep):
            len_batch = len(self.ldr_train)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                #---------forward prop-------------
                fx = net(images)
                client_fx = fx.clone().detach().requires_grad_(True)
                # Save paired smashed features and labels for EMD/FedWaD.
                with torch.no_grad():
                    _fx = fx.detach().cpu()
                    _labels = labels.detach().cpu().long()

                    if CONFIG.model == "charlstm":
                        _fx = _fx[:, -1, :]
                    else:
                        if _fx.dim() > 2:
                            _fx = F.adaptive_avg_pool2d(
                                _fx,
                                output_size=(
                                    DISTANCE_POOL_SIZE,
                                    DISTANCE_POOL_SIZE
                                ),
                            )
                        _fx = _fx.flatten(1)

                    should_save = (
                        CONFIG.model != "charlstm"
                        or batch_idx == 0
                    )

                    if smashed_dir is not None and should_save:
                        os.makedirs(smashed_dir, exist_ok=True)
                        torch.save(
                            {"features": _fx, "labels": _labels},
                            os.path.join(
                                smashed_dir,
                                f"client_{self.idx}_batch_{batch_idx}.pt"
                            )
                        )

                # --- End save ---

                
                # Sending activations to server and receiving gradients from server
                dfx = train_server(client_fx, labels, iter, self.local_ep, self.idx, len_batch)
                
                #--------backward prop -------------
                fx.backward(dfx)
                optimizer_client.step()
                            
            
            #prRed('Client{} Train => Epoch: {}'.format(self.idx, ell))
           
        return net.state_dict() 

# ============================================================
# Global evaluation after cluster-wise server FedAvg
# ============================================================

def evaluate_global_model(
    global_client_model,
    global_server_model,
    dataset_test
):
    test_loader = DataLoader(
        dataset_test,
        batch_size=CONFIG.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    criterion_eval = nn.CrossEntropyLoss()

    global_client_model.eval()
    global_server_model.eval()

    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Client-side global model
            smashed = global_client_model(images)

            # Cluster-aggregated global server model
            outputs = global_server_model(smashed)

            loss = criterion_eval(
                outputs,
                labels
            )

            predictions = outputs.argmax(
                dim=1
            )

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

            total_loss += (
                loss.item()
                * labels.size(0)
            )

    accuracy = (
        100.0
        * total_correct
        / total_samples
    )

    average_loss = (
        total_loss
        / total_samples
    )

    return accuracy, average_loss


#=====================================================================================================
# dataset_iid() will create a dictionary to collect the indices of the data samples randomly for each client
# IID HAM10000 datasets will be created based on this
# def dataset_iid(dataset, num_users):
    
#     num_items = int(len(dataset)/num_users)
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items, replace = False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users    


def dataset_noniid_dirichlet(
    labels,
    num_users,
    alpha,
    seed=42,
    min_samples_per_client=10,
    max_attempts=1000
):
    labels = np.asarray(labels, dtype=np.int64)
    class_ids = np.unique(labels)
    rng = np.random.default_rng(seed)

    for attempt in range(max_attempts):
        dict_users = {
            user_id: []
            for user_id in range(num_users)
        }

        for class_id in class_ids:
            class_indices = np.where(
                labels == class_id
            )[0]

            rng.shuffle(class_indices)

            proportions = rng.dirichlet(
                np.full(
                    num_users,
                    alpha,
                    dtype=np.float64
                )
            )

            split_points = (
                np.cumsum(proportions)[:-1]
                * len(class_indices)
            ).astype(int)

            class_splits = np.split(
                class_indices,
                split_points
            )

            for user_id, indices in enumerate(class_splits):
                dict_users[user_id].extend(
                    indices.tolist()
                )

        client_sizes = [
            len(dict_users[user_id])
            for user_id in range(num_users)
        ]

        if min(client_sizes) >= min_samples_per_client:
            break
    else:
        raise RuntimeError(
            "Unable to create a valid Dirichlet partition "
            f"after {max_attempts} attempts. "
            "Increase alpha or reduce min_samples_per_client."
        )

    for user_id in range(num_users):
        rng.shuffle(dict_users[user_id])
        dict_users[user_id] = set(dict_users[user_id])

    return dict_users
                          
#=============================================================================
#                         Data loading 
#============================================================================= 
# =============================================================================
# CIFAR-10 data loading — complete dataset
# =============================================================================

dataset_train, dataset_test, train_labels, test_labels = load_configured_datasets(CONFIG)

print(f"{CONFIG.dataset} training samples:", len(dataset_train))
print(f"{CONFIG.dataset} test samples:", len(dataset_test))

print(
    "Training class counts:",
    np.bincount(train_labels, minlength=CONFIG.num_classes).tolist()
)

print(
    "Test class counts:",
    np.bincount(test_labels, minlength=CONFIG.num_classes).tolist()
)

# =============================================================================
# Highly Non-IID CIFAR-10 partition
# =============================================================================



dict_users = dataset_noniid_dirichlet(
    labels=train_labels,
    num_users=num_users,
    alpha=alpha,
    seed=SEED,
    min_samples_per_client=10,
    max_attempts=1000
)

# Identical complete test set for every client
all_test_indices = set(range(len(dataset_test)))

dict_users_test = {
    user_id: all_test_indices.copy()
    for user_id in range(num_users)
}

valid_users = [
    user_id
    for user_id in range(num_users)
    if len(dict_users[user_id]) > 0
]

print(
    f"\nUsing complete {CONFIG.dataset} dataset with "
    f"Dirichlet Non-IID partition: "
    f"alpha={alpha}, seed={SEED}"
)

print("Valid clients:", valid_users)
print(
    "Identical test samples per client:",
    len(all_test_indices)
)

print(
    "\n========== Non-IID client distributions =========="
)

client_distribution_summary = {}

for user_id in range(num_users):
    client_indices = sorted(dict_users[user_id])
    client_labels = train_labels[client_indices]

    class_counts = np.bincount(
        client_labels,
        minlength=CONFIG.num_classes
    )

    client_distribution_summary[str(user_id)] = {
        "num_samples": len(client_indices),
        "class_counts": class_counts.tolist(),
        "indices": client_indices
    }

    print(
        f"Client {user_id:2d} | "
        f"samples={len(client_indices):5d} | "
        f"class counts={class_counts.tolist()}"
    )

total_partitioned_samples = sum(
    len(dict_users[user_id])
    for user_id in range(num_users)
)

print(
    "Total partitioned training samples:",
    total_partitioned_samples
)

assert total_partitioned_samples == len(dataset_train), (
    "Partition error: not all training samples "
    "were assigned."
)

all_assigned_indices = [
    sample_index
    for user_id in range(num_users)
    for sample_index in dict_users[user_id]
]

assert len(all_assigned_indices) == len(dataset_train), (
    "Partition error: incorrect total number of assigned samples."
)

assert len(set(all_assigned_indices)) == len(dataset_train), (
    "Partition error: duplicate or missing training indices detected."
)

assert set(all_assigned_indices) == set(range(len(dataset_train))), (
    "Partition error: assigned indices do not match the full "
    "CIFAR-10 training dataset."
)


#------------ Training And Testing  -----------------
net_glob_client.train()
#copy weights
w_glob_client = net_glob_client.state_dict()

# ============================================================
# TIMING SETUP
# ============================================================

round_times = []
cumulative_times = []

sync_cuda()
experiment_start_time = time.perf_counter()

# Persistent client-side models.
# No FedAvg is performed on the client side.
# client_models_by_id = {
#     client_id: copy.deepcopy(net_glob_client).to(device)
#     for client_id in valid_users
# }
# Federation takes place after certain local epochs in train() client-side
# this epoch is global epoch, also known as rounds
for global_round in range(epochs):

    sync_cuda()
    round_start_time = time.perf_counter()

    print(
        f"\n========== Global Round "
        f"{global_round + 1}/{epochs} =========="
    )

    smashed_dir, out_dir, distrib_dir = make_round_dirs(global_round)
    
    m = min(
        max(int(frac * num_users), 1),
        len(valid_users)
        )
    if m == len(valid_users):
        idxs_users = np.array(sorted(valid_users), dtype=np.int64)
    else:
        round_rng = np.random.default_rng(SEED + global_round)
        idxs_users = round_rng.choice(
            valid_users,
            size=m,
            replace=False
        )

    w_locals_client = []
    
    # client_weights_by_id = {}
    # w_locals_client = []

    for idx in idxs_users:
        local = Client(
            net_glob_client,
            idx,
            lr,
            device,
            dataset_train=dataset_train,
            dataset_test=dataset_test,
            idxs=dict_users[idx],
            idxs_test=dict_users_test[idx]
        )
        
        w_client = local.train(
            net=copy.deepcopy(
                net_glob_client
                ).to(device),
                smashed_dir=smashed_dir
        )
        
        w_locals_client.append(
            copy.deepcopy(w_client)
        )

        # # Start from this client's own model from the previous round.
        # client_model = copy.deepcopy(
        #     client_models_by_id[idx]
        # ).to(device)
        # w_client = local.train(
        #     net=client_model,
        #     smashed_dir=smashed_dir
        # )
        # # Keep the updated client-side model for the next round.
        # client_models_by_id[idx].load_state_dict(
        #     copy.deepcopy(w_client)
        # )

        # # local.evaluate(
        # #     net=copy.deepcopy(net_glob_client).to(device),
        # #     ell=global_round
        # # )

    print("-----------------------------------------------------------")
    print(
        f"------ Round-wise {DISTANCE_METHOD.upper()} Only "
        "+ Cluster FedAvg ------"
    )
    print("-----------------------------------------------------------")  
    
    # 1. Build one smashed-data distribution per client
    build_distrib_for_round(
        smashed_dir,
        distrib_dir
    )
    
    distribution_files = sorted(
        glob(
            os.path.join(
                distrib_dir,
                "client*.pt"
            )
        )
    )
    
    if len(distribution_files) != len(idxs_users):
        raise RuntimeError(
            "Smashed distribution count mismatch. "
            f"Expected {len(idxs_users)}, "
            f"found {len(distribution_files)}."
        )

    # 2. Calculate the selected pairwise client distance. Everything after
    # this point is shared by the baseline and FedWaD experiments.
    if DISTANCE_METHOD == "fedwad":
        client_distance, distance_client_ids = compute_fedwad_for_round(
            distrib_dir,
            out_dir,
        )
    else:
        client_distance, distance_client_ids = compute_emd_for_round(
            distrib_dir,
            out_dir,
        )

    if distance_client_ids != sorted(int(i) for i in idxs_users):
        raise RuntimeError(
            "Distance matrix client order does not match selected clients: "
            f"{distance_client_ids} vs {sorted(int(i) for i in idxs_users)}"
        )

    # 3. EMD-ONLY clustering
    # Directly use the EMD distance matrix.
    # No Ollivier-Ricci Curvature is calculated.
    labels, cluster_groups = cluster_clients_from_distance(
        client_distance,
        distance_client_ids,
        out_dir,
        n_clusters=CONFIG.n_clusters
    )

    print(
        f"Round {global_round + 1} cluster groups:",
        cluster_groups
    )

    # =========================================================
    # CLUSTER-WISE FEDAVG — SERVER SIDE ONLY
    # =========================================================

    expected_client_ids = set(
        int(client_id)
        for client_id in idxs_users
    )
    
    saved_server_ids = set(
        int(client_id)
        for client_id in server_weights_by_id.keys()
    )
    
    if saved_server_ids != expected_client_ids:
        raise RuntimeError(
            "Server model collection mismatch. "
            f"Expected {sorted(expected_client_ids)}, "
            f"found {sorted(saved_server_ids)}."
    )

    w_glob_server, cluster_models_server = clusterwise_fedavg(
        server_weights_by_id,
        cluster_groups
    )

    # Update global server model
    net_glob_server.load_state_dict(w_glob_server)

    # Use this global server model in next round
    net_model_server = [
        copy.deepcopy(net_glob_server)
        for _ in range(num_users)
    ]

    # =========================================================
    # ORIGINAL SFLV1 CLIENT-SIDE FEDAVG
    # =========================================================

    w_glob_client = FedAvg(
        w_locals_client
    )
    net_glob_client.load_state_dict(
        w_glob_client
    )

    # =========================================================
    # Evaluate the final model produced by this round
    # =========================================================

    round_test_accuracy, round_test_loss = evaluate_global_model(
        net_glob_client,
        net_glob_server,
        dataset_test
    )
    acc_test_collect.append(round_test_accuracy)
    loss_test_collect.append(round_test_loss)
    
    print("==========================================================")
    print(
        f"Round {global_round + 1:3d} | "
        f"Train Accuracy: {acc_avg_all_user_train:.3f}% | "
        f"Train Loss: {loss_avg_all_user_train:.4f}"
    )
    print(
        f"Round {global_round + 1:3d} | "
        f"Test Accuracy:  {round_test_accuracy:.3f}% | "
        f"Test Loss: {round_test_loss:.4f}"
    )
    print("==========================================================")

    # Save each cluster model
    for cluster_id, cluster_model in enumerate(cluster_models_server):
        torch.save(
            cluster_model,
            os.path.join(
                out_dir,
                f"server_cluster_model_{cluster_id}.pth"
            )
        )

    # Save final global server model
    torch.save(
        w_glob_server,
        os.path.join(
            out_dir,
            f"server_global_model_round_{global_round + 1:03d}.pth"
        )
    )

    print(
        f"Round {global_round + 1}: "
        "cluster-wise server FedAvg completed."
    )

    # Reset only after aggregation
    server_weights_by_id = {}

    # ========================================================
    # ROUND TIMING
    # ========================================================

    sync_cuda()
    round_end_time = time.perf_counter()

    round_elapsed = round_end_time - round_start_time
    cumulative_elapsed = round_end_time - experiment_start_time

    round_times.append(round_elapsed)
    cumulative_times.append(cumulative_elapsed)

    print("\n---------------- TIMING ----------------")
    print(
        f"Round {global_round + 1:3d} time: "
        f"{round_elapsed:.2f} sec "
        f"({round_elapsed / 60:.2f} min)"
    )

    print(
        f"Cumulative time after round {global_round + 1:3d}: "
        f"{cumulative_elapsed:.2f} sec "
        f"({cumulative_elapsed / 60:.2f} min)"
    )
    print("----------------------------------------")


#===================================================================================     

# ============================================================
# TOTAL EXPERIMENT TIMING
# ============================================================

sync_cuda()
experiment_end_time = time.perf_counter()

total_experiment_time = (
    experiment_end_time - experiment_start_time
)

print("\n==========================================================")
print(f"       {DISTANCE_METHOD.upper()} ONLY TIMING SUMMARY")
print("==========================================================")

print(
    f"Total time: "
    f"{total_experiment_time:.2f} sec"
)

print(
    f"Total time: "
    f"{total_experiment_time / 60:.2f} min"
)

print(
    f"Total time: "
    f"{total_experiment_time / 3600:.2f} hours"
)

print(
    f"Average round time: "
    f"{np.mean(round_times):.2f} sec"
)

print("==========================================================")

print("Training and Evaluation completed!") 

#===============================================================================
# Save output data to .excel file (we use for comparision plots)
assert len(acc_train_collect) == len(acc_test_collect), (
    f"Train/test metric length mismatch: "
    f"{len(acc_train_collect)} train vs "
    f"{len(acc_test_collect)} test"
)

assert len(acc_train_collect) == epochs, (
    f"Expected {epochs} rounds, "
    f"but collected {len(acc_train_collect)} train results"
)
round_process = list(range(1, epochs + 1))
df = DataFrame({
    'round': round_process,
    'acc_train': acc_train_collect,
    'acc_test': acc_test_collect,
    'round_time_sec': round_times,
    'cumulative_time_sec': cumulative_times,
    'distance_method': [DISTANCE_METHOD] * epochs,
    'fedwad_n_supp': [FEDWAD_N_SUPP if DISTANCE_METHOD == 'fedwad' else np.nan] * epochs,
    'fedwad_epochs': [FEDWAD_EPOCHS if DISTANCE_METHOD == 'fedwad' else np.nan] * epochs,
    'fedwad_t': [FEDWAD_T if DISTANCE_METHOD == 'fedwad' else np.nan] * epochs,
    'distance_pool_size': [DISTANCE_POOL_SIZE] * epochs,
    'alpha': [CONFIG.alpha] * epochs,
    'seed': [CONFIG.seed] * epochs,
    'num_users': [CONFIG.num_users] * epochs,
})
file_name = (
    f"{program}_"
    + (
        f"supp{FEDWAD_N_SUPP}_fwepochs{FEDWAD_EPOCHS}_"
        if DISTANCE_METHOD == "fedwad" else ""
    )
    + f"pool{DISTANCE_POOL_SIZE}_"
    f"alpha{CONFIG.alpha}_"
    f"clients{CONFIG.num_users}_"
    f"seed{CONFIG.seed}.xlsx"
) 
df.to_excel(file_name, sheet_name= "v1_test", index = False)     

#=============================================================================
#                         Program Completed
#=============================================================================
