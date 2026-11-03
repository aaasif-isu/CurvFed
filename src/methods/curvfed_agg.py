"""Geometric client distance, ORC graph construction, shortest-path Ricci metric,
noise-floor sanity check (Sanity Check A), and clustering utilities."""

import json
import os
import shutil
from glob import glob
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import ot
import torch
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from scipy.sparse.csgraph import floyd_warshall
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score


def _client_id_from_path(path: str) -> int:
    base = os.path.splitext(os.path.basename(path))[0]
    return int(base.replace("client_", "").replace("client", ""))


def make_round_dirs(global_round: int, experiment_root: str = "outputs_curvfed") -> Tuple[str, str, str]:
    round_name = f"round_{global_round + 1:03d}"
    smashed_dir = os.path.join(experiment_root, "smashed", round_name)
    out_dir = os.path.join(experiment_root, "outputs", round_name)
    distrib_dir = os.path.join(out_dir, "distrib")

    for directory in (smashed_dir, out_dir, distrib_dir):
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)

    return smashed_dir, out_dir, distrib_dir


def build_distrib_for_round(
    smashed_dir: str,
    distrib_dir: str,
    seed: int = 42,
    cap: int = 256,
    cleanup_smashed: bool = True,
) -> List[int]:
    os.makedirs(distrib_dir, exist_ok=True)
    client_batches: Dict[int, List[torch.Tensor]] = {}
    client_label_batches: Dict[int, List[torch.Tensor]] = {}

    for fname in sorted(os.listdir(smashed_dir)):
        if not (fname.startswith("client_") and fname.endswith(".pt")):
            continue
        parts = fname.replace(".pt", "").split("_")
        client_id = int(parts[1])
        x = torch.load(os.path.join(smashed_dir, fname), map_location="cpu").float()
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        client_batches.setdefault(client_id, []).append(x)

        lbl_fname = f"labels_{parts[1]}_batch_{parts[3]}.pt" if len(parts) >= 4 else ""
        lbl_path = os.path.join(smashed_dir, lbl_fname)
        if lbl_fname and os.path.exists(lbl_path):
            y = torch.load(lbl_path, map_location="cpu").long()
            client_label_batches.setdefault(client_id, []).append(y)

    for client_id, tensors in sorted(client_batches.items()):
        x_all = torch.cat(tensors, dim=0)
        y_list = client_label_batches.get(client_id, [])
        y_all = torch.cat(y_list, dim=0) if len(y_list) == len(tensors) and y_list else None
        if x_all.size(0) > cap:
            generator = torch.Generator().manual_seed(seed + int(client_id))
            selected = torch.randperm(x_all.size(0), generator=generator)[:cap]
            x_all = x_all[selected]
            if y_all is not None and y_all.size(0) >= selected.max().item() + 1:
                y_all = y_all[selected]
        torch.save(x_all, os.path.join(distrib_dir, f"client{client_id}.pt"))
        if y_all is not None:
            torch.save(y_all, os.path.join(distrib_dir, f"labels_client{client_id}.pt"))

    if cleanup_smashed and os.path.exists(smashed_dir):
        shutil.rmtree(smashed_dir, ignore_errors=True)

    return sorted(client_batches.keys())


def _standardize_cloud(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)


def _normalize_client_clouds(
    raw_clouds: List[np.ndarray], zscore_mode: str = "per_client"
) -> List[np.ndarray]:
    """Normalize client activation point clouds according to `zscore_mode` (Sanity Check C)."""
    zscore_mode = zscore_mode.lower()
    clouds = [np.asarray(X, dtype=np.float64) for X in raw_clouds]
    if zscore_mode == "per_client":
        return [_standardize_cloud(X) for X in clouds]
    elif zscore_mode == "pooled":
        pooled = np.concatenate(clouds, axis=0)
        mean_pool = pooled.mean(axis=0, keepdims=True)
        std_pool = pooled.std(axis=0, keepdims=True) + 1e-8
        return [(X - mean_pool) / std_pool for X in clouds]
    elif zscore_mode == "none":
        return clouds
    else:
        raise ValueError(f"Unsupported zscore_mode: {zscore_mode}")


def compute_emd_for_round(
    distrib_dir: str,
    out_dir: Optional[str] = None,
    zscore_mode: str = "per_client",
) -> np.ndarray:
    files = sorted(glob(os.path.join(distrib_dir, "client*.pt")), key=_client_id_from_path)
    if not files:
        raise RuntimeError(f"No client distribution files found in {distrib_dir}")

    raw_clouds = [torch.load(f, map_location="cpu").float().numpy() for f in files]
    Xs = _normalize_client_clouds(raw_clouds, zscore_mode=zscore_mode)

    k = len(Xs)
    emd = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        Xi, wi = Xs[i], ot.unif(len(Xs[i]))
        for j in range(i + 1, k):
            Xj, wj = Xs[j], ot.unif(len(Xs[j]))
            M = ot.dist(Xi, Xj, metric="euclidean")
            emd[i, j] = emd[j, i] = float(ot.emd2(wi, wj, M))

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "emd_matrix.npy"), emd)
        with open(os.path.join(out_dir, "emd_matrix_readable.txt"), "w") as f:
            f.write(f"Shape: {emd.shape}\n\n")
            for row in emd:
                f.write(" ".join([f"{v:.4f}" for v in row]) + "\n")

    return emd


def compute_emd_noise_floor(
    distrib_dir: str,
    emd_matrix: Optional[np.ndarray] = None,
    seed: int = 42,
    zscore_mode: str = "per_client",
) -> Dict[str, object]:
    """Sanity Check A (Fair Equal-Sample-Size):
    Split each client i into disjoint halves A_i and B_i of the SAME sample size h;
    compute within = EMD(A_i, B_i) and cross = EMD(A_i, A_j) for i < j."""
    files = sorted(glob(os.path.join(distrib_dir, "client*.pt")), key=_client_id_from_path)
    if not files:
        raise RuntimeError(f"No client distribution files found in {distrib_dir}")

    client_ids = [_client_id_from_path(f) for f in files]
    raw_clouds = [torch.load(f, map_location="cpu").float().numpy() for f in files]
    Xs = _normalize_client_clouds(raw_clouds, zscore_mode=zscore_mode)

    half_sizes = [X.shape[0] // 2 for X in Xs]
    common_half = min(half_sizes) if half_sizes else 0

    halves_A: List[np.ndarray] = []
    halves_B: List[np.ndarray] = []
    within_emds: Dict[int, float] = {}

    for cid, X in zip(client_ids, Xs):
        if common_half < 1:
            within_emds[cid] = 0.0
            halves_A.append(X)
            halves_B.append(X)
            continue
        rng = np.random.default_rng(seed + int(cid) * 101)
        perm = rng.permutation(X.shape[0])
        A_i = X[perm[:common_half]]
        B_i = X[perm[common_half : 2 * common_half]]
        halves_A.append(A_i)
        halves_B.append(B_i)

        w_h = ot.unif(common_half)
        M_ab = ot.dist(A_i, B_i, metric="euclidean")
        within_emds[cid] = float(ot.emd2(w_h, w_h, M_ab))

    k = len(halves_A)
    cross_vals: List[float] = []
    if common_half >= 1:
        w_h = ot.unif(common_half)
        for i in range(k):
            for j in range(i + 1, k):
                M_ij = ot.dist(halves_A[i], halves_A[j], metric="euclidean")
                cross_vals.append(float(ot.emd2(w_h, w_h, M_ij)))

    within_vals = list(within_emds.values())
    within_mean = float(np.mean(within_vals)) if within_vals else 0.0
    within_std = float(np.std(within_vals)) if within_vals else 0.0
    cross_mean = float(np.mean(cross_vals)) if cross_vals else 0.0
    cross_std = float(np.std(cross_vals)) if cross_vals else 0.0
    snr = cross_mean / max(within_mean, 1e-12)

    print(
        f"---------------- SANITY CHECK A: EMD NOISE FLOOR (zscore={zscore_mode}, half_n={common_half}) ----------------"
    )
    print(
        f"  Within-client EMD(A_i, B_i): mean={within_mean:.4f}, std={within_std:.4f}, "
        f"min={min(within_vals):.4f}, max={max(within_vals):.4f} | per-client={within_emds}"
    )
    print(
        f"  Cross-client  EMD(A_i, A_j): mean={cross_mean:.4f}, std={cross_std:.4f}, "
        f"min={min(cross_vals):.4f}, max={max(cross_vals):.4f}"
    )
    print(f"  EMD Signal-to-Noise Ratio (cross / within): {snr:.4f}x")
    print("------------------------------------------------------------------------------------------------------")

    return {
        "sample_size_half": common_half,
        "within_client_emd": within_emds,
        "within_mean": within_mean,
        "within_std": within_std,
        "within_min": float(min(within_vals)) if within_vals else 0.0,
        "within_max": float(max(within_vals)) if within_vals else 0.0,
        "cross_client_values": cross_vals,
        "cross_mean": cross_mean,
        "cross_std": cross_std,
        "cross_min": float(min(cross_vals)) if cross_vals else 0.0,
        "cross_max": float(max(cross_vals)) if cross_vals else 0.0,
        "snr": snr,
    }


def compute_ricci_for_round(
    emd: np.ndarray,
    out_dir: Optional[str] = None,
    k: Optional[int] = None,
    alpha: float = 0.5,
) -> nx.Graph:
    """Build k-NN client graph with `weight = emd / tau` and compute Ollivier-Ricci curvature."""
    emd = np.asarray(emd, dtype=np.float64)
    n = emd.shape[0]
    positives = emd[np.triu_indices(n, 1)]
    tau = float(np.median(positives[positives > 0])) if np.any(positives > 0) else 1.0
    tau = max(tau, 1e-8)

    scaled_emd = emd / tau
    sim = np.exp(-scaled_emd)
    np.fill_diagonal(sim, 0.0)

    G = nx.Graph()
    k_neighbors = min(3, n - 1) if k is None else min(int(k), n - 1)

    for i in range(n):
        G.add_node(i)
        nbrs = np.argsort(sim[i])[::-1][:k_neighbors]
        for j in nbrs:
            j_idx = int(j)
            if i != j_idx:
                G.add_edge(
                    i,
                    j_idx,
                    weight=float(scaled_emd[i, j_idx]),
                    similarity=float(sim[i, j_idx]),
                    distance=float(emd[i, j_idx]),
                )

    if n > 1 and not nx.is_connected(G):
        complete_G = nx.Graph()
        for i in range(n):
            for j in range(i + 1, n):
                complete_G.add_edge(
                    i,
                    j,
                    weight=float(scaled_emd[i, j]),
                    similarity=float(sim[i, j]),
                    distance=float(emd[i, j]),
                )
        for u, v, data in nx.minimum_spanning_edges(complete_G, weight="weight", data=True):
            if not G.has_edge(u, v):
                G.add_edge(u, v, **data)

    ricci = OllivierRicci(G, weight="weight", alpha=alpha, method="OTD", proc=1, verbose="ERROR")
    Gk = ricci.compute_ricci_curvature()

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        edge_kappa = {
            f"{u}-{v}": float(Gk[u][v].get("ricciCurvature", 0.0))
            for u, v in Gk.edges()
        }
        with open(os.path.join(out_dir, "ricci_edges.json"), "w") as f:
            json.dump(edge_kappa, f, indent=2)

    return Gk


def compute_ricci_distance_matrix(Gk: nx.Graph) -> np.ndarray:
    """Compute shortest-path completed Ricci distance matrix on `Gk`.

    The resulting matrix is symmetric, zero on the diagonal, and satisfies the
    triangle inequality `D[i, k] <= D[i, j] + D[j, k]` for all i, j, k.
    """
    nodes = list(sorted(Gk.nodes()))
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)

    raw_dist = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(raw_dist, 0.0)

    for u, v, data in Gk.edges(data=True):
        iu, iv = node_to_idx[u], node_to_idx[v]
        curvature = float(data.get("ricciCurvature", 0.0))
        distance = max(0.0, 1.0 - curvature)
        raw_dist[iu, iv] = distance
        raw_dist[iv, iu] = distance

    ricci_distance = floyd_warshall(raw_dist, directed=False)
    if not np.all(np.isfinite(ricci_distance)):
        finite_vals = raw_dist[np.isfinite(raw_dist) & (raw_dist > 0)]
        bridge_val = float(np.max(finite_vals)) if finite_vals.size > 0 else 1.0
        raw_dist[~np.isfinite(raw_dist)] = max(1.0, bridge_val)
        ricci_distance = floyd_warshall(raw_dist, directed=False)

    ricci_distance = 0.5 * (ricci_distance + ricci_distance.T)
    np.fill_diagonal(ricci_distance, 0.0)
    return ricci_distance


def cluster_clients_from_ricci(
    Gk: nx.Graph,
    out_dir: Optional[str] = None,
    n_clusters: int = 3,
    client_ids: Optional[List[int]] = None,
) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    n = Gk.number_of_nodes()
    if client_ids is None:
        client_ids = list(range(n))
    if len(client_ids) != n:
        raise RuntimeError(f"Expected {n} client IDs for the Ricci graph, found {len(client_ids)}.")

    ricci_distance = compute_ricci_distance_matrix(Gk)
    effective_clusters = min(n_clusters, n)

    try:
        model = AgglomerativeClustering(
            n_clusters=effective_clusters,
            metric="precomputed",
            linkage="average",
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=effective_clusters,
            affinity="precomputed",
            linkage="average",
        )

    labels = model.fit_predict(ricci_distance)
    cluster_groups: Dict[int, List[int]] = {}
    for cid, cluster_id in zip(client_ids, labels):
        cluster_groups.setdefault(int(cluster_id), []).append(int(cid))

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "cluster_summary.txt"), "w") as f:
            f.write("Ricci-based client clustering\n\n")
            for cid, cluster_id in zip(client_ids, labels):
                f.write(f"Client {cid} -> Cluster {cluster_id}\n")
            f.write("\nCluster groups\n")
            for cluster_id, clients in sorted(cluster_groups.items()):
                f.write(f"Cluster {cluster_id}: {clients}\n")
        np.savetxt(os.path.join(out_dir, "client_clusters.txt"), labels, fmt="%d")

    return labels, cluster_groups


def cluster_clients_from_emd(
    emd: np.ndarray,
    out_dir: Optional[str] = None,
    n_clusters: int = 3,
    client_ids: Optional[List[int]] = None,
) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    emd = np.asarray(emd, dtype=np.float64)
    n = emd.shape[0]
    if client_ids is None:
        client_ids = list(range(n))
    effective_clusters = min(n_clusters, n)

    try:
        model = AgglomerativeClustering(
            n_clusters=effective_clusters,
            metric="precomputed",
            linkage="average",
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=effective_clusters,
            affinity="precomputed",
            linkage="average",
        )

    labels = model.fit_predict(emd)
    cluster_groups: Dict[int, List[int]] = {}
    for cid, cluster_id in zip(client_ids, labels):
        cluster_groups.setdefault(int(cluster_id), []).append(int(cid))

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "cluster_summary.txt"), "w") as f:
            f.write("EMD-only client clustering\n\n")
            for cid, cluster_id in zip(client_ids, labels):
                f.write(f"Client {cid} -> Cluster {cluster_id}\n")
            f.write("\nCluster groups\n")
            for cluster_id, clients in sorted(cluster_groups.items()):
                f.write(f"Cluster {cluster_id}: {clients}\n")
        np.savetxt(os.path.join(out_dir, "client_clusters.txt"), labels, fmt="%d")

    return labels, cluster_groups


def apply_cluster_mode(
    cluster_groups: Dict[int, List[int]],
    cluster_mode: str = "geometry",
    seed: int = 42,
) -> Dict[int, List[int]]:
    """Sanity Check B: Either preserve geometric clusters ('geometry') or randomly permute
    clients into clusters of the exact same sizes ('random')."""
    cluster_mode = cluster_mode.lower()
    if cluster_mode == "geometry":
        return {int(k): [int(x) for x in v] for k, v in sorted(cluster_groups.items())}
    if cluster_mode != "random":
        raise ValueError(f"Unsupported cluster_mode: {cluster_mode}")

    sorted_keys = sorted(cluster_groups.keys())
    sizes = [len(cluster_groups[k]) for k in sorted_keys]
    all_clients = sorted(int(cid) for k in sorted_keys for cid in cluster_groups[k])

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(all_clients).tolist()

    random_groups: Dict[int, List[int]] = {}
    cursor = 0
    for k, sz in zip(sorted_keys, sizes):
        random_groups[int(k)] = sorted(shuffled[cursor : cursor + sz])
        cursor += sz
    return random_groups


def compute_cluster_ari(
    prev_cluster_groups: Optional[Dict[int, List[int]]],
    curr_cluster_groups: Optional[Dict[int, List[int]]],
) -> float:
    """Sanity Check D: Compute Adjusted Rand Index (ARI) between two clusterings over common clients."""
    if not prev_cluster_groups or not curr_cluster_groups:
        return float("nan")

    prev_map = {int(cid): int(k) for k, cids in prev_cluster_groups.items() for cid in cids}
    curr_map = {int(cid): int(k) for k, cids in curr_cluster_groups.items() for cid in cids}
    common_clients = sorted(set(prev_map.keys()).intersection(curr_map.keys()))
    if len(common_clients) < 2:
        return float("nan")

    prev_labels = [prev_map[cid] for cid in common_clients]
    curr_labels = [curr_map[cid] for cid in common_clients]
    return float(adjusted_rand_score(prev_labels, curr_labels))


def _augment_for_fedotdd(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Diagonal class-conditional OTDD feature augmentation for FedWaD."""
    augmented = np.empty((x.shape[0], 3 * x.shape[1]), dtype=np.float64)
    for class_id in np.unique(y):
        mask = y == class_id
        class_x = x[mask]
        mean = class_x.mean(axis=0)
        variance = class_x.var(axis=0, ddof=1) if class_x.shape[0] > 1 else np.zeros(x.shape[1], dtype=np.float64)
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


def compute_fedwad_for_round(
    distrib_dir: str,
    out_dir: Optional[str] = None,
    zscore_mode: str = "per_client",
    seed: int = 42,
    n_supp: int = 10,
    n_epoch: int = 20,
    t_val: float = 0.5,
) -> np.ndarray:
    """Compute pairwise FedWaD (FedOTDD interpolating measure) distance matrix across active clients."""
    from FedWaD import FedOT

    files = sorted(glob(os.path.join(distrib_dir, "client*.pt")), key=_client_id_from_path)
    if not files:
        raise RuntimeError(f"No client distribution files found in {distrib_dir}")

    client_ids = [_client_id_from_path(f) for f in files]
    raw_clouds = [torch.load(f, map_location="cpu").float().numpy() for f in files]
    Xs = _normalize_client_clouds(raw_clouds, zscore_mode=zscore_mode)

    augmented = []
    for cid, x in zip(client_ids, Xs):
        lbl_path = os.path.join(distrib_dir, f"labels_client{cid}.pt")
        if os.path.exists(lbl_path):
            y = torch.load(lbl_path, map_location="cpu").long().numpy()
            augmented.append(_augment_for_fedotdd(x, y))
        else:
            augmented.append(x)

    k = len(augmented)
    distances = np.zeros((k, k), dtype=np.float64)
    numpy_state = np.random.get_state()
    try:
        for i in range(k):
            for j in range(i + 1, k):
                np.random.seed(seed + i * k + j)
                fedwad = FedOT(
                    n_supp=n_supp,
                    n_epoch=n_epoch,
                    t_val=t_val,
                    verbose=False,
                    metric="sqeuclidean",
                )
                fedwad.fit(augmented[i], augmented[j], approx_interp=True)
                dist_val = float(fedwad.cost ** 2)
                distances[i, j] = distances[j, i] = dist_val
    finally:
        np.random.set_state(numpy_state)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f"fedwad_matrix_supp{n_supp}_epochs{n_epoch}.npy"), distances)

    return distances




