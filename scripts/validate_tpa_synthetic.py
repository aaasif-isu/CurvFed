"""
Standalone CPU-only synthetic validation experiment for:
Topology-Aware Population Aggregation (TPA) via Voronoi-Wasserstein Projection.

Tests the hypothesis:
"Curvature-adjusted client-graph distances provide better representative-client
assignments than ordinary EMD, thereby reducing the error between a partially
sampled global update and the full-population update."
"""

import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment, linprog

try:
    import ot
    HAS_POT = True
except ImportError:
    HAS_POT = False

try:
    from GraphRicciCurvature.OllivierRicci import OllivierRicci
    HAS_GRC = True
except ImportError:
    HAS_GRC = False


def exact_wasserstein_lp(p: np.ndarray, q: np.ndarray, cost_matrix: np.ndarray) -> float:
    """Solves exact Monge-Kantorovich OT LP via scipy HiGHS solver."""
    n_p, n_q = len(p), len(q)
    c = cost_matrix.reshape(-1)
    A_eq = np.zeros((n_p + n_q, n_p * n_q), dtype=np.float64)
    for i in range(n_p):
        A_eq[i, i * n_q : (i + 1) * n_q] = 1.0
    for j in range(n_q):
        A_eq[n_p + j, j::n_q] = 1.0
    b_eq = np.concatenate([p, q])
    res = linprog(
        c, A_eq=A_eq[:-1], b_eq=b_eq[:-1], bounds=(0, None), method="highs"
    )
    if not res.success:
        raise RuntimeError(f"OT LP failed: {res.message}")
    return float(res.fun)


def exact_emd_uniform(X_u: np.ndarray, X_v: np.ndarray) -> float:
    """Exact 1-Wasserstein distance (EMD) between two uniform empirical clouds."""
    cost_matrix = cdist(X_u, X_v, metric="euclidean")
    n_u, n_v = X_u.shape[0], X_v.shape[0]
    if HAS_POT:
        w_u = np.full(n_u, 1.0 / n_u, dtype=np.float64)
        w_v = np.full(n_v, 1.0 / n_v, dtype=np.float64)
        return float(ot.emd2(w_u, w_v, cost_matrix))
    if n_u == n_v:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return float(cost_matrix[row_ind, col_ind].mean())
    return exact_wasserstein_lp(
        np.full(n_u, 1.0 / n_u), np.full(n_v, 1.0 / n_v), cost_matrix
    )


def compute_pairwise_emd_matrix(client_clouds: list) -> np.ndarray:
    """Computes M x M exact EMD matrix across all client empirical distributions."""
    M = len(client_clouds)
    emd = np.zeros((M, M), dtype=np.float64)
    for i in range(M):
        for j in range(i + 1, M):
            d = exact_emd_uniform(client_clouds[i], client_clouds[j])
            emd[i, j] = emd[j, i] = d
    return emd


def build_knn_graph_and_orc(emd: np.ndarray, k: int = 5, alpha: float = 0.5):
    """
    Constructs the k-NN client graph weighted by EMD and computes exact
    Ollivier-Ricci Curvature on every edge:
        kappa_{ij} = 1 - W_1(m_i^alpha, m_j^alpha; d_G) / d_G(i, j)
    """
    M = emd.shape[0]
    positives = emd[np.triu_indices(M, 1)]
    tau = float(np.median(positives[positives > 0])) if np.any(positives > 0) else 1.0
    sim = np.exp(-emd / max(tau, 1e-8))
    np.fill_diagonal(sim, 0.0)

    G = nx.Graph()
    for i in range(M):
        G.add_node(i)
        nbrs = np.argsort(sim[i])[::-1][:k]
        for j in nbrs:
            if i != j:
                G.add_edge(i, int(j), weight=float(emd[i, j]))

    if not nx.is_connected(G):
        complete_G = nx.Graph()
        for i in range(M):
            for j in range(i + 1, M):
                complete_G.add_edge(i, j, weight=float(emd[i, j]))
        mst = nx.minimum_spanning_tree(complete_G, weight="weight")
        for u, v, data in mst.edges(data=True):
            if not G.has_edge(u, v):
                G.add_edge(u, v, weight=float(data["weight"]))

    sp_dict = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    d_graph = np.zeros((M, M), dtype=np.float64)
    for i in range(M):
        for j, dist_val in sp_dict[i].items():
            d_graph[i, j] = float(dist_val)

    if HAS_GRC:
        ricci = OllivierRicci(G, weight="weight", alpha=alpha, method="OTD", verbose="ERROR")
        Gk = ricci.compute_ricci_curvature()
    else:
        Gk = G.copy()
        measures = {}
        for u in Gk.nodes():
            nbrs = list(Gk.neighbors(u))
            support = [u] + nbrs
            probs = np.zeros(len(support), dtype=np.float64)
            probs[0] = alpha
            if len(nbrs) > 0:
                edge_dists = np.array([Gk[u][v]["weight"] for v in nbrs], dtype=np.float64)
                w_nbr = np.exp(-edge_dists / max(tau, 1e-8))
                w_nbr /= w_nbr.sum()
                probs[1:] = (1.0 - alpha) * w_nbr
            measures[u] = (support, probs)

        for u, v in Gk.edges():
            supp_u, p_u = measures[u]
            supp_v, p_v = measures[v]
            ground_M = d_graph[np.ix_(supp_u, supp_v)]
            w1_uv = exact_wasserstein_lp(p_u, p_v, ground_M)
            d_uv = max(d_graph[u, v], 1e-12)
            Gk[u][v]["ricciCurvature"] = float(1.0 - (w1_uv / d_uv))

    for u, v, data in Gk.edges(data=True):
        kappa = float(data["ricciCurvature"])
        raw_w = float(data["weight"])
        Gk[u][v]["orc_scaled_weight"] = max(1e-8, 1.0 - kappa) * raw_w
        Gk[u][v]["orc_unscaled_weight"] = max(1e-8, 1.0 - kappa)

    sp_orc_scaled = dict(nx.all_pairs_dijkstra_path_length(Gk, weight="orc_scaled_weight"))
    sp_orc_unscaled = dict(nx.all_pairs_dijkstra_path_length(Gk, weight="orc_unscaled_weight"))

    d_orc_scaled = np.zeros((M, M), dtype=np.float64)
    d_orc_unscaled = np.zeros((M, M), dtype=np.float64)
    for i in range(M):
        for j in range(M):
            d_orc_scaled[i, j] = float(sp_orc_scaled[i][j])
            d_orc_unscaled[i, j] = float(sp_orc_unscaled[i][j])

    return Gk, d_graph, d_orc_scaled, d_orc_unscaled


def voronoi_population_weights(
    dist_matrix: np.ndarray, active_indices: np.ndarray, pop_weights: np.ndarray
) -> np.ndarray:
    """
    Computes Voronoi-Wasserstein representative weights w in Delta^{m-1}.
    Each population client u in {0..M-1} assigns its mass p_u to its nearest
    active client(s) in active_indices (with uniform tie-breaking).
    """
    M = dist_matrix.shape[0]
    m = len(active_indices)
    sub_dists = dist_matrix[:, active_indices]  # (M, m)
    min_dists = sub_dists.min(axis=1, keepdims=True)
    ties = np.isclose(sub_dists, min_dists, rtol=1e-12, atol=1e-12)
    assignment = ties.astype(np.float64) / ties.sum(axis=1, keepdims=True)
    weights = (pop_weights[:, None] * assignment).sum(axis=0)
    return weights / weights.sum()


def generate_synthetic_population(
    M: int = 100, n_samples: int = 24, feat_dim: int = 4, seed: int = 42
):
    """
    Generates M=100 synthetic client empirical distributions X_u in R^{n_samples x feat_dim}
    arranged in 4 imbalanced non-convex / bridged manifold communities of sizes [45, 25, 20, 10].
    Communities 0 and 1 have a narrow 'neck' (bridge) where boundary clients come closer in
    ambient Euclidean space than the diameter of each community, creating negative-ORC bridge edges.
    """
    rng = np.random.default_rng(seed)
    sizes = [45, 25, 20, 10]
    assert sum(sizes) == M

    client_clouds = []
    cluster_ids = np.zeros(M, dtype=np.int64)
    latent_coords = np.zeros((M, feat_dim), dtype=np.float64)

    idx = 0
    for c_id, sz in enumerate(sizes):
        # Parameterize each community along a curved arc in R^feat_dim
        t = np.linspace(-1.0, 1.0, sz)
        for loc_i in range(sz):
            s = t[loc_i]
            if c_id == 0:
                # Upper crescent
                center = np.array([2.2 * s, 1.2 * (s**2) + 0.0, 0.0, 0.0])
            elif c_id == 1:
                # Lower crescent (approaches c_id=0 near s=0 at distance ~1.35, while ends are >4.0 apart)
                center = np.array([2.2 * s, -1.2 * (s**2) - 1.35, 0.0, 0.0])
            elif c_id == 2:
                # Third cluster offset in dim 2
                center = np.array([1.8 * s, 2.5, 2.5 + 0.8 * (s**2), 0.0])
            else:
                # Fourth minority cluster offset in dim 3
                center = np.array([-2.0 + 1.2 * s, -2.5, -2.0, 2.5])

            jitter = rng.normal(0.0, 0.08, size=feat_dim)
            mu_u = center + jitter
            # Local client sample cloud around mu_u
            cloud = mu_u[None, :] + rng.normal(0.0, 0.35, size=(n_samples, feat_dim))
            client_clouds.append(cloud)
            cluster_ids[idx] = c_id
            latent_coords[idx] = mu_u
            idx += 1

    return client_clouds, cluster_ids, latent_coords


def generate_updates_for_setting(
    setting: str,
    cluster_ids: np.ndarray,
    latent_coords: np.ndarray,
    update_dim: int = 64,
    seed: int = 42,
) -> np.ndarray:
    """
    Generates client update vectors Delta_u in R^{update_dim} for M=100 clients
    under the three experimental settings.
    """
    rng = np.random.default_rng(seed)
    M = len(cluster_ids)
    feat_dim = latent_coords.shape[1]

    # Orthogonal cluster prototype directions in R^{update_dim}
    raw_protos = rng.normal(0.0, 1.0, size=(4, update_dim))
    q_protos, _ = np.linalg.qr(raw_protos.T)
    cluster_protos = q_protos.T[:4] * 5.0  # norm 5.0 per community mode
    W_smooth = rng.normal(0.0, 0.35, size=(feat_dim, update_dim)) / np.sqrt(feat_dim)

    updates = np.zeros((M, update_dim), dtype=np.float64)

    if setting == "aligned":
        # Setting 1: Client distributions meaningfully predict update similarity
        # Updates depend strongly on community identity + smooth intra-manifold position
        for u in range(M):
            c = cluster_ids[u]
            updates[u] = (
                cluster_protos[c]
                + latent_coords[u] @ W_smooth
                + rng.normal(0.0, 0.25, size=update_dim)
            )
    elif setting == "uninformative":
        # Setting 2: Client distributions have NO relationship to update similarity
        perm = rng.permutation(M)
        for u in range(M):
            c_rand = cluster_ids[perm[u]]
            updates[u] = (
                cluster_protos[c_rand]
                + rng.normal(0.0, 1.0, size=update_dim)
            )
    elif setting == "misleading":
        # Setting 3: Client distances are misleading about update similarity
        # Neighboring clients in the same feature cluster alternate signs (+1 / -1 concept shift),
        # while clients with matching parity across distant clusters share the same update direction.
        for u in range(M):
            sign = 1.0 if (u % 2 == 0) else -1.0
            c = cluster_ids[u]
            updates[u] = (
                sign * cluster_protos[0]
                + 0.5 * sign * cluster_protos[c]
                + rng.normal(0.0, 0.25, size=update_dim)
            )
    else:
        raise ValueError(f"Unknown setting: {setting}")

    return updates

def evaluate_methods_across_cohorts(
    updates: np.ndarray,
    pop_weights: np.ndarray,
    cohorts: np.ndarray,
    d_emd: np.ndarray,
    d_graph: np.ndarray,
    d_orc_scaled: np.ndarray,
    d_orc_unscaled: np.ndarray,
    cluster_ids: np.ndarray,
):
    true_pop_update = (pop_weights[:, None] * updates).sum(axis=0)
    true_norm = float(np.linalg.norm(true_pop_update))
    num_cohorts, _ = cohorts.shape
    _, update_dim = updates.shape

    method_names = [
        "1. FedAvg (Uniform)",
        "2. TPA-EMD (Voronoi)",
        "3. TPA-Graph (SP, no ORC)",
        "4a. TPA-ORC (Scaled SP)",
        "4b. TPA-ORC (Unscaled SP)",
    ]

    est_updates = {n: np.zeros((num_cohorts, update_dim), dtype=np.float64) for n in method_names}
    errors = {n: np.zeros(num_cohorts, dtype=np.float64) for n in method_names}
    purity_map = {n: [] for n in method_names[1:]}

    dist_map = {
        "2. TPA-EMD (Voronoi)": d_emd,
        "3. TPA-Graph (SP, no ORC)": d_graph,
        "4a. TPA-ORC (Scaled SP)": d_orc_scaled,
        "4b. TPA-ORC (Unscaled SP)": d_orc_unscaled,
    }

    for r in range(num_cohorts):
        active = cohorts[r]
        active_updates = updates[active]

        w_fed = pop_weights[active] / pop_weights[active].sum()
        est_fed = (w_fed[:, None] * active_updates).sum(axis=0)
        est_updates["1. FedAvg (Uniform)"][r] = est_fed
        errors["1. FedAvg (Uniform)"][r] = float(np.linalg.norm(est_fed - true_pop_update))

        for name, d_mat in dist_map.items():
            w_vor = voronoi_population_weights(d_mat, active, pop_weights)
            est = (w_vor[:, None] * active_updates).sum(axis=0)
            est_updates[name][r] = est
            errors[name][r] = float(np.linalg.norm(est - true_pop_update))

            nearest_pos = np.argmin(d_mat[:, active], axis=1)
            assigned_active = active[nearest_pos]
            purity_map[name].append(float(np.mean(cluster_ids == cluster_ids[assigned_active])))

    results = {}
    fed_err = errors["1. FedAvg (Uniform)"]
    emd_err = errors["2. TPA-EMD (Voronoi)"]
    graph_err = errors["3. TPA-Graph (SP, no ORC)"]

    for name in method_names:
        err = errors[name]
        mean_est = est_updates[name].mean(axis=0)
        bias_l2 = float(np.linalg.norm(mean_est - true_pop_update))
        q75, q25 = np.percentile(err, [75, 25])
        results[name] = {
            "mean_error": float(np.mean(err)),
            "rel_error_pct": float(100.0 * np.mean(err) / max(true_norm, 1e-12)),
            "std_error": float(np.std(err, ddof=1)),
            "iqr_error": float(q75 - q25),
            "bias_l2": bias_l2,
            "win_vs_fedavg_pct": float(100.0 * np.mean(err < fed_err)) if name != method_names[0] else 0.0,
            "win_vs_emd_pct": float(100.0 * np.mean(err < emd_err)) if name not in method_names[:2] else 0.0,
            "win_vs_graph_pct": float(100.0 * np.mean(err < graph_err)) if "ORC" in name else 0.0,
            "cluster_purity_pct": float(100.0 * np.mean(purity_map[name])) if name in purity_map else float("nan"),
        }
    return results, true_norm


def main():
    M, m, num_cohorts, seed = 100, 10, 1000, 42
    print("=" * 96)
    print("TOPOLOGY-AWARE POPULATION AGGREGATION (TPA) — CPU SYNTHETIC VALIDATION")
    print(f"Clients: M={M}, Active/round: m={m}, Cohorts: R={num_cohorts}, kNN k=5, ORC alpha=0.5")
    print("=" * 96)

    client_clouds, cluster_ids, latent_coords = generate_synthetic_population(M=M, seed=seed)
    pop_weights = np.full(M, 1.0 / M, dtype=np.float64)

    d_emd = compute_pairwise_emd_matrix(client_clouds)
    Gk, d_graph, d_orc_scaled, d_orc_unscaled = build_knn_graph_and_orc(d_emd, k=5, alpha=0.5)

    intra_k, inter_k = [], []
    for u, v, data in Gk.edges(data=True):
        (intra_k if cluster_ids[u] == cluster_ids[v] else inter_k).append(data["ricciCurvature"])

    print(f"\nGraph Curvature Diagnostics:")
    print(f"  Edges: {Gk.number_of_edges()} (Intra: {len(intra_k)}, Inter-cluster bridges: {len(inter_k)})")
    print(f"  Intra-cluster ORC kappa: mean={np.mean(intra_k):+.4f}, min={np.min(intra_k):+.4f}, max={np.max(intra_k):+.4f}")
    if inter_k:
        print(f"  Inter-cluster ORC kappa: mean={np.mean(inter_k):+.4f}, min={np.min(inter_k):+.4f}, max={np.max(inter_k):+.4f}")

    cohort_rng = np.random.default_rng(seed + 999)
    cohorts = np.array([np.sort(cohort_rng.choice(M, size=m, replace=False)) for _ in range(num_cohorts)])

    settings = [
        ("aligned", "Setting 1: Geometry-Aligned (Distributions Predict Update Similarity)"),
        ("uninformative", "Setting 2: Uninformative (Distributions Unrelated to Updates)"),
        ("misleading", "Setting 3: Misleading Geometry (Local Concept Shift / Opposite Updates)"),
    ]

    for s_key, s_title in settings:
        updates = generate_updates_for_setting(s_key, cluster_ids, latent_coords, update_dim=64, seed=seed + 100)
        results, true_norm = evaluate_methods_across_cohorts(
            updates, pop_weights, cohorts, d_emd, d_graph, d_orc_scaled, d_orc_unscaled, cluster_ids
        )
        print(f"\n{s_title}  (||Delta_pop||_2 = {true_norm:.4f})")
        print("-" * 116)
        print(
            f"{'Method':<27} | {'Mean Err':>8} | {'Rel%':>6} | {'Std Dev':>7} | "
            f"{'IQR':>7} | {'Bias L2':>7} | {'Beat FedAvg':>11} | {'Beat EMD':>8} | {'Beat Graph':>10} | {'Purity%':>7}"
        )
        print("-" * 116)
        for name, st in results.items():
            pur = f"{st['cluster_purity_pct']:6.2f}%" if not np.isnan(st["cluster_purity_pct"]) else "   N/A "
            print(
                f"{name:<27} | {st['mean_error']:8.4f} | {st['rel_error_pct']:5.2f}% | "
                f"{st['std_error']:7.4f} | {st['iqr_error']:7.4f} | {st['bias_l2']:7.4f} | "
                f"{st['win_vs_fedavg_pct']:10.1f}% | {st['win_vs_emd_pct']:7.1f}% | "
                f"{st['win_vs_graph_pct']:9.1f}% | {pur}"
            )


if __name__ == "__main__":
    main()


