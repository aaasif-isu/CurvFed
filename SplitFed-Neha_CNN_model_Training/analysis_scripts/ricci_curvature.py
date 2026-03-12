import os, json
import numpy as np
import networkx as nx

# GraphRicciCurvature API
try:
    from GraphRicciCurvature.OllivierRicci import OllivierRicci
except Exception as e:
    raise SystemExit(
        "GraphRicciCurvature is not installed correctly. "
        "Install with:  python3 -m pip install GraphRicciCurvature"
    ) from e

HERE = os.path.dirname(__file__)
emd_path = os.path.join(HERE, "emd_matrix.npy")
if not os.path.exists(emd_path):
    raise FileNotFoundError(f"Missing {emd_path}. Run compute_emd_matrix.py first.")

emd = np.load(emd_path)
if emd.ndim != 2 or emd.shape[0] != emd.shape[1]:
    raise ValueError(f"emd_matrix.npy must be a square matrix, got {emd.shape}")

n = emd.shape[0]
# Convert distances to similarities for graph construction
positives = emd[np.triu_indices(n, 1)]
tau = np.median(positives[positives > 0]) if np.any(positives > 0) else 1.0
sim = np.exp(-emd / max(tau, 1e-8))
np.fill_diagonal(sim, 0.0)

# Build a small k-NN graph (undirected)
k = min(3, n - 1)  # keep it sparse and stable
G = nx.Graph()
for i in range(n):
    G.add_node(i)
    # pick top-k neighbors by similarity
    nbrs = np.argsort(sim[i])[::-1][:k]
    for j in nbrs:
        w = float(sim[i, j])
        d = float(emd[i, j])
        if w > 0 and i != j:
            # add edge once, keep the strongest weight if duplicates
            if G.has_edge(i, j):
                # keep max sim (min distance)
                if w > G[i][j].get("weight", 0.0):
                    G[i][j]["weight"] = w
                    G[i][j]["distance"] = d
            else:
                G.add_edge(i, j, weight=w, distance=d)

print(f"Built graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (k={k}).")

# Compute Ollivier–Ricci curvature
ricci = OllivierRicci(G, alpha=0.5, method="OTD", verbose="ERROR")  # OTD is fast and reliable
Gk = ricci.compute_ricci_curvature()  # returns the graph with curvature on edges

# Safely read curvature key (package sometimes changes key names)
def get_kappa(u, v):
    d = Gk[u][v]
    # Try common keys
    for key in ("ricciCurvature", "kappa", "ollivierRicciCurvature"):
        if key in d:
            return float(d[key])
    return None  # if missing

edge_kappa = {}
missing = 0
for u, v in Gk.edges():
    kappa = get_kappa(u, v)
    if kappa is None:
        missing += 1
    edge_kappa[f"{u}-{v}"] = kappa

print(f"Computed curvature. Edges: {len(edge_kappa)}, missing values: {missing}")

# Optional: simple node curvature as average of incident edge curvature
node_kappa = {}
for u in Gk.nodes():
    vals = []
    for v in Gk.neighbors(u):
        k = get_kappa(u, v)
        if k is not None:
            vals.append(k)
    node_kappa[str(u)] = (float(np.mean(vals)) if vals else None)

# Save outputs
out_json = os.path.join(HERE, "ricci_edges.json")
with open(out_json, "w") as f:
    json.dump(edge_kappa, f, indent=2)
print(f"Saved edge curvatures to {out_json}")

out_nodes = os.path.join(HERE, "ricci_nodes.json")
with open(out_nodes, "w") as f:
    json.dump(node_kappa, f, indent=2)
print(f"Saved node curvatures to {out_nodes}")

# Quick summary for report
finite_vals = [v for v in edge_kappa.values() if v is not None]
if finite_vals:
    arr = np.array(finite_vals)
    print(f"Edge curvature stats → mean: {arr.mean():.4f}, median: {np.median(arr):.4f}, "
          f"min: {arr.min():.4f}, max: {arr.max():.4f}")
else:
    print("Warning: No finite edge curvature values were produced.")
