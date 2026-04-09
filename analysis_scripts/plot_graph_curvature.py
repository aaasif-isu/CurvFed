import json, numpy as np, pathlib
import networkx as nx
import matplotlib.pyplot as plt


# --- load artifacts ---
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

emd = np.load(ROOT / "emd_matrix.npy")
edge_kappa = json.loads((HERE / "ricci_edges.json").read_text())

# rebuild the same k-NN graph (k=3)
n = emd.shape[0]
tau = np.median(emd[np.triu_indices(n,1)])
sim = np.exp(-emd/max(tau,1e-8))
np.fill_diagonal(sim, 0.0)
k = min(3, n-1)

G = nx.Graph()
for i in range(n):
    G.add_node(i)
    nbrs = np.argsort(sim[i])[::-1][:k]
    for j in nbrs:
        if i != j:
            G.add_edge(i, j, weight=float(sim[i, j]))

# attach curvature to edges
for (u, v) in G.edges():
    G[u][v]["kappa"] = edge_kappa.get(f"{u}-{v}") or edge_kappa.get(f"{v}-{u}")

pos = nx.spring_layout(G, seed=3)
edge_k = [G[u][v]["kappa"] for u, v in G.edges()]
edge_k = [x for x in edge_k if x is not None]  # ensure finite list

print(f"Edges: {len(edge_k)}, mean={np.mean(edge_k):.3f}, "
      f"median={np.median(edge_k):.3f}, min={np.min(edge_k):.3f}, max={np.max(edge_k):.3f}")

# draw
vmin, vmax = float(np.min(edge_k)), float(np.max(edge_k))
plt.figure(figsize=(5,4))
nx.draw_networkx_nodes(G, pos, node_size=600, node_color="#ddddff", edgecolors="#333")
ec = nx.draw_networkx_edges(
    G, pos,
    width=2,
    edge_color=[G[u][v]["kappa"] for u, v in G.edges()],
    edge_cmap=plt.cm.coolwarm,
    edge_vmin=vmin,   # <-- correct kwarg
    edge_vmax=vmax    # <-- correct kwarg
)
nx.draw_networkx_labels(G, pos, font_size=10)
cbar = plt.colorbar(ec); cbar.set_label("Ollivier–Ricci κ")
plt.axis("off"); plt.tight_layout()
plt.savefig("ricci_graph.png", dpi=200)
print("Saved ricci_graph.png")
