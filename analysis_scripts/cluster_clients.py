import numpy as np
from sklearn.cluster import AgglomerativeClustering
import pathlib
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve()

ROOT = None
for p in HERE.parents:
    if (p / "outputs" / "emd_matrix.npy").exists():
        ROOT = p
        break

if ROOT is None:
    raise FileNotFoundError("Could not find outputs/emd_matrix.npy in any parent directory.")

emd = np.load(ROOT / "outputs" / "emd_matrix.npy")

try:
    model = AgglomerativeClustering(
        n_clusters=3,
        metric="precomputed",
        linkage="average"
    )
except TypeError:
    model = AgglomerativeClustering(
        n_clusters=3,
        affinity="precomputed",
        linkage="average"
    )

labels = model.fit_predict(emd)

# Build reverse mapping: cluster -> list of clients
cluster_map = defaultdict(list)
for client_id, cluster_id in enumerate(labels):
    cluster_map[int(cluster_id)].append(client_id)

# Save raw labels if you still want them
np.savetxt(ROOT / "outputs" / "client_clusters_raw.txt", labels, fmt="%d")

# Save readable report
report_path = ROOT / "outputs" / "client_clusters.txt"
with open(report_path, "w") as f:
    f.write("Cluster assignments by client\n")
    for client_id, cluster_id in enumerate(labels):
        f.write(f"Client {client_id} -> Cluster {int(cluster_id)}\n")

    f.write("\nCluster groups\n")
    for cluster_id in sorted(cluster_map):
        f.write(f"Cluster {cluster_id}: {cluster_map[cluster_id]}\n")

print("Cluster assignments:", labels.tolist())
print(f"Saved readable clusters to {report_path}")