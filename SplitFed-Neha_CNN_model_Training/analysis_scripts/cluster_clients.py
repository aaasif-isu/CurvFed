import numpy as np
from sklearn.cluster import AgglomerativeClustering

emd = np.load("emd_matrix.npy")

# Prefer new API (scikit-learn ≥ 1.2)
try:
    model = AgglomerativeClustering(
        n_clusters=3,
        metric="precomputed",   # <-- was 'affinity'
        linkage="average"       # 'average' or 'complete' work with precomputed distances
    )
except TypeError:
    # Fallback for very old sklearn
    model = AgglomerativeClustering(
        n_clusters=3,
        affinity="precomputed",
        linkage="average"
    )

labels = model.fit_predict(emd)
print("Cluster assignments:", labels.tolist())
np.savetxt("client_clusters.txt", labels, fmt="%d")
