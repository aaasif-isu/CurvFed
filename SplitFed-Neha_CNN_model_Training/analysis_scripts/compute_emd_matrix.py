import glob, torch, numpy as np, ot
from tqdm import tqdm

files = sorted(glob.glob("distrib/client*.pt"))
Xs = []
for f in files:
    X = torch.load(f, map_location="cpu").float().numpy()
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True)+1e-8)
    Xs.append(X)

k = len(Xs)
emd = np.zeros((k, k), dtype=np.float64)
for i in tqdm(range(k), desc="EMD"):
    Xi, wi = Xs[i], ot.unif(len(Xs[i]))
    for j in range(i+1, k):
        Xj, wj = Xs[j], ot.unif(len(Xs[j]))
        M = ot.dist(Xi, Xj, metric="euclidean")
        emd2 = ot.emd2(wi, wj, M)
        emd[i,j] = emd[j,i] = float(emd2)

np.save("emd_matrix.npy", emd)
print("Saved emd_matrix.npy")
