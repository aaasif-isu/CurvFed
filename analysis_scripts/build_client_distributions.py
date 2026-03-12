import os, glob, torch
os.makedirs("distrib", exist_ok=True)

by_client = {}
for f in glob.glob("smashed/client_*/*.pt"):
    cid = int(os.path.basename(os.path.dirname(f)).replace("client_",""))
    x = torch.load(f, map_location="cpu")  # (n, D)
    by_client.setdefault(cid, []).append(x)

CAP = 2000
for cid, tensors in by_client.items():
    X = torch.cat(tensors, dim=0)
    if X.size(0) > CAP:
        X = X[torch.randperm(X.size(0))[:CAP]]
    out = f"distrib/client{cid}.pt"
    torch.save(X, out)
    print(f"{out}: {tuple(X.shape)}")