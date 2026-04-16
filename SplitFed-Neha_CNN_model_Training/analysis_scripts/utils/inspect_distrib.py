import torch
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "outputs"

distrib_path = ROOT / "distrib" / "client0.pt"
output_txt = OUT / "distrib_client0_readable.txt"

x = torch.load(distrib_path, map_location="cpu")

with open(output_txt, "w") as f:
    f.write(f"Type: {type(x)}\n")

    if hasattr(x, "shape"):
        f.write(f"Shape: {tuple(x.shape)}\n")
        f.write(f"Dtype: {x.dtype}\n\n")
        f.write("First 5 rows, first 10 values:\n\n")

        rows = min(5, x.shape[0])
        for i in range(rows):
            vals = x[i].flatten()[:10].tolist()
            f.write(f"Row {i}: {vals}\n")
    else:
        f.write(str(x))

print(f"Saved to {output_txt}")
