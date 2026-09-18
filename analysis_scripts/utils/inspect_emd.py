import numpy as np
import pathlib

# Paths
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "outputs"

emd_path = OUT / "emd_matrix.npy"
output_txt = OUT / "emd_matrix_readable.txt"

# Load matrix
emd = np.load(emd_path)

# Write readable version
with open(output_txt, "w") as f:
    f.write(f"Shape: {emd.shape}\n\n")
    f.write("EMD Matrix (rounded):\n\n")

    for row in emd:
        formatted = ["{:.2f}".format(v) for v in row]
        f.write(" ".join(formatted) + "\n")

print(f"Saved readable EMD matrix to {output_txt}")
