import torch, os

os.makedirs("smashed", exist_ok=True)
torch.manual_seed(42)

num_clients = 5
num_samples = 500
C, H, W = 64, 8, 8         # <-- spatial features to mimic real split output

for c in range(num_clients):
    mean = c * 0.3
    scale = 1.0 + 0.15 * c
    x = torch.randn(num_samples, C, H, W) * scale + mean
    torch.save(x, f"smashed/client{c}_round0.pt")
    print(f"client{c}_round0.pt -> {tuple(x.shape)}")
print("\nNew spatial smashed data saved in ./smashed/")
