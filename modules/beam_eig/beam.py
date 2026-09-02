import torch

device = torch.device("cuda")

eta = torch.tensor(
    [0.2, 0.5, 1.0],
    dtype=torch.float32,
    device=device,
    requires_grad=True
)

b = torch.exp(1j * eta)

loss = (b.real**2).sum()
loss.backward()

print("device:", b.device)
print("beam:", b)
print("gradient:", eta.grad)