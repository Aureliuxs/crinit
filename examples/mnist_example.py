"""Train a 64-wide, 32-deep tanh network on MNIST from the Crinit point and from the standard sigma_w^2 = 1.
Needs torch and torchvision (python mnist_example.py)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import crinit.torch_init as ct

L, N, EPOCHS = 32, 64, 1
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), transforms.Lambda(lambda x: x.view(-1))])
train = torch.utils.data.DataLoader(datasets.MNIST("data", train=True, download=True, transform=tf), batch_size=128, shuffle=True)
test = torch.utils.data.DataLoader(datasets.MNIST("data", train=False, transform=tf), batch_size=1000)


def run(model, lr=0.005):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for _ in range(EPOCHS):
        for x, y in train:
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
    correct = sum((model(x).argmax(1) == y).sum().item() for x, y in test)
    return correct / len(test.dataset)


torch.manual_seed(0)
model, r = ct.mlp(784, N, L, 10, "tanh")
print(r)
print("crinit:   test accuracy", run(model))

torch.manual_seed(0)
model = nn.Sequential(*[m for i in range(L) for m in (nn.Linear(784 if i == 0 else N, N), nn.Tanh())], nn.Linear(N, 10))
for lin in ct.linear_layers(model):
    nn.init.normal_(lin.weight, 0, (1.0 / lin.weight.shape[1]) ** 0.5)
    nn.init.zeros_(lin.bias)
print("standard: test accuracy", run(model))
