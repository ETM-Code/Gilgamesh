#!/usr/bin/env python3
"""Compare snnTorch Python vs gilgamesh Rust speed."""
import sys
sys.path.insert(0, '../snntorch_reference')

import time
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Match Rust configuration
INPUT_SIZE = 49  # 7x7 downsampled MNIST
HIDDEN_SIZE = 100
OUTPUT_SIZE = 10
BETA = 0.9
NUM_STEPS = 25
BATCH_SIZE = 128
LR = 0.001
EPOCHS = 5

# Downsample transform (7x7)
transform = transforms.Compose([
    transforms.Resize((7, 7)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten to 49
])

# Load MNIST
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Network matching Rust architecture
class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.lif1 = snn.Leaky(beta=BETA, spike_grad=spike_grad)
        self.fc2 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.lif2 = snn.Leaky(beta=BETA, spike_grad=spike_grad)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk_rec = []

        for _ in range(NUM_STEPS):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk_rec.append(spk2)

        return torch.stack(spk_rec, dim=0).sum(0)

# Training
device = torch.device('cpu')  # Match Rust (CPU only)
net = SNN().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

print(f"=== Python snnTorch (CPU) ===")
print(f"Architecture: {INPUT_SIZE}-{HIDDEN_SIZE}-{OUTPUT_SIZE}")
print(f"Timesteps: {NUM_STEPS}, Batch: {BATCH_SIZE}, Beta: {BETA}")
print()

start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    net.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        train_total += target.size(0)
        train_correct += predicted.eq(target).sum().item()

    # Evaluate
    net.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            _, predicted = output.max(1)
            test_total += target.size(0)
            test_correct += predicted.eq(target).sum().item()

    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch} | Loss: {train_loss/len(train_loader):.4f} | "
          f"Train: {100*train_correct/train_total:.2f}% | "
          f"Test: {100*test_correct/test_total:.2f}% | "
          f"Time: {epoch_time:.2f}s")

total_time = time.time() - start_time
print(f"\nTotal time: {total_time:.2f}s ({total_time/EPOCHS:.2f}s/epoch)")
