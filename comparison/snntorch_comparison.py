#!/usr/bin/env python3
"""
SNNTorch Comparison Networks for Gilgamesh

This script trains spiking neural networks using PyTorch and snntorch
with the same configuration as the gilgamesh Rust implementation:
- 2-layer feedforward LIF SNN
- 49 input (7x7 MNIST) -> 100 hidden -> 10 output
- Rate-coded input encoding
- Cross-entropy loss on spike counts
- Adam optimizer with cosine annealing

Trains multiple network variants and generates a comparison report.
"""

import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import snntorch as snn
from snntorch import surrogate
import numpy as np


# =============================================================================
# Configuration (matching gilgamesh defaults)
# =============================================================================

@dataclass
class NetworkConfig:
    """Network architecture configuration."""
    input_size: int = 49       # 7x7 downsampled MNIST
    hidden_size: int = 9       # Hidden layer neurons (matching gilgamesh)
    output_size: int = 10      # Output classes (digits 0-9)


@dataclass
class NeuronConfig:
    """LIF neuron configuration."""
    beta: float = 0.9          # Membrane decay rate
    threshold: float = 1.0     # Spike threshold
    slope: float = 25.0        # Surrogate gradient slope
    reset_mechanism: str = "subtract"  # "subtract", "zero", or "none"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    lr: float = 0.001          # Initial learning rate
    lr_min: float = 0.00001    # Minimum learning rate (1% of initial)
    epochs: int = 15           # Number of training epochs
    batch_size: int = 128      # Batch size
    num_steps: int = 25        # Timesteps per sample
    seed: int = 42             # Random seed
    weight_decay: float = 0.0  # AdamW weight decay
    beta1: float = 0.9         # Adam first moment
    beta2: float = 0.999       # Adam second moment
    grad_clip: float = 1.0     # Gradient clipping max norm


@dataclass
class Config:
    """Complete configuration."""
    network: NetworkConfig
    neuron: NeuronConfig
    training: TrainingConfig

    @classmethod
    def default(cls):
        return cls(
            network=NetworkConfig(),
            neuron=NeuronConfig(),
            training=TrainingConfig()
        )


# =============================================================================
# Dataset
# =============================================================================

class DownsampledMNIST:
    """MNIST dataset downsampled to 7x7 pixels (matching gilgamesh)."""

    def __init__(self, data_dir: str, target_size: int = 7):
        self.data_dir = data_dir
        self.target_size = target_size

        # Transform: resize to 7x7, normalize with MNIST stats
        self.transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten to 49
        ])

    def get_loaders(self, batch_size: int):
        """Get train and test data loaders."""
        train_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.transform
        )

        test_dataset = datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        return train_loader, test_loader


# =============================================================================
# Network Models
# =============================================================================

class GilgameshSNN(nn.Module):
    """
    2-layer feedforward LIF SNN matching gilgamesh architecture.

    Architecture: Input(49) -> FC1 -> LIF(100) -> FC2 -> LIF(10)
    Uses spike count output for classification.
    """

    def __init__(self, config: Config):
        super().__init__()

        net = config.network
        neuron = config.neuron

        # Surrogate gradient function (fast sigmoid)
        spike_grad = surrogate.fast_sigmoid(slope=neuron.slope)

        # Reset mechanism
        reset_map = {
            "subtract": "subtract",
            "zero": "zero",
            "none": "none"
        }
        reset = reset_map.get(neuron.reset_mechanism, "subtract")

        # Layer 1: Input -> Hidden
        self.fc1 = nn.Linear(net.input_size, net.hidden_size)
        self.lif1 = snn.Leaky(
            beta=neuron.beta,
            threshold=neuron.threshold,
            spike_grad=spike_grad,
            reset_mechanism=reset
        )

        # Layer 2: Hidden -> Output
        self.fc2 = nn.Linear(net.hidden_size, net.output_size)
        self.lif2 = snn.Leaky(
            beta=neuron.beta,
            threshold=neuron.threshold,
            spike_grad=spike_grad,
            reset_mechanism=reset
        )

        self.num_steps = config.training.num_steps

    def forward(self, x):
        """
        Forward pass with rate-coded input.

        Args:
            x: Input tensor [batch, 49]

        Returns:
            spike_count: Total spikes per output neuron [batch, 10]
            mem_record: Membrane potentials over time
            spike_record: Spikes over time
        """
        batch_size = x.size(0)

        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Output spike count accumulator
        spike_count = torch.zeros(batch_size, self.fc2.out_features, device=x.device)

        # Records for analysis
        mem_record = []
        spike_record = []

        # Simulate over time steps (rate-coded: same input each step)
        for _ in range(self.num_steps):
            # Layer 1
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            # Layer 2
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            # Accumulate output spikes
            spike_count = spike_count + spk2

            # Record
            mem_record.append(mem2.detach())
            spike_record.append(spk2.detach())

        return spike_count, mem_record, spike_record


class GilgameshSNN_Synaptic(nn.Module):
    """
    Variant using Synaptic neurons (dual exponential dynamics).
    """

    def __init__(self, config: Config):
        super().__init__()

        net = config.network
        neuron = config.neuron

        spike_grad = surrogate.fast_sigmoid(slope=neuron.slope)

        self.fc1 = nn.Linear(net.input_size, net.hidden_size)
        self.lif1 = snn.Synaptic(
            alpha=0.9,  # Synaptic decay
            beta=neuron.beta,
            threshold=neuron.threshold,
            spike_grad=spike_grad
        )

        self.fc2 = nn.Linear(net.hidden_size, net.output_size)
        self.lif2 = snn.Synaptic(
            alpha=0.9,
            beta=neuron.beta,
            threshold=neuron.threshold,
            spike_grad=spike_grad
        )

        self.num_steps = config.training.num_steps

    def forward(self, x):
        batch_size = x.size(0)

        syn1, mem1 = self.lif1.init_synaptic()
        syn2, mem2 = self.lif2.init_synaptic()

        spike_count = torch.zeros(batch_size, self.fc2.out_features, device=x.device)
        mem_record = []
        spike_record = []

        for _ in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)

            cur2 = self.fc2(spk1)
            spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)

            spike_count = spike_count + spk2
            mem_record.append(mem2.detach())
            spike_record.append(spk2.detach())

        return spike_count, mem_record, spike_record


class GilgameshSNN_Recurrent(nn.Module):
    """
    Variant with recurrent connections in hidden layer.
    """

    def __init__(self, config: Config):
        super().__init__()

        net = config.network
        neuron = config.neuron

        spike_grad = surrogate.fast_sigmoid(slope=neuron.slope)

        self.fc1 = nn.Linear(net.input_size, net.hidden_size)
        self.rec1 = nn.Linear(net.hidden_size, net.hidden_size, bias=False)
        self.lif1 = snn.Leaky(
            beta=neuron.beta,
            threshold=neuron.threshold,
            spike_grad=spike_grad
        )

        self.fc2 = nn.Linear(net.hidden_size, net.output_size)
        self.lif2 = snn.Leaky(
            beta=neuron.beta,
            threshold=neuron.threshold,
            spike_grad=spike_grad
        )

        self.num_steps = config.training.num_steps

    def forward(self, x):
        batch_size = x.size(0)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk1 = torch.zeros(batch_size, self.fc1.out_features, device=x.device)

        spike_count = torch.zeros(batch_size, self.fc2.out_features, device=x.device)
        mem_record = []
        spike_record = []

        for _ in range(self.num_steps):
            # Layer 1 with recurrence
            cur1 = self.fc1(x) + self.rec1(spk1)
            spk1, mem1 = self.lif1(cur1, mem1)

            # Layer 2
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spike_count = spike_count + spk2
            mem_record.append(mem2.detach())
            spike_record.append(spk2.detach())

        return spike_count, mem_record, spike_record


class GilgameshSNN_3Layer(nn.Module):
    """
    Deeper variant with 3 hidden layers.
    """

    def __init__(self, config: Config):
        super().__init__()

        net = config.network
        neuron = config.neuron

        spike_grad = surrogate.fast_sigmoid(slope=neuron.slope)

        hidden1 = net.hidden_size
        hidden2 = net.hidden_size // 2  # 50

        self.fc1 = nn.Linear(net.input_size, hidden1)
        self.lif1 = snn.Leaky(beta=neuron.beta, threshold=neuron.threshold, spike_grad=spike_grad)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.lif2 = snn.Leaky(beta=neuron.beta, threshold=neuron.threshold, spike_grad=spike_grad)

        self.fc3 = nn.Linear(hidden2, net.output_size)
        self.lif3 = snn.Leaky(beta=neuron.beta, threshold=neuron.threshold, spike_grad=spike_grad)

        self.num_steps = config.training.num_steps

    def forward(self, x):
        batch_size = x.size(0)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spike_count = torch.zeros(batch_size, self.fc3.out_features, device=x.device)
        mem_record = []
        spike_record = []

        for _ in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spike_count = spike_count + spk3
            mem_record.append(mem3.detach())
            spike_record.append(spk3.detach())

        return spike_count, mem_record, spike_record


class StandardANN(nn.Module):
    """
    Standard PyTorch ANN (non-spiking) for comparison.

    Same architecture as GilgameshSNN but with ReLU activations instead of LIF neurons.
    Architecture: Input(49) -> FC -> ReLU(100) -> FC -> Output(10)
    """

    def __init__(self, config: Config):
        super().__init__()

        net = config.network

        self.fc1 = nn.Linear(net.input_size, net.hidden_size)
        self.fc2 = nn.Linear(net.hidden_size, net.output_size)

    def forward(self, x):
        """
        Forward pass.

        Returns same format as SNN models for compatibility:
            output: Logits [batch, 10]
            mem_record: Empty list (no membrane potentials)
            spike_record: Empty list (no spikes)
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, [], []


class StandardANN_3Layer(nn.Module):
    """
    Deeper ANN variant with 3 layers for comparison with GilgameshSNN_3Layer.

    Architecture: Input(49) -> FC -> ReLU(100) -> FC -> ReLU(50) -> FC -> Output(10)
    """

    def __init__(self, config: Config):
        super().__init__()

        net = config.network
        hidden1 = net.hidden_size
        hidden2 = net.hidden_size // 2  # 50

        self.fc1 = nn.Linear(net.input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, net.output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, [], []


# =============================================================================
# Training
# =============================================================================

class CosineAnnealingLR:
    """Cosine annealing learning rate scheduler (matching gilgamesh)."""

    def __init__(self, optimizer, initial_lr: float, min_lr: float, total_epochs: int):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs

    def step(self, epoch: int):
        """Update learning rate based on epoch."""
        progress = epoch / self.total_epochs
        lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


class Trainer:
    """Training loop for SNN models."""

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        device: torch.device,
        name: str = "model"
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.name = name

        tc = config.training

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=tc.lr,
            betas=(tc.beta1, tc.beta2),
            weight_decay=tc.weight_decay
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            initial_lr=tc.lr,
            min_lr=tc.lr_min,
            total_epochs=tc.epochs
        )

        self.grad_clip = tc.grad_clip

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'lr': [],
            'epoch_time': []
        }

    def train_epoch(self, train_loader: DataLoader) -> tuple:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for data, targets in train_loader:
            data = data.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            spike_count, _, _ = self.model(data)

            # Cross-entropy loss on spike counts
            loss = F.cross_entropy(spike_count, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            # Statistics
            total_loss += loss.item() * data.size(0)
            pred = spike_count.argmax(dim=1)
            correct += (pred == targets).sum().item()
            total += data.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def run_evaluation(self, test_loader: DataLoader) -> tuple:
        """Evaluate on test set."""
        self.model.train(False)  # Set to evaluation mode

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in test_loader:
                data = data.to(self.device)
                targets = targets.to(self.device)

                spike_count, _, _ = self.model(data)

                loss = F.cross_entropy(spike_count, targets)

                total_loss += loss.item() * data.size(0)
                pred = spike_count.argmax(dim=1)
                correct += (pred == targets).sum().item()
                total += data.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        verbose: bool = True
    ) -> dict:
        """Full training loop."""
        epochs = self.config.training.epochs

        best_test_acc = 0.0
        best_weights = None

        for epoch in range(epochs):
            epoch_start = time.time()

            # Update learning rate
            lr = self.scheduler.step(epoch)

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Evaluate
            test_loss, test_acc = self.run_evaluation(test_loader)

            epoch_time = time.time() - epoch_start

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['lr'].append(lr)
            self.history['epoch_time'].append(epoch_time)

            # Track best
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if verbose:
                print(f"[{self.name}] Epoch {epoch+1:2d}/{epochs} | "
                      f"Train: {train_acc*100:.2f}% | "
                      f"Test: {test_acc*100:.2f}% | "
                      f"LR: {lr:.6f} | "
                      f"Time: {epoch_time:.1f}s")

        # Restore best weights
        if best_weights is not None:
            self.model.load_state_dict(best_weights)

        return {
            'best_test_acc': best_test_acc,
            'final_train_acc': self.history['train_acc'][-1],
            'final_test_acc': self.history['test_acc'][-1],
            'history': self.history
        }


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(results: dict, output_dir: Path):
    """Generate markdown comparison report."""

    report = []
    report.append("# SNNTorch Comparison Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    report.append("## Configuration (matching gilgamesh)")
    report.append("")
    report.append("| Parameter | Value |")
    report.append("|-----------|-------|")
    report.append("| Input Size | 49 (7×7 MNIST) |")
    report.append("| Hidden Size | 100 |")
    report.append("| Output Size | 10 |")
    report.append("| Beta (decay) | 0.9 |")
    report.append("| Threshold | 1.0 |")
    report.append("| Surrogate Slope | 25.0 |")
    report.append("| Learning Rate | 0.001 → 0.00001 (cosine) |")
    report.append("| Epochs | 15 |")
    report.append("| Batch Size | 128 |")
    report.append("| Timesteps | 25 |")
    report.append("| Optimizer | Adam (β1=0.9, β2=0.999) |")
    report.append("")

    report.append("## Results Summary")
    report.append("")
    report.append("| Model | Best Test Acc | Final Train Acc | Final Test Acc | Parameters |")
    report.append("|-------|---------------|-----------------|----------------|------------|")

    for name, res in results.items():
        report.append(
            f"| {name} | {res['best_test_acc']*100:.2f}% | "
            f"{res['final_train_acc']*100:.2f}% | "
            f"{res['final_test_acc']*100:.2f}% | "
            f"{res['params']:,} |"
        )

    report.append("")
    report.append("## Model Descriptions")
    report.append("")
    report.append("### Spiking Neural Networks (SNNs)")
    report.append("")
    report.append("#### GilgameshSNN (Baseline)")
    report.append("Standard 2-layer feedforward LIF SNN matching gilgamesh architecture exactly.")
    report.append("Architecture: Input(49) → FC → LIF(100) → FC → LIF(10)")
    report.append("")
    report.append("#### GilgameshSNN_Synaptic")
    report.append("Uses dual exponential synaptic dynamics with separate synaptic and membrane time constants.")
    report.append("")
    report.append("#### GilgameshSNN_Recurrent")
    report.append("Adds recurrent connections within the hidden layer for temporal processing.")
    report.append("")
    report.append("#### GilgameshSNN_3Layer")
    report.append("Deeper SNN with 3 layers: Input(49) → LIF(100) → LIF(50) → LIF(10)")
    report.append("")
    report.append("### Artificial Neural Networks (ANNs)")
    report.append("")
    report.append("#### StandardANN")
    report.append("Standard 2-layer feedforward ANN with ReLU activations (non-spiking baseline).")
    report.append("Architecture: Input(49) → FC → ReLU(100) → FC → Output(10)")
    report.append("")
    report.append("#### StandardANN_3Layer")
    report.append("Deeper ANN with 3 layers: Input(49) → ReLU(100) → ReLU(50) → Output(10)")
    report.append("")

    report.append("## Training Curves")
    report.append("")
    report.append("See `training_curves.png` for visualization of training and test accuracy over epochs.")
    report.append("")

    report.append("## Weight Files")
    report.append("")
    for name in results.keys():
        report.append(f"- `{name}_weights.pt`: PyTorch state dict")
        report.append(f"- `{name}_weights.json`: Weights exported as JSON (for gilgamesh compatibility)")
    report.append("")

    report.append("## Notes")
    report.append("")
    report.append("- All models use rate-coded input (same input at each timestep)")
    report.append("- Loss: Cross-entropy on spike counts")
    report.append("- Best weights (highest test accuracy) are saved")
    report.append("- JSON exports use gilgamesh-compatible format")
    report.append("")

    # Write report
    report_path = output_dir / "comparison_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    return report_path


def plot_training_curves(results: dict, output_dir: Path):
    """Plot training curves for all models."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

        # Training accuracy
        ax = axes[0, 0]
        for (name, res), color in zip(results.items(), colors):
            epochs = range(1, len(res['history']['train_acc']) + 1)
            ax.plot(epochs, [a*100 for a in res['history']['train_acc']],
                   label=name, color=color)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Training Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Test accuracy
        ax = axes[0, 1]
        for (name, res), color in zip(results.items(), colors):
            epochs = range(1, len(res['history']['test_acc']) + 1)
            ax.plot(epochs, [a*100 for a in res['history']['test_acc']],
                   label=name, color=color)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Test Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Training loss
        ax = axes[1, 0]
        for (name, res), color in zip(results.items(), colors):
            epochs = range(1, len(res['history']['train_loss']) + 1)
            ax.plot(epochs, res['history']['train_loss'], label=name, color=color)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Learning rate
        ax = axes[1, 1]
        # All models use same LR schedule, just plot one
        first_res = list(results.values())[0]
        epochs = range(1, len(first_res['history']['lr']) + 1)
        ax.plot(epochs, first_res['history']['lr'], color='black')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule (Cosine Annealing)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = output_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        return plot_path

    except ImportError:
        print("Warning: matplotlib not available, skipping plot generation")
        return None


def save_weights(model: nn.Module, name: str, output_dir: Path):
    """Save model weights in both PyTorch and JSON formats."""

    # PyTorch format
    pt_path = output_dir / f"{name}_weights.pt"
    torch.save(model.state_dict(), pt_path)

    # JSON format (gilgamesh compatible)
    state_dict = model.state_dict()
    weights_json = {}

    for key, tensor in state_dict.items():
        weights_json[key] = tensor.cpu().numpy().tolist()

    json_path = output_dir / f"{name}_weights.json"
    with open(json_path, 'w') as f:
        json.dump(weights_json, f, indent=2)

    return pt_path, json_path


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='SNNTorch Comparison for Gilgamesh')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory for MNIST data')
    parser.add_argument('--output-dir', type=str, default='./comparison/results',
                       help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--num-steps', type=int, default=25,
                       help='Number of simulation timesteps')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda, mps, cpu, or auto)')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['baseline', 'synaptic', 'recurrent', '3layer', 'ann', 'ann_3layer'],
                       help='Models to train (baseline, synaptic, recurrent, 3layer, ann, ann_3layer)')

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device selection
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Configuration
    config = Config.default()
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.lr = args.lr
    config.training.num_steps = args.num_steps
    config.training.seed = args.seed

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump({
            'network': asdict(config.network),
            'neuron': asdict(config.neuron),
            'training': asdict(config.training)
        }, f, indent=2)

    print(f"\nConfiguration saved to {config_path}")

    # Load data
    print("\nLoading MNIST dataset (7x7 downsampled)...")
    dataset = DownsampledMNIST(args.data_dir, target_size=7)
    train_loader, test_loader = dataset.get_loaders(args.batch_size)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Model configurations
    model_classes = {
        'baseline': ('GilgameshSNN', GilgameshSNN),
        'synaptic': ('GilgameshSNN_Synaptic', GilgameshSNN_Synaptic),
        'recurrent': ('GilgameshSNN_Recurrent', GilgameshSNN_Recurrent),
        '3layer': ('GilgameshSNN_3Layer', GilgameshSNN_3Layer),
        'ann': ('StandardANN', StandardANN),
        'ann_3layer': ('StandardANN_3Layer', StandardANN_3Layer),
    }

    # Train selected models
    results = {}

    for model_key in args.models:
        if model_key not in model_classes:
            print(f"Unknown model: {model_key}, skipping")
            continue

        name, ModelClass = model_classes[model_key]

        print(f"\n{'='*60}")
        print(f"Training {name}")
        print('='*60)

        model = ModelClass(config)
        params = count_parameters(model)
        print(f"Parameters: {params:,}")

        trainer = Trainer(model, config, device, name=name)
        result = trainer.train(train_loader, test_loader)
        result['params'] = params

        results[name] = result

        # Save weights
        pt_path, json_path = save_weights(model, name, output_dir)
        print(f"Weights saved to {pt_path}")
        print(f"JSON weights saved to {json_path}")

    # Generate report
    print(f"\n{'='*60}")
    print("Generating Report")
    print('='*60)

    report_path = generate_report(results, output_dir)
    print(f"Report saved to {report_path}")

    plot_path = plot_training_curves(results, output_dir)
    if plot_path:
        print(f"Training curves saved to {plot_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print('='*60)
    print(f"\n{'Model':<25} {'Best Test Acc':>15} {'Parameters':>12}")
    print('-'*55)
    for name, res in sorted(results.items(), key=lambda x: -x[1]['best_test_acc']):
        print(f"{name:<25} {res['best_test_acc']*100:>14.2f}% {res['params']:>12,}")

    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main()
