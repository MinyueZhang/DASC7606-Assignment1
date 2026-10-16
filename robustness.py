"""
DASC7606 Assignment 1 -- Robustness test
========================================
File: robustness.py
Purpose: Add small Gaussian noise / random occlusion to CIFAR-10
         test images and report how the test accuracy degrades.
         Demonstrates the regularisation effect of training-time
         augmentation.

Usage:
    python robustness.py --ckpt model.pt --model vgg16
"""

import argparse
import os

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

from model import build_model


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


class CorruptedTest(Dataset):
    """Wrap CIFAR-10 test set with optional Gaussian noise or cutout."""
    def __init__(self, base, sigma=0.0, cutout=0):
        self.base   = base
        self.sigma  = sigma
        self.cutout = cutout   # square patch size in pixels

    def __len__(self): return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]            # x is normalised tensor
        if self.sigma > 0:
            x = x + torch.randn_like(x) * self.sigma
        if self.cutout > 0:
            x = x.clone()
            top  = torch.randint(0, 32 - self.cutout, (1,)).item()
            left = torch.randint(0, 32 - self.cutout, (1,)).item()
            x[:, top:top + self.cutout, left:left + self.cutout] = 0
        return x, y


def evaluate(model, loader, device):
    model.eval()
    c, n = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            c += (model(x).argmax(1) == y).sum().item()
            n += y.size(0)
    return c / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',  default='model.pt')
    p.add_argument('--model', default='vgg16')
    p.add_argument('--data_root', default='./data')
    p.add_argument('--out',  default='figures/robustness.png')
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(args.model, num_classes=10).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state['model'] if 'model' in state else state)
    model.eval()

    base = torchvision.datasets.CIFAR10(
        root=args.data_root, train=False, download=True,
        transform=T.Compose([T.ToTensor(),
                             T.Normalize(CIFAR10_MEAN, CIFAR10_STD)]))

    sigmas  = [0.0, 0.05, 0.10, 0.20, 0.30]
    cutouts = [0, 4, 8, 12, 16]

    results = {'noise': [], 'cutout': []}
    for s in sigmas:
        ds = CorruptedTest(base, sigma=s)
        acc = evaluate(model, DataLoader(ds, batch_size=256,
                                         num_workers=2), device)
        results['noise'].append(acc)
        print(f"  Gaussian noise sigma={s:.2f}  acc={acc:.4f}")
    for c in cutouts:
        ds = CorruptedTest(base, cutout=c)
        acc = evaluate(model, DataLoader(ds, batch_size=256,
                                         num_workers=2), device)
        results['cutout'].append(acc)
        print(f"  cutout size={c:2d}        acc={acc:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(sigmas, results['noise'], 'o-', color='C0')
    axes[0].set_xlabel('Gaussian noise $\\sigma$')
    axes[0].set_ylabel('Test accuracy')
    axes[0].set_title('Robustness to noise')
    axes[0].grid(alpha=.3)
    axes[1].plot(cutouts, results['cutout'], 's-', color='C1')
    axes[1].set_xlabel('Cutout patch size (px)')
    axes[1].set_ylabel('Test accuracy')
    axes[1].set_title('Robustness to occlusion')
    axes[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"\nSaved {args.out}")


if __name__ == '__main__':
    main()
