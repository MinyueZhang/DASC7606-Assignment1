"""
DASC7606 Assignment 1 -- Learning-rate sensitivity sweep
========================================================
File: lr_sweep.py
Purpose: Train the VGG-16 model with three different initial
         learning rates (0.01, 0.1, 0.5) for a short number of
         epochs and plot the resulting validation curves.

Output: figures/lr_sweep.png

Usage:
    python lr_sweep.py --epochs 20
"""

import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from model import build_model
from train import (set_seed, get_loaders, mixup_or_cutmix, soft_ce_loss,
                   evaluate, train_one_epoch)


LR_LIST = [0.01, 0.05, 0.1, 0.5]


def run_one(lr, epochs, batch_size, device, num_classes=10):
    set_seed(42)
    tr, va, te = get_loaders(batch_size=batch_size)
    model = build_model('vgg16', num_classes=num_classes).to(device)
    opt   = torch.optim.SGD(model.parameters(), lr=lr,
                             momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    val_acc_hist = []
    for ep in range(1, epochs + 1):
        train_one_epoch(model, tr, opt, sched, device,
                        num_classes=num_classes, use_mix=True,
                        label_smoothing=0.1)
        _, v = evaluate(model, va, device)
        val_acc_hist.append(v)
        print(f"  lr={lr:>4}  epoch {ep:2d}/{epochs}  val_acc={v:.4f}")
    _, t = evaluate(model, te, device)
    return val_acc_hist, t


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--out', default='figures/lr_sweep.png')
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = {}
    for lr in LR_LIST:
        print(f"\n>> Learning rate = {lr}")
        hist, test = run_one(lr, args.epochs, args.batch_size, device)
        results[lr] = (hist, test)

    fig, ax = plt.subplots(figsize=(6, 4))
    for lr, (hist, _) in results.items():
        ax.plot(range(1, len(hist) + 1), hist, marker='o',
                label=f"lr={lr}  (test={results[lr][1]:.3f})")
    ax.set_xlabel('Epoch'); ax.set_ylabel('Validation accuracy')
    ax.set_title('Effect of initial learning rate (VGG-16, CIFAR-10)')
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"\nSaved {args.out}")


if __name__ == '__main__':
    main()
