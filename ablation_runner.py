"""
DASC7606 Assignment 1 -- Ablation runner
=========================================
File: ablation_runner.py
Purpose: Run a sweep of "change-one-thing" experiments on the VGG-16
         backbone so that the report can include an ablation table.

Each ablation RE-USES the VGG-16 model from model.py and only flips
one flag in the training recipe.  The output is a CSV
(ablation_results.csv) and a printed summary.

Usage on the GPU Farm:
    python ablation_runner.py --epochs 30
(The full model uses 100 epochs; the ablations only need 30 to show
 the relative trend, which is what matters for analysis.)

We deliberately keep epochs small here so all 8 ablations fit in
~6-8 hours of single-GPU time.  After the sweep, the FINAL model is
re-trained with the full 100-epoch recipe in train.py.
"""

import argparse
import csv
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from model import build_model
from train import (set_seed, get_loaders, mixup_or_cutmix, soft_ce_loss,
                   evaluate, train_one_epoch)


# Each row is (name, kwargs to override in the training recipe)
# We start from the FULL recipe and turn off ONE thing at a time.
ABLATIONS = [
    # name                   mix_alpha  label_smooth  auto_augment  rand_erase
    ("full",                  1.0,       0.1,          True,         True),
    ("no_batchnorm",          1.0,       0.1,          True,         True),
    ("no_dropout",            1.0,       0.1,          True,         True),
    ("no_autoaugment",        1.0,       0.1,          False,        True),
    ("no_mix",                0.0,       0.1,          True,         True),
    ("no_label_smooth",       1.0,       0.0,          True,         True),
    ("no_randerase",          1.0,       0.1,          True,         False),
    ("step_lr",               1.0,       0.1,          True,         True),  # scheduler override
]


def make_train_tf(use_autoaug: bool, use_randerase: bool):
    base = [T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip()]
    if use_autoaug:
        base.append(T.AutoAugment(T.AutoAugmentPolicy.CIFAR10))
    base += [T.ToTensor(),
             T.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))]
    if use_randerase:
        base.append(T.RandomErasing(p=0.25, scale=(0.02, 0.33),
                                    ratio=(0.3, 3.3)))
    return T.Compose(base)


def run_single(name, mix_alpha, label_smooth, use_autoaug, use_randerase,
               epochs, batch_size, lr, data_root, device):
    """Run one ablation and return (val_acc, test_acc, n_params)."""
    set_seed(42)
    # Rebuild data loaders so augmentation changes are picked up
    train_tf = make_train_tf(use_autoaug, use_randerase)
    test_tf  = T.Compose([T.ToTensor(),
                          T.Normalize((0.4914, 0.4822, 0.4465),
                                      (0.2470, 0.2435, 0.2616))])
    full_train_aug = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=train_tf)
    full_test      = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=False, transform=test_tf)
    full_train_eval = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=test_tf)
    n_train = len(full_train_aug); n_val = 5000
    g = torch.Generator().manual_seed(42)
    perm   = torch.randperm(n_train, generator=g).tolist()
    val_idx   = perm[:n_val]
    train_idx = perm[n_val:]
    train_set = torch.utils.data.Subset(full_train_aug, train_idx)
    val_set   = torch.utils.data.Subset(full_train_eval, val_idx)
    tr_loader = DataLoader(train_set, batch_size=batch_size,
                           shuffle=True, num_workers=2,
                           pin_memory=True, drop_last=True)
    va_loader = DataLoader(val_set,   batch_size=256, shuffle=False,
                           num_workers=2, pin_memory=True)
    te_loader = DataLoader(full_test, batch_size=256, shuffle=False,
                           num_workers=2, pin_memory=True)

    # Special case: "no_batchnorm" -- swap BatchNorm for Identity
    # Special case: "no_dropout" -- swap Dropout for Identity
    model = build_model('vgg16', num_classes=10).to(device)
    if name == 'no_batchnorm':
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False
                # Replace with Identity
                m.__class__ = nn.Identity
    if name == 'no_dropout':
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=0.9, weight_decay=5e-4,
                                nesterov=True)
    if name == 'step_lr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                     step_size=10, gamma=0.5)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)

    best_val, best_test = 0.0, 0.0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_one_epoch(model, tr_loader, optimizer, scheduler, device,
                        num_classes=10, use_mix=(mix_alpha > 0),
                        label_smoothing=label_smooth)
        val_loss, val_acc = evaluate(model, va_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            _, best_test = evaluate(model, te_loader, device)
    dt = time.time() - t0
    print(f"  [{datetime.now():%H:%M:%S}] {name:18s} "
          f"best_val={best_val:.4f}  best_test={best_test:.4f}  "
          f"({dt/60:.1f} min, {n_params/1e6:.2f}M params)")
    return best_val, best_test, n_params


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=30,
                   help='epochs per ablation (30 is enough for trends)')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--data_root', default='./data')
    p.add_argument('--out_csv', default='ablation_results.csv')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")
    print(f"epochs/abl = {args.epochs}\n")

    rows = []
    for cfg in ABLATIONS:
        name, mix_a, ls, aug, re_ = cfg
        print(f">> ablation = {name}")
        v, t, n = run_single(name, mix_a, ls, aug, re_,
                             args.epochs, args.batch_size, args.lr,
                             args.data_root, device)
        rows.append({'name': name, 'val_acc': v, 'test_acc': t,
                     'n_params_M': n / 1e6,
                     'epochs': args.epochs})

    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['name', 'val_acc', 'test_acc',
                                          'n_params_M', 'epochs'])
        w.writeheader()
        w.writerows(rows)
    print(f"\nResults written to {args.out_csv}")
    print(f"{'name':18s} {'val_acc':>8s} {'test_acc':>9s}")
    for r in rows:
        print(f"{r['name']:18s} {r['val_acc']:8.4f} {r['test_acc']:9.4f}")


if __name__ == '__main__':
    main()
