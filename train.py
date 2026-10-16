"""
DASC7606 Assignment 1 - Training script
========================================
File: train.py
Purpose: Train BaselineCNN / ImprovedCNN / VGG16 on CIFAR-10 with
         strong augmentation (AutoAugment + Mixup + CutMix +
         RandomErasing + label smoothing) and a cosine LR schedule.

Usage (on HKU GPU Farm):
    python train.py --model vgg16 --epochs 100 --batch_size 128 \
                    --lr 0.1 --save_path model.pt

Outputs:
    * model.pt                    -- best checkpoint (highest val acc)
    * checkpoints/last.pt         -- most recent checkpoint
    * logs/<model>_<timestamp>.log  -- training log
"""

import argparse
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from model import build_model


# ---------------------------------------------------------------------------
#  Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
#  Data
# ---------------------------------------------------------------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def get_loaders(batch_size: int = 128,
                num_workers: int = 2,
                data_root: str = './data'):
    """Return (train_loader, val_loader, test_loader).

    We split the official 50 000 train images into 45 000/5 000
    train/val and use the official 10 000 test images for reporting.
    """
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        # Strong augmentation: CIFAR-10 policy learned by AutoAugment
        T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        # RandomErasing: zero-out a random rectangle of the image
        T.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    full_train = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_tf)
    full_test  = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_tf)

    # 45k / 5k split
    n_train = len(full_train)
    n_val   = 5000
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(n_train, generator=g).tolist()
    val_idx   = perm[:n_val]
    train_idx = perm[n_val:]

    train_set = torch.utils.data.Subset(full_train, train_idx)
    val_set   = torch.utils.data.Subset(
        torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=False, transform=test_tf),
        val_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=256,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(full_test, batch_size=256,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
#  Mixup / CutMix
# ---------------------------------------------------------------------------
def rand_bbox(size, lam):
    H, W = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1, bby1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
    bbx2, bby2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def mixup_or_cutmix(x, y, num_classes, alpha=1.0, cutmix_prob=0.5):
    """With probability cutmix_prob apply CutMix, else Mixup, else identity."""
    if alpha <= 0:
        return x, F.one_hot(y, num_classes).float()
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size(0), device=x.device)
    if np.random.rand() < cutmix_prob:
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bby1:bby2, bbx1:bbx2] = x[rand_index, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(2) * x.size(3)))
    else:
        x = lam * x + (1 - lam) * x[rand_index]
    y_oh = F.one_hot(y, num_classes).float()
    y_oh = lam * y_oh + (1 - lam) * y_oh[rand_index]
    return x, y_oh


# ---------------------------------------------------------------------------
#  Training / evaluation
# ---------------------------------------------------------------------------
def soft_ce_loss(logits, target_oh, label_smoothing=0.1):
    """Cross-entropy with soft targets + label smoothing."""
    n_classes = logits.size(1)
    if label_smoothing > 0:
        target_oh = target_oh * (1 - label_smoothing) + label_smoothing / n_classes
    logp = F.log_softmax(logits, dim=-1)
    return -(target_oh * logp).sum(dim=-1).mean()


def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss_sum += F.cross_entropy(logits, y, reduction='sum').item()
            correct += (logits.argmax(1) == y).sum().item()
            total   += y.size(0)
    return loss_sum / total, correct / total


def train_one_epoch(model, loader, optimizer, scheduler, device,
                    num_classes, use_mix=True, label_smoothing=0.1):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if use_mix:
            x, y_oh = mixup_or_cutmix(x, y, num_classes, alpha=1.0)
        else:
            y_oh = F.one_hot(y, num_classes).float()
        logits = model(x)
        loss = soft_ce_loss(logits, y_oh, label_smoothing=label_smoothing)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss_sum += loss.item() * y.size(0)
        # for hard accuracy, just use the dominant target (approx)
        target = y_oh.argmax(1)
        correct += (logits.argmax(1) == target).sum().item()
        total   += y.size(0)
    return loss_sum / total, correct / total


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model',       default='vgg16',
                   choices=['baseline', 'improved', 'vgg16'])
    p.add_argument('--epochs',      type=int,   default=100)
    p.add_argument('--batch_size',  type=int,   default=128)
    p.add_argument('--lr',          type=float, default=0.1)
    p.add_argument('--weight_decay',type=float, default=5e-4)
    p.add_argument('--momentum',    type=float, default=0.9)
    p.add_argument('--label_smooth',type=float, default=0.1)
    p.add_argument('--mix_alpha',   type=float, default=1.0,
                   help='set to 0 to disable mixup/cutmix')
    p.add_argument('--cutmix_prob', type=float, default=0.5)
    p.add_argument('--workers',     type=int,   default=2)
    p.add_argument('--data_root',   default='./data')
    p.add_argument('--save_path',   default='model.pt')
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--num_classes', type=int,   default=10)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{datetime.now():%H:%M:%S}] device = {device}")

    train_loader, val_loader, test_loader = get_loaders(
        batch_size=args.batch_size, num_workers=args.workers,
        data_root=args.data_root)

    model = build_model(args.model, num_classes=args.num_classes).to(device)
    n_params = sum(x.numel() for x in model.parameters())
    print(f"[{datetime.now():%H:%M:%S}] model = {args.model}, "
          f"params = {n_params/1e6:.2f}M")

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    log_path = os.path.join('logs',
        f"{args.model}_{datetime.now():%Y%m%d_%H%M%S}.log")
    log_f = open(log_path, 'w')
    log_f.write(str(args) + '\n')

    best_val, best_test = 0.0, 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            num_classes=args.num_classes,
            use_mix=(args.mix_alpha > 0),
            label_smoothing=args.label_smooth)
        val_loss, val_acc = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        line = (f"[{datetime.now():%H:%M:%S}] "
                f"epoch {epoch:3d}/{args.epochs}  "
                f"tr_loss {tr_loss:.3f} tr_acc {tr_acc:.3f}  "
                f"val_loss {val_loss:.3f} val_acc {val_acc:.3f}  "
                f"lr {optimizer.param_groups[0]['lr']:.5f}  "
                f"time {elapsed:.1f}s")
        print(line)
        log_f.write(line + '\n'); log_f.flush()

        if val_acc > best_val:
            best_val = val_acc
            test_loss, test_acc = evaluate(model, test_loader, device)
            best_test = test_acc
            torch.save({'model': model.state_dict(),
                        'args':   vars(args),
                        'epoch':  epoch,
                        'val_acc':val_acc,
                        'test_acc':test_acc}, args.save_path)
            print(f"           -> new best val {val_acc:.3f}, "
                  f"test {test_acc:.3f}, saved to {args.save_path}")
        torch.save({'model': model.state_dict(),
                    'args':   vars(args),
                    'epoch':  epoch}, 'checkpoints/last.pt')

    log_f.write(f"\nBEST val_acc={best_val:.4f}  test_acc={best_test:.4f}\n")
    log_f.close()
    print(f"\nDone. Best val_acc = {best_val:.4f}, "
          f"test_acc = {best_test:.4f}")


if __name__ == '__main__':
    main()
