"""
DASC7606 Assignment 1 -- Visualisation
=======================================
File: visualize.py
Purpose: Produce every figure that goes into report.pdf.

Outputs (saved into ./figures/):
    * confusion_matrix.png  -- 10x10 normalised heat-map
    * per_class_accuracy.png -- bar chart
    * misclassified.png     -- grid of worst mistakes
    * augmentation_grid.png -- raw vs augmented samples
    * training_curves.png   -- tr/val loss & accuracy vs epoch

Usage (after training is done):
    python visualize.py --ckpt model.pt --model vgg16
                        --log logs/<your_training_log>.log
"""

import argparse
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import seaborn as sns
    HAVE_SEABORN = True
except Exception:
    HAVE_SEABORN = False

from model import build_model


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)
CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# ---------------------------------------------------------------------------
def get_test_loader(batch_size=256, data_root='./data'):
    tf = T.Compose([T.ToTensor(),
                    T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    ds = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=False, num_workers=2, pin_memory=True)


def denormalize(x):
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std  = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


# ---------------------------------------------------------------------------
def plot_confusion_matrix(model, loader, device, save_path):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            preds = model(x).argmax(1).cpu().numpy()
            all_pred.append(preds)
            all_true.append(y.numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    cm = np.zeros((10, 10), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    if HAVE_SEABORN:
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    else:
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        for i in range(10):
            for j in range(10):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha='center', va='center',
                        color='white' if cm_norm[i, j] > 0.5 else 'black')
        ax.set_xticks(range(10)); ax.set_xticklabels(CLASSES, rotation=45)
        ax.set_yticks(range(10)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Normalised Confusion Matrix (Test set)')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved {save_path}")


def plot_per_class_accuracy(model, loader, device, save_path):
    model.eval()
    correct = np.zeros(10); total = np.zeros(10)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True); y = y.to(device)
            p = model(x).argmax(1)
            for c in range(10):
                mask = (y == c)
                total[c] += mask.sum().item()
                correct[c] += ((p == y) & mask).sum().item()
    acc = correct / total
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bars = ax.bar(CLASSES, acc * 100, color='steelblue', edgecolor='k')
    for b, a in zip(bars, acc):
        ax.text(b.get_x() + b.get_width()/2, a*100 + 0.5,
                f"{a*100:.1f}", ha='center', va='bottom', fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-class accuracy on CIFAR-10 test set')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved {save_path}")


def plot_misclassified(model, loader, device, save_path, k=25):
    model.eval()
    worst = []  # (confidence, image, true, pred)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs  = F.softmax(logits, dim=1)
            top_p, top_y = probs.max(1)
            for i in range(x.size(0)):
                if top_y[i].item() != y[i].item():
                    worst.append((top_p[i].item(), x[i].cpu(),
                                  y[i].item(), top_y[i].item()))
    worst.sort(key=lambda r: -r[0])
    worst = worst[:k]
    n = len(worst)
    rows = int(np.ceil(n / 5))
    fig, axes = plt.subplots(rows, 5, figsize=(10, 2 * rows))
    axes = np.array(axes).reshape(-1)
    for ax in axes: ax.axis('off')
    for i, (conf, img, t, p) in enumerate(worst):
        ax = axes[i]
        ax.imshow(denormalize(img.unsqueeze(0)).squeeze(0)
                  .permute(1, 2, 0).numpy())
        ax.set_title(f"true: {CLASSES[t]}\npred: {CLASSES[p]}\n"
                     f"conf: {conf:.2f}", fontsize=8)
    fig.suptitle("Most confident misclassifications", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved {save_path}")


def plot_augmentation_grid(save_path):
    """Show 8 raw images and 8 augmented versions side-by-side."""
    raw_tf = T.Compose([T.ToTensor()])
    aug_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])
    raw_ds = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=raw_tf)
    fig, axes = plt.subplots(2, 8, figsize=(12, 3))
    for i in range(8):
        img_raw, label = raw_ds[i]
        img_aug, _     = aug_tf(raw_ds.data[i]) if False else (None, None)
        # Apply aug to a PIL version of the raw image
        pil = T.ToPILImage()(img_raw)
        img_aug = aug_tf(pil)
        axes[0, i].imshow(img_raw.permute(1, 2, 0).numpy())
        axes[0, i].axis('off')
        axes[1, i].imshow(denormalize(img_aug.unsqueeze(0))
                          .squeeze(0).permute(1, 2, 0).numpy())
        axes[1, i].axis('off')
        if i == 0:
            axes[0, i].set_title('raw', fontsize=8)
            axes[1, i].set_title('augmented', fontsize=8)
    fig.suptitle("Data augmentation pipeline", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved {save_path}")


def plot_training_curves(log_path, save_path):
    """Parse the .log written by train.py and plot loss/acc curves."""
    epochs, tr_loss, tr_acc, val_loss, val_acc = [], [], [], [], []
    pat = re.compile(
        r"epoch\s+(\d+).*?tr_loss\s+([\d.]+)\s+tr_acc\s+([\d.]+).*?"
        r"val_loss\s+([\d.]+)\s+val_acc\s+([\d.]+)")
    with open(log_path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                epochs.append(int(m.group(1)))
                tr_loss.append(float(m.group(2)))
                tr_acc.append(float(m.group(3)))
                val_loss.append(float(m.group(4)))
                val_acc.append(float(m.group(5)))
    if not epochs:
        print(f"  WARNING: no epoch lines found in {log_path}")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].plot(epochs, tr_loss, label='train', color='C0')
    axes[0].plot(epochs, val_loss, label='val',   color='C1')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss vs epoch'); axes[0].legend(); axes[0].grid(alpha=.3)
    axes[1].plot(epochs, tr_acc, label='train', color='C0')
    axes[1].plot(epochs, val_acc, label='val',   color='C1')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy vs epoch'); axes[1].legend(); axes[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved {save_path}  ({len(epochs)} epochs parsed)")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',  default='model.pt')
    p.add_argument('--model', default='vgg16')
    p.add_argument('--log',   default='logs/vgg16_<latest>.log',
                   help='path to the training log produced by train.py')
    p.add_argument('--data_root', default='./data')
    p.add_argument('--out_dir', default='figures')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(args.model, num_classes=10).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state['model'] if 'model' in state else state)
    model.eval()
    print(f"Loaded {args.model} from {args.ckpt}")

    loader = get_test_loader(data_root=args.data_root)
    print("Generating figures...")
    plot_confusion_matrix(model, loader, device,
        os.path.join(args.out_dir, 'confusion_matrix.png'))
    plot_per_class_accuracy(model, loader, device,
        os.path.join(args.out_dir, 'per_class_accuracy.png'))
    plot_misclassified(model, loader, device,
        os.path.join(args.out_dir, 'misclassified.png'))
    plot_augmentation_grid(os.path.join(args.out_dir, 'augmentation_grid.png'))
    if os.path.exists(args.log):
        plot_training_curves(args.log,
            os.path.join(args.out_dir, 'training_curves.png'))
    else:
        print(f"  log file {args.log} not found -- skipping training curves")
    print("Done. All figures saved to", args.out_dir)


if __name__ == '__main__':
    main()
