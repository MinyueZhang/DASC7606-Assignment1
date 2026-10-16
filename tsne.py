"""
DASC7606 Assignment 1 -- t-SNE feature visualisation
====================================================
File: tsne.py
Purpose: Extract penultimate-layer features from the trained VGG-16
         on CIFAR-10 test set and project them to 2D with t-SNE
         for a qualitative analysis figure.

Usage:
    python tsne.py --ckpt model.pt --model vgg16 --n 2000
"""

import argparse
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from model import build_model


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)
CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
COLORS  = plt.cm.tab10(np.linspace(0, 1, 10))


def get_features(model, loader, device):
    """Return the 512-d penultimate features (after avgpool, before fc)."""
    model.eval()
    feats, labels = [], []
    # We rebuild a forward hook to grab the input of the final Linear.
    feats_buf = []
    def hook(module, inp, out):
        feats_buf.append(inp[0].detach().cpu().numpy())
    handle = model.classifier[1].register_forward_hook(hook)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            model(x)
            labels.append(y.numpy())
    handle.remove()
    feats = np.concatenate(feats_buf, axis=0)        # (N, 512)
    labels = np.concatenate(labels, axis=0)
    return feats, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',  default='model.pt')
    p.add_argument('--model', default='vgg16')
    p.add_argument('--n', type=int, default=2000,
                   help='number of test points to project (for speed)')
    p.add_argument('--data_root', default='./data')
    p.add_argument('--out',  default='figures/tsne.png')
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(args.model, num_classes=10).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state['model'] if 'model' in state else state)

    tf = T.Compose([T.ToTensor(),
                    T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    test_set = torchvision.datasets.CIFAR10(
        root=args.data_root, train=False, download=True, transform=tf)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(test_set), size=args.n, replace=False)
    sub = Subset(test_set, idx.tolist())
    loader = DataLoader(sub, batch_size=256, shuffle=False, num_workers=2,
                        pin_memory=True)

    print("Extracting features ...")
    feats, labels = get_features(model, loader, device)
    print(f"  feature shape = {feats.shape}")

    print("Running t-SNE ...")
    z = TSNE(n_components=2, init='pca', random_state=42,
             perplexity=30).fit_transform(feats)

    fig, ax = plt.subplots(figsize=(6, 5))
    for c in range(10):
        mask = (labels == c)
        ax.scatter(z[mask, 0], z[mask, 1], s=5, alpha=0.7,
                   color=COLORS[c], label=CLASSES[c])
    ax.legend(markerscale=3, loc='best', fontsize=8)
    ax.set_title(f"t-SNE of {args.model} penultimate features (CIFAR-10 test)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"  saved {args.out}")


if __name__ == '__main__':
    main()
