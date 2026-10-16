"""
DASC7606 Assignment 1 - Evaluation script
=========================================
File: evaluate.py
Purpose: Load model.pt via build_model() and report test accuracy
         on CIFAR-10. This is the script the grader will run on
         the unseen evaluation data.

Usage (on HKU GPU Farm):
    python evaluate.py --ckpt model.pt --model vgg16
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from model import build_model


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = (
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
)


def get_test_loader(data_root: str = './data', batch_size: int = 256):
    tf = T.Compose([T.ToTensor(),
                    T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=tf)
    return DataLoader(test_set, batch_size=batch_size,
                      shuffle=False, num_workers=2, pin_memory=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',  default='model.pt')
    p.add_argument('--model', default='vgg16')
    p.add_argument('--data_root', default='./data')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--save_preds', default='test_predictions.npy',
                   help='if set, save predicted class indices to this .npy')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # IMPORTANT: this is how the grader will load the model.
    model = build_model(args.model, num_classes=10).to(device)
    state = torch.load(args.ckpt, map_location=device)
    if isinstance(state, dict) and 'model' in state:
        model.load_state_dict(state['model'])
    else:
        model.load_state_dict(state)
    model.eval()
    print(f"Loaded {args.model} from {args.ckpt}")

    loader = get_test_loader(args.data_root, args.batch_size)
    correct, total = 0, 0
    all_preds = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
            all_preds.append(preds.cpu().numpy())

    acc = correct / total
    print(f"Test accuracy: {acc:.4f}  ({correct}/{total})")

    if args.save_preds:
        all_preds = np.concatenate(all_preds)
        np.save(args.save_preds, all_preds)
        print(f"Saved predictions to {args.save_preds}  shape={all_preds.shape}")


if __name__ == '__main__':
    main()
