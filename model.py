"""
DASC7606 Assignment 1 - CIFAR-10 Image Classification
======================================================
File: model.py
Purpose: Provide build_model() for the assignment submission.

Usage:
    from model import build_model
    model = build_model()
    # model.load_state_dict(torch.load('model.pt', map_location='cpu'))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Exercise 1: Baseline (3 conv + 2 FC, NO BN, NO Dropout)
# ---------------------------------------------------------------------------
class BaselineCNN(nn.Module):
    """The simplest baseline given in the assignment notebook.

    Three conv layers (each followed by ReLU + MaxPool) and two fully
    connected layers. NO batch-norm and NO dropout -- to show the
    starting point that we will improve on in Exercise 2/3.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # input: 3 x 32 x 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # 32 x 32 x 32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 32 x 16 x 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64 x 16 x 16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 64 x 8 x 8
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 128 x 8 x 8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 128 x 4 x 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
#  Exercise 2: Improved (5 conv + 3 FC, with BatchNorm + Dropout)
# ---------------------------------------------------------------------------
class ImprovedCNN(nn.Module):
    """Deeper network with BatchNorm and Dropout.

    This is the "example of possible improvements" mentioned in the
    assignment. It serves as a stepping stone towards Exercise 3.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            # block 1: 3 -> 64, 32x32
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 16x16
            # block 2: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 8x8
            # block 3: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
#  VGG-style configuration (used by ImprovedCNN and the final model)
# ---------------------------------------------------------------------------
VGG16_CFG = [64, 64, 'M',
             128, 128, 'M',
             256, 256, 256, 'M',
             512, 512, 512, 'M',
             512, 512, 512, 'M']


# ---------------------------------------------------------------------------
#  Exercise 3: Open-ended design -- the model submitted for grading
# ---------------------------------------------------------------------------
class VGG16(nn.Module):
    """VGG-16 style network (adapted for 32x32 CIFAR-10 images).

    Key design choices:
      * Every conv layer followed by BatchNorm + ReLU
      * Dropout (0.3) before the classifier head
      * AdaptiveAvgPool -> flatten -> Linear
      * ~14.7M parameters

    This is the model that build_model() returns.  Train it with
    `train.py` (uses Mixup/CutMix + AutoAugment + label smoothing +
    cosine LR schedule) for the best test accuracy.
    """

    def __init__(self,
                 cfg=VGG16_CFG,
                 num_classes: int = 10,
                 init_weights: bool = True,
                 dropout: float = 0.3):
        super().__init__()
        self.features = self._make_layers(cfg)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    @staticmethod
    def _make_layers(cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers += [
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True),
                ]
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
#  Public API -- THE function the grader will call
# ---------------------------------------------------------------------------
# The grader imports `build_model` and calls it with NO arguments:
#
#     from model import build_model
#     model = build_model()            # <-- this exact form
#     state = torch.load('model.pt', map_location='cpu')
#     state = state['model'] if isinstance(state, dict) and 'model' in state else state
#     model.load_state_dict(state)
#
# We also expose the same function with an optional `name` keyword so our
# own scripts (train.py, evaluate.py, ablation_runner.py, ...) can still
# pick the three exercise architectures (`baseline`, `improved`, `vgg16`)
# without having to import all three classes individually.

def build_model(*args, num_classes: int = 10, name: str = None, **_ignored):
    """Build the model used for this assignment.

    Calling forms
    -------------
      build_model()                       -> FinalModel (the submitted one)
      build_model('vgg16')                -> FinalModel
      build_model('improved')             -> ImprovedCNN
      build_model('baseline')             -> BaselineCNN
      build_model(name='vgg16')           -> FinalModel
      build_model(name='vgg16', num_classes=10)
    """
    # Decide which architecture was requested.
    if name is None:
        if args:
            name = args[0]
        else:
            name = 'vgg16'   # default == the submitted one

    name = name.lower()
    if name in ('vgg16', 'final', 'finalmodel'):
        return VGG16(num_classes=num_classes)
    if name == 'improved':
        return ImprovedCNN(num_classes=num_classes)
    if name == 'baseline':
        return BaselineCNN(num_classes=num_classes)
    raise ValueError(
        f"Unknown model name {name!r}. Choose from: 'vgg16', 'improved', 'baseline'."
    )


# Public alias for the architecture that build_model() returns by default.
FinalModel = VGG16


# ---------------------------------------------------------------------------
#  Sanity-check
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import torch
    # grader-style call
    m = build_model()
    n_params = sum(p.numel() for p in m.parameters())
    x = torch.randn(2, 3, 32, 32)
    y = m(x)
    print(f"build_model()            -> params={n_params/1e6:6.2f}M  out={tuple(y.shape)}")

    # compat-style calls
    for n in ('baseline', 'improved', 'vgg16'):
        m = build_model(n)
        n_params = sum(p.numel() for p in m.parameters())
        x = torch.randn(2, 3, 32, 32)
        y = m(x)
        print(f"build_model({n!r:>10s})    -> params={n_params/1e6:6.2f}M  out={tuple(y.shape)}")
