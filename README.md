# DASC7606 Assignment 1 -- CIFAR-10 Image Classification

This repository contains a complete solution for the DASC7606
Assignment 1, organised around three experiments plus an extensive
analysis suite.  The **submitted model is `VGG16`** (built by
`build_model("vgg16")`).

## File layout

```
DASC7606_A1/
├── model.py             # build_model()  <-- the function the grader calls
├── train.py             # Training script (SGD + cosine LR + Mixup/CutMix/AutoAugment)
├── evaluate.py          # Re-evaluates the saved model.pt on CIFAR-10 test set
│
├── ablation_runner.py   # Sweep of 8 "change-one-thing" ablations -> ablation_results.csv
├── model_stats.py       # Print parameter counts (+ FLOPs) for the three models
├── lr_sweep.py          # Train with 4 initial LRs to study sensitivity
├── visualize.py         # Confusion matrix, per-class acc, misclassified grid,
│                        # augmentation grid, training curves
├── tsne.py              # t-SNE of penultimate features
├── robustness.py        # Robustness to Gaussian noise and cutout
│
├── report.tex           # LaTeX source for the 2-page report
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── SUBMIT.md            # Submission checklist + time budget
```

## Environment (HKU GPU Farm Phase 2)

```bash
pip install -r requirements.txt
```

## End-to-end pipeline

```bash
# 1. Train the final model (submitted as model.pt) -- ~30-45 min on 1 GPU
python train.py --model vgg16 --epochs 100 --batch_size 128 --lr 0.1 \
                --save_path model.pt

# 2. Re-evaluate (this is what the grader runs)
python evaluate.py --ckpt model.pt --model vgg16

# 3. Generate every figure for the report
python visualize.py --ckpt model.pt --model vgg16 \
                    --log logs/vgg16_<latest>.log
python tsne.py       --ckpt model.pt --model vgg16
python robustness.py --ckpt model.pt --model vgg16
python model_stats.py
python lr_sweep.py   --epochs 20

# 4. (Optional but recommended) run the full ablation study
python ablation_runner.py --epochs 30
```

## Reproducing the smaller experiments

```bash
python train.py --model baseline  --epochs 20 --lr 1e-3 --mix_alpha 0 --save_path exp1.pt
python train.py --model improved  --epochs 50 --lr 0.05 --mix_alpha 0 --save_path exp2.pt
```

## Hardware / software notes

* Single GPU is sufficient (VGG-16 / CIFAR-10 fits in <2 GB).
* Approximate training time on one GPU of the Farm: 30-45 min
  for 100 epochs.
* Ablation sweep (8 runs × 30 epochs): ~5-6 hours on one GPU.
* No pre-trained weights are used.  Only CIFAR-10 is used for
  training, in line with the assignment rules.

## Marking-scheme coverage

| Marking item                                       | Where it lives                                |
|----------------------------------------------------|-----------------------------------------------|
| Coding (network, training, evaluation)             | `model.py`, `train.py`, `evaluate.py`         |
| Test-set accuracy                                  | `evaluate.py` -> `model.pt`                   |
| **Reasonable number of experiments** (90-100% rep.)| `ablation_runner.py` (8 ablations)            |
| Visualisation / evaluation                         | `visualize.py`, `tsne.py`                     |
| Hyper-parameter tuning                             | `lr_sweep.py`                                 |
| Per-class error analysis                           | `visualize.py:plot_per_class_accuracy`        |
| Robustness                                         | `robustness.py`                               |
