"""
DASC7606 Assignment 1 -- Model statistics
=========================================
File: model_stats.py
Purpose: Print parameter counts and FLOPs for the three models.
         Output is a LaTeX-ready table that can be copy-pasted
         into report.pdf.

Usage:
    python model_stats.py
"""

import torch

from model import build_model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def try_count_flops(model, input_size=(1, 3, 32, 32)):
    """Best-effort FLOPs estimation using thop if available, else None."""
    try:
        from thop import profile
        macs, _ = profile(model, inputs=torch.randn(*input_size),
                          verbose=False)
        return int(macs) * 2  # 1 MAC ~ 2 FLOPs
    except Exception as e:
        print(f"  (thop not available: {e})")
        return None


def main():
    rows = []
    for name in ('baseline', 'improved', 'vgg16'):
        m = build_model(name)
        total, trainable = count_params(m)
        flops = try_count_flops(m)
        rows.append((name, total / 1e6, flops / 1e9 if flops else None))
        print(f"{name:9s}  params = {total/1e6:6.2f} M"
              f"{f'  FLOPs = {flops/1e9:5.2f} G' if flops else ''}")

    # LaTeX table
    print("\n--- copy-paste into report.tex ---")
    print(r"\begin{tabular}{l r r}")
    print(r"\toprule")
    print(r"Model & Params (M) & FLOPs (G) \\")
    print(r"\midrule")
    for name, p, f in rows:
        f_str = f"{f:.2f}" if f is not None else "--"
        print(f"{name:9s} & {p:6.2f} & {f_str:>5s} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == '__main__':
    main()
