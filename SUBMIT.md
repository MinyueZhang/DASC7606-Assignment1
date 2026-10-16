# DASC 7606 Assignment 1 -- Submission checklist

Deadline: **2026-10-18 (Sun) 23:59 GMT+8**

## Files to submit (Moodle)

| # | Item                        | Status |
|---|-----------------------------|--------|
| 1 | `report.pdf` (up to 2 pages) | TODO -- compile `report.tex` |
| 2 | `Assignment_1.ipynb`        | TODO -- copy from GPU Farm, add our code & outputs |
| 3 | `model.py`                  | DONE   -- contains `build_model()` |
| 4 | `README.md`                 | DONE   -- environment + how to reproduce |
| 5 | `model.pt`                  | TODO -- produced by `train.py` on the GPU Farm |

## One-time submission

1. SSH into the GPU Farm.
2. Get an interactive GPU node: `gpu-interactive` (1 GPU / 4 CPU / 28 GB RAM).
3. Set up the env (see `README.md`).
4. `python train.py --model vgg16 --epochs 100 --save_path model.pt`
5. `python evaluate.py --ckpt model.pt --model vgg16`
6. **If test acc looks reasonable, commit.  Only one submission is allowed.**

## Local pre-flight (on the Farm, before submitting)

```bash
# make sure build_model() loads the saved weights
python - <<'PY'
import torch
from model import build_model
m = build_model('vgg16')
state = torch.load('model.pt', map_location='cpu')
m.load_state_dict(state['model'] if 'model' in state else state)
print("OK -- keys match:", m.classifier[-1].out_features)
PY
```

If the script above prints `OK`, the model is loadable by the
grader's evaluator.

## Time budget

| When        | What                                                |
|-------------|-----------------------------------------------------|
| Fri evening | Get GPU access, run training script, check it converges |
| Sat morning | Fill in ablation numbers in `report.tex`           |
| Sat evening | Compile `report.pdf`, build the zip, double-check  |
| Sun 22:00   | Submit on Moodle                                    |
