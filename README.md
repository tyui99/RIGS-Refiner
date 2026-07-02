# RIGS-Refiner

RIGS-Refiner is a lightweight frozen-host prediction-space refinement plugin for polyp segmentation.
This repository provides the training, evaluation, mask refinement, and dataset preparation code used for the paper.

Install:

```bash
pip install -r requirements.txt
```

Data:
- `docs/datasets.md`

Training:

```bash
python main.py --preset paper_main --data-root ./data/light/kvasirseg --host-checkpoint ./runs/checkpoints/host_a.pth
python main.py --preset ablation_t1 --data-root ./data/light/kvasirseg --host-checkpoint ./runs/checkpoints/host_a.pth
python main.py --preset ablation_t2 --data-root ./data/light/kvasirseg --host-checkpoint ./runs/checkpoints/host_a.pth
python main.py --preset ablation_no_risk_update --data-root ./data/light/kvasirseg --host-checkpoint ./runs/checkpoints/host_a.pth
python main.py --preset paper_alt_protocol --data-root ./data/light/kvasirseg --host-checkpoint ./runs/checkpoints/host_b.pth
```

Validation:

```bash
python scripts/eval.py --preset paper_main --checkpoint ./runs/checkpoints/paper_main.pth --data-root ./data/light/kvasirseg
```

Mask Refinement:

```bash
python scripts/postprocess.py --input-dir predictions/raw --output-dir predictions/refined
```
