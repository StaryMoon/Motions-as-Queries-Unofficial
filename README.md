# Motions-as-Queries-Unofficial

> Unofficial PyTorch implementation starter for **Motions as Queries: One-Stage Multi-Person Holistic Human Motion Capture** (CVPR 2025).
>
> If this repo helps you understand the "motion query" idea faster, please star it and follow [@StaryMoon](https://github.com/StaryMoon). I am building honest open reproduction starters for recent CVPR papers.

## Status

This repository is an **independent, unofficial, work-in-progress starter**.

- Paper: [Motions as Queries: One-Stage Multi-Person Holistic Human Motion Capture](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Motions_as_Queries_One-Stage_Multi-Person_Holistic_Human_Motion_Capture_CVPR_2025_paper.html)
- Venue: CVPR 2025, pp. 17529-17539
- Official code: the CVPR page says the method code will be made publicly available; this repo is not affiliated with the authors.
- Reproduction status: **benchmarks are not reproduced yet**.

## What Is Implemented

The first release is a compact toy implementation of the paper's high-level idea:

- video frame encoder
- learnable motion queries
- temporal cross-attention over frame tokens
- per-query trajectory head
- per-query pose head
- simple matching-free toy training loss
- smoke-test script

The goal is to expose the structure of "one query tracks one person's long motion" in readable PyTorch.

## What Is Not Implemented Yet

- exact paper architecture
- detector-free full-body / hand / face representation
- SMPL-X or body model fitting
- real datasets
- tracking metrics
- official checkpoints

## Quick Start

```bash
git clone https://github.com/StaryMoon/Motions-as-Queries-Unofficial.git
cd Motions-as-Queries-Unofficial
pip install -r requirements.txt
python scripts/smoke_test.py
```

Expected output:

```text
loss: ...
trajectory: torch.Size([2, 3, 8, 2])
pose: torch.Size([2, 3, 8, 34])
```

## Minimal Usage

```python
import torch

from motions_as_queries_unofficial import MotionQueryCapture

video = torch.rand(2, 8, 3, 64, 64)
model = MotionQueryCapture(num_queries=3, pose_dim=34)
out = model(video)
```

## Roadmap

- [ ] Add Hungarian matching for variable-person supervision.
- [ ] Add dataset loader for video clips and person tracks.
- [ ] Add SMPL-X / whole-body output adapter.
- [ ] Add tracking and pose evaluation metrics.
- [ ] Add visualization utilities.
- [ ] Reproduce a small benchmark table.

## Citation

If you use the method, please cite the original paper:

```bibtex
@InProceedings{Liu_2025_CVPR,
  author = {Liu, Kenkun and Fu, Yurong and Yuan, Weihao and Lin, Jing and Li, Peihao and Gu, Xiaodong and Qiu, Lingteng and Wang, Haoqian and Dong, Zilong and Han, Xiaoguang},
  title = {Motions as Queries: One-Stage Multi-Person Holistic Human Motion Capture},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2025},
  pages = {17529--17539}
}
```

## License

MIT License. The paper and official materials remain owned by their respective authors / publishers.
