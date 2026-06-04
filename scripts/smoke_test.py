from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from motions_as_queries_unofficial import MotionQueryCapture


def main() -> None:
    torch.manual_seed(2026)
    video = torch.rand(2, 8, 3, 64, 64)
    target_trajectory = torch.rand(2, 3, 8, 2)
    target_pose = torch.rand(2, 3, 8, 34)

    model = MotionQueryCapture(num_queries=3, pose_dim=34, embed_dim=64)
    loss = model.training_loss(video, target_trajectory, target_pose)
    loss.backward()
    output = model(video)

    print(f"loss: {loss.item():.6f}")
    print(f"trajectory: {output.trajectory.shape}")
    print(f"pose: {output.pose.shape}")


if __name__ == "__main__":
    main()
