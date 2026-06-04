from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class MotionQueryOutput:
    trajectory: torch.Tensor
    pose: torch.Tensor
    query_features: torch.Tensor


class FrameEncoder(nn.Module):
    def __init__(self, image_channels: int, embed_dim: int):
        super().__init__()
        hidden = embed_dim // 2
        self.net = nn.Sequential(
            nn.Conv2d(image_channels, hidden, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8 if hidden % 8 == 0 else 1, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8 if embed_dim % 8 == 0 else 1, embed_dim),
            nn.SiLU(),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = video.shape
        x = video.view(batch * frames, channels, height, width)
        features = self.net(x).mean(dim=(-1, -2))
        return features.view(batch, frames, -1)


class MotionQueryCapture(nn.Module):
    """Toy one-stage multi-person motion-query model."""

    def __init__(
        self,
        image_channels: int = 3,
        num_queries: int = 3,
        embed_dim: int = 128,
        pose_dim: int = 34,
        num_heads: int = 4,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.pose_dim = pose_dim
        self.encoder = FrameEncoder(image_channels=image_channels, embed_dim=embed_dim)
        self.motion_queries = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(embed_dim)
        self.traj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, 2),
        )
        self.pose_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, pose_dim),
        )

    def forward(self, video: torch.Tensor) -> MotionQueryOutput:
        batch, frames = video.shape[:2]
        frame_tokens = self.encoder(video)
        queries = self.motion_queries[None].expand(batch, -1, -1)

        per_frame_features = []
        for t in range(frames):
            context = frame_tokens[:, : t + 1]
            attended, _ = self.temporal_attention(queries, context, context)
            per_frame_features.append(self.query_norm(attended + queries))
            queries = attended + queries

        query_features = torch.stack(per_frame_features, dim=2)
        trajectory = torch.sigmoid(self.traj_head(query_features))
        pose = self.pose_head(query_features)
        return MotionQueryOutput(
            trajectory=trajectory,
            pose=pose,
            query_features=query_features,
        )

    def training_loss(
        self,
        video: torch.Tensor,
        target_trajectory: torch.Tensor,
        target_pose: torch.Tensor,
    ) -> torch.Tensor:
        output = self.forward(video)
        traj_loss = F.smooth_l1_loss(output.trajectory, target_trajectory)
        pose_loss = F.smooth_l1_loss(output.pose, target_pose)
        return traj_loss + pose_loss
