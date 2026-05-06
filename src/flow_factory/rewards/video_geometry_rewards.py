# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VGGRPO-style video geometry rewards.

This module provides draft implementations of the two reward terms introduced in
VGGRPO (arXiv:2603.26599):
1. Camera motion smoothness reward.
2. Geometry reprojection consistency reward.

The implementation is designed to be practical in Flow-Factory today:
- If geometry predictions (camera/point/depth) are provided in Sample fields,
  the reward uses them directly.
- Otherwise, it falls back to deterministic video-space approximations so the
  reward can run without an external geometry foundation model.

Note on geometry field routing:
  Geometry fields (camera_centers, camera_angular_velocity, point_maps_world,
  depths, camera_w2c, intrinsics, scene_flow) are not declared on any task-level
  Sample dataclass. To use the geometry-aware code path, populate these fields
  via ``BaseSample.extra_kwargs`` (e.g. ``sample.extra_kwargs["camera_centers"] =
  ...``). The RewardProcessor calls ``filter_kwargs(model.__call__,
  **sample.to_dict())``, which merges extra_kwargs into the top-level dict, so
  any field matching a named parameter on ``__call__`` is forwarded automatically.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from PIL import Image

from .abc import PointwiseRewardModel, RewardModelOutput
from ..hparams import RewardArguments


def _safe_float(value: Any, default: float, name: str) -> float:
    """Return ``value`` as float when provided, else ``default``.

    Args:
        value: Value to coerce to float.
        default: Default when value is ``None``.
        name: Configuration field name for error reporting.

    Returns:
        Float value from ``value`` or ``default``.

    Raises:
        ValueError: If ``value`` is provided but cannot be converted to float.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name} value: {value!r}. Expected a float.") from exc


class CameraMotionSmoothnessReward(PointwiseRewardModel):
    """Compute VGGRPO-like camera smoothness reward from video or camera tracks.

    The reward follows the paper's structure:
    - translational smoothness from normalized acceleration,
    - rotational smoothness from normalized angular acceleration,
    - mapped to ``[0, 1]`` via ``e -> 1 / (1 + e)``.

    Inputs are expected to be batched by ``RewardProcessor``.
    """

    required_fields = ("video",)
    use_tensor_inputs = True

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        """Initialize reward hyperparameters.

        Args:
            config: Reward configuration for this instance.
            accelerator: Accelerator handle for device placement.
        """
        super().__init__(config, accelerator)
        self.eps = _safe_float(config.extra_kwargs.get("eps"), 1e-6, "eps")

    @staticmethod
    def _to_video_tensor(frames: torch.Tensor) -> torch.Tensor:
        """Normalize one sample to shape ``(T, C, H, W)`` and float dtype."""
        if frames.ndim != 4:
            raise ValueError(
                f"Expected video tensor with shape (T,C,H,W), got {tuple(frames.shape)}"
            )
        return frames.float()

    def _motion_from_camera_centers(
        self,
        centers: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return translational velocity and acceleration from camera centers."""
        if centers.ndim != 2 or centers.shape[-1] != 3:
            return None, None
        if centers.shape[0] < 3:
            return None, None

        vel = centers[1:] - centers[:-1]
        acc = vel[1:] - vel[:-1]
        return vel, acc

    def _rotation_from_video(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate angular velocity and acceleration from frame moments."""
        t, _, h, w = frames.shape
        gray = frames.mean(dim=1)
        gray = gray - gray.amin(dim=(1, 2), keepdim=True)
        weight = gray + self.eps

        ys = torch.linspace(-1.0, 1.0, h, device=frames.device, dtype=frames.dtype).view(1, h, 1)
        xs = torch.linspace(-1.0, 1.0, w, device=frames.device, dtype=frames.dtype).view(1, 1, w)

        denom = weight.sum(dim=(1, 2), keepdim=True) + self.eps
        cx = (weight * xs).sum(dim=(1, 2), keepdim=True) / denom
        cy = (weight * ys).sum(dim=(1, 2), keepdim=True) / denom

        x0 = xs - cx
        y0 = ys - cy
        mu20 = (weight * x0.square()).sum(dim=(1, 2)) / (denom.squeeze(-1).squeeze(-1) + self.eps)
        mu02 = (weight * y0.square()).sum(dim=(1, 2)) / (denom.squeeze(-1).squeeze(-1) + self.eps)
        mu11 = (weight * x0 * y0).sum(dim=(1, 2)) / (denom.squeeze(-1).squeeze(-1) + self.eps)

        theta = 0.5 * torch.atan2(2.0 * mu11, mu20 - mu02 + self.eps)
        omega = theta[1:] - theta[:-1]
        omega = (omega + torch.pi) % (2 * torch.pi) - torch.pi
        alpha = omega[1:] - omega[:-1]

        if t < 3:
            return torch.zeros(0, device=frames.device), torch.zeros(0, device=frames.device)
        return omega, alpha

    def _translation_from_video(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate translational velocity and acceleration from center-of-mass."""
        t, _, h, w = frames.shape
        gray = frames.mean(dim=1)
        gray = gray - gray.amin(dim=(1, 2), keepdim=True)
        weight = gray + self.eps

        ys = torch.linspace(-1.0, 1.0, h, device=frames.device, dtype=frames.dtype).view(1, h, 1)
        xs = torch.linspace(-1.0, 1.0, w, device=frames.device, dtype=frames.dtype).view(1, 1, w)

        denom = weight.sum(dim=(1, 2), keepdim=True) + self.eps
        cx = (weight * xs).sum(dim=(1, 2), keepdim=True) / denom
        cy = (weight * ys).sum(dim=(1, 2), keepdim=True) / denom
        centers = torch.cat([cx.squeeze(-1), cy.squeeze(-1)], dim=-1)

        if t < 3:
            return torch.zeros(0, 2, device=frames.device), torch.zeros(0, 2, device=frames.device)

        vel = centers[1:] - centers[:-1]
        acc = vel[1:] - vel[:-1]
        return vel, acc

    def _normalized_acc_error(self, vel: torch.Tensor, acc: torch.Tensor) -> torch.Tensor:
        """Compute scale-normalized acceleration error used by VGGRPO."""
        if vel.shape[0] < 2 or acc.shape[0] < 1:
            return torch.tensor(1.0, device=self.device)

        num = torch.linalg.norm(acc, dim=-1)
        den = torch.linalg.norm(vel[1:], dim=-1) + torch.linalg.norm(vel[:-1], dim=-1) + self.eps
        return (num / den).mean()

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[torch.Tensor]] = None,
        condition_images: Optional[List[Union[List[Image.Image], torch.Tensor]]] = None,
        condition_videos: Optional[List[Union[List[List[Image.Image]], torch.Tensor]]] = None,
        camera_centers: Optional[List[torch.Tensor]] = None,
        camera_angular_velocity: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        """Compute per-sample camera smoothness reward.

        Args:
            prompt: Prompt list used to infer batch size.
            video: List of tensors with shape ``(T, C, H, W)``.
            camera_centers: Optional list of camera trajectories ``(T, 3)``.
            camera_angular_velocity: Optional list of angular velocity tracks.
            **kwargs: Unused extra fields from samples.

        Returns:
            RewardModelOutput: Rewards in shape ``(batch_size,)`` with range ``[0, 1]``.
        """
        del image, condition_images, condition_videos, kwargs

        batch_size = len(prompt)
        rewards = torch.zeros(batch_size, device=self.device, dtype=torch.float32)

        if video is None:
            return RewardModelOutput(rewards=rewards)

        for idx, frames in enumerate(video):
            frames = self._to_video_tensor(frames).to(self.device)

            vel_trans, acc_trans = self._translation_from_video(frames)
            if camera_centers is not None and camera_centers[idx] is not None:
                candidate_vel, candidate_acc = self._motion_from_camera_centers(
                    camera_centers[idx].to(self.device).float()
                )
                if candidate_vel is not None and candidate_acc is not None:
                    vel_trans, acc_trans = candidate_vel, candidate_acc

            if camera_angular_velocity is not None and camera_angular_velocity[idx] is not None:
                omega = camera_angular_velocity[idx].to(self.device).float().reshape(-1)
                alpha = omega[1:] - omega[:-1]
            else:
                omega, alpha = self._rotation_from_video(frames)

            e_trans = self._normalized_acc_error(vel_trans, acc_trans)
            e_rot = self._normalized_acc_error(omega.unsqueeze(-1), alpha.unsqueeze(-1))

            r_motion = 0.5 * (1.0 / (1.0 + e_trans) + 1.0 / (1.0 + e_rot))
            rewards[idx] = r_motion

        return RewardModelOutput(
            rewards=rewards,
            extra_info={"reward_type": "camera_motion_smoothness"},
        )


class GeometryReprojectionConsistencyReward(PointwiseRewardModel):
    """Compute VGGRPO-like geometry reprojection consistency reward.

    The reward maps per-view reprojection error to ``[0, 1]`` via
    ``e -> 1 / (1 + e)``, mirroring the motion reward's formulation.

    This draft supports two modes:
    1. Geometry-aware mode when point/depth/camera tensors are provided.
    2. Video-only fallback using translation-aligned depth proxies.
    """

    required_fields = ("video",)
    use_tensor_inputs = True

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        """Initialize reprojection reward hyperparameters.

        Args:
            config: Reward configuration for this instance.
            accelerator: Accelerator handle for device placement.
        """
        super().__init__(config, accelerator)
        self.eps = _safe_float(config.extra_kwargs.get("eps"), 1e-6, "eps")
        self.max_points = int(config.extra_kwargs.get("max_points", 20000))
        self.static_flow_threshold = _safe_float(
            config.extra_kwargs.get("static_flow_threshold"), 0.02, "static_flow_threshold"
        )

    @staticmethod
    def _to_chw_depth(depth: torch.Tensor) -> torch.Tensor:
        """Normalize depth tensor to shape ``(T, H, W)``."""
        if depth.ndim == 4 and depth.shape[1] == 1:
            depth = depth[:, 0]
        if depth.ndim != 3:
            raise ValueError(f"Expected depth shape (T,H,W) or (T,1,H,W), got {tuple(depth.shape)}")
        return depth.float()

    @staticmethod
    def _to_point_map(points: torch.Tensor) -> torch.Tensor:
        """Normalize point map tensor to shape ``(T, H, W, 3)``."""
        if points.ndim != 4:
            raise ValueError(f"Expected point maps with 4 dims, got {tuple(points.shape)}")

        if points.shape[-1] == 3:
            return points.float()
        if points.shape[1] == 3:
            return points.permute(0, 2, 3, 1).float()

        raise ValueError(
            "Point maps must have channel=3 either in dim=1 or dim=-1, "
            f"got shape {tuple(points.shape)}"
        )

    def _estimate_translation(self, frame_a: torch.Tensor, frame_b: torch.Tensor) -> torch.Tensor:
        """Estimate global frame translation from intensity center-of-mass."""
        _, h, w = frame_a.shape
        gray_a = frame_a.mean(dim=0)
        gray_b = frame_b.mean(dim=0)
        gray_a = gray_a - gray_a.min()
        gray_b = gray_b - gray_b.min()

        wa = gray_a + self.eps
        wb = gray_b + self.eps

        ys = torch.linspace(-1.0, 1.0, h, device=frame_a.device, dtype=frame_a.dtype).view(h, 1)
        xs = torch.linspace(-1.0, 1.0, w, device=frame_a.device, dtype=frame_a.dtype).view(1, w)

        ca_x = (wa * xs).sum() / (wa.sum() + self.eps)
        ca_y = (wa * ys).sum() / (wa.sum() + self.eps)
        cb_x = (wb * xs).sum() / (wb.sum() + self.eps)
        cb_y = (wb * ys).sum() / (wb.sum() + self.eps)

        return torch.stack([cb_x - ca_x, cb_y - ca_y], dim=0)

    def _warp_by_translation(self, frame: torch.Tensor, shift_xy: torch.Tensor) -> torch.Tensor:
        """Warp one frame by normalized xy shift using bilinear sampling."""
        c, h, w = frame.shape
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=frame.device, dtype=frame.dtype),
            torch.linspace(-1.0, 1.0, w, device=frame.device, dtype=frame.dtype),
            indexing="ij",
        )
        grid = torch.stack([x - shift_xy[0], y - shift_xy[1]], dim=-1).unsqueeze(0)
        warped = F.grid_sample(
            frame.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return warped[0]

    def _video_fallback_error(self, frames: torch.Tensor) -> torch.Tensor:
        """Compute per-view consistency errors from aligned frame differences."""
        t = frames.shape[0]
        if t < 2:
            return torch.zeros(1, device=frames.device)

        errors = []
        for i in range(t - 1):
            shift = self._estimate_translation(frames[i], frames[i + 1])
            aligned_next = self._warp_by_translation(frames[i + 1], shift)
            error = torch.abs(frames[i] - aligned_next).mean()
            errors.append(error)

        return torch.stack(errors, dim=0)

    def _infer_intrinsics(
        self, h: int, w: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Create simple pinhole intrinsics when none are provided."""
        fx = float(w)
        fy = float(h)
        cx = float(w) / 2.0
        cy = float(h) / 2.0
        return torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        )

    def _build_static_cloud(
        self,
        points_world: torch.Tensor,
        scene_flow: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Aggregate a static point cloud from per-frame world point maps."""
        t, h, w, _ = points_world.shape
        flat_points = points_world.reshape(t, h * w, 3)

        if scene_flow is not None:
            flow = scene_flow.float()
            if flow.ndim == 5 and flow.shape[-1] == 3:
                flow_mag = torch.linalg.norm(flow, dim=-1)
            elif flow.ndim == 5 and flow.shape[2] == 3:
                flow_mag = torch.linalg.norm(flow.permute(0, 1, 3, 4, 2), dim=-1)
            else:
                raise ValueError(
                    "scene_flow must have shape (T-1,H,W,3) or (T-1,3,H,W), "
                    f"got {tuple(flow.shape)}"
                )
            static_mask = flow_mag < self.static_flow_threshold
            static_mask = torch.cat([static_mask, static_mask[-1:]], dim=0)
            flat_mask = static_mask.reshape(t, h * w)
            cloud = flat_points[flat_mask]
        else:
            cloud = flat_points.reshape(-1, 3)

        if cloud.numel() == 0:
            return torch.zeros(0, 3, device=points_world.device, dtype=points_world.dtype)

        if cloud.shape[0] > self.max_points:
            idx = torch.linspace(0, cloud.shape[0] - 1, self.max_points, device=cloud.device)
            cloud = cloud[idx.long()]

        return cloud

    def _render_depth(
        self,
        cloud_world: torch.Tensor,
        cam_w2c: torch.Tensor,
        intrinsics: torch.Tensor,
        h: int,
        w: int,
    ) -> torch.Tensor:
        """Render z-buffer depth map by projecting cloud into one camera view."""
        if cloud_world.numel() == 0:
            return torch.full((h, w), torch.inf, device=cam_w2c.device, dtype=cam_w2c.dtype)

        ones = torch.ones(
            cloud_world.shape[0], 1, device=cloud_world.device, dtype=cloud_world.dtype
        )
        cloud_h = torch.cat([cloud_world, ones], dim=-1)

        cam_xyz_h = (cam_w2c @ cloud_h.t()).t()
        cam_xyz = cam_xyz_h[:, :3]
        z = cam_xyz[:, 2]

        valid_z = z > self.eps
        if valid_z.sum() == 0:
            return torch.full((h, w), torch.inf, device=cam_w2c.device, dtype=cam_w2c.dtype)

        cam_xyz = cam_xyz[valid_z]
        z = z[valid_z]

        u = (intrinsics[0, 0] * (cam_xyz[:, 0] / z) + intrinsics[0, 2]).round().long()
        v = (intrinsics[1, 1] * (cam_xyz[:, 1] / z) + intrinsics[1, 2]).round().long()

        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if valid.sum() == 0:
            return torch.full((h, w), torch.inf, device=cam_w2c.device, dtype=cam_w2c.dtype)

        u = u[valid]
        v = v[valid]
        z = z[valid]

        depth = torch.full((h * w,), torch.inf, device=cam_w2c.device, dtype=cam_w2c.dtype)
        linear_idx = v * w + u
        depth.scatter_reduce_(0, linear_idx, z, reduce="amin", include_self=True)
        return depth.view(h, w)

    def _geometry_error(
        self,
        point_maps_world: torch.Tensor,
        depths: torch.Tensor,
        camera_w2c: torch.Tensor,
        intrinsics: Optional[torch.Tensor],
        scene_flow: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute per-view reprojection depth error as in VGGRPO Eq. 14."""
        t, h, w, _ = point_maps_world.shape
        cloud = self._build_static_cloud(point_maps_world, scene_flow)

        if intrinsics is None:
            intrinsics = self._infer_intrinsics(
                h, w, device=point_maps_world.device, dtype=point_maps_world.dtype
            )
            intrinsics = intrinsics.unsqueeze(0).repeat(t, 1, 1)

        per_view_error = []
        for i in range(t):
            rendered = self._render_depth(cloud, camera_w2c[i], intrinsics[i], h, w)
            depth_i = depths[i]
            valid = torch.isfinite(rendered) & torch.isfinite(depth_i) & (depth_i > 0)

            if valid.sum() == 0:
                per_view_error.append(torch.tensor(1.0, device=point_maps_world.device))
            else:
                per_view_error.append(torch.abs(rendered[valid] - depth_i[valid]).mean())

        return torch.stack(per_view_error, dim=0)

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[torch.Tensor]] = None,
        condition_images: Optional[List[Union[List[Image.Image], torch.Tensor]]] = None,
        condition_videos: Optional[List[Union[List[List[Image.Image]], torch.Tensor]]] = None,
        point_maps_world: Optional[List[torch.Tensor]] = None,
        depths: Optional[List[torch.Tensor]] = None,
        camera_w2c: Optional[List[torch.Tensor]] = None,
        intrinsics: Optional[List[torch.Tensor]] = None,
        scene_flow: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        """Compute per-sample geometry consistency reward.

        Args:
            prompt: Prompt list used to infer batch size.
            video: List of tensors with shape ``(T, C, H, W)``.
            point_maps_world: Optional world-space point maps ``(T,H,W,3)`` or ``(T,3,H,W)``.
            depths: Optional depth maps ``(T,H,W)`` or ``(T,1,H,W)``.
            camera_w2c: Optional camera extrinsics ``(T,4,4)``.
            intrinsics: Optional intrinsics ``(T,3,3)``.
            scene_flow: Optional scene flow ``(T-1,H,W,3)`` or ``(T-1,3,H,W)``.
            **kwargs: Unused extra fields from samples.

        Returns:
            RewardModelOutput: Rewards in shape ``(batch_size,)`` with range ``[0, 1]``.
            The value follows VGGRPO Eq. 15, i.e., ``1 / (1 + mean_error)`` over worst views.
        """
        del image, condition_images, condition_videos, kwargs

        batch_size = len(prompt)
        rewards = torch.zeros(batch_size, device=self.device, dtype=torch.float32)

        if video is None:
            return RewardModelOutput(rewards=rewards)

        for idx, frames in enumerate(video):
            frames = frames.to(self.device).float()

            have_geometry = (
                point_maps_world is not None
                and depths is not None
                and camera_w2c is not None
                and point_maps_world[idx] is not None
                and depths[idx] is not None
                and camera_w2c[idx] is not None
            )

            if have_geometry:
                points_i = self._to_point_map(point_maps_world[idx].to(self.device))
                depths_i = self._to_chw_depth(depths[idx].to(self.device))
                cam_i = camera_w2c[idx].to(self.device).float()

                intr_i = None
                if intrinsics is not None and intrinsics[idx] is not None:
                    intr_i = intrinsics[idx].to(self.device).float()

                flow_i = None
                if scene_flow is not None and scene_flow[idx] is not None:
                    flow_i = scene_flow[idx].to(self.device).float()

                per_view_errors = self._geometry_error(
                    point_maps_world=points_i,
                    depths=depths_i,
                    camera_w2c=cam_i,
                    intrinsics=intr_i,
                    scene_flow=flow_i,
                )
            else:
                per_view_errors = self._video_fallback_error(frames)

            k = min(3, per_view_errors.shape[0])
            worst = torch.topk(per_view_errors, k=k, largest=True).values
            rewards[idx] = 1.0 / (1.0 + worst.mean())

        return RewardModelOutput(
            rewards=rewards,
            extra_info={"reward_type": "geometry_reprojection_consistency"},
        )


class CombinedVideoGeometryReward(PointwiseRewardModel):
    """Combine the two VGGRPO reward components into one reward signal.

    Both sub-rewards output values in ``[0, 1]``, so the weighted sum
    lies in ``[0, motion_weight + geometry_weight]``.
    """

    required_fields = ("video",)
    use_tensor_inputs = True

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        """Initialize sub-rewards and combination weights.

        Args:
            config: Reward configuration for this instance.
            accelerator: Accelerator handle for device placement.
        """
        super().__init__(config, accelerator)
        self.motion_reward = CameraMotionSmoothnessReward(config, accelerator)
        self.geometry_reward = GeometryReprojectionConsistencyReward(config, accelerator)

        self.motion_weight = _safe_float(
            config.extra_kwargs.get("motion_weight"), 0.5, "motion_weight"
        )
        self.geometry_weight = _safe_float(
            config.extra_kwargs.get("geometry_weight"), 0.5, "geometry_weight"
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[torch.Tensor]] = None,
        condition_images: Optional[List[Union[List[Image.Image], torch.Tensor]]] = None,
        condition_videos: Optional[List[Union[List[List[Image.Image]], torch.Tensor]]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        """Compute weighted sum of camera and geometry reward components.

        Args:
            prompt: Prompt list.
            image: Optional image list.
            video: Optional video list.
            condition_images: Optional conditioning images.
            condition_videos: Optional conditioning videos.
            **kwargs: Additional optional geometry fields passed through.

        Returns:
            RewardModelOutput: Weighted reward tensor with range ``[0, motion_weight + geometry_weight]``.
                Per-component values are included in extra_info.
        """
        motion = self.motion_reward(
            prompt=prompt,
            image=image,
            video=video,
            condition_images=condition_images,
            condition_videos=condition_videos,
            **kwargs,
        )
        geometry = self.geometry_reward(
            prompt=prompt,
            image=image,
            video=video,
            condition_images=condition_images,
            condition_videos=condition_videos,
            **kwargs,
        )

        combined = self.motion_weight * motion.rewards + self.geometry_weight * geometry.rewards

        return RewardModelOutput(
            rewards=combined,
            extra_info={
                "camera_motion": motion.rewards,
                "geometry_reprojection": geometry.rewards,
            },
        )
