"""Simple 3D behavior metrics for pre/post condition comparisons."""

from __future__ import annotations

import numpy as np


def centroid_trajectory(points_xyz: np.ndarray) -> np.ndarray:
    """Return the per-frame centroid for one mouse, ignoring missing joints."""
    points_xyz = np.asarray(points_xyz, dtype=float)
    with np.errstate(invalid="ignore"):
        return np.nanmean(points_xyz, axis=-2)


def compute_trajectory_metrics(points_xyz: np.ndarray, fps: float = 30.0) -> dict[str, float]:
    """Compute interpretable movement descriptors for one mouse trajectory."""
    centroid = centroid_trajectory(points_xyz)
    valid = np.isfinite(centroid).all(axis=1)
    if valid.sum() < 2:
        return {"valid_frame_fraction": float(valid.mean()), "path_length": float("nan"), "mean_speed": float("nan"), "peak_speed": float("nan")}
    trajectory = centroid[valid]
    displacement = np.diff(trajectory, axis=0)
    speed = np.linalg.norm(displacement, axis=1) * fps
    return {
        "valid_frame_fraction": float(valid.mean()),
        "path_length": float(np.linalg.norm(displacement, axis=1).sum()),
        "mean_speed": float(np.mean(speed)),
        "peak_speed": float(np.max(speed)),
    }


def compare_conditions(pre_points: np.ndarray, post_points: np.ndarray, fps: float = 30.0) -> dict[str, float]:
    """Return post-minus-pre changes for shared trajectory metrics."""
    pre = compute_trajectory_metrics(pre_points, fps)
    post = compute_trajectory_metrics(post_points, fps)
    return {f"delta_{key}": post[key] - pre[key] for key in pre}
