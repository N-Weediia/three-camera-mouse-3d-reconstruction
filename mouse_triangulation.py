"""Public, data-free core for three-view mouse keypoint reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TriangulationConfig:
    confidence_threshold: float = 0.5
    reprojection_threshold_px: float = 3.0
    ransac_iterations: int = 80
    max_interpolation_gap: int = 5


def project_point(point_xyz: np.ndarray, projection: np.ndarray) -> np.ndarray:
    homogeneous = projection @ np.append(np.asarray(point_xyz, dtype=float), 1.0)
    if abs(homogeneous[2]) < 1e-12:
        return np.array([np.nan, np.nan])
    return homogeneous[:2] / homogeneous[2]


def _triangulate(observations: np.ndarray, projections: np.ndarray, views: list[int]) -> np.ndarray | None:
    rows = []
    for view in views:
        x, y = observations[view, :2]
        matrix = projections[view]
        rows.extend((x * matrix[2] - matrix[0], y * matrix[2] - matrix[1]))
    try:
        _, _, vh = np.linalg.svd(np.asarray(rows, dtype=float))
    except np.linalg.LinAlgError:
        return None
    point = vh[-1]
    if abs(point[3]) < 1e-12:
        return None
    return point[:3] / point[3]


def _reprojection_errors(point_xyz: np.ndarray, observations: np.ndarray, projections: np.ndarray) -> np.ndarray:
    errors = np.full(len(observations), np.inf, dtype=float)
    for view, observation in enumerate(observations):
        if np.isfinite(observation[:2]).all():
            projected = project_point(point_xyz, projections[view])
            errors[view] = float(np.linalg.norm(projected - observation[:2]))
    return errors


def ransac_triangulate(
    observations: np.ndarray,
    projections: np.ndarray,
    config: TriangulationConfig | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Reconstruct one 3D keypoint from ``(x, y, confidence)`` observations.

    At least two confident views are required. The returned value is NaN when
    the views cannot produce a geometrically consistent reconstruction.
    """
    cfg = config or TriangulationConfig()
    observations = np.asarray(observations, dtype=float)
    projections = np.asarray(projections, dtype=float)
    valid = [
        view for view, item in enumerate(observations)
        if item[2] >= cfg.confidence_threshold and np.isfinite(item[:2]).all()
    ]
    if len(valid) < 2:
        return np.full(3, np.nan)

    generator = rng or np.random.default_rng(42)
    sample_size = min(3, len(valid))
    best_point = None
    best_inliers: list[int] = []
    for _ in range(cfg.ransac_iterations):
        sample = generator.choice(valid, size=sample_size, replace=False).tolist()
        point = _triangulate(observations, projections, sample)
        if point is None:
            continue
        errors = _reprojection_errors(point, observations, projections)
        inliers = [view for view in valid if errors[view] <= cfg.reprojection_threshold_px]
        if len(inliers) > len(best_inliers):
            best_point, best_inliers = point, inliers

    if best_point is None or len(best_inliers) < 2:
        return np.full(3, np.nan)
    refined = _triangulate(observations, projections, best_inliers)
    return refined if refined is not None else np.full(3, np.nan)


def reconstruct_sequence(
    observations: np.ndarray,
    projections: np.ndarray,
    config: TriangulationConfig | None = None,
) -> np.ndarray:
    """Reconstruct an array shaped ``(frames, mice, joints, views, 3)``."""
    observations = np.asarray(observations, dtype=float)
    frames, mice, joints, views, _ = observations.shape
    if projections.shape != (views, 3, 4):
        raise ValueError("projections must have shape (views, 3, 4)")
    output = np.full((frames, mice, joints, 3), np.nan, dtype=float)
    for frame in range(frames):
        for mouse in range(mice):
            for joint in range(joints):
                output[frame, mouse, joint] = ransac_triangulate(
                    observations[frame, mouse, joint], projections, config
                )
    return output


def interpolate_short_gaps(points_xyz: np.ndarray, max_gap: int = 5) -> np.ndarray:
    """Linearly fill short gaps in a ``(frames, ..., 3)`` trajectory."""
    result = np.array(points_xyz, dtype=float, copy=True)
    flat = result.reshape(result.shape[0], -1, 3)
    for track in range(flat.shape[1]):
        valid = np.isfinite(flat[:, track]).all(axis=1)
        valid_indices = np.flatnonzero(valid)
        if len(valid_indices) < 2:
            continue
        for left, right in zip(valid_indices[:-1], valid_indices[1:]):
            if 1 < right - left <= max_gap + 1:
                for frame in range(left + 1, right):
                    ratio = (frame - left) / (right - left)
                    flat[frame, track] = flat[left, track] * (1 - ratio) + flat[right, track] * ratio
    return flat.reshape(result.shape)
