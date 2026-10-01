"""End-to-end synthetic three-camera reconstruction demo."""

import numpy as np

from mouse_triangulation import TriangulationConfig, reconstruct_sequence
from trajectory_metrics import compute_trajectory_metrics


def make_projection_matrices() -> np.ndarray:
    focal = 800.0
    center = np.array([320.0, 240.0])
    matrices = []
    for baseline in (0.0, 0.8, -0.7):
        matrices.append(np.array([[focal, 0, center[0], -focal * baseline], [0, focal, center[1], 0], [0, 0, 1, 0]], dtype=float))
    return np.stack(matrices)


def main() -> None:
    rng = np.random.default_rng(7)
    frames, mice, joints, views = 40, 2, 8, 3
    projections = make_projection_matrices()
    truth = np.zeros((frames, mice, joints, 3), dtype=float)
    for frame in range(frames):
        for mouse in range(mice):
            center = np.array([-0.25 + mouse * 0.5 + frame * 0.005, 0.05, 3.0])
            truth[frame, mouse] = center + rng.normal(0, 0.08, (joints, 3))

    observations = np.full((frames, mice, joints, views, 3), np.nan)
    for f in range(frames):
        for m in range(mice):
            for j in range(joints):
                for v in range(views):
                    homogeneous = projections[v] @ np.append(truth[f, m, j], 1.0)
                    observations[f, m, j, v, :2] = homogeneous[:2] / homogeneous[2] + rng.normal(0, 0.25, 2)
                    observations[f, m, j, v, 2] = 0.98
    observations[8, 0, 2, 1, 2] = 0.1
    reconstructed = reconstruct_sequence(observations, projections, TriangulationConfig(reprojection_threshold_px=2.0),)
    valid = np.isfinite(reconstructed).all(axis=-1)
    median_error = float(np.median(np.linalg.norm(reconstructed[valid] - truth[valid], axis=-1)))
    metrics = compute_trajectory_metrics(reconstructed[:, 0])
    print(f"valid reconstructions: {valid.mean():.1%}")
    print(f"median 3D error: {median_error:.5f} m")
    print(f"mouse-0 mean speed: {metrics['mean_speed']:.4f} m/s")


if __name__ == "__main__":
    main()
