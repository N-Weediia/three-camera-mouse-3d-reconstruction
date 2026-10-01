import numpy as np

from demo_synthetic import make_projection_matrices
from mouse_triangulation import TriangulationConfig, ransac_triangulate


def test_three_view_ransac_reconstructs_point_with_one_low_confidence_view():
    projections = make_projection_matrices()
    point = np.array([0.15, 0.05, 3.0])
    observations = []
    for projection in projections:
        homogeneous = projection @ np.append(point, 1.0)
        observations.append([homogeneous[0] / homogeneous[2], homogeneous[1] / homogeneous[2], 0.98])
    observations = np.asarray(observations)
    observations[1, 2] = 0.1
    reconstructed = ransac_triangulate(observations, projections, TriangulationConfig(reprojection_threshold_px=1.0))
    assert np.linalg.norm(reconstructed - point) < 1e-5
