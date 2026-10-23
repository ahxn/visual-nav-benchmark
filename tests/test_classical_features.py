import unittest


class ClassicalFeatureTests(unittest.TestCase):
    def setUp(self):
        try:
            import cv2  # noqa: F401
            import numpy as np  # noqa: F401
        except ImportError:
            self.skipTest("vision extras are not installed")

    def test_harris_detects_checkerboard_corners(self):
        import numpy as np
        from visual_nav.classical_features import harris_corners

        image = np.zeros((80, 80), dtype=np.uint8)
        for y in range(0, 80, 20):
            for x in range(0, 80, 20):
                if (x // 20 + y // 20) % 2:
                    image[y:y + 20, x:x + 20] = 255
        self.assertGreaterEqual(len(harris_corners(image)), 8)

    def test_sift_and_sparse_stereo_recover_shifted_texture(self):
        import cv2
        import numpy as np
        from visual_nav.classical_features import match_sift, sparse_stereo_depth

        rng = np.random.default_rng(9)
        left = rng.integers(0, 256, size=(160, 240), dtype=np.uint8)
        left = cv2.GaussianBlur(left, (3, 3), 0)
        right = np.zeros_like(left)
        right[:, :-8] = left[:, 8:]
        matches = match_sift(left, right)
        result = sparse_stereo_depth(left, right, focal_length_px=120.0, baseline_m=0.08)
        finite = result.depth[np.isfinite(result.depth)]
        self.assertGreater(len(matches.points_a), 20)
        self.assertGreater(len(finite), 10)
        self.assertAlmostEqual(float(np.median(finite)), 1.2, delta=0.25)

    def test_essential_matrix_ransac_rejects_outliers(self):
        import cv2
        import numpy as np
        from visual_nav.classical_features import camera_matrix, estimate_pose_from_correspondences

        rng = np.random.default_rng(5)
        intrinsics = camera_matrix(320, 240, 70)
        points_3d = np.column_stack((
            rng.uniform(-1, 1, 80), rng.uniform(-0.7, 0.7, 80), rng.uniform(3, 7, 80)
        ))

        def project(points):
            pixels = (intrinsics @ points.T).T
            return pixels[:, :2] / pixels[:, 2, None]

        points_a = project(points_3d)
        rotation, _ = cv2.Rodrigues(np.array([0.0, 0.035, 0.0]))
        points_b = project((rotation @ points_3d.T).T + np.array([0.2, 0.0, 0.0]))
        points_b[:12] = rng.uniform([0, 0], [320, 240], size=(12, 2))
        pose = estimate_pose_from_correspondences(points_a, points_b, intrinsics)
        self.assertIsNotNone(pose)
        self.assertGreater(pose.inliers, 55)
        self.assertLess(pose.inliers, 80)

    def test_visual_odometry_angle_metrics(self):
        import numpy as np
        from visual_nav.vo_benchmark import direction_error_degrees, rotation_error_degrees

        self.assertAlmostEqual(direction_error_degrees([0, 0, -1]), 0.0)
        self.assertAlmostEqual(direction_error_degrees([1, 0, 0]), 90.0)
        self.assertAlmostEqual(rotation_error_degrees(np.eye(3)), 0.0)


if __name__ == "__main__":
    unittest.main()
