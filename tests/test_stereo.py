import unittest


class StereoTests(unittest.TestCase):
    def test_axial_depth_converts_to_longer_off_axis_range(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("vision extras are not installed")
        from visual_nav.stereo import axial_depth_to_range

        ranges = axial_depth_to_range(np.ones((3, 3), dtype=np.float32), 1.0)
        self.assertAlmostEqual(float(ranges[1, 1]), 1.0)
        self.assertGreater(float(ranges[0, 0]), float(ranges[1, 1]))

    def test_shifted_texture_produces_finite_depth(self):
        try:
            import cv2  # noqa: F401
            import numpy as np
        except ImportError:
            self.skipTest("vision extras are not installed")
        from visual_nav.stereo import StereoDepthEstimator

        rng = np.random.default_rng(4)
        left = rng.integers(0, 256, size=(80, 128), dtype=np.uint8)
        right = np.zeros_like(left)
        right[:, :-8] = left[:, 8:]
        estimator = StereoDepthEstimator(100.0, 0.08, num_disparities=16, block_size=5)
        depth = estimator.estimate(left, right)
        finite = depth[np.isfinite(depth)]
        self.assertGreater(finite.size, 100)
        self.assertGreater(float(np.median(finite)), 0.5)


if __name__ == "__main__":
    unittest.main()

