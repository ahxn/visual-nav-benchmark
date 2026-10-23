from __future__ import annotations


def axial_depth_to_range(depth, focal_length_px: float):
    """Convert pinhole z-depth to Euclidean camera-ray distance."""
    import numpy as np

    values = np.asarray(depth, dtype=np.float32)
    height, width = values.shape
    u = np.arange(width, dtype=np.float32) + 0.5 - width / 2.0
    v = np.arange(height, dtype=np.float32) + 0.5 - height / 2.0
    uu, vv = np.meshgrid(u, v)
    scale = np.sqrt(1.0 + (uu / focal_length_px) ** 2 + (vv / focal_length_px) ** 2)
    return values * scale


class StereoDepthEstimator:
    """OpenCV stereo baseline producing metric camera-ray ranges in meters."""

    def __init__(self, focal_length_px: float, baseline_m: float, num_disparities: int = 64, block_size: int = 9):
        if baseline_m <= 0 or focal_length_px <= 0:
            raise ValueError("focal length and baseline must be positive")
        if num_disparities <= 0 or num_disparities % 16:
            raise ValueError("num_disparities must be a positive multiple of 16")
        self.focal_length_px = focal_length_px
        self.baseline_m = baseline_m
        self.num_disparities = num_disparities
        self.block_size = block_size

    def estimate(self, left, right):
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Install the 'vision' optional dependencies") from exc
        if left.shape[:2] != right.shape[:2]:
            raise ValueError("stereo images must have equal dimensions")
        # AI2-THOR's event.frame and third-party frames are RGB arrays.
        left_gray = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY) if left.ndim == 3 else left
        right_gray = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY) if right.ndim == 3 else right
        matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=8 * self.block_size**2,
            P2=32 * self.block_size**2,
        )
        disparity = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
        depth = np.full(disparity.shape, np.nan, dtype=np.float32)
        valid = disparity > 0
        depth[valid] = self.focal_length_px * self.baseline_m / disparity[valid]
        return axial_depth_to_range(depth, self.focal_length_px)

