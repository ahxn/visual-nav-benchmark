from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureMatches:
    points_a: object
    points_b: object
    keypoints_a: int
    keypoints_b: int


@dataclass(frozen=True)
class RelativePose:
    rotation: object
    translation_direction: object
    inlier_mask: object

    @property
    def inliers(self) -> int:
        return int(self.inlier_mask.sum())


@dataclass(frozen=True)
class SparseStereoResult:
    depth: object
    keypoints_left: int
    keypoints_right: int
    ratio_matches: int
    epipolar_inliers: int


def camera_matrix(width: int, height: int, vertical_fov_degrees: float):
    import numpy as np

    if width <= 0 or height <= 0 or not 0 < vertical_fov_degrees < 180:
        raise ValueError("invalid image dimensions or field of view")
    focal = 0.5 * height / math.tan(math.radians(vertical_fov_degrees) / 2.0)
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def harris_corners(rgb, max_corners: int = 500, quality: float = 0.01):
    """Return non-max-suppressed Harris corners as Nx2 (x, y) coordinates."""
    import cv2
    import numpy as np

    image = np.asarray(rgb)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    response = cv2.cornerHarris(np.float32(gray), blockSize=2, ksize=3, k=0.04)
    local_maximum = response == cv2.dilate(response, None)
    candidates = np.argwhere(local_maximum & (response > quality * float(response.max())))
    if not len(candidates):
        return np.empty((0, 2), dtype=np.float32)
    scores = response[candidates[:, 0], candidates[:, 1]]
    selected = candidates[np.argsort(scores)[::-1][:max_corners]]
    return selected[:, ::-1].astype(np.float32)


def match_sift(rgb_a, rgb_b, ratio_threshold: float = 0.75) -> FeatureMatches:
    """Match SIFT descriptors with Lowe's nearest-neighbor ratio test."""
    import cv2
    import numpy as np

    if not 0 < ratio_threshold < 1:
        raise ValueError("ratio threshold must lie between zero and one")
    image_a, image_b = np.asarray(rgb_a), np.asarray(rgb_b)
    cv2.setRNGSeed(0)
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_RGB2GRAY) if image_a.ndim == 3 else image_a
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_RGB2GRAY) if image_b.ndim == 3 else image_b
    sift = cv2.SIFT_create()
    keypoints_a, descriptors_a = sift.detectAndCompute(gray_a, None)
    keypoints_b, descriptors_b = sift.detectAndCompute(gray_b, None)
    if descriptors_a is None or descriptors_b is None or len(descriptors_b) < 2:
        empty = np.empty((0, 2), dtype=np.float32)
        return FeatureMatches(empty, empty.copy(), len(keypoints_a), len(keypoints_b))
    candidates = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors_a, descriptors_b, k=2)
    accepted = [first for first, second in candidates if first.distance < ratio_threshold * second.distance]
    points_a = np.float32([keypoints_a[match.queryIdx].pt for match in accepted]).reshape(-1, 2)
    points_b = np.float32([keypoints_b[match.trainIdx].pt for match in accepted]).reshape(-1, 2)
    return FeatureMatches(points_a, points_b, len(keypoints_a), len(keypoints_b))


def estimate_pose_from_correspondences(
    points_a,
    points_b,
    intrinsics,
    ransac_threshold_px: float = 1.0,
) -> RelativePose | None:
    """Estimate calibrated relative pose with essential-matrix RANSAC."""
    import cv2
    import numpy as np

    a, b = np.asarray(points_a, dtype=np.float32), np.asarray(points_b, dtype=np.float32)
    if len(a) < 5 or a.shape != b.shape:
        return None
    cv2.setRNGSeed(0)
    essential, mask = cv2.findEssentialMat(
        a, b, np.asarray(intrinsics, dtype=np.float64), method=cv2.RANSAC,
        prob=0.999, threshold=ransac_threshold_px,
    )
    if essential is None or mask is None:
        return None
    # findEssentialMat can return several stacked 3x3 candidates.
    essential = essential[:3, :3]
    _, rotation, translation, pose_mask = cv2.recoverPose(
        essential, a, b, np.asarray(intrinsics, dtype=np.float64), mask=mask
    )
    return RelativePose(rotation, translation[:, 0], pose_mask[:, 0].astype(bool))


def estimate_relative_pose(rgb_a, rgb_b, intrinsics) -> tuple[FeatureMatches, RelativePose | None]:
    matches = match_sift(rgb_a, rgb_b)
    return matches, estimate_pose_from_correspondences(matches.points_a, matches.points_b, intrinsics)


def sparse_stereo_depth(
    left_rgb,
    right_rgb,
    focal_length_px: float,
    baseline_m: float,
    epipolar_tolerance_px: float = 2.0,
) -> SparseStereoResult:
    """Triangulate sparse metric camera-ray ranges from SIFT stereo matches.

    Lowe ratio filtering rejects ambiguous descriptors. Fundamental-matrix
    RANSAC rejects geometrically inconsistent matches before triangulation.
    """
    import cv2
    import numpy as np

    if focal_length_px <= 0 or baseline_m <= 0:
        raise ValueError("focal length and baseline must be positive")
    matches = match_sift(left_rgb, right_rgb)
    left, right = matches.points_a, matches.points_b
    rectified = (np.abs(left[:, 1] - right[:, 1]) <= epipolar_tolerance_px) & (
        left[:, 0] > right[:, 0]
    )
    left, right = left[rectified], right[rectified]
    inliers = np.ones(len(left), dtype=bool)
    if len(left) >= 8:
        cv2.setRNGSeed(0)
        _, mask = cv2.findFundamentalMat(left, right, cv2.FM_RANSAC, 1.0, 0.999)
        if mask is not None:
            inliers = mask[:, 0].astype(bool)
    left, right = left[inliers], right[inliers]
    height, width = np.asarray(left_rgb).shape[:2]
    depth = np.full((height, width), np.nan, dtype=np.float32)
    if len(left):
        disparity = left[:, 0] - right[:, 0]
        axial = focal_length_px * baseline_m / disparity
        cx, cy = width / 2.0, height / 2.0
        ray_scale = np.sqrt(
            1.0 + ((left[:, 0] - cx) / focal_length_px) ** 2
            + ((left[:, 1] - cy) / focal_length_px) ** 2
        )
        ranges = axial * ray_scale
        pixels = np.rint(left).astype(int)
        valid = (
            (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
            & np.isfinite(ranges) & (ranges > 0)
        )
        depth[pixels[valid, 1], pixels[valid, 0]] = ranges[valid]
    return SparseStereoResult(
        depth, matches.keypoints_a, matches.keypoints_b,
        len(matches.points_a), int(inliers.sum()),
    )
