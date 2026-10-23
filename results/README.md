# Result artifacts

## Current calibrated artifacts

- `pointcloud_floorplan1_5_calibrated.*`: five paired seeds in the development
  kitchen, with aggregate, by-scene, and paired-bootstrap summaries.
- `pointcloud_cross_category_calibrated_smoke.*`: one paired seed in each of a
  held-out kitchen, living room, bedroom, and bathroom.
- `pointcloud_floorplan2_no_recovery_calibrated_smoke.*`: matching
  perception-only navigation ablation for one scene and seed.
- `cross_category_calibrated_with_ablation.summary.*`: combined recovery and
  no-recovery report.
- `diagnostics_floorplan1_seed1_calibrated/`: representative RGB, depth,
  absolute-error, and occupancy figures.
- `pointcloud_cross_category_20_calibrated.*`: full five-seed-by-four-category
  result, including the report-ready summary figure.
- `pointcloud_cross_category_20_no_recovery_calibrated.*`: matching full run
  without the collision-only fallback.
- `cross_category_20_recovery_ablation.summary.*`: direct paired comparison of
  the two recovery policies.
- `classical_sparse_features_cross_category_20.*`: sparse SIFT/RANSAC stereo
  results on the same 20 held-out cases.
- `cv_methods_cross_category_20.*`: combined four-method CV comparison.
- `vo_*.json`, `vo_*.steps.csv`, and `vo_cross_category_summary.*`: temporal
  essential-matrix visual-odometry evaluation.
- `diagnostics_floorplan1_seed1_cv_complete/`: Harris, SIFT, epipolar-RANSAC,
  sparse-depth, temporal-pose, and occupancy diagnostics.

All current stereo results use RGB grayscale conversion and convert triangulated
axial depth to Euclidean camera-ray distance before evaluation and mapping.

## Legacy development artifacts

Files without `calibrated` in their name are retained only as implementation
history. They include either the original scanline mapper or the pre-calibration
stereo convention and must not be mixed with current results.
