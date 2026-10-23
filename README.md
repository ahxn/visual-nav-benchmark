# Visual Navigation Benchmark

A reproducible comparison of classical and learned computer-vision methods for
indoor robot navigation. AI2-THOR supplies paired stereo observations and
ground-truth evaluation data; every method shares the same occupancy mapper,
A* planner, controller, starts, goals, and episode budget.

## Implemented methods

- `simulator_depth`: simulator depth as an explicit upper-bound baseline.
- `classical_stereo`: calibrated OpenCV SGBM dense stereo.
- `classical_sparse_features`: Harris corners, SIFT descriptors, Lowe-ratio
  matching, fundamental-matrix RANSAC, and sparse triangulation.
- `learned_metric_depth`: pretrained Depth Anything V2 indoor metric depth.

The repository also includes temporal feature matching, essential-matrix
RANSAC, and visual-odometry evaluation. The perception output is projected to a
height-filtered 3D point cloud and a 2D occupancy grid, then an inflated-grid A*
planner runs in a closed observe-map-plan-act loop.

## Evaluation

Navigation is evaluated with success rate, collisions, path length, and SPL.
Perception is evaluated independently with depth AbsRel/RMSE/coverage,
occupancy IoU, point-cloud Chamfer distance, match and RANSAC-inlier counts, and
visual-odometry translation/rotation error. Reported benchmark comparisons use
paired episodes across all four iTHOR room categories and paired bootstrap
confidence intervals.

The core CV comparison, held-out trials, recovery-policy ablation, qualitative
diagnostics, and visual-odometry evaluation are complete. The synthetic harness
also provides `oracle_depth` and `noisy_depth` smoke-test methods without
requiring AI2-THOR.

## Quick start

Create a local environment for the synthetic benchmark and tests:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[vision,dev]"
.venv\Scripts\python.exe -m pytest -q
```

Run the deterministic synthetic benchmark:

```powershell
.venv\Scripts\visual-nav.exe --episodes 20 --methods oracle_depth noisy_depth
```

Results are written to `results/summary.csv` and `results/episodes.json`.

## AI2-THOR on Windows

AI2-THOR 5.0 publishes Linux and macOS Unity builds. On Windows, create the full
environment under Ubuntu 22.04 with WSLg:

```powershell
wsl.exe -d Ubuntu-22.04 -- python3 -m venv .venv-linux
wsl.exe -d Ubuntu-22.04 -- .venv-linux/bin/pip install -e '.[vision,ai2thor,deep,dev]'
```

Run paired trials for all four methods:

```powershell
wsl.exe -d Ubuntu-22.04 -- .venv-linux/bin/visual-nav-thor `
  --scenes FloorPlan2 FloorPlan202 FloorPlan302 FloorPlan402 `
  --episodes 5 --seed 11 `
  --methods simulator_depth classical_stereo classical_sparse_features learned_metric_depth `
  --output results/cv_methods_cross_category_20.json
```

The learned method uses the pretrained 24.8M-parameter Depth Anything V2 indoor
metric model. Model weights are downloaded once and cached by Hugging Face.

Generate qualitative diagnostics:

```powershell
wsl.exe -d Ubuntu-22.04 -- .venv-linux/bin/visual-nav-diagnostics `
  --scene FloorPlan1 --seed 1 `
  --output results/diagnostics_floorplan1_seed1_cv_complete
```

Add `--disable-recovery` to a simulator run for the perception-only navigation
ablation. See [the experiment protocol](docs/EXPERIMENT.md) for the controls and
full command set.

## Included results

- [Four-method CV summary](results/cv_methods_cross_category_20.summary.csv)
  and [figure](results/cv_methods_cross_category_20.png)
- [Dense three-method benchmark](results/pointcloud_cross_category_20_calibrated.summary.csv)
  and [paired confidence intervals](results/pointcloud_cross_category_20_calibrated.paired_ci.csv)
- [Recovery ablation](results/cross_category_20_recovery_ablation.summary.recovery_ci.csv)
- [Visual-odometry summary](results/vo_cross_category_summary.json)
- Representative [depth](results/diagnostics_floorplan1_seed1_cv_complete/depth_comparison_panel.png),
  [classical-feature](results/diagnostics_floorplan1_seed1_cv_complete/classical_feature_panel.png),
  and [occupancy](results/diagnostics_floorplan1_seed1_cv_complete/occupancy_comparison_panel.png)
  figures

Generated intermediate outputs are ignored by Git; the report-ready artifacts
listed above are versioned so the main conclusions can be inspected without
rerunning the simulator or downloading model weights.

## Experimental controls

Keep scene seeds, starts, goals, action space, robot radius, map resolution, and
step budget identical across methods. Tune thresholds on separate development
scenes. Simulator ground-truth geometry is used only for evaluation and for the
explicitly named upper-bound baseline.

## Optional extensions

Potential follow-up work includes increasing the paired sample count on a GPU,
testing a larger frozen metric-depth model without changing the controller, and
adding the proposal's convex-corridor trajectory optimization.
