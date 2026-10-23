import math
import unittest

from visual_nav.benchmark import EpisodeResult, run_episode
from visual_nav.depth_projection import scanline_depth_to_observation
from visual_nav.grid import GridMap
from visual_nav.metrics import chamfer_distance, depth_errors, occupancy_iou, spl
from visual_nav.perception import EvidenceOccupancyMapper, OracleDepth, RangeObservation
from visual_nav.planning import astar
from visual_nav.pointcloud import pointcloud_depth_to_observation, project_depth_to_world
from visual_nav.statistics import paired_bootstrap_comparisons, paired_recovery_comparisons


class PlanningTests(unittest.TestCase):
    def test_astar_routes_around_wall(self):
        grid = GridMap(5, 4, frozenset({(2, 0), (2, 1), (2, 2)}))
        path = astar(grid, (0, 1), (4, 1))
        self.assertIsNotNone(path)
        self.assertEqual(path[0], (0, 1))
        self.assertEqual(path[-1], (4, 1))
        self.assertTrue(all(grid.free(cell) for cell in path))

    def test_astar_reports_impossible_goal(self):
        grid = GridMap(3, 3, frozenset({(1, 0), (1, 1), (1, 2)}))
        self.assertIsNone(astar(grid, (0, 1), (2, 1)))


class MetricTests(unittest.TestCase):
    def test_metrics(self):
        self.assertEqual(occupancy_iou({(0, 0)}, {(0, 0)}), 1.0)
        self.assertEqual(chamfer_distance({(0, 0)}, {(1, 0)}), 1.0)
        self.assertEqual(spl(True, 5, 10), 0.5)
        self.assertEqual(spl(False, 5, 5), 0.0)
        self.assertTrue(math.isinf(chamfer_distance(set(), {(1, 1)})))

    def test_depth_errors(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("vision extras are not installed")
        predicted = np.array([[1.0, 4.0], [np.nan, 2.0]])
        truth = np.array([[1.0, 2.0], [3.0, 2.0]])
        abs_rel, rmse, coverage = depth_errors(predicted, truth)
        self.assertAlmostEqual(abs_rel, 1.0 / 3.0)
        self.assertAlmostEqual(rmse, math.sqrt(4.0 / 3.0))
        self.assertAlmostEqual(coverage, 0.75)

    def test_paired_bootstrap_preserves_constant_difference(self):
        common = dict(
            success=True, collisions=0, shortest_path=5, spl=1.0,
            occupancy_iou=0.5, chamfer_distance=1.0, scene="FloorPlan1",
        )
        results = [
            EpisodeResult(method=method, seed=seed, path_length=length, **common)
            for seed in (1, 2, 3)
            for method, length in (("a", 5), ("b", 7))
        ]
        rows = paired_bootstrap_comparisons(results, iterations=50)
        path_row = next(row for row in rows if row["metric"] == "path_length")
        self.assertEqual(path_row["mean_difference_b_minus_a"], 2.0)
        self.assertEqual(path_row["ci95_low"], 2.0)
        self.assertEqual(path_row["ci95_high"], 2.0)

    def test_recovery_comparison_pairs_matching_scene_seed_and_method(self):
        common = dict(
            method="classical", seed=1, collisions=0, shortest_path=5,
            occupancy_iou=0.5, chamfer_distance=1.0, scene="FloorPlan1",
        )
        results = [
            EpisodeResult(success=False, path_length=3, spl=0.0, recovery_policy="disabled", **common),
            EpisodeResult(success=True, path_length=5, spl=1.0, recovery_policy="collision_only", **common),
        ]
        rows = paired_recovery_comparisons(results, iterations=20)
        success_row = next(row for row in rows if row["metric"] == "success")
        self.assertEqual(success_row["mean_difference_collision_only_minus_disabled"], 1.0)


class ProjectionTests(unittest.TestCase):
    def test_scanline_marks_free_ray_and_obstacle_endpoint(self):
        depth = [[2.0] * 5 for _ in range(3)]
        observation = scanline_depth_to_observation(
            depth, origin=(5, 5), yaw_degrees=0, field_of_view_degrees=0,
            grid_size_m=1.0, max_range_m=4.0, column_stride=1,
        )
        self.assertIn((5, 6), observation.visible_free)
        self.assertIn((5, 7), observation.visible_blocked)

    def test_evidence_mapper_requires_repeated_hits_and_keeps_collisions(self):
        mapper = EvidenceOccupancyMapper(5, 5, occupied_threshold=2)
        occupied = RangeObservation((0, 0), frozenset({(2, 2)}), frozenset())
        mapper.update(occupied)
        self.assertNotIn((2, 2), mapper.blocked)
        mapper.update(occupied)
        self.assertIn((2, 2), mapper.blocked)
        mapper.record_collision((3, 3))
        self.assertIn((3, 3), mapper.as_grid(include_depth=False).blocked)

    def test_center_depth_projects_forward_and_respects_yaw(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("vision extras are not installed")
        depth = np.array([[2.0]], dtype=np.float32)
        forward = project_depth_to_world(depth, (0, 1, 0), 0, 0, 90, stride=1)
        right = project_depth_to_world(depth, (0, 1, 0), 90, 0, 90, stride=1)
        np.testing.assert_allclose(forward[0], [0, 1, 2], atol=1e-6)
        np.testing.assert_allclose(right[0], [2, 1, 0], atol=1e-6)

    def test_pointcloud_height_filter_excludes_floor(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("vision extras are not installed")
        obstacle = pointcloud_depth_to_observation(
            np.array([[1.0]]), (0, 0), (0, 1, 0), 0, 0, 0, 90,
            0, 0, 0.5, 6, 6, stride=1,
        )
        floor = pointcloud_depth_to_observation(
            np.array([[1.0]]), (0, 0), (0, 1, 0), 0, 0, 90, 90,
            0, 0, 0.5, 6, 6, stride=1,
        )
        self.assertIn((0, 2), obstacle.visible_blocked)
        self.assertFalse(floor.visible_blocked)


class IntegrationTests(unittest.TestCase):
    def test_oracle_episode_reaches_goal(self):
        result = run_episode(OracleDepth(), seed=7)
        self.assertTrue(result.success)
        self.assertGreater(result.shortest_path, 0)
        self.assertGreater(result.spl, 0.0)


if __name__ == "__main__":
    unittest.main()
