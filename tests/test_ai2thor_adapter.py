import unittest
from types import SimpleNamespace

from visual_nav.ai2thor_env import AI2ThorEnvironment, ThorConfig


def make_event(position=None, yaw=0, action_return=None, success=True):
    position = position or {"x": 0.0, "y": 0.9, "z": 0.0}
    metadata = {
        "lastActionSuccess": success,
        "errorMessage": "",
        "actionReturn": action_return,
        "agent": {
            "position": dict(position),
            "cameraPosition": dict(position),
            "rotation": {"x": 0, "y": yaw, "z": 0},
            "cameraHorizon": 0,
        },
    }
    return SimpleNamespace(
        metadata=metadata,
        depth_frame=[[5.0] * 8 for _ in range(8)],
        frame=[[[0, 0, 0]] * 8 for _ in range(8)],
        third_party_camera_frames=[[[[0, 0, 0]] * 8 for _ in range(8)]],
    )


class FakeController:
    def __init__(self):
        self.positions = [
            {"x": float(x), "y": 0.9, "z": float(z)}
            for x in range(3) for z in range(3)
        ]
        self.last_event = make_event()
        self.stopped = False

    def reset(self, **kwargs):
        self.last_event = make_event()
        return self.last_event

    def step(self, action, **kwargs):
        agent = self.last_event.metadata["agent"]
        position = dict(agent["position"])
        yaw = agent["rotation"]["y"]
        action_return = None
        if action == "GetReachablePositions":
            action_return = self.positions
        elif action == "Teleport":
            position = dict(kwargs["position"])
            yaw = kwargs.get("rotation", {}).get("y", yaw)
        elif action == "RotateRight":
            yaw = (yaw + 90) % 360
        elif action == "RotateLeft":
            yaw = (yaw - 90) % 360
        elif action == "MoveAhead":
            delta = {0: (0, 1), 90: (1, 0), 180: (0, -1), 270: (-1, 0)}[yaw]
            position["x"] += delta[0]
            position["z"] += delta[1]
        self.last_event = make_event(position, yaw, action_return)
        return self.last_event

    def stop(self):
        self.stopped = True


class AI2ThorAdapterTests(unittest.TestCase):
    def test_reset_builds_grid_and_movement_translates_to_actions(self):
        controller = FakeController()
        env = AI2ThorEnvironment(ThorConfig(grid_size=1.0), controller=controller)
        env.reset(seed=2)
        self.assertEqual((env.ground_truth.width, env.ground_truth.height), (3, 3))
        start = env.position
        candidates = env.ground_truth.neighbors(start)
        self.assertTrue(candidates)
        self.assertTrue(env.move_to_neighbor(candidates[0]))
        self.assertEqual(env.position, candidates[0])

    def test_stereo_camera_returns_pair(self):
        env = AI2ThorEnvironment(ThorConfig(grid_size=1.0), controller=FakeController())
        env.reset(seed=1)
        left, right = env.capture_stereo_pair()
        self.assertEqual(len(left), len(right))


if __name__ == "__main__":
    unittest.main()

