from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from .grid import Cell, GridMap
from .planning import astar


@dataclass(frozen=True)
class ThorConfig:
    scene: str = "FloorPlan1"
    width: int = 300
    height: int = 300
    field_of_view: float = 90.0
    grid_size: float = 0.25
    rotate_step_degrees: int = 90
    baseline_m: float = 0.08
    headless: bool = False


class AI2ThorEnvironment:
    """Small, injectable wrapper around AI2-THOR's Controller API."""

    def __init__(self, config: ThorConfig, controller: Any | None = None):
        self.config = config
        self.controller = controller or self._create_controller()
        self._origin_x = 0.0
        self._origin_z = 0.0
        self.start: Cell = (0, 0)
        self.goal: Cell = (0, 0)
        self.ground_truth = GridMap(1, 1, frozenset())
        self.collisions = 0
        self.action_count = 0
        self._stereo_camera_id: int | None = None

    def _create_controller(self):
        try:
            from ai2thor.controller import Controller
        except ImportError as exc:
            raise RuntimeError("Install the 'ai2thor' optional dependencies") from exc
        kwargs: dict[str, Any] = {}
        if self.config.headless:
            from ai2thor.platform import CloudRendering
            kwargs["platform"] = CloudRendering
        return Controller(
            scene=self.config.scene,
            agentMode="default",
            gridSize=self.config.grid_size,
            snapToGrid=True,
            rotateStepDegrees=self.config.rotate_step_degrees,
            renderDepthImage=True,
            width=self.config.width,
            height=self.config.height,
            fieldOfView=self.config.field_of_view,
            **kwargs,
        )

    def reset(self, seed: int) -> None:
        self.controller.reset(
            scene=self.config.scene,
            gridSize=self.config.grid_size,
            rotateStepDegrees=self.config.rotate_step_degrees,
            renderDepthImage=True,
            width=self.config.width,
            height=self.config.height,
            fieldOfView=self.config.field_of_view,
        )
        event = self.controller.step(action="GetReachablePositions")
        positions = event.metadata.get("actionReturn") or []
        if not event.metadata.get("lastActionSuccess", True) or len(positions) < 2:
            raise RuntimeError(f"could not query reachable positions: {event.metadata.get('errorMessage', '')}")
        self._build_ground_truth(positions)
        rng = random.Random(seed)
        start_world = rng.choice(positions)
        candidates = sorted(
            positions,
            key=lambda p: (p["x"] - start_world["x"]) ** 2 + (p["z"] - start_world["z"]) ** 2,
            reverse=True,
        )
        goal_world = candidates[rng.randrange(max(1, min(10, len(candidates))))]
        self.start = self.world_to_cell(start_world["x"], start_world["z"])
        self.goal = self.world_to_cell(goal_world["x"], goal_world["z"])
        teleport = self.controller.step(action="Teleport", position=start_world, rotation=dict(x=0, y=0, z=0))
        if not teleport.metadata.get("lastActionSuccess", False):
            raise RuntimeError(f"could not set start pose: {teleport.metadata.get('errorMessage', '')}")
        self.collisions = 0
        self.action_count = 0
        self._stereo_camera_id = None

    def _build_ground_truth(self, positions: list[dict[str, float]]) -> None:
        self._origin_x = min(p["x"] for p in positions)
        self._origin_z = min(p["z"] for p in positions)
        reachable = {self.world_to_cell(p["x"], p["z"]) for p in positions}
        width = max(x for x, _ in reachable) + 1
        height = max(z for _, z in reachable) + 1
        blocked = {(x, z) for x in range(width) for z in range(height)} - reachable
        self.ground_truth = GridMap(width, height, frozenset(blocked))

    def world_to_cell(self, x: float, z: float) -> Cell:
        return (
            round((x - self._origin_x) / self.config.grid_size),
            round((z - self._origin_z) / self.config.grid_size),
        )

    @property
    def event(self):
        return self.controller.last_event

    @property
    def position(self) -> Cell:
        pose = self.event.metadata["agent"]["position"]
        return self.world_to_cell(pose["x"], pose["z"])

    @property
    def yaw(self) -> float:
        return float(self.event.metadata["agent"]["rotation"]["y"]) % 360

    @property
    def horizon(self) -> float:
        return float(self.event.metadata["agent"].get("cameraHorizon", 0.0))

    @property
    def camera_position(self) -> tuple[float, float, float]:
        position = self.event.metadata.get(
            "cameraPosition", self.event.metadata["agent"]["position"]
        )
        return float(position["x"]), float(position["y"]), float(position["z"])

    @property
    def floor_y(self) -> float:
        # The default iTHOR agent origin is approximately 0.901 m above its floor.
        return float(self.event.metadata["agent"]["position"]["y"]) - 0.901

    @property
    def grid_origin(self) -> tuple[float, float]:
        return self._origin_x, self._origin_z

    def depth_frame(self):
        frame = getattr(self.event, "depth_frame", None)
        if frame is None:
            raise RuntimeError("AI2-THOR returned no depth frame; renderDepthImage must be enabled")
        return frame

    def move_to_neighbor(self, target: Cell) -> bool:
        x, z = self.position
        delta = target[0] - x, target[1] - z
        desired = {(0, 1): 0, (1, 0): 90, (0, -1): 180, (-1, 0): 270}.get(delta)
        if desired is None:
            raise ValueError(f"target {target} is not a cardinal neighbor of {(x, z)}")
        turns = int(round(((desired - self.yaw) % 360) / self.config.rotate_step_degrees))
        reverse_turns = int(round(((self.yaw - desired) % 360) / self.config.rotate_step_degrees))
        action = "RotateRight" if turns <= reverse_turns else "RotateLeft"
        for _ in range(min(turns, reverse_turns)):
            self.controller.step(action=action)
            self.action_count += 1
        result = self.controller.step(action="MoveAhead")
        self.action_count += 1
        success = bool(result.metadata.get("lastActionSuccess", False))
        if not success:
            self.collisions += 1
        return success

    def capture_stereo_pair(self):
        """Return rectified-style RGB views separated along the camera-right axis."""
        event = self.event
        agent = event.metadata["agent"]
        position = event.metadata.get("cameraPosition", agent["position"])
        yaw = self.yaw
        yaw_rad = math.radians(yaw)
        right_position = {
            "x": position["x"] + self.config.baseline_m * math.cos(yaw_rad),
            "y": position["y"],
            "z": position["z"] - self.config.baseline_m * math.sin(yaw_rad),
        }
        rotation = {"x": agent.get("cameraHorizon", 0), "y": yaw, "z": 0}
        if self._stereo_camera_id is None:
            stereo_event = self.controller.step(
                action="AddThirdPartyCamera",
                position=right_position,
                rotation=rotation,
                fieldOfView=self.config.field_of_view,
            )
            self._stereo_camera_id = 0
        else:
            stereo_event = self.controller.step(
                action="UpdateThirdPartyCamera",
                thirdPartyCameraId=self._stereo_camera_id,
                position=right_position,
                rotation=rotation,
                fieldOfView=self.config.field_of_view,
            )
        if not stereo_event.metadata.get("lastActionSuccess", False):
            raise RuntimeError(f"stereo camera update failed: {stereo_event.metadata.get('errorMessage', '')}")
        return stereo_event.frame, stereo_event.third_party_camera_frames[self._stereo_camera_id]

    def shortest_path_length(self) -> int:
        path = astar(self.ground_truth, self.start, self.goal)
        if path is None:
            raise RuntimeError("selected AI2-THOR poses are disconnected")
        return len(path) - 1

    def stop(self) -> None:
        stop = getattr(self.controller, "stop", None)
        if stop is not None:
            stop()

