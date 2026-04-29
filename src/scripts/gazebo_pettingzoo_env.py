"""PettingZoo wrapper for the Gazebo + ROS2 multi-UAV maze environment.

This module provides a simple `pettingzoo.ParallelEnv` wrapper that converts the
existing Gazebo/ROS2 multi-UAV simulation into a multi-agent RL environment.

Each UAV is treated as an agent. Actions are body-frame
(linear_x, linear_y, linear_z, yaw_rate), and observations include odometry
(x, y, z, yaw, vx, vy, vz, yaw_rate) plus a fixed-length laser scan.

Usage (example):

    from src.scripts.gazebo_pettingzoo_env import GazeboMultiUAVParallelEnv

    env = GazeboMultiUAVParallelEnv(uav_names=["uav1", "uav2", "uav3", "uav4"])
    obs = env.reset()
    actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
    obs, rewards, dones, infos = env.step(actions)

Note: This wrapper assumes the Gazebo world is already running and ROS2 bridges are
active (e.g., via `setup_bridges.sh`).
"""

import logging
import os
import subprocess
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

import rclpy
from rclpy.exceptions import InvalidServiceNameException
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from ros_gz_interfaces.msg import WorldControl
from ros_gz_interfaces.srv import ControlWorld
from sensor_msgs.msg import LaserScan


LOGGER = logging.getLogger(__name__)


def _make_box(low, high, shape, dtype=np.float32):
    return spaces.Box(low=low, high=high, shape=shape, dtype=dtype)


def _quat_from_rpy(roll, pitch, yaw):
    half_roll = roll * 0.5
    half_pitch = pitch * 0.5
    half_yaw = yaw * 0.5
    cr = np.cos(half_roll)
    sr = np.sin(half_roll)
    cp = np.cos(half_pitch)
    sp = np.sin(half_pitch)
    cy = np.cos(half_yaw)
    sy = np.sin(half_yaw)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


class GazeboMultiUAVParallelEnv(ParallelEnv):
    metadata = {"render_modes": []}

    def __init__(
        self,
        uav_names=None,
        lidar_size=36,
        max_xy_vel=1.0,
        max_z_vel=0.6,
        max_yaw_rate=1.2,
        spin_timeout=0.08,
        idle_spin_timeout=0.01,
        action_update_timeout=0.03,
        require_step_lidar_update=False,
        world_name="maze_1.sdf",
        reset_timeout=2.0,
        max_steps=500,
        min_height=0.2,
        max_height=3.0,
        collision_lidar_threshold=0.25,
        step_penalty=0.01,
        progress_reward_scale=2.0,
        coverage_cell_size=1.0,
        coverage_target_ratio=0.85,
        coverage_new_cell_reward=1.2,
        coverage_completion_bonus=8.0,
        revisit_cell_penalty=0.08,
        overlap_cell_penalty=0.15,
        assignment_bonus=0.2,
        local_coverage_radius=1,
        sensor_coverage_radius=1,
        local_map_size=7,
        coverage_delta_scale=0.0,
        sensor_wait_timeout=5.0,
        startup_grace_steps=15,
        arena_x_limits=(-10.0, 10.0),
        arena_y_limits=(-10.0, 10.0),
        out_of_bounds_penalty=2.0,
        reset_position_tolerance=1.5,
        reset_validation_retries=1,
        reset_validation_allow_failure=False,
        reset_stabilization_timeout=1.5,
        reset_poll_interval=0.05,
        reset_initial_sensor_timeout=1.5,
        reset_initial_sensor_timeout_first=4.0,
        pre_reset_brake_wait=0.03,
        min_height_enforce_steps=60,
        hard_floor_height=0.08,
        min_step_duration=0.05,
        altitude_safety_threshold=0.4,
        low_altitude_penalty=0.3,
        collision_penalty=1.5,
        collision_terminal=True,
        collision_cooldown_steps=10,
        survival_bonus=0.0,
        boundary_penalty_margin=2.0,
        boundary_penalty_scale=0.3,
        min_spawn_pair_clearance=0.28,
        min_spawn_altitude=0.14,
        spawn_layout_validation_retries=2,
        spawn_layout_allow_risky=False,
    ):
        super().__init__()
        self.uav_names = uav_names or ["uav1", "uav2", "uav3", "uav4"]
        self.agents = list(self.uav_names)
        self.possible_agents = list(self.agents)
        self.world_name = str(world_name)
        self._project_root = Path(__file__).resolve().parent.parent.parent
        self._world_sdf_path = self._project_root / "src" / "worlds" / "mazes" / self.world_name
        self._service_world_name = (
            self.world_name[:-4] if self.world_name.endswith(".sdf") else self.world_name
        )
        self._reset_timeout = float(reset_timeout)
        self.max_steps = int(max_steps)
        self.min_height = float(min_height)
        self.max_height = float(max_height)
        self.collision_lidar_threshold = float(collision_lidar_threshold)
        self.step_penalty = float(step_penalty)
        self.progress_reward_scale = float(progress_reward_scale)
        self.coverage_cell_size = max(float(coverage_cell_size), 0.1)
        self.coverage_target_ratio = float(np.clip(coverage_target_ratio, 0.05, 1.0))
        self.coverage_new_cell_reward = float(coverage_new_cell_reward)
        self.coverage_completion_bonus = float(coverage_completion_bonus)
        self.revisit_cell_penalty = float(revisit_cell_penalty)
        self.overlap_cell_penalty = float(overlap_cell_penalty)
        self.assignment_bonus = float(assignment_bonus)
        self.local_coverage_radius = max(0, int(local_coverage_radius))
        self.sensor_coverage_radius = max(0, int(sensor_coverage_radius))
        self.local_map_size = max(3, int(local_map_size))
        self.coverage_delta_scale = float(coverage_delta_scale)
        self.sensor_wait_timeout = float(sensor_wait_timeout)
        self.startup_grace_steps = max(0, int(startup_grace_steps))
        self.arena_x_limits = (float(arena_x_limits[0]), float(arena_x_limits[1]))
        self.arena_y_limits = (float(arena_y_limits[0]), float(arena_y_limits[1]))
        self.out_of_bounds_penalty = float(out_of_bounds_penalty)
        self.reset_position_tolerance = float(reset_position_tolerance)
        self.reset_validation_retries = max(0, int(reset_validation_retries))
        self.reset_validation_allow_failure = bool(reset_validation_allow_failure)
        self.reset_stabilization_timeout = float(reset_stabilization_timeout)
        self.reset_poll_interval = float(reset_poll_interval)
        self.reset_initial_sensor_timeout = float(reset_initial_sensor_timeout)
        self.reset_initial_sensor_timeout_first = float(reset_initial_sensor_timeout_first)
        self.pre_reset_brake_wait = float(pre_reset_brake_wait)
        self.min_height_enforce_steps = max(0, int(min_height_enforce_steps))
        self.hard_floor_height = float(hard_floor_height)
        self._min_step_duration = max(float(min_step_duration), 0.0)
        self.altitude_safety_threshold = float(altitude_safety_threshold)
        self.low_altitude_penalty = float(low_altitude_penalty)
        self.collision_penalty = float(collision_penalty)
        self.collision_terminal = bool(collision_terminal)
        self.collision_cooldown_steps = max(0, int(collision_cooldown_steps))
        self.survival_bonus = float(survival_bonus)
        self.boundary_penalty_margin = float(boundary_penalty_margin)
        self.boundary_penalty_scale = float(boundary_penalty_scale)
        self.min_spawn_pair_clearance = float(min_spawn_pair_clearance)
        self.min_spawn_altitude = float(min_spawn_altitude)
        self.spawn_layout_validation_retries = max(0, int(spawn_layout_validation_retries))
        self.spawn_layout_allow_risky = bool(spawn_layout_allow_risky)

        # Shared spaces
        self.action_spaces = {
            agent: spaces.Box(
                low=np.array([-max_xy_vel, -max_xy_vel, -max_z_vel, -max_yaw_rate], dtype=np.float32),
                high=np.array([max_xy_vel, max_xy_vel, max_z_vel, max_yaw_rate], dtype=np.float32),
                dtype=np.float32,
            )
            for agent in self.agents
        }

        # Observation: [x, y, z, yaw, vx, vy, vz, yaw_rate] + lidar ranges
        n_others = max(len(self.uav_names) - 1, 1)
        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "pose": spaces.Box(
                        low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
                    ),
                    "lidar": spaces.Box(
                        low=0.0, high=np.inf, shape=(lidar_size,), dtype=np.float32
                    ),
                    "coverage": spaces.Box(
                        low=0.0, high=1.0, shape=(5,), dtype=np.float32
                    ),
                    "local_map": spaces.Box(
                        low=0.0, high=1.0,
                        shape=(self.local_map_size * self.local_map_size,),
                        dtype=np.float32,
                    ),
                    "other_agents": spaces.Box(
                        low=-1.0, high=1.0,
                        shape=(n_others * 4,),
                        dtype=np.float32,
                    ),
                }
            )
            for agent in self.agents
        }

        self._spin_timeout = float(spin_timeout)
        self._idle_spin_timeout = max(float(idle_spin_timeout), 0.001)
        self._action_update_timeout = max(float(action_update_timeout), 0.0)
        self._require_step_lidar_update = bool(require_step_lidar_update)
        self._last_odom = {agent: None for agent in self.agents}
        self._last_lidar = {agent: None for agent in self.agents}
        self._odom_update_count = {agent: 0 for agent in self.agents}
        self._lidar_update_count = {agent: 0 for agent in self.agents}
        self._step_count = 0
        self._coverage_complete = False
        self._coverage_completion_step = None
        self._warned_reset_unavailable = False
        self._warned_sensor_unavailable = False
        self._first_reset = True
        self._warned_gz_reset_unavailable = False
        self._warned_spawn_layout = False
        self._spawn_pose_specs = self._load_spawn_pose_specs()
        self._spawn_reference_positions = (
            {
                agent: np.asarray(spec["position"][:2], dtype=np.float32)
                for agent, spec in self._spawn_pose_specs.items()
            }
            if self._spawn_pose_specs
            else None
        )
        self._coverage_visit_counts = None
        self._coverage_owner = None
        self._coverage_grid_shape = self._build_coverage_grid_shape()

        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node("gazebo_pettingzoo_env")
        self._py_logger = LOGGER

        # Optional Reset Service (Gazebo / ros_gz_bridge)
        self._reset_client = None
        # ROS service name must be ROS-valid, so it uses the sanitized world token.
        self._reset_service_name = f"/world/{self._service_world_name}/control"
        # Gazebo transport service may expose world names in different forms.
        self._gz_reset_service_candidates = []
        for candidate_world in [self.world_name, self._service_world_name, "default"]:
            service = f"/world/{candidate_world}/control"
            if service not in self._gz_reset_service_candidates:
                self._gz_reset_service_candidates.append(service)
        self._active_gz_reset_service = None
        if self._service_world_name != self.world_name:
            self._py_logger.info(
                "[ResetService] world_name='%s' mapped to service world '%s'",
                self.world_name,
                self._service_world_name,
            )
        try:
            self._reset_client = self._node.create_client(
                ControlWorld, self._reset_service_name
            )
        except InvalidServiceNameException:
            self._warn_reset_unavailable(
                f"cannot be created because '{self._reset_service_name}' is not a valid ROS service name"
            )

        if self._reset_client is not None and not self._reset_client.wait_for_service(timeout_sec=1.0):
            self._warn_reset_unavailable("not available")

        self._publishers = {}
        for agent in self.agents:
            self._publishers[agent] = self._node.create_publisher(
                Twist, f"/{agent}/cmd_vel", 10
            )
            self._node.create_subscription(
                Odometry, f"/{agent}/odom", self._make_odom_cb(agent), 10
            )
            self._node.create_subscription(
                LaserScan, f"/{agent}/lidar", self._make_lidar_cb(agent), 10
            )

        self._dones = {agent: False for agent in self.agents}
        self._collision_cooldown = {agent: 0 for agent in self.agents}

    def _make_odom_cb(self, agent):
        def cb(msg: Odometry):
            self._last_odom[agent] = msg
            self._odom_update_count[agent] += 1

        return cb

    def _make_lidar_cb(self, agent):
        def cb(msg: LaserScan):
            self._last_lidar[agent] = msg
            self._lidar_update_count[agent] += 1

        return cb

    def _spin_once(self):
        rclpy.spin_once(self._node, timeout_sec=self._spin_timeout)

    def _spin_for(self, duration):
        deadline = time.time() + duration
        while time.time() < deadline:
            self._spin_once()

    def _sensor_counter_snapshot(self):
        return {
            agent: (self._odom_update_count[agent], self._lidar_update_count[agent])
            for agent in self.agents
        }

    def _has_fresh_post_action_observation(self, previous_counts):
        for agent in self.agents:
            prev_odom, prev_lidar = previous_counts[agent]
            odom_advanced = self._odom_update_count[agent] > prev_odom
            lidar_advanced = self._lidar_update_count[agent] > prev_lidar
            if not odom_advanced:
                return False
            if self._require_step_lidar_update and not lidar_advanced:
                return False
            if not self._has_agent_sensor_data(agent):
                return False
        return True

    def _wait_for_post_action_observation(self):
        start = time.time()

        # Wait for fresh sensor data (existing logic)
        if self._action_update_timeout > 0.0:
            previous_counts = self._sensor_counter_snapshot()
            deadline = start + self._action_update_timeout
            while time.time() < deadline:
                if self._has_fresh_post_action_observation(previous_counts):
                    break
                rclpy.spin_once(self._node, timeout_sec=self._idle_spin_timeout)

        # Enforce minimum step duration so Gazebo physics advances enough
        # between RL steps (real-time simulation coupling).
        remaining = self._min_step_duration - (time.time() - start)
        if remaining > 0:
            self._spin_for(remaining)

    def _warn_reset_unavailable(self, reason):
        if not self._warned_reset_unavailable:
            message = f"Reset service {self._reset_service_name} {reason}; reset() will fall back to a soft reset."
            self._node.get_logger().warning(
                message
            )
            self._py_logger.warning(message)
            self._warned_reset_unavailable = True

    def _warn_gz_reset_unavailable(self, reason):
        if not self._warned_gz_reset_unavailable:
            message = f"Gazebo native reset via {self._reset_service_name} {reason}; reset() will fall back to a soft reset."
            self._node.get_logger().warning(
                message
            )
            self._py_logger.warning(message)
            self._warned_gz_reset_unavailable = True

    def _warn_sensor_unavailable(self):
        if not self._warned_sensor_unavailable:
            message = (
                "No odom/lidar data received from ROS2 topics. Check that Gazebo is running and start bridges with "
                "'bash setup_bridges.sh' from the project root."
            )
            self._node.get_logger().warning(
                message
            )
            self._py_logger.warning(message)
            self._warned_sensor_unavailable = True

    def _check_required_topics(self):
        """Check if all required ROS2 topics exist before training starts."""
        all_topics = [name for name, _ in self._node.get_topic_names_and_types()]
        missing_topics = []
        for agent in self.agents:
            for topic_suffix in ["odom", "lidar", "cmd_vel"]:
                topic_name = f"/{agent}/{topic_suffix}"
                if topic_name not in all_topics:
                    missing_topics.append(topic_name)
        
        if missing_topics:
            error_msg = (
                f"ERROR: The following ROS2 topics are missing:\n"
                f"  {', '.join(missing_topics)}\n\n"
                f"This usually means:\n"
                f"  1. Gazebo is not running: 'gz sim src/worlds/mazes/maze_1.sdf' in terminal 1\n"
                f"  2. ROS2 bridges are not active: 'bash ./setup_bridges.sh' in terminal 2\n"
                f"  3. UAV models are not spawned in Gazebo\n\n"
                f"Available topics:\n"
            )
            available = sorted([t for t in all_topics if "uav" in t])
            if available:
                error_msg += "  " + ", ".join(available)
            else:
                error_msg += "  (no UAV-related topics found)"
            
            self._node.get_logger().error(error_msg)
            self._py_logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        self._node.get_logger().info(f"All required topics present: {self.agents}")
        self._py_logger.info("All required topics present: %s", self.agents)

    def _has_agent_sensor_data(self, agent):
        return self._last_odom.get(agent) is not None and self._last_lidar.get(agent) is not None

    def _build_coverage_grid_shape(self):
        x_span = max(self.arena_x_limits[1] - self.arena_x_limits[0], self.coverage_cell_size)
        y_span = max(self.arena_y_limits[1] - self.arena_y_limits[0], self.coverage_cell_size)
        x_bins = max(1, int(np.ceil(x_span / self.coverage_cell_size)))
        y_bins = max(1, int(np.ceil(y_span / self.coverage_cell_size)))
        return x_bins, y_bins

    def _load_spawn_pose_specs(self):
        if not self._world_sdf_path.exists():
            self._py_logger.warning(
                "[ResetService] World SDF not found for spawn pose parsing: %s",
                self._world_sdf_path,
            )
            return None

        try:
            tree = ET.parse(self._world_sdf_path)
        except ET.ParseError as exc:
            self._py_logger.warning(
                "[ResetService] Failed to parse world SDF for spawn poses: %s",
                exc,
            )
            return None

        root = tree.getroot()
        spawn_specs = {}
        for include in root.findall(".//include"):
            name_node = include.find("name")
            pose_node = include.find("pose")
            if name_node is None or pose_node is None or not name_node.text:
                continue
            agent = name_node.text.strip()
            if agent not in self.agents:
                continue
            pose_tokens = pose_node.text.split()
            if len(pose_tokens) < 6:
                continue
            pose_values = [float(token) for token in pose_tokens[:6]]
            quat = _quat_from_rpy(pose_values[3], pose_values[4], pose_values[5])
            spawn_specs[agent] = {
                "position": np.asarray(pose_values[:3], dtype=np.float32),
                "orientation": np.asarray(quat, dtype=np.float32),
            }

        if len(spawn_specs) != len(self.agents):
            missing_agents = [agent for agent in self.agents if agent not in spawn_specs]
            if missing_agents:
                self._py_logger.warning(
                    "[ResetService] Missing spawn poses for agents in %s: %s",
                    self._world_sdf_path,
                    missing_agents,
                )
        return spawn_specs or None

    def _build_pose_vector_request(self):
        if not self._spawn_pose_specs:
            return None

        pose_entries = []
        for agent in self.agents:
            spec = self._spawn_pose_specs.get(agent)
            if spec is None:
                return None
            position = spec["position"]
            orientation = spec["orientation"]
            pose_entries.append(
                "{"
                f'name: "{agent}" '
                f'position: {{x: {position[0]:.6f} y: {position[1]:.6f} z: {position[2]:.6f}}} '
                f'orientation: {{x: {orientation[0]:.8f} y: {orientation[1]:.8f} z: {orientation[2]:.8f} w: {orientation[3]:.8f}}}'
                "}"
            )
        return f"pose: [{', '.join(pose_entries)}]"

    def _set_pose_service_candidates(self, suffix):
        candidates = []
        base_services = []
        if self._active_gz_reset_service is not None:
            base_services.append(self._active_gz_reset_service)
        base_services.extend(self._gz_reset_service_candidates)
        for service in base_services:
            if service.startswith("/world/") and service.endswith("/control"):
                candidate = f"{service[:-len('/control')]}/{suffix}"
                if candidate not in candidates:
                    candidates.append(candidate)
        for world_token in [self.world_name, self._service_world_name, "default"]:
            candidate = f"/world/{world_token}/{suffix}"
            if candidate not in candidates:
                candidates.append(candidate)
        return candidates

    def _call_gz_service(self, service, reqtype, reptype, request, timeout_ms):
        command = [
            "gz",
            "service",
            "-s",
            service,
            "--reqtype",
            reqtype,
            "--reptype",
            reptype,
            "--timeout",
            str(timeout_ms),
            "--req",
            request,
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=max((timeout_ms / 1000.0) + 2.0, 3.0),
        )
        output = ((completed.stdout or "") + (completed.stderr or "")).strip()
        return completed.returncode == 0 and "data: true" in output, output or f"exit code {completed.returncode}"

    def _restore_spawn_poses_via_gz(self):
        if not self._spawn_pose_specs:
            return True

        timeout_ms = max(int(self._reset_timeout * 1000), 1500)
        vector_request = self._build_pose_vector_request()
        last_output = ""

        if vector_request:
            for service in self._set_pose_service_candidates("set_pose_vector"):
                try:
                    ok, output = self._call_gz_service(
                        service,
                        "gz.msgs.Pose_V",
                        "gz.msgs.Boolean",
                        vector_request,
                        timeout_ms,
                    )
                except (OSError, subprocess.SubprocessError) as exc:
                    last_output = str(exc)
                    continue
                last_output = output
                if ok:
                    self._py_logger.info("[ResetService] Restored UAV spawn poses via %s", service)
                    return True

        for agent in self.agents:
            spec = self._spawn_pose_specs.get(agent)
            if spec is None:
                return False
            request = (
                f'name: "{agent}" '
                f'position: {{x: {spec["position"][0]:.6f} y: {spec["position"][1]:.6f} z: {spec["position"][2]:.6f}}} '
                f'orientation: {{x: {spec["orientation"][0]:.8f} y: {spec["orientation"][1]:.8f} z: {spec["orientation"][2]:.8f} w: {spec["orientation"][3]:.8f}}}'
            )
            moved = False
            for service in self._set_pose_service_candidates("set_pose"):
                try:
                    ok, output = self._call_gz_service(
                        service,
                        "gz.msgs.Pose",
                        "gz.msgs.Boolean",
                        request,
                        timeout_ms,
                    )
                except (OSError, subprocess.SubprocessError) as exc:
                    last_output = str(exc)
                    continue
                last_output = output
                if ok:
                    moved = True
                    break
            if not moved:
                self._py_logger.warning(
                    "[ResetService] Failed to restore spawn pose for %s: %s",
                    agent,
                    last_output or "unknown error",
                )
                return False

        self._py_logger.info("[ResetService] Restored UAV spawn poses via per-agent set_pose calls.")
        return True

    def _reset_coverage_state(self):
        x_bins, y_bins = self._coverage_grid_shape
        self._coverage_visit_counts = np.zeros((x_bins, y_bins), dtype=np.int32)
        owner = np.zeros((x_bins, y_bins), dtype=np.int32)
        for x_index in range(x_bins):
            owner[x_index, :] = min((x_index * len(self.agents)) // x_bins, len(self.agents) - 1)
        self._coverage_owner = owner
        self._coverage_complete = False
        self._coverage_completion_step = None
        self._prev_coverage_ratio = 0.0

    def _position_to_cell_index(self, x_value, y_value):
        if (
            x_value < self.arena_x_limits[0]
            or x_value > self.arena_x_limits[1]
            or y_value < self.arena_y_limits[0]
            or y_value > self.arena_y_limits[1]
        ):
            return None

        x_bins, y_bins = self._coverage_grid_shape
        x_index = int((x_value - self.arena_x_limits[0]) / self.coverage_cell_size)
        y_index = int((y_value - self.arena_y_limits[0]) / self.coverage_cell_size)
        x_index = int(np.clip(x_index, 0, x_bins - 1))
        y_index = int(np.clip(y_index, 0, y_bins - 1))
        return x_index, y_index

    def _coverage_ratio(self):
        if self._coverage_visit_counts is None:
            return 0.0
        covered = np.count_nonzero(self._coverage_visit_counts > 0)
        total = max(self._coverage_visit_counts.size, 1)
        return float(covered / total)

    def _sector_coverage_ratio(self, agent_index):
        if self._coverage_visit_counts is None or self._coverage_owner is None:
            return 0.0
        sector_mask = self._coverage_owner == agent_index
        sector_total = int(np.count_nonzero(sector_mask))
        if sector_total <= 0:
            return 0.0
        sector_covered = int(np.count_nonzero(self._coverage_visit_counts[sector_mask] > 0))
        return float(sector_covered / sector_total)

    def _local_coverage_ratio_from_cell(self, cell_index):
        if self._coverage_visit_counts is None or cell_index is None:
            return 0.0
        x_index, y_index = cell_index
        x_min = max(0, x_index - self.local_coverage_radius)
        x_max = min(self._coverage_visit_counts.shape[0], x_index + self.local_coverage_radius + 1)
        y_min = max(0, y_index - self.local_coverage_radius)
        y_max = min(self._coverage_visit_counts.shape[1], y_index + self.local_coverage_radius + 1)
        patch = self._coverage_visit_counts[x_min:x_max, y_min:y_max]
        if patch.size == 0:
            return 0.0
        return float(np.count_nonzero(patch > 0) / patch.size)

    def _register_coverage(self, obs):
        step_stats = {}
        cell_to_agents = {}
        coverage_completed_now = False
        sr = self.sensor_coverage_radius
        x_bins, y_bins = self._coverage_grid_shape

        for agent_index, agent in enumerate(self.agents):
            pose = obs[agent]["pose"]
            sensor_ready = self._has_agent_sensor_data(agent)
            cell_index = self._position_to_cell_index(float(pose[0]), float(pose[1])) if sensor_ready else None
            revisit_cell = False
            newly_covered_cell = False
            new_cells_count = 0
            assigned_cell = False
            assigned_new_count = 0
            local_ratio = 0.0
            sector_ratio = self._sector_coverage_ratio(agent_index)
            visit_count = 0
            if cell_index is not None and self._coverage_visit_counts is not None:
                cx, cy = cell_index
                visit_count = int(self._coverage_visit_counts[cell_index])
                # Mark all cells within sensor_coverage_radius
                for dx in range(-sr, sr + 1):
                    for dy in range(-sr, sr + 1):
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < x_bins and 0 <= ny < y_bins:
                            if self._coverage_visit_counts[nx, ny] == 0:
                                new_cells_count += 1
                                if self._coverage_owner[nx, ny] == agent_index:
                                    assigned_new_count += 1
                            self._coverage_visit_counts[nx, ny] += 1
                            cell_to_agents.setdefault((nx, ny), []).append(agent)
                newly_covered_cell = new_cells_count > 0
                revisit_cell = (not newly_covered_cell) and visit_count > 0
                assigned_cell = assigned_new_count > 0
                local_ratio = self._local_coverage_ratio_from_cell(cell_index)
                sector_ratio = self._sector_coverage_ratio(agent_index)

            step_stats[agent] = {
                "cell_index": cell_index,
                "newly_covered_cell": newly_covered_cell,
                "new_cells_count": int(new_cells_count),
                "revisit_cell": revisit_cell,
                "assigned_cell": assigned_cell,
                "assigned_new_count": int(assigned_new_count),
                "local_coverage_ratio": local_ratio,
                "sector_coverage_ratio": sector_ratio,
                "visit_count": visit_count + (1 if cell_index is not None else 0),
            }

        coverage_ratio = self._coverage_ratio()
        if not self._coverage_complete and coverage_ratio >= self.coverage_target_ratio:
            self._coverage_complete = True
            self._coverage_completion_step = int(self._step_count)
            coverage_completed_now = True

        for stats in step_stats.values():
            cell_index = stats["cell_index"]
            overlap_cell = cell_index is not None and len(cell_to_agents.get(cell_index, [])) > 1
            overlap_count = len(cell_to_agents.get(cell_index, [])) if cell_index is not None else 0
            stats["overlap_cell"] = bool(overlap_cell)
            stats["overlap_count"] = int(max(0, overlap_count - 1))
            stats["coverage_ratio"] = coverage_ratio
            stats["coverage_complete"] = bool(self._coverage_complete)
            stats["coverage_completed_now"] = bool(coverage_completed_now)
            stats["coverage_completion_step"] = self._coverage_completion_step
            stats["covered_cells"] = int(np.count_nonzero(self._coverage_visit_counts > 0))
            stats["total_coverable_cells"] = int(self._coverage_visit_counts.size)

        return step_stats

    def _coverage_features(self, agent, pose):
        agent_index = self.agents.index(agent)
        cell_index = self._position_to_cell_index(float(pose[0]), float(pose[1]))
        global_ratio = self._coverage_ratio()
        sector_ratio = self._sector_coverage_ratio(agent_index)
        local_ratio = self._local_coverage_ratio_from_cell(cell_index)
        current_cell_visited = 0.0
        if cell_index is not None and self._coverage_visit_counts is not None:
            current_cell_visited = 1.0 if self._coverage_visit_counts[cell_index] > 0 else 0.0
        x_bins, _ = self._coverage_grid_shape
        sector_width = max(1, x_bins // len(self.agents))
        sector_center_index = min(agent_index * sector_width + sector_width / 2.0, x_bins - 1)
        sector_center_x = self.arena_x_limits[0] + sector_center_index * self.coverage_cell_size
        arena_width = max(self.arena_x_limits[1] - self.arena_x_limits[0], 1.0)
        sector_offset = float(np.clip((pose[0] - sector_center_x) / arena_width, -1.0, 1.0))
        return np.array(
            [global_ratio, sector_ratio, local_ratio, current_cell_visited, (sector_offset + 1.0) * 0.5],
            dtype=np.float32,
        )

    def _local_coverage_map(self, pose):
        """Return a flattened binary grid (local_map_size × local_map_size)
        centred on the agent's current cell.
        Values: 0.0 = unvisited, 1.0 = visited, 0.5 = outside arena.
        """
        size = self.local_map_size
        half = size // 2
        result = np.full(size * size, 0.5, dtype=np.float32)  # default: boundary
        cell = self._position_to_cell_index(float(pose[0]), float(pose[1]))
        if cell is None or self._coverage_visit_counts is None:
            return result
        cx, cy = cell
        x_bins, y_bins = self._coverage_grid_shape
        idx = 0
        for dx in range(-half, half + 1):
            for dy in range(-half, half + 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < x_bins and 0 <= ny < y_bins:
                    result[idx] = 1.0 if self._coverage_visit_counts[nx, ny] > 0 else 0.0
                idx += 1
        return result

    def _other_agents_features(self, agent, all_poses):
        """Encode relative (dx, dy) and heading (cos_yaw, sin_yaw) of every
        other agent, normalised by arena diagonal."""
        n_others = len(self.agents) - 1
        features = np.zeros(n_others * 4, dtype=np.float32)
        my_pose = all_poses[agent]
        arena_diag = np.sqrt(
            (self.arena_x_limits[1] - self.arena_x_limits[0]) ** 2
            + (self.arena_y_limits[1] - self.arena_y_limits[0]) ** 2
        )
        arena_diag = max(arena_diag, 1.0)
        idx = 0
        for other in self.agents:
            if other == agent:
                continue
            op = all_poses[other]
            features[idx * 4 + 0] = float(np.clip((op[0] - my_pose[0]) / arena_diag, -1.0, 1.0))
            features[idx * 4 + 1] = float(np.clip((op[1] - my_pose[1]) / arena_diag, -1.0, 1.0))
            features[idx * 4 + 2] = float(np.cos(op[3]))
            features[idx * 4 + 3] = float(np.sin(op[3]))
            idx += 1
        return features

    def describe_task(self):
        return {
            "task_mode": "area_coverage",
            "coverage_cell_size": self.coverage_cell_size,
            "coverage_target_ratio": self.coverage_target_ratio,
            "coverage_new_cell_reward": self.coverage_new_cell_reward,
            "coverage_completion_bonus": self.coverage_completion_bonus,
            "revisit_cell_penalty": self.revisit_cell_penalty,
            "overlap_cell_penalty": self.overlap_cell_penalty,
            "assignment_bonus": self.assignment_bonus,
            "local_coverage_radius": self.local_coverage_radius,
            "sensor_coverage_radius": self.sensor_coverage_radius,
            "local_map_size": self.local_map_size,
            "coverage_delta_scale": self.coverage_delta_scale,
            "action_update_timeout": self._action_update_timeout,
            "require_step_lidar_update": self._require_step_lidar_update,
            "reset_stabilization_timeout": self.reset_stabilization_timeout,
            "reset_poll_interval": self.reset_poll_interval,
            "reset_initial_sensor_timeout": self.reset_initial_sensor_timeout,
            "reset_initial_sensor_timeout_first": self.reset_initial_sensor_timeout_first,
            "pre_reset_brake_wait": self.pre_reset_brake_wait,
            "arena_x_limits": list(self.arena_x_limits),
            "arena_y_limits": list(self.arena_y_limits),
            "max_steps": self.max_steps,
            "world_name": self.world_name,
            "min_step_duration": self._min_step_duration,
        }

    def _wait_for_sensor_data(self, timeout=None):
        """Wait for all agents to receive at least one sensor data message.
        
        Spins more aggressively to ensure ROS2 has time to deliver messages from bridges.
        """
        wait_timeout = self.sensor_wait_timeout if timeout is None else float(timeout)
        deadline = time.time() + wait_timeout
        while time.time() < deadline:
            if all(self._has_agent_sensor_data(agent) for agent in self.agents):
                return True
            # Spin multiple times per iteration to process more messages
            for _ in range(5):
                self._spin_once()
        return all(self._has_agent_sensor_data(agent) for agent in self.agents)

    def _collect_observation(self):
        obs = {}
        all_poses = {}
        for agent in self.agents:
            odom = self._last_odom.get(agent)
            scan = self._last_lidar.get(agent)

            if odom is None:
                pose = np.zeros(8, dtype=np.float32)
            else:
                q = odom.pose.pose.orientation
                yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
                pose = np.array(
                    [
                        odom.pose.pose.position.x,
                        odom.pose.pose.position.y,
                        odom.pose.pose.position.z,
                        yaw,
                        odom.twist.twist.linear.x,
                        odom.twist.twist.linear.y,
                        odom.twist.twist.linear.z,
                        odom.twist.twist.angular.z,
                    ],
                    dtype=np.float32,
                )

            if scan is None:
                lidar = np.full(self.observation_spaces[agent]["lidar"].shape, np.inf, dtype=np.float32)
            else:
                ranges = np.array(scan.ranges, dtype=np.float32)
                ranges = np.nan_to_num(ranges, nan=np.inf, posinf=np.inf, neginf=0.0)
                if ranges.size != self.observation_spaces[agent]["lidar"].shape[0]:
                    # Pad or trim to match expected size
                    target = self.observation_spaces[agent]["lidar"].shape[0]
                    if ranges.size < target:
                        padded = np.full(target, np.inf, dtype=np.float32)
                        padded[: ranges.size] = ranges
                        ranges = padded
                    else:
                        ranges = ranges[:target]
                lidar = ranges

            all_poses[agent] = pose
            obs[agent] = {
                "pose": pose,
                "lidar": lidar,
                "coverage": self._coverage_features(agent, pose),
                "local_map": self._local_coverage_map(pose),
            }

        # Second pass: cross-agent relative features (needs all poses)
        for agent in self.agents:
            obs[agent]["other_agents"] = self._other_agents_features(agent, all_poses)

        return obs

    def _assess_spawn_layout_risk(self, obs):
        if not all(self._has_agent_sensor_data(agent) for agent in self.agents):
            return False, None, False, False

        positions = {agent: obs[agent]["pose"][:3] for agent in self.agents}
        min_pair_dist = np.inf
        for i, a1 in enumerate(self.agents):
            for a2 in self.agents[i + 1 :]:
                d = float(np.linalg.norm(positions[a1] - positions[a2]))
                min_pair_dist = min(min_pair_dist, d)

        min_alt = min(float(obs[a]["pose"][2]) for a in self.agents)
        approx_lidar_clearance = min_pair_dist - 0.18
        risky_alt = min_alt < self.min_spawn_altitude
        risky_spacing = approx_lidar_clearance < self.min_spawn_pair_clearance
        if risky_alt or risky_spacing:
            message = (
                "Potentially risky spawn layout detected: "
                f"min_alt={min_alt:.3f} (threshold={self.min_spawn_altitude:.3f}), "
                f"min_pair_dist={min_pair_dist:.3f}, "
                f"approx_lidar_clearance={approx_lidar_clearance:.3f} "
                f"(threshold={self.min_spawn_pair_clearance:.3f})."
            )
            return True, message, risky_alt, risky_spacing
        return False, None, False, False

    def _warn_if_spawn_layout_risky(self, message):
        if not message:
            return
        self._node.get_logger().warning(message)
        self._py_logger.warning(message)
        self._warned_spawn_layout = True

    def _compute_reset_xy_drift(self, obs):
        if not all(self._has_agent_sensor_data(agent) for agent in self.agents):
            return None

        current_positions = {agent: obs[agent]["pose"][:2].copy() for agent in self.agents}
        if self._spawn_reference_positions is None:
            self._spawn_reference_positions = current_positions
            return 0.0

        dists = [
            float(np.linalg.norm(current_positions[agent] - self._spawn_reference_positions[agent]))
            for agent in self.agents
        ]
        return max(dists) if dists else 0.0

    def _wait_for_reset_stabilization(self, initial_obs):
        best_obs = initial_obs
        best_drift = self._compute_reset_xy_drift(initial_obs)
        if best_drift is None or best_drift <= self.reset_position_tolerance:
            return best_obs, best_drift

        deadline = time.time() + max(self.reset_stabilization_timeout, 0.0)
        while time.time() < deadline:
            for pub in self._publishers.values():
                pub.publish(Twist())
            self._spin_for(max(self.reset_poll_interval, self._spin_timeout))
            candidate_obs = self._collect_observation()
            candidate_drift = self._compute_reset_xy_drift(candidate_obs)
            if candidate_drift is None:
                continue
            if best_drift is None or candidate_drift < best_drift:
                best_drift = candidate_drift
                best_obs = candidate_obs
            if candidate_drift <= self.reset_position_tolerance:
                return candidate_obs, candidate_drift

        return best_obs, best_drift

    def _validate_reset_positions(self, obs, drift):
        if drift is None:
            return
        if drift > self.reset_position_tolerance:
            raise RuntimeError(
                "Environment reset failed: UAVs did not return to initial XY positions. "
                f"max XY drift={drift:.3f}m > tolerance={self.reset_position_tolerance:.3f}m. "
                f"Please check Gazebo reset service '{self._reset_service_name}' and world reset semantics."
            )

    def _build_reset_drift_error(self, drift):
        return (
            "Environment reset failed: UAVs did not return to initial XY positions. "
            f"max XY drift={drift:.3f}m > tolerance={self.reset_position_tolerance:.3f}m. "
            f"Please check Gazebo reset service '{self._reset_service_name}' and world reset semantics."
        )

    def _reset_simulation(self):
        # Attempt to reset via ROS service first, then fall back to Gazebo native service.
        if self._reset_client is None:
            return self._reset_simulation_via_gz()

        if not self._reset_client.service_is_ready():
            if not self._reset_client.wait_for_service(timeout_sec=self._reset_timeout):
                self._warn_reset_unavailable(f"not ready after {self._reset_timeout}s")
                return self._reset_simulation_via_gz()

        req = ControlWorld.Request()
        req.world_control.reset.all = True
        # For deterministic resets, you can set seed (recommended for training)
        # req.world_control.seed = 12345

        future = self._reset_client.call_async(req)
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=self._reset_timeout)
        if future.done():
            resp = future.result()
            if resp is not None and getattr(resp, "success", False):
                return True
        return self._reset_simulation_via_gz()

    def _refresh_gz_reset_service_candidates(self):
        command = ["gz", "service", "-l"]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=max(self._reset_timeout + 1.0, 3.0),
            )
        except (OSError, subprocess.SubprocessError):
            return

        if completed.returncode != 0:
            return

        discovered = []
        for line in (completed.stdout or "").splitlines():
            service = line.strip()
            if service.startswith("/world/") and service.endswith("/control"):
                discovered.append(service)

        preferred = [
            f"/world/{self._service_world_name}/control",
            f"/world/{self.world_name}/control",
            "/world/default/control",
        ]
        merged = []
        for service in preferred + discovered + self._gz_reset_service_candidates:
            if service and service not in merged:
                merged.append(service)
        self._gz_reset_service_candidates = merged

    def _reset_simulation_via_gz(self):
        timeout_ms = max(int(self._reset_timeout * 1000), 1500)
        self._refresh_gz_reset_service_candidates()

        candidate_services = []
        if self._active_gz_reset_service is not None:
            candidate_services.append(self._active_gz_reset_service)
        for service in self._gz_reset_service_candidates:
            if service not in candidate_services:
                candidate_services.append(service)

        last_output = ""
        for service in candidate_services:
            for attempt in range(3):
                attempt_timeout_ms = timeout_ms * (attempt + 1)
                command = [
                    "gz",
                    "service",
                    "-s",
                    service,
                    "--reqtype",
                    "gz.msgs.WorldControl",
                    "--reptype",
                    "gz.msgs.Boolean",
                    "--timeout",
                    str(attempt_timeout_ms),
                    "--req",
                    "reset: {all: true}",
                ]
                try:
                    completed = subprocess.run(
                        command,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=max(self._reset_timeout * (attempt + 2), 3.0),
                    )
                except (OSError, subprocess.SubprocessError) as exc:
                    last_output = str(exc)
                    time.sleep(0.2)
                    continue

                output = ((completed.stdout or "") + (completed.stderr or "")).strip()
                last_output = output or f"exit code {completed.returncode}"
                if completed.returncode == 0 and "data: true" in output:
                    if self._active_gz_reset_service != service:
                        self._active_gz_reset_service = service
                        self._py_logger.info("[ResetService] Using Gazebo reset service: %s", service)
                    return True
                if "Service call timed out" in output:
                    time.sleep(0.2)
                    continue
                if "service call failed" not in output.lower() and "not found" not in output.lower():
                    break

        self._warn_gz_reset_unavailable(
            f"failed across {candidate_services}: {last_output or 'unknown error'}"
        )
        return False

    def _restart_simulation_stack(self):
        """Kill and restart Gazebo + ROS-GZ bridges when Gazebo becomes unresponsive."""
        project_root = self._project_root
        world_sdf_path = self._world_sdf_path
        bridges_script = project_root / "src" / "scripts" / "setup_bridges.sh"

        if not world_sdf_path.exists():
            self._py_logger.error(
                "[AutoRestart] Cannot restart: world SDF not found at %s", world_sdf_path,
            )
            return False

        self._py_logger.warning("[AutoRestart] Gazebo unresponsive — restarting simulation stack...")

        # 1. Kill existing Gazebo server processes
        for pattern in ["gz sim -s", "gz-sim-server", "ruby.*gz.*sim"]:
            subprocess.run(
                ["pkill", "-9", "-f", pattern],
                capture_output=True, timeout=5,
            )
        time.sleep(2)

        # 2. Restart Gazebo headless
        gz_log_path = project_root / "artifacts" / "gz_autorestart.log"
        gz_log_path.parent.mkdir(parents=True, exist_ok=True)
        gz_log_f = open(gz_log_path, "a")  # noqa: SIM115
        self._gz_restart_log_f = gz_log_f  # prevent GC
        subprocess.Popen(
            ["gz", "sim", "-s", str(world_sdf_path)],
            stdout=gz_log_f, stderr=gz_log_f,
            preexec_fn=os.setpgrp,
        )
        self._py_logger.info("[AutoRestart] Gazebo server relaunched on world %s", world_sdf_path.name)
        time.sleep(5)

        # 3. Kill existing bridge processes and restart
        subprocess.run(["pkill", "-9", "-f", "parameter_bridge"], capture_output=True, timeout=5)
        time.sleep(1)

        if bridges_script.exists():
            subprocess.Popen(
                ["bash", str(bridges_script)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                preexec_fn=os.setpgrp,
            )
            self._py_logger.info("[AutoRestart] ROS-GZ bridges relaunched.")
            time.sleep(5)

        # 4. Clear cached reset service so discovery runs fresh
        self._active_gz_reset_service = None

        self._py_logger.info("[AutoRestart] Simulation stack restarted. Attempting reset...")
        return True

    def reset(self, seed=None, return_info=False, options=None):
        # On first reset, check if all required topics exist
        is_first_reset = self._first_reset
        if self._first_reset:
            try:
                self._check_required_topics()
            except RuntimeError as e:
                self._node.get_logger().error(str(e))
                self._node.get_logger().warning("Proceeding without topic verification...")
                self._py_logger.exception("Topic verification failed; proceeding without topic verification.")
            self._first_reset = False
        
        # Stop all UAVs before reset to reduce drift carry-over.
        for pub in self._publishers.values():
            pub.publish(Twist())
        self._spin_for(max(self._spin_timeout, self.pre_reset_brake_wait))

        # Reset the Gazebo world (if supported via ros_gz_bridge), then clear
        # stored observations and publish zero commands for safety.
        reset_ok = self._reset_simulation()
        if not reset_ok:
            # One retry helps with transient service delays in long headless runs.
            self._py_logger.warning("Reset failed once; retrying after short wait.")
            time.sleep(0.4)
            for pub in self._publishers.values():
                pub.publish(Twist())
            self._spin_for(max(self._spin_timeout, self.pre_reset_brake_wait))
            reset_ok = self._reset_simulation()
        if not reset_ok:
            # Last resort: restart the entire Gazebo + bridge stack and retry.
            if self._restart_simulation_stack():
                reset_ok = self._reset_simulation()
        if not reset_ok:
            raise RuntimeError(
                f"Failed to reset world '{self.world_name}'. "
                "Training would continue from old state, so it has been stopped intentionally."
            )

        if not self._restore_spawn_poses_via_gz():
            self._py_logger.warning(
                "[ResetService] Spawn pose restoration did not fully succeed; relying on world reset only."
            )

        obs = None
        reset_drift = None
        max_attempts = max(1, max(self.reset_validation_retries, self.spawn_layout_validation_retries) + 1)
        for attempt in range(max_attempts):
            # Clear stored sensor data so we don't return stale observations.
            self._last_odom = {agent: None for agent in self.agents}
            self._last_lidar = {agent: None for agent in self.agents}
            self._dones = {agent: False for agent in self.agents}
            self._collision_cooldown = {agent: 0 for agent in self.agents}
            self._step_count = 0
            self._reset_coverage_state()

            # Publish zero command to stop movement after reset.
            for pub in self._publishers.values():
                pub.publish(Twist())

            # Give gazebo / ROS a chance to update after reset.
            # Use longer timeout on first reset to allow bridges to stabilize
            initial_wait_timeout = self.reset_initial_sensor_timeout_first if is_first_reset else self.reset_initial_sensor_timeout
            sensor_data_ok = self._wait_for_sensor_data(timeout=initial_wait_timeout)
            if not sensor_data_ok:
                self._warn_sensor_unavailable()

            obs = self._collect_observation()
            obs, reset_drift = self._wait_for_reset_stabilization(obs)
            risky_spawn, risky_message, risky_alt, risky_spacing = self._assess_spawn_layout_risk(obs)

            drift_ok = (reset_drift is None or reset_drift <= self.reset_position_tolerance)
            spawn_ok = not risky_spawn
            mild_spawn_risk = risky_alt and not risky_spacing

            if drift_ok and (spawn_ok or mild_spawn_risk):
                if mild_spawn_risk and risky_message:
                    self._py_logger.warning(
                        "[ResetValidation] Mild spawn altitude risk accepted: %s",
                        risky_message,
                    )
                break

            if attempt < max_attempts - 1:
                issues = []
                severe_drift = False
                if not drift_ok:
                    issues.append(
                        f"drift {reset_drift:.3f}m > tol {self.reset_position_tolerance:.3f}m"
                    )
                    severe_drift = reset_drift > (self.reset_position_tolerance * 1.8)
                if risky_spawn:
                    if risky_spacing:
                        issues.append("risky spawn spacing")
                    elif risky_alt:
                        issues.append("risky spawn altitude")
                self._py_logger.warning(
                    "[ResetValidation] %s (attempt %d/%d); retrying reset.",
                    ", ".join(issues),
                    attempt + 1,
                    max_attempts,
                )
                reset_ok = False
                spacing_needs_escalation = risky_spacing and attempt >= 1
                if severe_drift or spacing_needs_escalation:
                    self._py_logger.warning(
                        "[ResetValidation] Escalating recovery (severe_drift=%s, risky_spacing=%s). Restarting simulation stack before retry.",
                        severe_drift,
                        risky_spacing,
                    )
                    if self._restart_simulation_stack():
                        reset_ok = self._reset_simulation()
                if not reset_ok:
                    reset_ok = self._reset_simulation()
                if not reset_ok and self._restart_simulation_stack():
                    reset_ok = self._reset_simulation()
                if not reset_ok:
                    raise RuntimeError(
                        f"Failed to reset world '{self.world_name}' during drift recovery retry."
                    )
                continue

            if not spawn_ok:
                if risky_alt and not risky_spacing:
                    self._py_logger.warning(
                        "[ResetValidation] Proceeding with mild spawn altitude risk after retries: %s",
                        risky_message,
                    )
                elif self.spawn_layout_allow_risky:
                    self._py_logger.warning(
                        "[ResetValidation] %s Proceeding due to spawn_layout_allow_risky=true.",
                        risky_message,
                    )
                else:
                    raise RuntimeError(
                        "Environment reset failed: spawn layout remains risky after retries. "
                        f"{risky_message}"
                    )

            if not drift_ok and self.reset_validation_allow_failure:
                self._py_logger.warning(
                    "[ResetValidation] %s Proceeding due to reset_validation_allow_failure=true.",
                    self._build_reset_drift_error(reset_drift),
                )
            elif not drift_ok:
                raise RuntimeError(self._build_reset_drift_error(reset_drift))

        if not all(self._has_agent_sensor_data(agent) for agent in self.agents):
            self._warn_sensor_unavailable()
        self._register_coverage(obs)
        obs = self._collect_observation()
        if return_info:
            return obs, {agent: {} for agent in self.agents}
        return obs

    def step(self, actions):
        # actions: dict(agent->np.array([vx, vy, vz, yaw_rate]))
        for agent, action in actions.items():
            if self._dones.get(agent, False):
                # Dead agent: brake, do not apply requested action.
                self._publishers[agent].publish(Twist())
                continue
            twist = Twist()
            twist.linear.x = float(action[0])
            twist.linear.y = float(action[1])
            # Altitude safety: clamp descent when altitude approaches min_height
            vz = float(action[2])
            odom = self._last_odom.get(agent)
            if odom is not None and vz < 0:
                alt = odom.pose.pose.position.z
                if alt < self.altitude_safety_threshold:
                    vz = 0.0
            twist.linear.z = vz
            twist.angular.z = float(action[3])
            self._publishers[agent].publish(twist)

        # Give ROS time to process and update observations.
        # _wait_for_post_action_observation also enforces min_step_duration
        # so Gazebo physics advances enough between RL steps.
        self._wait_for_post_action_observation()

        obs = self._collect_observation()
        rewards = {}
        dones = {}
        infos = {}
        self._step_count += 1

        coverage_step_stats = self._register_coverage(obs)

        # Δcoverage reward (team-level coverage increment)
        current_coverage_ratio = self._coverage_ratio()
        delta_coverage = current_coverage_ratio - self._prev_coverage_ratio
        self._prev_coverage_ratio = current_coverage_ratio

        reached_or_failed = 0
        in_startup_grace = self._step_count <= self.startup_grace_steps
        for agent in self.agents:
            # Already-dead agents: zero reward, stay done, skip penalty logic.
            if self._dones.get(agent, False):
                rewards[agent] = 0.0
                dones[agent] = True
                infos[agent] = {"already_dead": True, "collided": True,
                                "coverage_ratio": float(coverage_step_stats.get(agent, {}).get("coverage_ratio", 0.0)),
                                "coverage_complete": bool(coverage_step_stats.get(agent, {}).get("coverage_complete", False))}
                reached_or_failed += 1
                continue

            pose = obs[agent]["pose"]
            lidar = obs[agent]["lidar"]
            sensor_ready = self._has_agent_sensor_data(agent)
            coverage_stats = coverage_step_stats.get(agent, {})
            reward = -self.step_penalty + self.survival_bonus
            lidar_min = float(np.min(lidar)) if lidar.size > 0 else np.inf
            out_of_bounds = sensor_ready and (
                pose[0] < self.arena_x_limits[0]
                or pose[0] > self.arena_x_limits[1]
                or pose[1] < self.arena_y_limits[0]
                or pose[1] > self.arena_y_limits[1]
            )
            min_height_active = self._step_count > self.min_height_enforce_steps
            hard_floor_hit = sensor_ready and (pose[2] < self.hard_floor_height)
            altitude_violation = sensor_ready and (
                pose[2] > self.max_height
                or hard_floor_hit
                or (min_height_active and pose[2] < self.min_height)
            )
            collided_raw = sensor_ready and (lidar_min < self.collision_lidar_threshold or altitude_violation)
            in_cooldown = self._collision_cooldown.get(agent, 0) > 0
            collided = bool(collided_raw and not in_startup_grace and not in_cooldown)

            if self.collision_terminal:
                done = collided or out_of_bounds
            else:
                # Non-terminal collisions: only OOB and hard_floor_hit are terminal
                done = out_of_bounds or (hard_floor_hit and not in_startup_grace)

            # Manage collision cooldown
            if collided and not self.collision_terminal:
                self._collision_cooldown[agent] = self.collision_cooldown_steps
            elif self._collision_cooldown.get(agent, 0) > 0:
                self._collision_cooldown[agent] -= 1

            if self.coverage_delta_scale > 0:
                # Team-level Δcoverage reward (same for all alive agents)
                reward += self.coverage_delta_scale * delta_coverage
                # Keep assignment bonus as per-agent incentive
                assigned_cnt = coverage_stats.get("assigned_new_count", 0)
                if assigned_cnt > 0:
                    reward += self.assignment_bonus * assigned_cnt
                if coverage_stats.get("revisit_cell", False):
                    reward -= self.revisit_cell_penalty
            else:
                # Original per-cell reward mode
                if coverage_stats.get("newly_covered_cell", False):
                    new_cnt = coverage_stats.get("new_cells_count", 1)
                    reward += self.coverage_new_cell_reward * new_cnt
                    assigned_cnt = coverage_stats.get("assigned_new_count", 0)
                    if assigned_cnt > 0:
                        reward += self.assignment_bonus * assigned_cnt
                elif coverage_stats.get("revisit_cell", False):
                    reward -= self.revisit_cell_penalty

            if coverage_stats.get("overlap_cell", False):
                reward -= self.overlap_cell_penalty * max(1, coverage_stats.get("overlap_count", 1))

            # Soft altitude penalty: penalise flying too low (gradient signal)
            if sensor_ready and pose[2] < self.altitude_safety_threshold:
                reward -= self.low_altitude_penalty

            if collided:
                reward -= self.collision_penalty
            if out_of_bounds:
                reward -= self.out_of_bounds_penalty
            elif sensor_ready and self.boundary_penalty_margin > 0:
                # Soft gradient penalty near arena edges
                margin = self.boundary_penalty_margin
                dx = max(0, self.arena_x_limits[0] + margin - pose[0],
                         pose[0] - (self.arena_x_limits[1] - margin))
                dy = max(0, self.arena_y_limits[0] + margin - pose[1],
                         pose[1] - (self.arena_y_limits[1] - margin))
                boundary_closeness = (dx + dy) / margin  # 0 at margin edge, 1 at arena edge
                if boundary_closeness > 0:
                    reward -= self.boundary_penalty_scale * boundary_closeness
            if coverage_stats.get("coverage_completed_now", False):
                reward += self.coverage_completion_bonus

            rewards[agent] = float(reward)
            dones[agent] = bool(done)
            infos[agent] = {
                "altitude": float(pose[2]),
                "collided": collided,
                "collided_raw": bool(collided_raw),
                "altitude_violation": altitude_violation,
                "min_height_active": bool(min_height_active),
                "hard_floor_hit": bool(hard_floor_hit),
                "out_of_bounds": out_of_bounds,
                "sensor_ready": sensor_ready,
                "startup_grace_active": in_startup_grace,
                "coverage_ratio": float(coverage_stats.get("coverage_ratio", 0.0)),
                "coverage_complete": bool(coverage_stats.get("coverage_complete", False)),
                "coverage_completion_step": coverage_stats.get("coverage_completion_step"),
                "covered_cells": int(coverage_stats.get("covered_cells", 0)),
                "total_coverable_cells": int(coverage_stats.get("total_coverable_cells", 0)),
                "newly_covered_cell": bool(coverage_stats.get("newly_covered_cell", False)),
                "new_cells_count": int(coverage_stats.get("new_cells_count", 0)),
                "revisit_cell": bool(coverage_stats.get("revisit_cell", False)),
                "overlap_cell": bool(coverage_stats.get("overlap_cell", False)),
                "overlap_count": int(coverage_stats.get("overlap_count", 0)),
                "sector_coverage_ratio": float(coverage_stats.get("sector_coverage_ratio", 0.0)),
                "local_coverage_ratio": float(coverage_stats.get("local_coverage_ratio", 0.0)),
                "assigned_cell": bool(coverage_stats.get("assigned_cell", False)),
            }
            if done:
                # Record per-agent death so subsequent steps skip this agent.
                self._dones[agent] = True
                reached_or_failed += 1

        if not all(self._has_agent_sensor_data(agent) for agent in self.agents):
            self._warn_sensor_unavailable()

        all_done = (
            reached_or_failed == len(self.agents)
            or self._step_count >= self.max_steps
            or self._coverage_complete
        )
        dones["__all__"] = bool(all_done)
        return obs, rewards, dones, infos

    def render(self):
        # Rendering is handled by Gazebo itself.
        return None

    def close(self):
        if hasattr(self, "_node") and self._node is not None:
            self._node.destroy_node()
            self._node = None


def make_env(**kwargs):
    return GazeboMultiUAVParallelEnv(**kwargs)
