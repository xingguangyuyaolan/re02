#!/usr/bin/env python3
"""
Training script for Attention-QMIX on Gazebo multi-UAV environment.
"""

import argparse
from datetime import datetime
import json
import logging
import os
import random
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(__file__))

from src.scripts.attention_qmix import QMIXConfig, train_attention_qmix
from src.scripts.gazebo_pettingzoo_env import GazeboMultiUAVParallelEnv
from src.scripts.logging_utils import setup_project_logging


LOGGER = logging.getLogger(__name__)


def _set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _parse_args():
    parser = argparse.ArgumentParser(description="Train Attention-QMIX in Gazebo multi-UAV environment")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON training config")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from, or 'latest'")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--world", type=str, default="maze_1.sdf", help="World file name under src/worlds/mazes")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Training device")
    return parser.parse_args()


def _load_json_config(path: str):
    project_root = os.path.dirname(__file__)
    raw_path = str(path)

    candidate_paths = []

    def _append_candidate(p: str):
        norm = os.path.normpath(p)
        if norm not in candidate_paths:
            candidate_paths.append(norm)

    # 1) exactly as provided
    _append_candidate(raw_path)

    # 2) if user omitted extension, try .json
    if not raw_path.endswith(".json"):
        _append_candidate(raw_path + ".json")

    # 3) if user passed only filename, try inside configs/
    base_name = os.path.basename(raw_path)
    _append_candidate(os.path.join(project_root, "configs", raw_path))
    _append_candidate(os.path.join(project_root, "configs", base_name))
    if not raw_path.endswith(".json"):
        _append_candidate(os.path.join(project_root, "configs", raw_path + ".json"))
        _append_candidate(os.path.join(project_root, "configs", base_name + ".json"))

    for candidate in candidate_paths:
        if os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                return json.load(f)

    raise FileNotFoundError(
        "Config file not found. Tried: " + ", ".join(candidate_paths)
    )


def _get_value(cfg_dict, key, fallback):
    return cfg_dict.get(key, fallback) if cfg_dict is not None else fallback


def _default_run_name(seed: int):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"maze_qmix_seed{seed}_{timestamp}"


def _timestamped_run_name(base_name: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}"


def _run_name_from_resume_path(resume_path: str):
    if not resume_path or resume_path == "latest":
        return None

    normalized = os.path.normpath(resume_path)
    parent_dir = os.path.dirname(normalized)
    if os.path.basename(parent_dir) == "checkpoints":
        return os.path.basename(os.path.dirname(parent_dir))
    return os.path.basename(parent_dir) or None


def _find_latest_run_name(output_root: str, base_run_name: str):
    if not os.path.isdir(output_root):
        return None

    candidates = []
    for entry in os.listdir(output_root):
        full_path = os.path.join(output_root, entry)
        if not os.path.isdir(full_path):
            continue
        if entry == base_run_name or entry.startswith(base_run_name + "_"):
            candidates.append((os.path.getmtime(full_path), entry))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _is_cross_stage_resume(base_run_name: str, resume_path: str) -> bool:
    """Return True when resume checkpoint comes from a different run (curriculum stage transfer)."""
    if not resume_path or resume_path == "latest":
        return False
    resumed_run_name = _run_name_from_resume_path(resume_path)
    if resumed_run_name is None:
        return False
    # Strip timestamp suffix for comparison: "p0_stage1_open_seed7_20260419_113356" → "p0_stage1_open_seed7"
    # base_run_name is already without timestamp, e.g. "p0_stage2_easy_seed7"
    return not resumed_run_name.startswith(base_run_name)


def _resolve_run_name(base_run_name: str, output_root: str, resume_path: str):
    if resume_path == "latest":
        latest_run_name = _find_latest_run_name(output_root, base_run_name)
        return latest_run_name or base_run_name

    if resume_path:
        if _is_cross_stage_resume(base_run_name, resume_path):
            # Curriculum transfer: create a new directory for the new stage
            return _timestamped_run_name(base_run_name)
        resumed_run_name = _run_name_from_resume_path(resume_path)
        return resumed_run_name or base_run_name
    return _timestamped_run_name(base_run_name)

def main():
    args = _parse_args()
    cfg_file = _load_json_config(args.config) if args.config else None

    episodes = int(_get_value(cfg_file, "episodes", args.episodes))
    max_steps = int(_get_value(cfg_file, "max_steps", args.max_steps))
    seed = int(_get_value(cfg_file, "seed", args.seed))
    startup_grace_steps = int(_get_value(cfg_file, "startup_grace_steps", 8))
    world_name = str(_get_value(cfg_file, "world_name", args.world))
    arena_x_limits = _get_value(cfg_file, "arena_x_limits", [-10.0, 10.0])
    arena_y_limits = _get_value(cfg_file, "arena_y_limits", [-10.0, 10.0])
    out_of_bounds_penalty = float(_get_value(cfg_file, "out_of_bounds_penalty", 2.0))
    reset_timeout = float(_get_value(cfg_file, "reset_timeout", 2.0))
    reset_position_tolerance = float(_get_value(cfg_file, "reset_position_tolerance", 5))
    reset_stabilization_timeout = float(_get_value(cfg_file, "reset_stabilization_timeout", 4.0))
    reset_poll_interval = float(_get_value(cfg_file, "reset_poll_interval", 0.2))
    reset_initial_sensor_timeout = float(_get_value(cfg_file, "reset_initial_sensor_timeout", 1.5))
    reset_initial_sensor_timeout_first = float(_get_value(cfg_file, "reset_initial_sensor_timeout_first", 4.0))
    pre_reset_brake_wait = float(_get_value(cfg_file, "pre_reset_brake_wait", 0.03))
    action_update_timeout = float(_get_value(cfg_file, "action_update_timeout", 0.03))
    require_step_lidar_update = bool(_get_value(cfg_file, "require_step_lidar_update", False))
    min_height_enforce_steps = int(_get_value(cfg_file, "min_height_enforce_steps", 60))
    hard_floor_height = float(_get_value(cfg_file, "hard_floor_height", 0.08))
    min_step_duration = float(_get_value(cfg_file, "min_step_duration", 0.1))
    altitude_safety_threshold = float(_get_value(cfg_file, "altitude_safety_threshold", 0.4))
    low_altitude_penalty = float(_get_value(cfg_file, "low_altitude_penalty", 0.3))
    collision_penalty = float(_get_value(cfg_file, "collision_penalty", 1.5))
    collision_terminal = bool(_get_value(cfg_file, "collision_terminal", True))
    collision_cooldown_steps = int(_get_value(cfg_file, "collision_cooldown_steps", 10))
    survival_bonus = float(_get_value(cfg_file, "survival_bonus", 0.0))
    boundary_penalty_margin = float(_get_value(cfg_file, "boundary_penalty_margin", 2.0))
    boundary_penalty_scale = float(_get_value(cfg_file, "boundary_penalty_scale", 0.3))

    coverage_cell_size = float(_get_value(cfg_file, "coverage_cell_size", 1.0))
    coverage_target_ratio = float(_get_value(cfg_file, "coverage_target_ratio", 0.85))
    coverage_new_cell_reward = float(_get_value(cfg_file, "coverage_new_cell_reward", 1.2))
    coverage_completion_bonus = float(_get_value(cfg_file, "coverage_completion_bonus", 8.0))
    revisit_cell_penalty = float(_get_value(cfg_file, "revisit_cell_penalty", 0.08))
    overlap_cell_penalty = float(_get_value(cfg_file, "overlap_cell_penalty", 0.15))
    assignment_bonus = float(_get_value(cfg_file, "assignment_bonus", 0.2))
    local_coverage_radius = int(_get_value(cfg_file, "local_coverage_radius", 1))
    sensor_coverage_radius = int(_get_value(cfg_file, "sensor_coverage_radius", 1))
    local_map_size = int(_get_value(cfg_file, "local_map_size", 7))
    coverage_delta_scale = float(_get_value(cfg_file, "coverage_delta_scale", 0.0))

    output_root = str(_get_value(cfg_file, "output_root", "artifacts/qmix"))
    resume_target = _get_value(cfg_file, "resume_path", args.resume)
    configured_run_name = str(_get_value(cfg_file, "run_name", f"maze_qmix_seed{seed}"))

    qmix_cfg = QMIXConfig(
        seed=seed,
        batch_size=int(_get_value(cfg_file, "batch_size", args.batch_size)),
        lr=float(_get_value(cfg_file, "lr", args.lr)),
        epsilon=float(_get_value(cfg_file, "epsilon", 1.0)),
        epsilon_min=float(_get_value(cfg_file, "epsilon_min", 0.05)),
        epsilon_decay_steps=int(_get_value(cfg_file, "epsilon_decay_steps", 50_000)),
        gamma=float(_get_value(cfg_file, "gamma", 0.99)),
        obs_clip=float(_get_value(cfg_file, "obs_clip", 20.0)),
        q_clip=float(_get_value(cfg_file, "q_clip", 1e4)),
        td_clip=float(_get_value(cfg_file, "td_clip", 1e3)),
        grad_clip_norm=float(_get_value(cfg_file, "grad_clip_norm", 10.0)),
        diagnostics_interval=int(_get_value(cfg_file, "diagnostics_interval", 100)),
        output_root=output_root,
        run_name=_resolve_run_name(configured_run_name, output_root, resume_target),
        checkpoint_interval=int(_get_value(cfg_file, "checkpoint_interval", 50)),
        save_best=bool(_get_value(cfg_file, "save_best", True)),
        resume_path=resume_target,
        device=str(_get_value(cfg_file, "device", args.device)),
        rnn_hidden_dim=int(_get_value(cfg_file, "rnn_hidden_dim", 64)),
        buffer_size=int(_get_value(cfg_file, "buffer_size", 5000)),
        target_update_freq=int(_get_value(cfg_file, "target_update_freq", 200)),
        qmix_hidden_dim=int(_get_value(cfg_file, "qmix_hidden_dim", 32)),
        hyper_hidden_dim=int(_get_value(cfg_file, "hyper_hidden_dim", 64)),
        hyper_layers_num=int(_get_value(cfg_file, "hyper_layers_num", 1)),
        use_self_attention=bool(_get_value(cfg_file, "use_self_attention", False)),
        self_attn_heads=int(_get_value(cfg_file, "self_attn_heads", 4)),
        self_attn_tokens=int(_get_value(cfg_file, "self_attn_tokens", 4)),
        use_cross_agent_attention=bool(_get_value(cfg_file, "use_cross_agent_attention", False)),
        cross_agent_attn_heads=int(_get_value(cfg_file, "cross_agent_attn_heads", 4)),
        use_mixing_attention=bool(_get_value(cfg_file, "use_mixing_attention", False)),
        train_interval=int(_get_value(cfg_file, "train_interval", 2)),
        early_stop_enabled=bool(_get_value(cfg_file, "early_stop_enabled", False)),
        early_stop_min_episodes=int(_get_value(cfg_file, "early_stop_min_episodes", 80)),
        early_stop_window=int(_get_value(cfg_file, "early_stop_window", 20)),
        early_stop_patience_windows=int(_get_value(cfg_file, "early_stop_patience_windows", 3)),
        early_stop_min_delta=float(_get_value(cfg_file, "early_stop_min_delta", 2.0)),
        early_stop_success_threshold=float(_get_value(cfg_file, "early_stop_success_threshold", 0.5)),
        early_stop_oob_threshold=float(_get_value(cfg_file, "early_stop_oob_threshold", 0.6)),
        early_stop_fail_oob_threshold=float(_get_value(cfg_file, "early_stop_fail_oob_threshold", 0.9)),
        early_stop_fail_patience_windows=int(_get_value(cfg_file, "early_stop_fail_patience_windows", 4)),
    )

    log_paths = setup_project_logging(qmix_cfg.output_root, qmix_cfg.run_name)

    _set_global_seed(seed)
    LOGGER.info("Global random seed set to %d", seed)

    env = GazeboMultiUAVParallelEnv(
        uav_names=["uav1", "uav2", "uav3", "uav4"],
        lidar_size=36,
        world_name=world_name,
        reset_timeout=reset_timeout,
        max_steps=max_steps,
        startup_grace_steps=startup_grace_steps,
        arena_x_limits=(float(arena_x_limits[0]), float(arena_x_limits[1])),
        arena_y_limits=(float(arena_y_limits[0]), float(arena_y_limits[1])),
        out_of_bounds_penalty=out_of_bounds_penalty,
        coverage_cell_size=coverage_cell_size,
        coverage_target_ratio=coverage_target_ratio,
        coverage_new_cell_reward=coverage_new_cell_reward,
        coverage_completion_bonus=coverage_completion_bonus,
        revisit_cell_penalty=revisit_cell_penalty,
        overlap_cell_penalty=overlap_cell_penalty,
        assignment_bonus=assignment_bonus,
        local_coverage_radius=local_coverage_radius,
        sensor_coverage_radius=sensor_coverage_radius,
        local_map_size=local_map_size,
        coverage_delta_scale=coverage_delta_scale,
        reset_position_tolerance=reset_position_tolerance,
        reset_stabilization_timeout=reset_stabilization_timeout,
        reset_poll_interval=reset_poll_interval,
        reset_initial_sensor_timeout=reset_initial_sensor_timeout,
        reset_initial_sensor_timeout_first=reset_initial_sensor_timeout_first,
        pre_reset_brake_wait=pre_reset_brake_wait,
        action_update_timeout=action_update_timeout,
        require_step_lidar_update=require_step_lidar_update,
        min_height_enforce_steps=min_height_enforce_steps,
        hard_floor_height=hard_floor_height,
        min_step_duration=min_step_duration,
        altitude_safety_threshold=altitude_safety_threshold,
        low_altitude_penalty=low_altitude_penalty,
        collision_penalty=collision_penalty,
        collision_terminal=collision_terminal,
        collision_cooldown_steps=collision_cooldown_steps,
        survival_bonus=survival_bonus,
        boundary_penalty_margin=boundary_penalty_margin,
        boundary_penalty_scale=boundary_penalty_scale,
    )

    LOGGER.info(
        "[TrainConfig] "
        f"episodes={episodes} max_steps={max_steps} seed={seed} "
        f"batch_size={qmix_cfg.batch_size} lr={qmix_cfg.lr} "
        f"device={qmix_cfg.device} "
        f"world_name={world_name} "
        f"reset_timeout={reset_timeout} "
        f"startup_grace_steps={startup_grace_steps} "
        f"arena_x_limits=({arena_x_limits[0]}, {arena_x_limits[1]}) "
        f"arena_y_limits=({arena_y_limits[0]}, {arena_y_limits[1]}) "
        f"out_of_bounds_penalty={out_of_bounds_penalty} "
        f"reset_position_tolerance={reset_position_tolerance} "
        f"reset_stabilization_timeout={reset_stabilization_timeout} "
        f"reset_poll_interval={reset_poll_interval} "
        f"min_height_enforce_steps={min_height_enforce_steps} "
        f"hard_floor_height={hard_floor_height} "
        f"coverage_cell_size={coverage_cell_size} "
        f"coverage_target_ratio={coverage_target_ratio} "
        f"coverage_new_cell_reward={coverage_new_cell_reward} "
        f"coverage_completion_bonus={coverage_completion_bonus} "
        f"revisit_cell_penalty={revisit_cell_penalty} "
        f"overlap_cell_penalty={overlap_cell_penalty} "
        f"assignment_bonus={assignment_bonus} "
        f"local_coverage_radius={local_coverage_radius} "
        f"sensor_coverage_radius={sensor_coverage_radius} "
        f"local_map_size={local_map_size} "
        f"coverage_delta_scale={coverage_delta_scale} "
        f"action_update_timeout={action_update_timeout} "
        f"require_step_lidar_update={require_step_lidar_update} "
        f"reset_initial_sensor_timeout={reset_initial_sensor_timeout} "
        f"reset_initial_sensor_timeout_first={reset_initial_sensor_timeout_first} "
        f"pre_reset_brake_wait={pre_reset_brake_wait} "
        f"run_name={qmix_cfg.run_name} output_root={qmix_cfg.output_root} "
        f"resume={qmix_cfg.resume_path}"
    )
    LOGGER.info(
        "Log files ready: runtime=%s error=%s",
        log_paths["runtime_log_path"],
        log_paths["error_log_path"],
    )
    train_attention_qmix(
        env,
        n_episodes=episodes,
        max_steps=max_steps,
        config=qmix_cfg,
        resume_path=qmix_cfg.resume_path,
    )

if __name__ == "__main__":
    main()