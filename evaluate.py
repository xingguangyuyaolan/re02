#!/usr/bin/env python3
"""
Evaluation script for trained Attention-QMIX models.

Usage examples:
  # Use best model from a run directory
  python evaluate.py --run-dir artifacts/qmix/maze_qmix_seed0_20260409_111739

  # Specify a checkpoint file directly
  python evaluate.py --model artifacts/qmix/maze_qmix_seed0_20260409_111739/best_model.pt

  # Set number of eval episodes and max steps per episode
  python evaluate.py --run-dir artifacts/qmix/... --episodes 10 --max-steps 300

  # Force CPU regardless of GPU availability
  python evaluate.py --run-dir artifacts/qmix/... --device cpu
"""

import argparse
from datetime import datetime
import json
import logging
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(__file__))

from src.scripts.attention_qmix import (
    QMIXConfig,
    QMIXForUAV,
    _apply_collision_shield,
    _init_agent_task_stats,
    _obs_dict_to_matrix,
    _summarize_agent_task_stats,
    _update_agent_task_stats,
)
from src.scripts.gazebo_pettingzoo_env import GazeboMultiUAVParallelEnv
from src.scripts.logging_utils import setup_project_logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger(__name__)


def _prepare_eval_output_dir(args):
    if args.run_dir:
        base_dir = os.path.normpath(args.run_dir)
    else:
        base_dir = os.path.dirname(os.path.normpath(args.model))
    eval_dir = os.path.join(base_dir, "evaluation", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(eval_dir, exist_ok=True)
    return eval_dir


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def _render_eval_plots(output_dir, episode_rows):
    if not episode_rows:
        return []

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        LOGGER.warning("Evaluation plot rendering skipped because matplotlib is unavailable: %s", exc)
        return []

    episodes = np.asarray([row["episode"] for row in episode_rows], dtype=np.int32)
    generated_paths = []

    def _save_plot(filename, specs, title):
        fig, axes = plt.subplots(len(specs), 1, figsize=(10, 3.2 * len(specs)), sharex=True)
        if len(specs) == 1:
            axes = [axes]

        for axis, spec in zip(axes, specs):
            values = [row.get(spec["key"]) for row in episode_rows]
            numeric = [np.nan if value is None else float(value) for value in values]
            arr = np.asarray(numeric, dtype=np.float32)
            axis.plot(episodes, arr, marker="o", linewidth=1.8, color=spec["color"])
            axis.set_ylabel(spec["label"])
            axis.set_title(spec["title"])
            axis.grid(True, alpha=0.25)

        axes[-1].set_xlabel("Episode")
        fig.suptitle(title, fontsize=13, y=0.98)
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
        output_path = os.path.join(output_dir, filename)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        generated_paths.append(output_path)

    _save_plot(
        "eval_performance.png",
        [
            {"key": "reward", "label": "Reward", "title": "Evaluation Reward", "color": "#1f77b4"},
            {"key": "steps", "label": "Steps", "title": "Evaluation Episode Steps", "color": "#9467bd"},
            {"key": "coverage_rate", "label": "Coverage", "title": "Coverage Rate", "color": "#2ca02c"},
            {"key": "full_coverage_success", "label": "Full Coverage", "title": "Full-Coverage Success", "color": "#17a589"},
        ],
        "Evaluation Performance",
    )
    _save_plot(
        "eval_task_metrics.png",
        [
            {"key": "collision_rate", "label": "Rate", "title": "Collision Rate", "color": "#ff7f0e"},
            {"key": "collision_step_rate", "label": "Rate", "title": "Collision Step Rate Trend", "color": "#d95f02"},
            {"key": "out_of_bounds_rate", "label": "Rate", "title": "Out-of-Bounds Rate", "color": "#8c564b"},
            {"key": "repeated_coverage_rate", "label": "Rate", "title": "Repeated Coverage Rate", "color": "#17becf"},
            {"key": "overlap_rate", "label": "Rate", "title": "Agent Overlap Rate", "color": "#bcbd22"},
            {"key": "coverage_completion_time", "label": "Steps", "title": "Coverage Completion Time", "color": "#9467bd"},
        ],
        "Evaluation Task Metrics (Including Collision Step Rate Trend)",
    )
    return generated_paths


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained Attention-QMIX model")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run-dir",
        type=str,
        help="Path to a training run directory (e.g. artifacts/qmix/maze_qmix_seed0_...). "
             "Loads best_model.pt if present, otherwise final_model.pt.",
    )
    group.add_argument(
        "--model",
        type=str,
        help="Direct path to a .pt checkpoint file.",
    )

    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes (default: 5)")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode (default: 500)")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device (default: auto)",
    )
    parser.add_argument(
        "--world", type=str, default=None,
        help="Gazebo world SDF file name (default: use saved env config)",
    )
    return parser.parse_args()


def _resolve_model_path(args):
    if args.model:
        path = args.model
        config_path = os.path.join(os.path.dirname(path), "config.json")
    else:
        run_dir = args.run_dir
        best = os.path.join(run_dir, "best_model.pt")
        final = os.path.join(run_dir, "final_model.pt")
        path = best if os.path.exists(best) else final
        config_path = os.path.join(run_dir, "config.json")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    return path, config_path


def _load_run_config(config_path):
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _cfg_get(*sources, key, default):
    """Get key from multiple dict-like sources with graceful fallback."""
    for source in sources:
        if isinstance(source, dict) and key in source and source[key] is not None:
            return source[key]
    return default


def main():
    args = _parse_args()
    model_path, config_path = _resolve_model_path(args)
    run_config = _load_run_config(config_path)
    eval_output_dir = _prepare_eval_output_dir(args)

    LOGGER.info("Model checkpoint : %s", model_path)
    LOGGER.info("Run config        : %s", config_path)
    LOGGER.info("Evaluation output : %s", eval_output_dir)

    # --- Reconstruct trainer from saved config ---
    if run_config is not None:
        n_agents = int(run_config["n_agents"])
        obs_dim = int(run_config["obs_dim"])
        state_dim = int(run_config["state_dim"])
        action_table = run_config["action_table"]
        saved_cfg = run_config.get("config", {})
        saved_env_cfg = run_config.get("env_config", {})
        saved_train_cfg = run_config.get("config", {})
    else:
        LOGGER.warning("config.json not found; using default architecture parameters")
        n_agents, obs_dim, state_dim = 4, 110, 440
        action_table = [
            [0.0, 0.0, 0.0, 0.0],
            [0.8, 0.0, 0.0, 0.0],
            [-0.6, 0.0, 0.0, 0.0],
            [0.0, 0.8, 0.0, 0.0],
            [0.0, -0.8, 0.0, 0.0],
            [0.0, 0.0, 0.4, 0.0],
            [0.0, 0.0, -0.3, 0.0],
            [0.6, 0.6, 0.0, 0.0],
            [0.6, -0.6, 0.0, 0.0],
            [-0.6, 0.6, 0.0, 0.0],
            [-0.6, -0.6, 0.0, 0.0],
        ]
        saved_cfg = {}
        saved_env_cfg = {}
        saved_train_cfg = {}

    cfg = QMIXConfig(
        rnn_hidden_dim=int(saved_cfg.get("rnn_hidden_dim", 64)),
        qmix_hidden_dim=int(saved_cfg.get("qmix_hidden_dim", 32)),
        hyper_hidden_dim=int(saved_cfg.get("hyper_hidden_dim", 64)),
        hyper_layers_num=int(saved_cfg.get("hyper_layers_num", 1)),
        use_orthogonal_init=bool(saved_cfg.get("use_orthogonal_init", True)),
        add_last_action=bool(saved_cfg.get("add_last_action", True)),
        add_agent_id=bool(saved_cfg.get("add_agent_id", True)),
        batch_size=int(saved_cfg.get("batch_size", 32)),
        use_self_attention=bool(saved_cfg.get("use_self_attention", False)),
        self_attn_heads=int(saved_cfg.get("self_attn_heads", 4)),
        self_attn_tokens=int(saved_cfg.get("self_attn_tokens", 4)),
        use_cross_agent_attention=bool(saved_cfg.get("use_cross_agent_attention", False)),
        cross_agent_attn_heads=int(saved_cfg.get("cross_agent_attn_heads", 4)),
        use_mixing_attention=bool(saved_cfg.get("use_mixing_attention", False)),
        use_vdn=bool(saved_cfg.get("use_vdn", False)),
        safety_shield_enabled=bool(saved_cfg.get("safety_shield_enabled", True)),
        safety_shield_horizon_sec=float(saved_cfg.get("safety_shield_horizon_sec", 0.6)),
        safety_shield_danger_dist=float(saved_cfg.get("safety_shield_danger_dist", 0.8)),
        safety_shield_emergency_dist=float(saved_cfg.get("safety_shield_emergency_dist", 0.45)),
        safety_shield_lidar_dist=float(saved_cfg.get("safety_shield_lidar_dist", 0.45)),
        safety_shield_boundary_margin=float(saved_cfg.get("safety_shield_boundary_margin", 1.2)),
        # Force epsilon=0 during evaluation (pure greedy, no exploration)
        epsilon=0.0,
        epsilon_min=0.0,
        epsilon_decay_steps=1,
        device=args.device,
    )

    cfg.max_episode_steps = args.max_steps
    trainer = QMIXForUAV(n_agents, obs_dim, state_dim, action_table, cfg)
    meta = trainer.load_checkpoint(model_path)
    # Override epsilon to 0 after loading (checkpoint may have stored a >0 value)
    trainer.epsilon = 0.0
    LOGGER.info(
        "Loaded checkpoint: episode=%d total_steps=%d best_reward=%s device=%s",
        meta["episode"], meta["total_steps"], meta["best_reward"], trainer.device,
    )

    # --- Build environment ---
    world_name = str(args.world if args.world is not None else saved_env_cfg.get("world_name", "maze_1.sdf"))
    LOGGER.info("Coverage evaluation | World: %s", world_name)

    env = GazeboMultiUAVParallelEnv(
        uav_names=["uav1", "uav2", "uav3", "uav4"],
        lidar_size=int(_cfg_get(saved_env_cfg, saved_train_cfg, key="lidar_size", default=36)),
        world_name=world_name,
        max_steps=args.max_steps,
        reset_timeout=float(_cfg_get(saved_env_cfg, saved_train_cfg, key="reset_timeout", default=2.0)),
        startup_grace_steps=int(_cfg_get(saved_env_cfg, saved_train_cfg, key="startup_grace_steps", default=8)),
        arena_x_limits=tuple(_cfg_get(saved_env_cfg, saved_train_cfg, key="arena_x_limits", default=[-10.0, 10.0])),
        arena_y_limits=tuple(_cfg_get(saved_env_cfg, saved_train_cfg, key="arena_y_limits", default=[-10.0, 10.0])),
        out_of_bounds_penalty=float(_cfg_get(saved_env_cfg, saved_train_cfg, key="out_of_bounds_penalty", default=2.0)),
        coverage_cell_size=float(_cfg_get(saved_env_cfg, saved_train_cfg, key="coverage_cell_size", default=1.0)),
        coverage_target_ratio=float(_cfg_get(saved_env_cfg, saved_train_cfg, key="coverage_target_ratio", default=0.85)),
        coverage_new_cell_reward=float(_cfg_get(saved_env_cfg, saved_train_cfg, key="coverage_new_cell_reward", default=1.2)),
        coverage_completion_bonus=float(_cfg_get(saved_env_cfg, saved_train_cfg, key="coverage_completion_bonus", default=8.0)),
        revisit_cell_penalty=float(_cfg_get(saved_env_cfg, saved_train_cfg, key="revisit_cell_penalty", default=0.08)),
        overlap_cell_penalty=float(_cfg_get(saved_env_cfg, saved_train_cfg, key="overlap_cell_penalty", default=0.15)),
        assignment_bonus=float(_cfg_get(saved_env_cfg, saved_train_cfg, key="assignment_bonus", default=0.2)),
        local_coverage_radius=int(_cfg_get(saved_env_cfg, saved_train_cfg, key="local_coverage_radius", default=1)),
        sensor_coverage_radius=int(_cfg_get(saved_env_cfg, saved_train_cfg, key="sensor_coverage_radius", default=1)),
        local_map_size=int(_cfg_get(saved_env_cfg, saved_train_cfg, key="local_map_size", default=7)),
        coverage_delta_scale=float(_cfg_get(saved_env_cfg, saved_train_cfg, key="coverage_delta_scale", default=0.0)),
        reset_position_tolerance=float(_cfg_get(saved_env_cfg, saved_train_cfg, key="reset_position_tolerance", default=5.0)),
        reset_validation_retries=int(_cfg_get(saved_env_cfg, saved_train_cfg, key="reset_validation_retries", default=1)),
        reset_validation_allow_failure=bool(_cfg_get(saved_env_cfg, saved_train_cfg, key="reset_validation_allow_failure", default=False)),
        min_spawn_pair_clearance=float(_cfg_get(saved_env_cfg, saved_train_cfg, key="min_spawn_pair_clearance", default=0.28)),
        min_spawn_altitude=float(_cfg_get(saved_env_cfg, saved_train_cfg, key="min_spawn_altitude", default=0.14)),
        spawn_layout_validation_retries=int(_cfg_get(saved_env_cfg, saved_train_cfg, key="spawn_layout_validation_retries", default=2)),
        spawn_layout_allow_risky=bool(_cfg_get(saved_env_cfg, saved_train_cfg, key="spawn_layout_allow_risky", default=False)),
        reset_stabilization_timeout=float(_cfg_get(saved_env_cfg, saved_train_cfg, key="reset_stabilization_timeout", default=1.5)),
        reset_poll_interval=float(_cfg_get(saved_env_cfg, saved_train_cfg, key="reset_poll_interval", default=0.05)),
        reset_initial_sensor_timeout=float(_cfg_get(saved_env_cfg, saved_train_cfg, key="reset_initial_sensor_timeout", default=1.5)),
        reset_initial_sensor_timeout_first=float(_cfg_get(saved_env_cfg, saved_train_cfg, key="reset_initial_sensor_timeout_first", default=4.0)),
        pre_reset_brake_wait=float(_cfg_get(saved_env_cfg, saved_train_cfg, key="pre_reset_brake_wait", default=0.03)),
        action_update_timeout=float(_cfg_get(saved_env_cfg, saved_train_cfg, key="action_update_timeout", default=0.03)),
        require_step_lidar_update=bool(_cfg_get(saved_env_cfg, saved_train_cfg, key="require_step_lidar_update", default=False)),
        min_height_enforce_steps=int(_cfg_get(saved_env_cfg, saved_train_cfg, key="min_height_enforce_steps", default=60)),
        hard_floor_height=float(_cfg_get(saved_env_cfg, saved_train_cfg, key="hard_floor_height", default=0.08)),
        min_step_duration=float(_cfg_get(saved_env_cfg, saved_train_cfg, key="min_step_duration", default=0.05)),
        altitude_safety_threshold=float(_cfg_get(saved_env_cfg, saved_train_cfg, key="altitude_safety_threshold", default=0.4)),
        low_altitude_penalty=float(_cfg_get(saved_env_cfg, saved_train_cfg, key="low_altitude_penalty", default=0.3)),
        collision_penalty=float(_cfg_get(saved_env_cfg, saved_train_cfg, key="collision_penalty", default=1.5)),
        collision_terminal=bool(_cfg_get(saved_env_cfg, saved_train_cfg, key="collision_terminal", default=True)),
        collision_cooldown_steps=int(_cfg_get(saved_env_cfg, saved_train_cfg, key="collision_cooldown_steps", default=10)),
        survival_bonus=float(_cfg_get(saved_env_cfg, saved_train_cfg, key="survival_bonus", default=0.0)),
        boundary_penalty_margin=float(_cfg_get(saved_env_cfg, saved_train_cfg, key="boundary_penalty_margin", default=2.0)),
        boundary_penalty_scale=float(_cfg_get(saved_env_cfg, saved_train_cfg, key="boundary_penalty_scale", default=0.3)),
        step_penalty=float(_cfg_get(saved_env_cfg, saved_train_cfg, key="step_penalty", default=0.01)),
    )

    # --- Evaluation loop (epsilon=0, greedy) ---
    LOGGER.info("Starting evaluation: %d episodes, max %d steps each", args.episodes, args.max_steps)
    episode_rewards = []
    episode_successes = []
    episode_step_counts = []
    episode_collision_rates = []
    episode_collision_step_rates = []
    episode_out_of_bounds_rates = []
    episode_coverage_rates = []
    episode_repeat_rates = []
    episode_overlap_rates = []
    episode_completion_times = []
    episode_full_coverage_successes = []
    episode_rows = []
    episode_safety_overrides = []
    episode_safety_override_inter_agent = []
    episode_safety_override_lidar = []
    episode_safety_override_boundary = []

    def _save_results():
        """Save whatever results have been collected so far."""
        if not episode_rows:
            LOGGER.warning("No episodes completed — nothing to save.")
            return
        eval_summary = {
            "model_path": model_path,
            "config_path": config_path,
            "output_dir": eval_output_dir,
            "episodes_requested": int(args.episodes),
            "episodes_completed": len(episode_rows),
            "max_steps": int(args.max_steps),
            "world": world_name,
            "mean_reward": float(np.mean(episode_rewards)),
            "max_reward": float(np.max(episode_rewards)),
            "min_reward": float(np.min(episode_rewards)),
            "success_rate": float(np.mean(episode_successes)),
            "mean_steps": float(np.mean(episode_step_counts)),
            "coverage_rate_mean": float(np.mean(episode_coverage_rates)),
            "repeated_coverage_rate_mean": float(np.mean(episode_repeat_rates)),
            "overlap_rate_mean": float(np.mean(episode_overlap_rates)),
            "collision_rate_mean": float(np.mean(episode_collision_rates)),
            "collision_step_rate_mean": float(np.mean(episode_collision_step_rates)),
            "out_of_bounds_rate_mean": float(np.mean(episode_out_of_bounds_rates)),
            "coverage_completion_time_mean": None if not episode_completion_times else float(np.mean(episode_completion_times)),
            "full_coverage_success_rate": float(np.mean(episode_full_coverage_successes)),
            "safety_override_count_mean": float(np.mean(episode_safety_overrides)) if episode_safety_overrides else 0.0,
            "safety_override_inter_agent_count_mean": float(np.mean(episode_safety_override_inter_agent)) if episode_safety_override_inter_agent else 0.0,
            "safety_override_lidar_count_mean": float(np.mean(episode_safety_override_lidar)) if episode_safety_override_lidar else 0.0,
            "safety_override_boundary_count_mean": float(np.mean(episode_safety_override_boundary)) if episode_safety_override_boundary else 0.0,
        }
        _write_jsonl(os.path.join(eval_output_dir, "episode_metrics.jsonl"), episode_rows)
        _write_json(os.path.join(eval_output_dir, "summary.json"), eval_summary)
        generated_plots = _render_eval_plots(eval_output_dir, episode_rows)
        if generated_plots:
            LOGGER.info("Saved evaluation plots: %s", ", ".join(generated_plots))
        LOGGER.info("Results saved to %s (%d/%d episodes)", eval_output_dir, len(episode_rows), args.episodes)

    try:
        for ep in range(1, args.episodes + 1):
            obs = env.reset()
            agents = list(env.agents)
            obs_n = _obs_dict_to_matrix(obs, agents)
            last_onehot_a = np.zeros((n_agents, len(action_table)), dtype=np.float32)
            avail_a_n = np.ones((n_agents, len(action_table)), dtype=np.float32)
            agent_task_stats = _init_agent_task_stats(agents)
            ep_reward = 0.0
            safety_overrides = 0
            safety_override_inter_agent = 0
            safety_override_lidar = 0
            safety_override_boundary = 0
            trainer.eval_q.rnn_hidden = None  # Reset recurrent state at episode start

            for step in range(args.max_steps):
                a_n = trainer.choose_action(obs_n, last_onehot_a, avail_a_n, epsilon=0.0)
                if cfg.safety_shield_enabled:
                    a_n, override_count, override_by_type = _apply_collision_shield(
                        obs,
                        agents,
                        a_n,
                        action_table=trainer.action_table,
                        horizon_sec=cfg.safety_shield_horizon_sec,
                        danger_dist=cfg.safety_shield_danger_dist,
                        emergency_dist=cfg.safety_shield_emergency_dist,
                        lidar_danger_dist=cfg.safety_shield_lidar_dist,
                        boundary_margin=cfg.safety_shield_boundary_margin,
                        arena_x_limits=getattr(env, "arena_x_limits", None),
                        arena_y_limits=getattr(env, "arena_y_limits", None),
                    )
                    safety_overrides += int(override_count)
                    safety_override_inter_agent += int(override_by_type.get("inter_agent", 0))
                    safety_override_lidar += int(override_by_type.get("lidar", 0))
                    safety_override_boundary += int(override_by_type.get("boundary", 0))

                # Build last action one-hot
                new_onehot = np.zeros_like(last_onehot_a)
                for i, a in enumerate(a_n):
                    new_onehot[i, a] = 1.0
                last_onehot_a = new_onehot

                actions_cont = trainer.discrete_to_continuous(a_n)
                actions_dict = {agent: actions_cont[i] for i, agent in enumerate(agents)}

                obs, rewards, dones, infos = env.step(actions_dict)
                obs_n = _obs_dict_to_matrix(obs, agents)
                _update_agent_task_stats(agent_task_stats, infos, step + 1)

                ep_reward += float(np.mean([rewards[a] for a in agents]))
                done_all = bool(dones.get("__all__", False))

                if done_all:
                    break

            task_metrics = _summarize_agent_task_stats(agent_task_stats)
            episode_rewards.append(ep_reward)
            episode_successes.append(1.0 if task_metrics["full_coverage_success"] else 0.0)
            episode_step_counts.append(step + 1)
            episode_collision_rates.append(task_metrics["collision_rate"])
            episode_collision_step_rates.append(task_metrics["collision_step_rate"])
            episode_out_of_bounds_rates.append(task_metrics["out_of_bounds_rate"])
            episode_coverage_rates.append(task_metrics["coverage_rate"])
            episode_repeat_rates.append(task_metrics["repeated_coverage_rate"])
            episode_overlap_rates.append(task_metrics["overlap_rate"])
            if task_metrics["coverage_completion_time"] is not None:
                episode_completion_times.append(task_metrics["coverage_completion_time"])
            episode_full_coverage_successes.append(1.0 if task_metrics["full_coverage_success"] else 0.0)
            episode_safety_overrides.append(float(safety_overrides))
            episode_safety_override_inter_agent.append(float(safety_override_inter_agent))
            episode_safety_override_lidar.append(float(safety_override_lidar))
            episode_safety_override_boundary.append(float(safety_override_boundary))
            episode_rows.append(
                {
                    "episode": int(ep),
                    "reward": float(ep_reward),
                    "steps": int(step + 1),
                    "success": 1.0 if task_metrics["full_coverage_success"] else 0.0,
                    "collision_rate": float(task_metrics["collision_rate"]),
                    "collision_step_rate": float(task_metrics["collision_step_rate"]),
                    "out_of_bounds_rate": float(task_metrics["out_of_bounds_rate"]),
                    "coverage_rate": float(task_metrics["coverage_rate"]),
                    "repeated_coverage_rate": float(task_metrics["repeated_coverage_rate"]),
                    "overlap_rate": float(task_metrics["overlap_rate"]),
                    "coverage_completion_time": None if task_metrics["coverage_completion_time"] is None else float(task_metrics["coverage_completion_time"]),
                    "full_coverage_success": 1.0 if task_metrics["full_coverage_success"] else 0.0,
                    "safety_override_count": int(safety_overrides),
                    "safety_override_inter_agent_count": int(safety_override_inter_agent),
                    "safety_override_lidar_count": int(safety_override_lidar),
                    "safety_override_boundary_count": int(safety_override_boundary),
                }
            )

            LOGGER.info(
                "Episode %d/%d | Reward=%.3f | Steps=%d | Coverage=%.2f | FullCoverage=%s | Repeat=%.2f | Overlap=%.2f | CollisionRate=%.2f | OOBRate=%.2f | CompletionStep=%s | ShieldOverrides=%d",
                ep, args.episodes, ep_reward, step + 1,
                task_metrics["coverage_rate"],
                task_metrics["full_coverage_success"],
                task_metrics["repeated_coverage_rate"],
                task_metrics["overlap_rate"],
                task_metrics["collision_rate"],
                task_metrics["out_of_bounds_rate"],
                "None" if task_metrics["coverage_completion_time"] is None else str(task_metrics["coverage_completion_time"]),
                safety_overrides,
            )

    except (KeyboardInterrupt, Exception) as exc:
        if isinstance(exc, KeyboardInterrupt):
            LOGGER.warning("Evaluation interrupted by user after %d/%d episodes", len(episode_rows), args.episodes)
        else:
            LOGGER.error("Evaluation failed at episode %d: %s", len(episode_rows) + 1, exc, exc_info=True)
    finally:
        _save_results()

    # --- Summary ---
    if episode_rows:
        n_done = len(episode_rows)
        print("\n" + "=" * 50)
        print(f"Evaluation Summary ({n_done}/{args.episodes} episodes)")
        print("=" * 50)
        print(f"  Mean reward  : {np.mean(episode_rewards):.3f}")
        print(f"  Max  reward  : {np.max(episode_rewards):.3f}")
        print(f"  Min  reward  : {np.min(episode_rewards):.3f}")
        print(f"  Full coverage success : {np.mean(episode_successes) * 100:.1f}%  ({int(sum(episode_successes))}/{n_done})")
        print(f"  Mean steps   : {np.mean(episode_step_counts):.1f}")
        print(f"  Coverage rate        : {np.mean(episode_coverage_rates) * 100:.1f}%")
        print(f"  Repeat coverage rate : {np.mean(episode_repeat_rates) * 100:.1f}%")
        print(f"  Overlap rate         : {np.mean(episode_overlap_rates) * 100:.1f}%")
        print(f"  Collision rate        : {np.mean(episode_collision_rates) * 100:.1f}%")
        print(f"  Out-of-bounds rate    : {np.mean(episode_out_of_bounds_rates) * 100:.1f}%")
        print(f"  Coverage completion   : {np.mean(episode_completion_times):.1f} steps" if episode_completion_times else "  Coverage completion   : N/A")
        print(f"  Full-coverage success : {np.mean(episode_full_coverage_successes) * 100:.1f}%")
        print(f"  Safety overrides      : {np.mean(episode_safety_overrides):.1f} per episode")
        print(f"  Output dir            : {eval_output_dir}")
        print("=" * 50)
    else:
        print("\nNo episodes completed. No results to display.")

    env.close()


if __name__ == "__main__":
    main()
