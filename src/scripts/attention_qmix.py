"""QMIX trainer adapted from the SMAC project for Gazebo multi-UAV path planning."""

from dataclasses import dataclass
import json
import logging
import os
import random
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LOGGER = logging.getLogger(__name__)


def _resolve_compute_device(device_name: str) -> torch.device:
    requested = (device_name or "auto").lower()

    def _cuda_probe_or_cpu() -> torch.device:
        if not torch.cuda.is_available():
            return torch.device("cpu")
        try:
            _ = torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except RuntimeError as exc:
            LOGGER.warning("[QMIX] CUDA runtime probe failed (%s), falling back to CPU.", exc)
            return torch.device("cpu")

    if requested == "auto":
        return _cuda_probe_or_cpu()
    if requested == "cuda":
        resolved = _cuda_probe_or_cpu()
        if resolved.type != "cuda":
            LOGGER.warning("[QMIX] CUDA requested but unavailable/incompatible, falling back to CPU.")
        return resolved
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device option: {device_name}")


def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0)
        elif "weight" in name:
            nn.init.orthogonal_(param, gain=gain)


class ObservationSelfAttention(nn.Module):
    """Self-attention on feature tokens from GRU hidden state.

    Splits hidden_dim into n_tokens groups and applies multi-head attention
    across tokens, helping the agent focus on key observation features
    (target position, obstacles, teammate coordinates).
    """

    def __init__(self, hidden_dim, n_heads=4, n_tokens=4):
        super().__init__()
        self.n_tokens = n_tokens
        self.token_dim = hidden_dim // n_tokens
        assert hidden_dim % n_tokens == 0
        assert self.token_dim % n_heads == 0
        self.attn = nn.MultiheadAttention(self.token_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        batch = x.size(0)
        tokens = x.view(batch, self.n_tokens, self.token_dim)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        return self.norm(x + attn_out.reshape(batch, -1))


class CrossAgentGAT(nn.Module):
    """Graph Attention Network for inter-agent communication.

    Each agent attends to all agents' hidden states to model cooperative
    relationships, producing enhanced representations for coordination.
    """

    def __init__(self, hidden_dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        assert hidden_dim % n_heads == 0
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, n_agents):
        hidden_dim = h.size(1)
        batch_size = h.size(0) // n_agents
        h_3d = h.view(batch_size, n_agents, hidden_dim)
        Q = self.W_q(h_3d).view(batch_size, n_agents, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(h_3d).view(batch_size, n_agents, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(h_3d).view(batch_size, n_agents, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, n_agents, hidden_dim)
        out = self.out_proj(out)
        return self.norm(h_3d + out).view(batch_size * n_agents, hidden_dim)


def _remap_grucell_to_gru(state_dict):
    """Remap GRUCell keys (rnn.weight_ih) to GRU keys (rnn.weight_ih_l0) for backward compat."""
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        for suffix in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            old = f"rnn.{suffix}"
            new = f"rnn.{suffix}_l0"
            if key == old:
                new_key = new
                break
        remapped[new_key] = value
    return remapped


class QNetworkRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim, use_orthogonal_init=True,
                 use_self_attention=False, self_attn_heads=4, self_attn_tokens=4):
        super().__init__()
        self.rnn_hidden = None
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.self_attn = (
            ObservationSelfAttention(hidden_dim, n_heads=self_attn_heads, n_tokens=self_attn_tokens)
            if use_self_attention else None
        )
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        if use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        h0 = self.rnn_hidden.unsqueeze(0) if self.rnn_hidden is not None else None
        out, h_n = self.rnn(x.unsqueeze(1), h0)
        self.rnn_hidden = h_n.squeeze(0)
        h = self.rnn_hidden
        if self.self_attn is not None:
            h = self.self_attn(h)
        return self.fc2(h)

    def forward_hidden(self, inputs):
        """Forward through fc1 -> GRU -> self-attention; return hidden before fc2."""
        x = F.relu(self.fc1(inputs))
        h0 = self.rnn_hidden.unsqueeze(0) if self.rnn_hidden is not None else None
        out, h_n = self.rnn(x.unsqueeze(1), h0)
        self.rnn_hidden = h_n.squeeze(0)
        h = self.rnn_hidden
        if self.self_attn is not None:
            h = self.self_attn(h)
        return h

    def forward_sequence(self, inputs_seq):
        """Process full sequence via fused CUDA GRU kernel.

        Args:
            inputs_seq: (batch, T, input_dim)
        Returns:
            h_all: (batch, T, hidden_dim) — hidden states after self-attention
        """
        x = F.relu(self.fc1(inputs_seq))
        h_all, _ = self.rnn(x)
        if self.self_attn is not None:
            B, T, H = h_all.shape
            h_all = self.self_attn(h_all.reshape(B * T, H)).reshape(B, T, H)
        return h_all

    def q_from_hidden(self, hidden):
        """Compute Q-values from hidden state (after optional cross-agent attention)."""
        return self.fc2(hidden)


class QMixNet(nn.Module):
    def __init__(self, n_agents, state_dim, batch_size, qmix_hidden_dim=32, hyper_hidden_dim=64, hyper_layers_num=1):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.qmix_hidden_dim = qmix_hidden_dim

        if hyper_layers_num == 2:
            self.hyper_w1 = nn.Sequential(
                nn.Linear(state_dim, hyper_hidden_dim),
                nn.ReLU(),
                nn.Linear(hyper_hidden_dim, n_agents * qmix_hidden_dim),
            )
            self.hyper_w2 = nn.Sequential(
                nn.Linear(state_dim, hyper_hidden_dim),
                nn.ReLU(),
                nn.Linear(hyper_hidden_dim, qmix_hidden_dim),
            )
        else:
            self.hyper_w1 = nn.Linear(state_dim, n_agents * qmix_hidden_dim)
            self.hyper_w2 = nn.Linear(state_dim, qmix_hidden_dim)

        self.hyper_b1 = nn.Linear(state_dim, qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, qmix_hidden_dim),
            nn.ReLU(),
            nn.Linear(qmix_hidden_dim, 1),
        )

    def forward(self, q, s):
        q = q.view(-1, 1, self.n_agents)
        s = s.reshape(-1, self.state_dim)

        w1 = torch.abs(self.hyper_w1(s)).view(-1, self.n_agents, self.qmix_hidden_dim)
        b1 = self.hyper_b1(s).view(-1, 1, self.qmix_hidden_dim)
        hidden = F.elu(torch.bmm(q, w1) + b1)

        w2 = torch.abs(self.hyper_w2(s)).view(-1, self.qmix_hidden_dim, 1)
        b2 = self.hyper_b2(s).view(-1, 1, 1)
        q_total = torch.bmm(hidden, w2) + b2
        return q_total.view(self.batch_size, -1, 1)


class StateAttentionQMixNet(nn.Module):
    """QMIX mixing network with state-attention for dynamic per-agent importance.

    Uses state information to compute attention weights that scale each agent's
    Q-value before standard QMIX mixing.  Critical agents receive higher weights.
    Monotonicity is preserved because softmax weights are strictly positive.
    """

    def __init__(self, n_agents, state_dim, batch_size, qmix_hidden_dim=32, hyper_hidden_dim=64, hyper_layers_num=1):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.qmix_hidden_dim = qmix_hidden_dim

        # State-attention: produces per-agent importance weights from global state
        self.state_attn = nn.Sequential(
            nn.Linear(state_dim, qmix_hidden_dim),
            nn.ReLU(),
            nn.Linear(qmix_hidden_dim, n_agents),
        )

        if hyper_layers_num == 2:
            self.hyper_w1 = nn.Sequential(
                nn.Linear(state_dim, hyper_hidden_dim),
                nn.ReLU(),
                nn.Linear(hyper_hidden_dim, n_agents * qmix_hidden_dim),
            )
            self.hyper_w2 = nn.Sequential(
                nn.Linear(state_dim, hyper_hidden_dim),
                nn.ReLU(),
                nn.Linear(hyper_hidden_dim, qmix_hidden_dim),
            )
        else:
            self.hyper_w1 = nn.Linear(state_dim, n_agents * qmix_hidden_dim)
            self.hyper_w2 = nn.Linear(state_dim, qmix_hidden_dim)

        self.hyper_b1 = nn.Linear(state_dim, qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, qmix_hidden_dim),
            nn.ReLU(),
            nn.Linear(qmix_hidden_dim, 1),
        )

    def forward(self, q, s):
        q = q.view(-1, 1, self.n_agents)
        s = s.reshape(-1, self.state_dim)

        # Dynamic per-agent importance: softmax > 0; *n_agents keeps mean ~= 1
        importance = F.softmax(self.state_attn(s), dim=-1) * self.n_agents
        q = q * importance.unsqueeze(1)

        w1 = torch.abs(self.hyper_w1(s)).view(-1, self.n_agents, self.qmix_hidden_dim)
        b1 = self.hyper_b1(s).view(-1, 1, self.qmix_hidden_dim)
        hidden = F.elu(torch.bmm(q, w1) + b1)

        w2 = torch.abs(self.hyper_w2(s)).view(-1, self.qmix_hidden_dim, 1)
        b2 = self.hyper_b2(s).view(-1, 1, 1)
        q_total = torch.bmm(hidden, w2) + b2
        return q_total.view(self.batch_size, -1, 1)


class EpisodeReplayBuffer:
    def __init__(self, n_agents, obs_dim, state_dim, action_dim, episode_limit, buffer_size, batch_size):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_limit = episode_limit
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.episode_num = 0
        self.current_size = 0

        self.buffer = {
            "obs_n": np.zeros([buffer_size, episode_limit + 1, n_agents, obs_dim], dtype=np.float32),
            "s": np.zeros([buffer_size, episode_limit + 1, state_dim], dtype=np.float32),
            "avail_a_n": np.ones([buffer_size, episode_limit + 1, n_agents, action_dim], dtype=np.float32),
            "last_onehot_a_n": np.zeros([buffer_size, episode_limit + 1, n_agents, action_dim], dtype=np.float32),
            "a_n": np.zeros([buffer_size, episode_limit, n_agents], dtype=np.int64),
            "r": np.zeros([buffer_size, episode_limit, 1], dtype=np.float32),
            "dw": np.zeros([buffer_size, episode_limit, 1], dtype=np.float32),
            "active": np.zeros([buffer_size, episode_limit, 1], dtype=np.float32),
        }
        self.episode_len = np.zeros(buffer_size, dtype=np.int32)

    def store_transition(self, episode_step, obs_n, s, avail_a_n, last_onehot_a_n, a_n, r, dw):
        idx = self.episode_num
        self.buffer["obs_n"][idx][episode_step] = obs_n
        self.buffer["s"][idx][episode_step] = s
        self.buffer["avail_a_n"][idx][episode_step] = avail_a_n
        self.buffer["last_onehot_a_n"][idx][episode_step + 1] = last_onehot_a_n
        self.buffer["a_n"][idx][episode_step] = a_n
        self.buffer["r"][idx][episode_step] = r
        self.buffer["dw"][idx][episode_step] = float(dw)
        self.buffer["active"][idx][episode_step] = 1.0

    def store_last_step(self, episode_step, obs_n, s, avail_a_n):
        idx = self.episode_num
        self.buffer["obs_n"][idx][episode_step] = obs_n
        self.buffer["s"][idx][episode_step] = s
        self.buffer["avail_a_n"][idx][episode_step] = avail_a_n
        self.episode_len[idx] = episode_step
        self.episode_num = (self.episode_num + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self):
        index = np.random.choice(self.current_size, size=self.batch_size, replace=False)
        max_episode_len = int(np.max(self.episode_len[index]))
        batch = {}
        for key in self.buffer:
            if key in ["obs_n", "s", "avail_a_n", "last_onehot_a_n"]:
                t = torch.from_numpy(self.buffer[key][index, : max_episode_len + 1])
            elif key == "a_n":
                t = torch.from_numpy(self.buffer[key][index, :max_episode_len]).long()
            else:
                t = torch.from_numpy(self.buffer[key][index, :max_episode_len])
            batch[key] = t.pin_memory()
        return batch, max_episode_len


@dataclass
class QMIXConfig:
    max_train_steps: int = 200_000
    max_episode_steps: int = 500
    evaluate_freq: int = 5000
    evaluate_times: int = 5
    epsilon: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay_steps: int = 50_000
    buffer_size: int = 5000
    batch_size: int = 32
    lr: float = 5e-4
    gamma: float = 0.99
    qmix_hidden_dim: int = 32
    hyper_hidden_dim: int = 64
    hyper_layers_num: int = 1
    rnn_hidden_dim: int = 64
    use_orthogonal_init: bool = True
    use_grad_clip: bool = True
    target_update_freq: int = 200
    add_last_action: bool = True
    add_agent_id: bool = True
    use_double_q: bool = True
    seed: int = 0
    obs_clip: float = 20.0
    q_clip: float = 1e4
    td_clip: float = 1e3
    grad_clip_norm: float = 10.0
    diagnostics_interval: int = 100
    output_root: str = "artifacts/qmix"
    run_name: str = "default"
    checkpoint_interval: int = 50
    save_best: bool = True
    resume_path: Optional[str] = None
    device: str = "auto"
    # Attention mechanisms
    use_self_attention: bool = False
    self_attn_heads: int = 4
    self_attn_tokens: int = 4
    use_cross_agent_attention: bool = False
    cross_agent_attn_heads: int = 4
    use_mixing_attention: bool = False
    train_interval: int = 2
    # Early stopping controls
    early_stop_enabled: bool = False
    early_stop_min_episodes: int = 80
    early_stop_window: int = 20
    early_stop_patience_windows: int = 3
    early_stop_min_delta: float = 2.0
    early_stop_success_threshold: float = 0.5
    early_stop_oob_threshold: float = 0.6
    early_stop_fail_oob_threshold: float = 0.9
    early_stop_fail_patience_windows: int = 4


class QMIXForUAV:
    def __init__(self, n_agents, obs_dim, state_dim, action_table, config: QMIXConfig):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_table = np.asarray(action_table, dtype=np.float32)
        self.action_dim = self.action_table.shape[0]
        self.cfg = config
        self.device = _resolve_compute_device(config.device)
        self.cfg.device = str(self.device)
        self.epsilon = config.epsilon
        self.epsilon_decay = (config.epsilon - config.epsilon_min) / max(config.epsilon_decay_steps, 1)
        self.train_step_count = 0

        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)

        self.input_dim = obs_dim
        if config.add_last_action:
            self.input_dim += self.action_dim
        if config.add_agent_id:
            self.input_dim += n_agents

        self.eval_q = QNetworkRNN(
            self.input_dim, config.rnn_hidden_dim, self.action_dim,
            config.use_orthogonal_init,
            use_self_attention=config.use_self_attention,
            self_attn_heads=config.self_attn_heads,
            self_attn_tokens=config.self_attn_tokens,
        ).to(self.device)
        self.target_q = QNetworkRNN(
            self.input_dim, config.rnn_hidden_dim, self.action_dim,
            config.use_orthogonal_init,
            use_self_attention=config.use_self_attention,
            self_attn_heads=config.self_attn_heads,
            self_attn_tokens=config.self_attn_tokens,
        ).to(self.device)
        self.target_q.load_state_dict(self.eval_q.state_dict())

        # Cross-agent graph attention
        if config.use_cross_agent_attention:
            self.cross_agent_attn = CrossAgentGAT(config.rnn_hidden_dim, config.cross_agent_attn_heads).to(self.device)
            self.target_cross_agent_attn = CrossAgentGAT(config.rnn_hidden_dim, config.cross_agent_attn_heads).to(self.device)
            self.target_cross_agent_attn.load_state_dict(self.cross_agent_attn.state_dict())
        else:
            self.cross_agent_attn = None
            self.target_cross_agent_attn = None

        # Mixing network (with or without state attention)
        MixClass = StateAttentionQMixNet if config.use_mixing_attention else QMixNet
        self.eval_mix = MixClass(n_agents, state_dim, config.batch_size, config.qmix_hidden_dim, config.hyper_hidden_dim, config.hyper_layers_num).to(self.device)
        self.target_mix = MixClass(n_agents, state_dim, config.batch_size, config.qmix_hidden_dim, config.hyper_hidden_dim, config.hyper_layers_num).to(self.device)
        self.target_mix.load_state_dict(self.eval_mix.state_dict())

        # Include cross-agent attention parameters in the optimizer so the
        # attention mechanism is actually trained.  To stabilise learning we
        # use stop-gradient on neighbouring agents' hidden states (see
        # _apply_cross_agent_attention_batched).
        self.eval_parameters = list(self.eval_q.parameters()) + list(self.eval_mix.parameters())
        if self.cross_agent_attn is not None:
            self.eval_parameters += list(self.cross_agent_attn.parameters())
        self.optimizer = torch.optim.Adam(self.eval_parameters, lr=config.lr)

        self.replay_buffer = EpisodeReplayBuffer(
            n_agents=n_agents,
            obs_dim=obs_dim,
            state_dim=state_dim,
            action_dim=self.action_dim,
            episode_limit=config.max_episode_steps,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
        )
        self.nan_skip_count = 0

    def build_checkpoint(self, episode, total_steps, best_reward=None):
        ckpt = {
            "episode": int(episode),
            "total_steps": int(total_steps),
            "epsilon": float(self.epsilon),
            "nan_skip_count": int(self.nan_skip_count),
            "train_step_count": int(self.train_step_count),
            "best_reward": None if best_reward is None else float(best_reward),
            "config": vars(self.cfg),
            "eval_q": self.eval_q.state_dict(),
            "target_q": self.target_q.state_dict(),
            "eval_mix": self.eval_mix.state_dict(),
            "target_mix": self.target_mix.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.cross_agent_attn is not None:
            ckpt["cross_agent_attn"] = self.cross_agent_attn.state_dict()
            ckpt["target_cross_agent_attn"] = self.target_cross_agent_attn.state_dict()
        return ckpt

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        # Remap old GRUCell keys to GRU keys for backward compatibility
        for net_key in ("eval_q", "target_q"):
            if net_key in checkpoint:
                checkpoint[net_key] = _remap_grucell_to_gru(checkpoint[net_key])
        # strict=False: gracefully handle added attention layers absent from older checkpoints
        missing = self.eval_q.load_state_dict(checkpoint["eval_q"], strict=False)
        if missing.missing_keys:
            LOGGER.warning("[QMIX] New layers in eval_q use random init: %s", missing.missing_keys)
        self.target_q.load_state_dict(checkpoint["target_q"], strict=False)
        missing = self.eval_mix.load_state_dict(checkpoint["eval_mix"], strict=False)
        if missing.missing_keys:
            LOGGER.warning("[QMIX] New layers in eval_mix use random init: %s", missing.missing_keys)
        self.target_mix.load_state_dict(checkpoint["target_mix"], strict=False)

        if self.cross_agent_attn is not None:
            if "cross_agent_attn" in checkpoint:
                self.cross_agent_attn.load_state_dict(checkpoint["cross_agent_attn"])
                self.target_cross_agent_attn.load_state_dict(checkpoint["target_cross_agent_attn"])
            else:
                LOGGER.warning("[QMIX] No cross-agent attention in checkpoint; using random init.")

        optimizer_state = checkpoint.get("optimizer")
        if optimizer_state is not None:
            try:
                self.optimizer.load_state_dict(optimizer_state)
            except (ValueError, KeyError):
                LOGGER.warning("[QMIX] Optimizer state mismatch (architecture changed); reinitializing.")

        self.epsilon = float(checkpoint.get("epsilon", self.epsilon))
        self.nan_skip_count = int(checkpoint.get("nan_skip_count", self.nan_skip_count))
        self.train_step_count = int(checkpoint.get("train_step_count", self.train_step_count))

        return {
            "episode": int(checkpoint.get("episode", 0)),
            "total_steps": int(checkpoint.get("total_steps", 0)),
            "best_reward": checkpoint.get("best_reward"),
        }

    def _sanitize_tensor(self, tensor, clip_abs=None):
        if clip_abs is None:
            return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        cleaned = torch.nan_to_num(tensor, nan=0.0, posinf=clip_abs, neginf=-clip_abs)
        return torch.clamp(cleaned, -clip_abs, clip_abs)

    def _apply_cross_attn_stable(self, h_all, gat, n_agents):
        """Apply cross-agent GAT with stop-gradient on other agents' states.

        For each agent *i* the query comes from agent *i*'s hidden state
        (with gradient), while keys/values from all other agents are
        detached.  This prevents the "moving target" instability while
        still allowing the attention weights and projections to learn.

        Args:
            h_all: (B*N, T, H) hidden states
            gat: CrossAgentGAT module
            n_agents: N
        Returns:
            h_out: (B*N, T, H) enhanced hidden states
        """
        BN, T, H = h_all.shape
        B = BN // n_agents
        # (B, N, T, H)
        h_4d = h_all.view(B, n_agents, T, H)
        results = []
        for t in range(T):
            h_t = h_4d[:, :, t, :]  # (B, N, H)
            h_flat = h_t.reshape(B * n_agents, H)
            # Detach other agents for stable gradients
            h_detached = h_t.detach().unsqueeze(1).expand(B, n_agents, n_agents, H)
            h_query = h_t.unsqueeze(2).expand(B, n_agents, n_agents, H)
            # Build input where for agent i: own state has grad, others detached
            mask = torch.eye(n_agents, device=h_all.device).bool().unsqueeze(0).unsqueeze(-1)
            h_mixed = torch.where(mask, h_query, h_detached)  # (B, N, N, H)
            # GAT expects (B*N, H) and reshapes internally using n_agents
            h_out_t = gat(h_flat, n_agents)  # (B*N, H)
            results.append(h_out_t)
        h_out = torch.stack(results, dim=1)  # (B*N, T, H)
        return h_out

    def choose_action(self, obs_n, last_onehot_a_n, avail_a_n, epsilon):
        with torch.no_grad():
            if np.random.uniform() < epsilon:
                return np.array([np.random.choice(np.nonzero(avail_a)[0]) for avail_a in avail_a_n], dtype=np.int64)

            obs_n_t = torch.tensor(obs_n, dtype=torch.float32, device=self.device)
            inputs = [obs_n_t]
            if self.cfg.add_last_action:
                inputs.append(torch.tensor(last_onehot_a_n, dtype=torch.float32, device=self.device))
            if self.cfg.add_agent_id:
                inputs.append(torch.eye(self.n_agents, device=self.device))
            inputs = torch.cat(inputs, dim=-1)

            # Keep acting hidden-state shape aligned with per-step agent batch.
            if self.eval_q.rnn_hidden is not None and self.eval_q.rnn_hidden.size(0) != inputs.size(0):
                self.eval_q.rnn_hidden = None

            # Forward through GRU + self-attention, get hidden, then apply cross-agent attention
            h = self.eval_q.forward_hidden(inputs)
            if self.cross_agent_attn is not None:
                h = self.cross_agent_attn(h, self.n_agents)
            q_value = self.eval_q.q_from_hidden(h)

            avail_a_t = torch.tensor(avail_a_n, dtype=torch.float32, device=self.device)
            q_value[avail_a_t == 0] = -float("inf")
            return q_value.argmax(dim=-1).cpu().numpy()

    def discrete_to_continuous(self, a_n):
        return self.action_table[np.asarray(a_n, dtype=np.int64)]

    def get_inputs(self, batch, max_episode_len):
        inputs = [batch["obs_n"]]
        if self.cfg.add_last_action:
            inputs.append(batch["last_onehot_a_n"])
        if self.cfg.add_agent_id:
            agent_id = torch.eye(self.n_agents, device=self.device).unsqueeze(0).unsqueeze(0)
            agent_id = agent_id.repeat(self.cfg.batch_size, max_episode_len + 1, 1, 1)
            inputs.append(agent_id)
        return torch.cat(inputs, dim=-1)

    def train_step(self, total_steps):
        if self.replay_buffer.current_size < self.cfg.batch_size:
            return None

        batch, max_episode_len = self.replay_buffer.sample()
        for key in batch:
            batch[key] = batch[key].to(self.device, non_blocking=True)
        self.train_step_count += 1

        batch["obs_n"] = self._sanitize_tensor(batch["obs_n"], self.cfg.obs_clip)
        batch["s"] = self._sanitize_tensor(batch["s"], self.cfg.obs_clip)
        batch["r"] = self._sanitize_tensor(batch["r"], self.cfg.td_clip)

        inputs = self.get_inputs(batch, max_episode_len)
        inputs = self._sanitize_tensor(inputs, self.cfg.obs_clip)

        # Preserve online acting hidden state; training uses its own B*n_agents rollout state.
        acting_hidden = self.eval_q.rnn_hidden
        self.eval_q.rnn_hidden = None
        self.target_q.rnn_hidden = None

        B = self.cfg.batch_size
        N = self.n_agents
        T = max_episode_len
        # inputs shape: (B, T+1, N, input_dim) -> permute to (B, N, T+1, input_dim) -> (B*N, T+1, input_dim)
        inputs_flat = inputs.permute(0, 2, 1, 3).contiguous().reshape(B * N, T + 1, -1)

        # --- Vectorized GRU: fused CUDA kernel over all timesteps ---
        # Both eval and target process ALL steps 0..T so GRU hidden state
        # propagates identically to the original step-by-step loop.
        eval_h_all = self.eval_q.forward_sequence(inputs_flat)  # (B*N, T+1, H)
        target_h_all_full = self.target_q.forward_sequence(inputs_flat)  # (B*N, T+1, H)
        target_h_all = target_h_all_full[:, 1:]  # take steps 1..T → (B*N, T, H)

        # --- Cross-agent attention (stable: stop-gradient on neighbours) ---
        if self.cross_agent_attn is not None:
            eval_h_all = self._apply_cross_attn_stable(eval_h_all, self.cross_agent_attn, N)
            with torch.no_grad():
                target_h_all = self._apply_cross_attn_stable(target_h_all, self.target_cross_agent_attn, N)

        # --- Vectorized Q-value computation ---
        q_evals_all = self.eval_q.q_from_hidden(eval_h_all)  # (B*N, T+1, action_dim)
        q_targets_all = self.target_q.q_from_hidden(target_h_all)  # (B*N, T, action_dim)

        # Reshape to (B, T, N, action_dim)
        A = q_evals_all.size(-1)
        q_evals = q_evals_all[:, :T].reshape(B, N, T, A).permute(0, 2, 1, 3)  # (B, T, N, A)
        q_targets = q_targets_all.reshape(B, N, T, A).permute(0, 2, 1, 3)  # (B, T, N, A)

        q_evals = self._sanitize_tensor(q_evals, self.cfg.q_clip)
        q_targets = self._sanitize_tensor(q_targets, self.cfg.q_clip)

        with torch.no_grad():
            if self.cfg.use_double_q:
                # Use eval Q-values at steps 1..T for action selection (clone to avoid in-place on graph tensor)
                q_evals_next = q_evals_all[:, 1:].reshape(B, N, T, A).permute(0, 2, 1, 3).clone()
                q_evals_next[batch["avail_a_n"][:, 1:] == 0] = -999999
                a_argmax = torch.argmax(q_evals_next, dim=-1, keepdim=True)
                q_targets = torch.gather(q_targets, dim=-1, index=a_argmax).squeeze(-1)
            else:
                q_targets = q_targets.clone()
                q_targets[batch["avail_a_n"][:, 1:] == 0] = -999999
                q_targets = q_targets.max(dim=-1)[0]

        q_evals = torch.gather(q_evals, dim=-1, index=batch["a_n"].unsqueeze(-1)).squeeze(-1)
        q_total_eval = self.eval_mix(q_evals, batch["s"][:, :-1])
        q_total_target = self.target_mix(q_targets, batch["s"][:, 1:])
        q_total_eval = self._sanitize_tensor(q_total_eval, self.cfg.q_clip)
        q_total_target = self._sanitize_tensor(q_total_target, self.cfg.q_clip)
        targets = batch["r"] + self.cfg.gamma * (1 - batch["dw"]) * q_total_target
        targets = self._sanitize_tensor(targets, self.cfg.td_clip)

        td_error = q_total_eval - targets.detach()
        td_error = self._sanitize_tensor(td_error, self.cfg.td_clip)
        masked_td_error = td_error * batch["active"]
        active_sum = torch.clamp(batch["active"].sum(), min=1.0)
        loss = (masked_td_error**2).sum() / active_sum

        if not torch.isfinite(loss):
            self.nan_skip_count += 1
            LOGGER.warning(
                "[QMIX][WARN] Non-finite loss detected; skip update "
                f"(step={total_steps}, skip_count={self.nan_skip_count}, "
                f"q_eval_max={q_total_eval.abs().max().item():.3e}, "
                f"q_target_max={q_total_target.abs().max().item():.3e}, "
                f"td_max={td_error.abs().max().item():.3e}, "
                f"reward_max={batch['r'].abs().max().item():.3e})"
            )
            self.eval_q.rnn_hidden = acting_hidden
            return None

        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.use_grad_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.cfg.grad_clip_norm)
            if not torch.isfinite(grad_norm):
                self.nan_skip_count += 1
                LOGGER.warning(
                    "[QMIX][WARN] Non-finite grad norm; skip optimizer step "
                    f"(step={total_steps}, skip_count={self.nan_skip_count}, grad_norm={grad_norm})"
                )
                self.optimizer.zero_grad(set_to_none=True)
                self.eval_q.rnn_hidden = acting_hidden
                return None
        self.optimizer.step()

        if self.cfg.diagnostics_interval > 0 and total_steps % self.cfg.diagnostics_interval == 0:
            LOGGER.info(
                "[QMIX][Diag] "
                f"step={total_steps} loss={loss.item():.3e} "
                f"q_eval_mean={q_total_eval.mean().item():.3e} "
                f"q_target_mean={q_total_target.mean().item():.3e} "
                f"td_abs_mean={td_error.abs().mean().item():.3e}"
            )

        if self.train_step_count % self.cfg.target_update_freq == 0:
            self.target_q.load_state_dict(self.eval_q.state_dict())
            self.target_mix.load_state_dict(self.eval_mix.state_dict())
            if self.cross_agent_attn is not None:
                self.target_cross_agent_attn.load_state_dict(self.cross_agent_attn.state_dict())

        # NOTE: epsilon decay moved to training loop (per env step, not per update)

        # Restore acting hidden state so action selection sequence is not corrupted by training.
        self.eval_q.rnn_hidden = acting_hidden
        return float(loss.item())

    def decay_epsilon(self):
        """Decay epsilon by one step. Call once per env step."""
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon - self.epsilon_decay)


def _flatten_agent_obs(agent_obs):
    pose = np.asarray(agent_obs["pose"], dtype=np.float32)
    lidar = np.asarray(agent_obs["lidar"], dtype=np.float32)
    coverage = np.asarray(agent_obs.get("coverage", np.zeros(0, dtype=np.float32)), dtype=np.float32)
    local_map = np.asarray(agent_obs.get("local_map", np.zeros(0, dtype=np.float32)), dtype=np.float32)
    other_agents = np.asarray(agent_obs.get("other_agents", np.zeros(0, dtype=np.float32)), dtype=np.float32)

    pose = np.nan_to_num(pose, nan=0.0, posinf=20.0, neginf=-20.0)
    # Normalize pose: xy by arena half-size, z by max_height, yaw by pi, velocities by max_vel
    pose[0] /= 10.0   # x: [-10, 10] -> [-1, 1]
    pose[1] /= 10.0   # y: [-10, 10] -> [-1, 1]
    pose[2] /= 3.0    # z: [0, 3] -> [0, 1]
    pose[3] /= np.pi  # yaw: [-pi, pi] -> [-1, 1]
    pose[4] /= 1.0    # vx: already ~[-1, 1]
    pose[5] /= 1.0    # vy
    pose[6] /= 0.6    # vz: normalized by max_z_vel
    pose[7] /= 1.2    # yaw_rate: normalized by max_yaw_rate
    pose = np.clip(pose, -2.0, 2.0)

    lidar = np.nan_to_num(lidar, nan=20.0, posinf=20.0, neginf=0.0)
    lidar = np.clip(lidar, 0.0, 20.0)
    lidar /= 20.0  # Normalize to [0, 1]

    coverage = np.nan_to_num(coverage, nan=0.0, posinf=1.0, neginf=0.0)
    coverage = np.clip(coverage, 0.0, 1.0)
    local_map = np.nan_to_num(local_map, nan=0.5, posinf=1.0, neginf=0.0)
    local_map = np.clip(local_map, 0.0, 1.0)
    other_agents = np.nan_to_num(other_agents, nan=0.0, posinf=1.0, neginf=-1.0)
    other_agents = np.clip(other_agents, -1.0, 1.0)
    return np.concatenate([pose, lidar, coverage, local_map, other_agents], axis=0)


def _obs_dict_to_matrix(obs_dict, agents):
    return np.stack([_flatten_agent_obs(obs_dict[a]) for a in agents], axis=0)


def _state_from_obs_matrix(obs_matrix):
    return obs_matrix.reshape(-1).astype(np.float32)


def _init_agent_task_stats(agents):
    return {
        "agents": {
            agent: {
                "collided": False,
                "out_of_bounds": False,
                "revisit_steps": 0,
                "overlap_steps": 0,
                "new_cells": 0,
                "active_steps": 0,
            }
            for agent in agents
        },
        "coverage_ratio": 0.0,
        "full_coverage": False,
        "coverage_completion_step": None,
    }


def _update_agent_task_stats(agent_task_stats, infos, step_index):
    for agent, stats in agent_task_stats["agents"].items():
        info = infos.get(agent, {})
        if info.get("collided", False):
            stats["collided"] = True
        if info.get("out_of_bounds", False):
            stats["out_of_bounds"] = True
        stats["revisit_steps"] += int(bool(info.get("revisit_cell", False)))
        stats["overlap_steps"] += int(bool(info.get("overlap_cell", False)))
        stats["new_cells"] += int(bool(info.get("newly_covered_cell", False)))
        stats["active_steps"] += 1
        agent_task_stats["coverage_ratio"] = float(info.get("coverage_ratio", agent_task_stats["coverage_ratio"]))
        if info.get("coverage_complete", False):
            agent_task_stats["full_coverage"] = True
            if agent_task_stats["coverage_completion_step"] is None:
                agent_task_stats["coverage_completion_step"] = int(step_index)


def _summarize_agent_task_stats(agent_task_stats):
    stats_list = list(agent_task_stats["agents"].values())
    n_agents = max(len(stats_list), 1)
    total_active_steps = max(sum(item["active_steps"] for item in stats_list), 1)
    total_revisit_steps = sum(item["revisit_steps"] for item in stats_list)
    total_overlap_steps = sum(item["overlap_steps"] for item in stats_list)
    mean_new_cells = float(np.mean([item["new_cells"] for item in stats_list])) if stats_list else 0.0
    return {
        "collision_rate": float(sum(1 for item in stats_list if item["collided"]) / n_agents),
        "out_of_bounds_rate": float(sum(1 for item in stats_list if item["out_of_bounds"]) / n_agents),
        "coverage_rate": float(agent_task_stats["coverage_ratio"]),
        "repeated_coverage_rate": float(total_revisit_steps / total_active_steps),
        "overlap_rate": float(total_overlap_steps / total_active_steps),
        "coverage_completion_time": agent_task_stats["coverage_completion_step"],
        "full_coverage_success": bool(agent_task_stats["full_coverage"]),
        "mean_new_cells_per_agent": mean_new_cells,
    }


def _prepare_run_dirs(cfg: QMIXConfig):
    run_dir = os.path.join(cfg.output_root, cfg.run_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    return {
        "run_dir": run_dir,
        "checkpoint_dir": checkpoint_dir,
        "plots_dir": plots_dir,
        "metrics_path": os.path.join(run_dir, "metrics.jsonl"),
        "summary_path": os.path.join(run_dir, "summary.json"),
        "config_path": os.path.join(run_dir, "config.json"),
        "latest_model_path": os.path.join(run_dir, "latest_model.pt"),
        "best_model_path": os.path.join(run_dir, "best_model.pt"),
        "final_model_path": os.path.join(run_dir, "final_model.pt"),
    }


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def _append_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            text = line.strip()
            if not text:
                continue
            try:
                records.append(json.loads(text))
            except json.JSONDecodeError:
                continue
    return records


def _save_checkpoint(path, trainer: QMIXForUAV, episode, total_steps, best_reward=None):
    torch.save(trainer.build_checkpoint(episode=episode, total_steps=total_steps, best_reward=best_reward), path)


def _moving_average(values, window_size):
    if window_size <= 1:
        return np.asarray(values, dtype=np.float32)
    result = np.full(len(values), np.nan, dtype=np.float32)
    for index in range(len(values)):
        start = max(0, index - window_size + 1)
        window = np.asarray(values[start : index + 1], dtype=np.float32)
        valid = window[np.isfinite(window)]
        if valid.size > 0:
            result[index] = float(np.mean(valid))
    return result


def _extract_metric_series(episode_metrics, key):
    episodes = []
    values = []
    for item in episode_metrics:
        if key not in item:
            continue
        raw_value = item.get(key)
        if raw_value is None:
            continue
        episodes.append(int(item.get("episode", len(episodes) + 1)))
        values.append(float(raw_value))
    return np.asarray(episodes, dtype=np.int32), np.asarray(values, dtype=np.float32)


def render_training_plots(run_dir, episode_metrics):
    if not episode_metrics:
        return []

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        LOGGER.warning("[QMIX] Plot rendering skipped because matplotlib is unavailable: %s", exc)
        return []

    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    generated_paths = []
    smoothing_window = max(1, min(20, len(episode_metrics) // 10 if len(episode_metrics) >= 10 else 1))

    def _plot_group(filename, specs, figure_title):
        fig, axes = plt.subplots(len(specs), 1, figsize=(10, 3.2 * len(specs)), sharex=True)
        if len(specs) == 1:
            axes = [axes]

        plotted_any = False
        for axis, spec in zip(axes, specs):
            episodes, values = _extract_metric_series(episode_metrics, spec["key"])
            if values.size == 0:
                axis.set_visible(False)
                continue

            plotted_any = True
            axis.plot(episodes, values, color=spec.get("color", "#1f77b4"), linewidth=1.2, alpha=0.45, label=spec.get("raw_label", "raw"))
            if values.size >= 2:
                smoothed = _moving_average(values, smoothing_window)
                axis.plot(episodes, smoothed, color=spec.get("color", "#1f77b4"), linewidth=2.0, label=f"moving avg ({smoothing_window})")
            axis.set_ylabel(spec["label"])
            axis.set_title(spec["title"])
            axis.grid(True, alpha=0.25)
            axis.legend(loc="best")

        if not plotted_any:
            plt.close(fig)
            return None

        axes[-1].set_xlabel("Episode")
        fig.suptitle(figure_title)
        fig.tight_layout()
        output_path = os.path.join(plots_dir, filename)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    learning_plot = _plot_group(
        "learning_curves.png",
        [
            {"key": "reward", "label": "Reward", "title": "Episode Reward", "color": "#1f77b4"},
            {"key": "loss", "label": "Loss", "title": "Training Loss", "color": "#d62728"},
            {"key": "epsilon", "label": "Epsilon", "title": "Exploration Rate", "color": "#2ca02c"},
            {"key": "episode_steps", "label": "Steps", "title": "Episode Length", "color": "#9467bd"},
        ],
        "Learning Curves",
    )
    if learning_plot:
        generated_paths.append(learning_plot)

    task_plot = _plot_group(
        "task_metrics.png",
        [
            {"key": "coverage_rate", "label": "Rate", "title": "Coverage Rate", "color": "#17becf"},
            {"key": "repeated_coverage_rate", "label": "Rate", "title": "Repeated Coverage Rate", "color": "#bcbd22"},
            {"key": "overlap_rate", "label": "Rate", "title": "Agent Overlap Rate", "color": "#1f78b4"},
            {"key": "collision_rate", "label": "Rate", "title": "Collision Rate", "color": "#ff7f0e"},
            {"key": "out_of_bounds_rate", "label": "Rate", "title": "Out-of-Bounds Rate", "color": "#8c564b"},
            {"key": "coverage_completion_time", "label": "Steps", "title": "Coverage Completion Time", "color": "#9467bd"},
            {"key": "full_coverage_success", "label": "Success", "title": "Full Coverage Success", "color": "#2ca02c"},
        ],
        "Task Metrics",
    )
    if task_plot:
        generated_paths.append(task_plot)

    benchmark_plot = _plot_group(
        "benchmark_metrics.png",
        [
            {"key": "benchmark_env_step_time_sec", "label": "Seconds", "title": "Environment Step Time", "color": "#7f7f7f"},
            {"key": "benchmark_train_update_time_sec", "label": "Seconds", "title": "Train Update Time", "color": "#e377c2"},
            {"key": "benchmark_train_updates", "label": "Updates", "title": "Train Updates Per Episode", "color": "#1f78b4"},
        ],
        "Runtime Benchmarks",
    )
    if benchmark_plot:
        generated_paths.append(benchmark_plot)

    return generated_paths


def train_attention_qmix(env, n_episodes=1000, max_steps=500, config: Optional[QMIXConfig] = None, resume_path: Optional[str] = None):
    agents = list(env.agents)
    obs_dim = int(sum(np.prod(space.shape) for space in env.observation_spaces[agents[0]].spaces.values()))
    n_agents = len(agents)
    state_dim = n_agents * obs_dim

    action_table = [
        # Hover (stop)
        [0.0, 0.0, 0.0, 0.0],
        # Cardinal directions (main movement for coverage)
        [0.8, 0.0, 0.0, 0.0],    # forward x
        [-0.6, 0.0, 0.0, 0.0],   # backward x
        [0.0, 0.8, 0.0, 0.0],    # forward y
        [0.0, -0.8, 0.0, 0.0],   # backward y
        # Altitude control (minimal, coverage is 2D)
        [0.0, 0.0, 0.4, 0.0],    # ascend
        [0.0, 0.0, -0.3, 0.0],   # descend
        # Diagonal movement (useful for efficient coverage)
        [0.6, 0.6, 0.0, 0.0],    # diagonal NE
        [0.6, -0.6, 0.0, 0.0],   # diagonal SE
        [-0.6, 0.6, 0.0, 0.0],   # diagonal NW
        [-0.6, -0.6, 0.0, 0.0],  # diagonal SW
    ]

    cfg = config if config is not None else QMIXConfig(max_train_steps=n_episodes * max_steps, max_episode_steps=max_steps)
    cfg.max_train_steps = n_episodes * max_steps
    cfg.max_episode_steps = max_steps
    trainer = QMIXForUAV(n_agents, obs_dim, state_dim, action_table, cfg)
    LOGGER.info("[QMIX] Using compute device: %s", trainer.device)
    run_paths = _prepare_run_dirs(cfg)
    _write_json(
        run_paths["config_path"],
        {
            "n_agents": n_agents,
            "obs_dim": obs_dim,
            "state_dim": state_dim,
            "action_table": action_table,
            "config": vars(cfg),
            "env_config": env.describe_task() if hasattr(env, "describe_task") else {},
        },
    )
    LOGGER.info("[QMIX] Saving artifacts to %s", run_paths["run_dir"])

    resume_target = resume_path if resume_path is not None else cfg.resume_path
    if resume_target == "latest":
        resume_target = run_paths["latest_model_path"]

    total_steps = 0
    best_reward = None
    episode_metrics = []
    completed_episodes = 0
    start_episode = 0

    # Detect cross-stage curriculum transfer vs same-stage resume
    cross_stage = False
    if resume_target:
        from_run = None
        norm_path = os.path.normpath(resume_target)
        parent = os.path.dirname(norm_path)
        if os.path.basename(parent) == "checkpoints":
            from_run = os.path.basename(os.path.dirname(parent))
        else:
            from_run = os.path.basename(parent) or None
        if from_run and not from_run.startswith(cfg.run_name.rsplit("_", 2)[0] if "_" in cfg.run_name else cfg.run_name):
            cross_stage = True

    if resume_target:
        if not os.path.exists(resume_target):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_target}")
        resume_state = trainer.load_checkpoint(resume_target)
        if cross_stage:
            # Curriculum transfer: load network weights but reset training state
            trainer.epsilon = cfg.epsilon
            start_episode = 0
            total_steps = 0
            best_reward = None
            episode_metrics = []
            LOGGER.info(
                "[QMIX] Cross-stage curriculum transfer from %s "
                "(weights loaded, epsilon reset to %.3f, episode/steps reset to 0)",
                resume_target, trainer.epsilon,
            )
        else:
            # Same-stage resume: restore full training state
            episode_metrics = _read_jsonl(run_paths["metrics_path"])
            completed_episodes = int(episode_metrics[-1]["episode"]) if episode_metrics else 0
            start_episode = int(resume_state["episode"])
            completed_episodes = max(completed_episodes, start_episode)
            total_steps = int(resume_state["total_steps"])
            best_reward = resume_state["best_reward"]
            LOGGER.info(
                "[QMIX] Resumed from checkpoint "
                f"{resume_target} (episode={start_episode}, total_steps={total_steps}, epsilon={trainer.epsilon:.3f})"
            )

    interrupted = False
    pending_exception = None
    early_stopped = False
    early_stop_reason = None
    best_window_reward = -np.inf
    stagnant_windows = 0
    fail_windows = 0
    try:
        for episode in range(start_episode, n_episodes):
            obs = env.reset()
            agents = list(env.agents)
            obs_n = _obs_dict_to_matrix(obs, agents)
            s = _state_from_obs_matrix(obs_n)
            last_onehot_a_n = np.zeros((n_agents, len(action_table)), dtype=np.float32)
            agent_task_stats = _init_agent_task_stats(agents)
            episode_reward = 0.0
            last_loss = None
            env_step_time_sec = 0.0
            train_update_time_sec = 0.0
            train_updates = 0
            train_interval = cfg.train_interval
            steps_since_train = 0

            trainer.eval_q.rnn_hidden = None
            agent_alive = {agent: True for agent in agents}

            for episode_step in range(max_steps):
                avail_a_n = np.ones((n_agents, len(action_table)), dtype=np.float32)
                # Dead agents can only select the stop action (index 0).
                for i, agent in enumerate(agents):
                    if not agent_alive[agent]:
                        avail_a_n[i] = 0.0
                        avail_a_n[i, 0] = 1.0

                a_n = trainer.choose_action(obs_n, last_onehot_a_n, avail_a_n, trainer.epsilon)
                last_onehot_a_n = np.eye(len(action_table), dtype=np.float32)[a_n]

                continuous = trainer.discrete_to_continuous(a_n)
                action_dict = {agent: continuous[i] for i, agent in enumerate(agents)}

                env_t0 = time.perf_counter()
                next_obs, rewards, dones, infos = env.step(action_dict)
                env_step_time_sec += time.perf_counter() - env_t0
                next_obs_n = _obs_dict_to_matrix(next_obs, agents)
                next_s = _state_from_obs_matrix(next_obs_n)
                _update_agent_task_stats(agent_task_stats, infos, episode_step + 1)

                # Track per-agent death for subsequent steps.
                for agent in agents:
                    if dones.get(agent, False):
                        agent_alive[agent] = False

                # Team reward: weight by alive agents so dead agents' zero
                # reward does not dilute positive exploration signals.
                alive_list = [a for a in agents if agent_alive[a]]
                if alive_list:
                    team_reward = float(np.mean([rewards[a] for a in alive_list]))
                else:
                    team_reward = float(np.mean([rewards[a] for a in agents]))

                done_all = bool(dones.get("__all__", False))
                dw = done_all and (episode_step + 1 != max_steps)

                trainer.replay_buffer.store_transition(
                    episode_step,
                    obs_n,
                    s,
                    avail_a_n,
                    last_onehot_a_n,
                    a_n,
                    team_reward,
                    dw,
                )

                obs_n = next_obs_n
                s = next_s
                episode_reward += team_reward
                total_steps += 1
                steps_since_train += 1
                trainer.decay_epsilon()

                if trainer.replay_buffer.current_size >= cfg.batch_size and steps_since_train >= train_interval:
                    train_t0 = time.perf_counter()
                    last_loss = trainer.train_step(total_steps)
                    train_update_time_sec += time.perf_counter() - train_t0
                    train_updates += 1
                    steps_since_train = 0

                if done_all:
                    break

            trainer.replay_buffer.store_last_step(episode_step + 1, obs_n, s, np.ones((n_agents, len(action_table)), dtype=np.float32))
            task_metrics = _summarize_agent_task_stats(agent_task_stats)
            metrics = {
                "episode": int(episode + 1),
                "total_steps": int(total_steps),
                "episode_steps": int(episode_step + 1),
                "reward": float(episode_reward),
                "epsilon": float(trainer.epsilon),
                "loss": None if last_loss is None else float(last_loss),
                "nan_skip_count": int(trainer.nan_skip_count),
                "benchmark_env_step_time_sec": float(env_step_time_sec),
                "benchmark_train_update_time_sec": float(train_update_time_sec),
                "benchmark_train_updates": int(train_updates),
                **task_metrics,
            }
            _append_jsonl(run_paths["metrics_path"], metrics)
            episode_metrics.append(metrics)
            completed_episodes = episode + 1

            _save_checkpoint(run_paths["latest_model_path"], trainer, episode + 1, total_steps, best_reward)

            if cfg.save_best and (best_reward is None or episode_reward > best_reward):
                best_reward = float(episode_reward)
                _save_checkpoint(run_paths["best_model_path"], trainer, episode + 1, total_steps, best_reward)

            if cfg.checkpoint_interval > 0 and (episode + 1) % cfg.checkpoint_interval == 0:
                checkpoint_path = os.path.join(run_paths["checkpoint_dir"], f"episode_{episode + 1:04d}.pt")
                _save_checkpoint(checkpoint_path, trainer, episode + 1, total_steps, best_reward)
                render_training_plots(run_paths["run_dir"], episode_metrics)

            LOGGER.info(
                "Episode %d/%d | Reward=%.3f | Epsilon=%.3f | Loss=%s | Bench(env=%.3fs train=%.3fs updates=%d)",
                episode + 1,
                n_episodes,
                episode_reward,
                trainer.epsilon,
                str(last_loss),
                env_step_time_sec,
                train_update_time_sec,
                train_updates,
            )

            if cfg.early_stop_enabled and completed_episodes >= cfg.early_stop_min_episodes:
                window = max(1, int(cfg.early_stop_window))
                if len(episode_metrics) >= window:
                    recent = episode_metrics[-window:]
                    window_reward_mean = float(np.mean([m["reward"] for m in recent]))
                    window_success_rate = float(np.mean([1.0 if m.get("full_coverage_success", False) else 0.0 for m in recent]))
                    window_oob_rate = float(np.mean([m.get("out_of_bounds_rate", 1.0) for m in recent]))

                    if window_reward_mean > best_window_reward + float(cfg.early_stop_min_delta):
                        best_window_reward = window_reward_mean
                        stagnant_windows = 0
                    else:
                        stagnant_windows += 1

                    if window_oob_rate >= float(cfg.early_stop_fail_oob_threshold):
                        fail_windows += 1
                    else:
                        fail_windows = 0

                    converged = (
                        window_success_rate >= float(cfg.early_stop_success_threshold)
                        and window_oob_rate <= float(cfg.early_stop_oob_threshold)
                        and stagnant_windows >= int(cfg.early_stop_patience_windows)
                    )
                    failed = (
                        trainer.epsilon <= float(cfg.epsilon_min) + 1e-8
                        and fail_windows >= int(cfg.early_stop_fail_patience_windows)
                        and stagnant_windows >= int(cfg.early_stop_patience_windows)
                    )

                    if converged or failed:
                        early_stopped = True
                        early_stop_reason = {
                            "type": "converged" if converged else "failed_oob",
                            "window": window,
                            "window_reward_mean": window_reward_mean,
                            "window_success_rate": window_success_rate,
                            "window_oob_rate": window_oob_rate,
                            "stagnant_windows": int(stagnant_windows),
                            "fail_windows": int(fail_windows),
                        }
                        LOGGER.info(
                            "[QMIX] Early stopping triggered (%s): reward_mean=%.3f success=%.3f oob=%.3f stagnant=%d fail=%d",
                            early_stop_reason["type"],
                            window_reward_mean,
                            window_success_rate,
                            window_oob_rate,
                            stagnant_windows,
                            fail_windows,
                        )
                        break
    except BaseException as exc:
        interrupted = True
        pending_exception = exc
        LOGGER.exception(
            "[QMIX] Training interrupted by %s; saving latest artifacts.",
            type(exc).__name__,
        )
    finally:
        latest_episode = completed_episodes
        _save_checkpoint(run_paths["latest_model_path"], trainer, latest_episode, total_steps, best_reward)
        _save_checkpoint(run_paths["final_model_path"], trainer, latest_episode, total_steps, best_reward)
        rewards = [item["reward"] for item in episode_metrics]
        losses = [item["loss"] for item in episode_metrics if item["loss"] is not None]
        env_times = [item["benchmark_env_step_time_sec"] for item in episode_metrics if "benchmark_env_step_time_sec" in item]
        train_times = [item["benchmark_train_update_time_sec"] for item in episode_metrics if "benchmark_train_update_time_sec" in item]
        train_counts = [item["benchmark_train_updates"] for item in episode_metrics if "benchmark_train_updates" in item]
        coverage_rates = [item["coverage_rate"] for item in episode_metrics if "coverage_rate" in item]
        repeated_coverage_rates = [item["repeated_coverage_rate"] for item in episode_metrics if "repeated_coverage_rate" in item]
        overlap_rates = [item["overlap_rate"] for item in episode_metrics if "overlap_rate" in item]
        collision_rates = [item["collision_rate"] for item in episode_metrics if "collision_rate" in item]
        out_of_bounds_rates = [item["out_of_bounds_rate"] for item in episode_metrics if "out_of_bounds_rate" in item]
        coverage_completion_times = [item["coverage_completion_time"] for item in episode_metrics if item.get("coverage_completion_time") is not None]
        full_coverage_successes = [1.0 if item["full_coverage_success"] else 0.0 for item in episode_metrics if "full_coverage_success" in item]
        _write_json(
            run_paths["summary_path"],
            {
                "requested_episodes": int(n_episodes),
                "completed_episodes": int(completed_episodes),
                "total_steps": int(total_steps),
                "resumed": bool(resume_target),
                "resume_path": resume_target,
                "start_episode": int(start_episode),
                "interrupted": interrupted,
                "early_stopped": bool(early_stopped),
                "early_stop_reason": early_stop_reason,
                "interrupt_type": None if pending_exception is None else type(pending_exception).__name__,
                "best_reward": None if best_reward is None else float(best_reward),
                "final_epsilon": float(trainer.epsilon),
                "nan_skip_count": int(trainer.nan_skip_count),
                "reward_mean": None if not rewards else float(np.mean(rewards)),
                "reward_min": None if not rewards else float(np.min(rewards)),
                "reward_max": None if not rewards else float(np.max(rewards)),
                "loss_mean": None if not losses else float(np.mean(losses)),
                "loss_min": None if not losses else float(np.min(losses)),
                "loss_max": None if not losses else float(np.max(losses)),
                "benchmark_env_step_time_mean_sec": None if not env_times else float(np.mean(env_times)),
                "benchmark_train_update_time_mean_sec": None if not train_times else float(np.mean(train_times)),
                "benchmark_train_updates_mean": None if not train_counts else float(np.mean(train_counts)),
                "coverage_rate_mean": None if not coverage_rates else float(np.mean(coverage_rates)),
                "repeated_coverage_rate_mean": None if not repeated_coverage_rates else float(np.mean(repeated_coverage_rates)),
                "overlap_rate_mean": None if not overlap_rates else float(np.mean(overlap_rates)),
                "collision_rate_mean": None if not collision_rates else float(np.mean(collision_rates)),
                "out_of_bounds_rate_mean": None if not out_of_bounds_rates else float(np.mean(out_of_bounds_rates)),
                "coverage_completion_time_mean": None if not coverage_completion_times else float(np.mean(coverage_completion_times)),
                "full_coverage_success_rate": None if not full_coverage_successes else float(np.mean(full_coverage_successes)),
            },
        )
        generated_plots = render_training_plots(run_paths["run_dir"], episode_metrics)
        if generated_plots:
            LOGGER.info("[QMIX] Saved plots: %s", ", ".join(generated_plots))

    if pending_exception is not None:
        raise pending_exception


if __name__ == "__main__":
    from src.scripts.gazebo_pettingzoo_env import GazeboMultiUAVParallelEnv

    env = GazeboMultiUAVParallelEnv()
    train_attention_qmix(env)