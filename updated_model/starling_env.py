from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from pettingzoo import ParallelEnv
from gymnasium.spaces import Box


class StarlingFlockEnv(ParallelEnv):
    metadata = {"name": "starling_flock_v0"}

    def __init__(
        self,
        num_birds: int = 32,
        world_size: float = 20.0,
        max_speed: float = 1.0,
        neighbor_sigma: float = 3.0,
        max_steps: int = 500,
        target_nnd: float = 1.0,
        target_speed: float = 0.6,
        spawn_range: float | None = None,
        device: str | torch.device = "cpu",
    ):
        self.num_birds = num_birds
        self.world_size = world_size
        self.max_speed = max_speed
        self.sigma2 = neighbor_sigma**2
        self.max_steps = max_steps
        self.target_nnd = target_nnd
        self.target_speed = target_speed
        self.device = torch.device(device)

        self.agents = [f"bird_{i}" for i in range(num_birds)]
        self.possible_agents = self.agents[:]
        self.spawn_range = spawn_range if spawn_range is not None else world_size

        self.observation_spaces = {
            a: Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
            for a in self.agents
        }
        self.action_spaces = {
            a: Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
            for a in self.agents
        }

        self.pos: torch.Tensor = None
        self.vel: torch.Tensor = None
        self.step_count: int = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            torch.manual_seed(seed)

        self.step_count = 0
        dev = self.device
        N = self.num_birds

        half = self.world_size / 2.0
        lo = half - self.spawn_range / 2.0
        hi = half + self.spawn_range / 2.0

        self.pos = lo + torch.rand(N, 3, device=dev) * (hi - lo)
        self.vel = torch.rand(N, 3, device=dev) - 0.5

        w = self._compute_weights()
        return self._all_obs(w), {a: {} for a in self.agents}

    def step(self, actions: dict):
        self.step_count += 1
        dev = self.device

        if isinstance(actions, np.ndarray):
            act = torch.tensor(actions, dtype=torch.float32, device=dev)
        else:
            act = torch.tensor(
                np.stack([actions[a] for a in self.agents]),
                dtype=torch.float32,
                device=dev,
            )
        act = act.clamp(-1.0, 1.0)

        self.vel = self.vel + act * 0.05
        speeds = self.vel.norm(dim=1, keepdim=True)
        self.vel = torch.where(
            speeds > self.max_speed,
            self.vel / speeds * self.max_speed,
            self.vel,
        )

        self.pos = (self.pos + self.vel) % self.world_size

        w = self._compute_weights()
        obs = self._all_obs(w)
        rewards = self._all_rewards(w)

        truncated = self.step_count >= self.max_steps
        return (
            obs,
            rewards,
            {a: False for a in self.agents},
            {a: truncated for a in self.agents},
            {a: {} for a in self.agents},
        )

    def _compute_weights(self) -> torch.Tensor:
        diff = self.pos.unsqueeze(1) - self.pos.unsqueeze(0)
        diff = diff - self.world_size * torch.round(diff / self.world_size)
        dist2 = (diff**2).sum(dim=-1)

        w = torch.exp(-dist2 / (2.0 * self.sigma2))
        w.fill_diagonal_(0.0)
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return w

    def get_obs_tensor(self, weights: torch.Tensor | None = None) -> torch.Tensor:
        w = weights if weights is not None else self._compute_weights()
        rel_pos = self.pos.unsqueeze(0) - self.pos.unsqueeze(1)
        rel_pos = rel_pos - self.world_size * torch.round(rel_pos / self.world_size)
        avg_rel_pos = (w.unsqueeze(-1) * rel_pos).sum(dim=1)
        avg_vel = w @ self.vel
        return torch.cat([self.vel, avg_rel_pos, avg_vel], dim=1)

    def _all_obs(self, weights: torch.Tensor) -> dict:
        obs_np = self.get_obs_tensor(weights).cpu().numpy().astype(np.float32)
        return {self.agents[i]: obs_np[i] for i in range(self.num_birds)}

    def reward_tensor(self, weights: torch.Tensor | None = None) -> torch.Tensor:
        w = weights if weights is not None else self._compute_weights()

        # cohesion
        diff = self.pos.unsqueeze(1) - self.pos.unsqueeze(0)
        diff = diff - self.world_size * torch.round(diff / self.world_size)
        dists = diff.norm(dim=-1)
        mean_dist = (w * dists).sum(dim=1)
        cohesion = 1.0 / (1.0 + (mean_dist - self.target_nnd).abs())

        # speed
        speeds = self.vel.norm(dim=1)
        speed_r = 1.0 / (1.0 + (speeds - self.target_speed).abs())

        # polarity
        avg_nv = w @ self.vel
        self_norm = F.normalize(self.vel, dim=1)
        neigh_norm = F.normalize(avg_nv, dim=1)
        polarity = ((self_norm * neigh_norm).sum(dim=1) + 1.0) / 2.0

        return (cohesion + speed_r + polarity) / 3.0

    def _all_rewards(self, weights: torch.Tensor) -> dict:
        r = self.reward_tensor(weights).cpu().numpy()
        return {self.agents[i]: float(r[i]) for i in range(self.num_birds)}

    def compute_stats(self) -> dict[str, float]:
        with torch.no_grad():
            speeds = self.vel.norm(dim=1)
            mean_speed = speeds.mean().item()

            diff = self.pos.unsqueeze(1) - self.pos.unsqueeze(0)
            diff = diff - self.world_size * torch.round(diff / self.world_size)
            dist = diff.norm(dim=-1)
            dist.fill_diagonal_(float("inf"))
            mean_nnd = dist.min(dim=1).values.mean().item()

            flock_mean_vel = self.vel.mean(dim=0, keepdim=True)
            self_norm = F.normalize(self.vel, dim=1)
            flock_norm = F.normalize(flock_mean_vel, dim=1)
            global_polarity = (self_norm * flock_norm).sum(dim=1).mean().item()

            centroid = self.pos.mean(dim=0, keepdim=True)
            flock_radius = (self.pos - centroid).norm(dim=1).mean().item()

            w = self._compute_weights()
            mean_reward = self.reward_tensor(w).mean().item()

        return {
            "mean_speed": mean_speed,
            "mean_nnd": mean_nnd,
            "global_polarity": global_polarity,
            "flock_radius": flock_radius,
            "mean_reward": mean_reward,
        }

    @property
    def pos_np(self) -> np.ndarray:
        return self.pos.cpu().numpy()

    @property
    def vel_np(self) -> np.ndarray:
        return self.vel.cpu().numpy()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
