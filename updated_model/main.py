"""
main.py — Actor-Critic training for StarlingFlockEnv

Key fixes vs previous version:
  - Per-bird returns and advantages (not averaged across birds)
  - Tanh removed from actor head (saturates gradients)
  - Rewards kept as (N,) tensors throughout; no premature averaging
  - Critic trained against per-bird discounted returns
  - Entropy coef annealed to keep exploration alive early on
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from starling_env import StarlingFlockEnv


class ActorCritic(nn.Module):
    """
    Shared trunk → separate actor and critic heads.

    Actor  : outputs raw mean (no Tanh) — Tanh saturates gradients.
             Actions are clamped in the env step; sampling from N(mean, std)
             handles stochasticity without bounding the mean.
    Critic : scalar value per bird, used as per-bird advantage baseline.
    log_std: learnable, observation-independent, clamped to [-2, 0.5].
    """

    def __init__(self, obs_dim: int = 9, action_dim: int = 3, hidden: int = 256):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden // 2),
            nn.ELU(),
        )

        self.actor = nn.Linear(hidden // 2, action_dim)
        self.critic = nn.Linear(hidden // 2, 1)

        self.log_std = nn.Parameter(torch.full((action_dim,), 0.5))

        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        mean = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return mean, value

    def log_std_clamped(self) -> torch.Tensor:
        return self.log_std.clamp(-2.0, 0.5)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    num_birds: int = 150,
    episodes: int = 300,
    gamma: float = 0.99,
    lr: float = 3e-4,
    hidden: int = 256,
    value_coef: float = 0.5,
    entropy_coef: float = 0.05,
    entropy_min: float = 0.005,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    onnx_path: str = "starling_policy.onnx",
):
    device = torch.device(device_str)
    print(f"Training on: {device}")

    env = StarlingFlockEnv(num_birds=num_birds, device=device)
    model = ActorCritic(obs_dim=9, action_dim=3, hidden=hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=episodes, eta_min=lr * 0.1
    )

    N = num_birds

    for episode in range(episodes):

        frac = episode / max(episodes - 1, 1)
        ent_coef_now = entropy_coef + frac * (entropy_min - entropy_coef)

        obs_dict, _ = env.reset()

        log_probs_buf: list[torch.Tensor] = []
        values_buf: list[torch.Tensor] = []
        entropy_buf: list[torch.Tensor] = []
        rewards_buf: list[torch.Tensor] = []

        done = False

        while not done:
            obs_t = torch.tensor(
                np.stack([obs_dict[a] for a in env.agents]),
                dtype=torch.float32,
                device=device,
            )

            mean, values = model(obs_t)
            std = model.log_std_clamped().exp()
            dist = torch.distributions.Normal(mean, std)

            actions_t = dist.sample()

            log_probs_buf.append(dist.log_prob(actions_t).sum(dim=-1))
            values_buf.append(values)
            entropy_buf.append(dist.entropy().sum(dim=-1))

            actions_np = actions_t.clamp(-1.0, 1.0).cpu().numpy()
            actions = {env.agents[i]: actions_np[i] for i in range(N)}

            obs_dict, rewards, _, truncs, _ = env.step(actions)

            r_tensor = torch.tensor(
                [rewards[a] for a in env.agents],
                dtype=torch.float32,
                device=device,
            )
            rewards_buf.append(r_tensor)

            done = all(truncs.values())

        T = len(rewards_buf)

        returns = torch.zeros(T, N, device=device)
        G = torch.zeros(N, device=device)
        for t in reversed(range(T)):
            G = rewards_buf[t] + gamma * G
            returns[t] = G

        returns = (returns - returns.mean(dim=0)) / (returns.std(dim=0) + 1e-8)

        log_probs_t = torch.stack(log_probs_buf)
        values_t = torch.stack(values_buf)
        entropy_t = torch.stack(entropy_buf)

        advantage = (returns - values_t).detach()

        actor_loss = -(log_probs_t * advantage).mean()
        critic_loss = F.mse_loss(values_t, returns)
        entropy_loss = -entropy_t.mean()

        loss = actor_loss + value_coef * critic_loss + ent_coef_now * entropy_loss

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        opt.step()
        scheduler.step()

        mean_reward = torch.stack(rewards_buf).mean().item()
        print(
            f"Ep {episode:4d} | "
            f"MeanR: {mean_reward:.4f} | "
            f"Actor: {actor_loss.item():7.4f} | "
            f"Critic: {critic_loss.item():6.4f} | "
            f"Std: {model.log_std_clamped().exp().mean().item():.3f} | "
            f"EntCoef: {ent_coef_now:.4f}"
        )

    class ActorOnly(nn.Module):
        def __init__(self, ac):
            super().__init__()
            self.ac = ac

        def forward(self, x):
            mean, _ = self.ac(x)
            return mean.clamp(-1.0, 1.0)

    actor_only = ActorOnly(model).eval().to(device)
    dummy = torch.randn(1, 9, device=device)

    torch.onnx.export(
        actor_only,
        dummy,
        onnx_path,
        opset_version=11,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
    )
    print(f"\nExported → {onnx_path}")
    return model


if __name__ == "__main__":
    train()
