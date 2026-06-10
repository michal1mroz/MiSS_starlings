from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from starling_env import StarlingFlockEnv

try:
    import onnxruntime
except ImportError:  # pragma: no cover
    onnxruntime = None


class ActorCritic(nn.Module):
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


def _load_onnx_session(onnx_path: str):
    if onnxruntime is None:
        raise ImportError(
            "onnxruntime is required to run test mode. Install it with 'pip install onnxruntime'"
        )

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, input_name, output_name


def test(
    onnx_path: str = "starling_policy.onnx",
    num_birds: int = 150,
    warmup_steps: int = 100,
    record_steps: int = 50,
    device_str: str = "cpu",
):
    session, input_name, output_name = _load_onnx_session(onnx_path)
    env = StarlingFlockEnv(num_birds=num_birds, device=device_str)
    output_csv = f"recorded_positions_{num_birds}birds.csv"


    obs_dict, _ = env.reset()
    num_agents = len(env.agents)

    print(f"Testing ONNX model: {onnx_path}")
    print(f"Warmup steps: {warmup_steps}, Record steps: {record_steps}")

    for step in range(warmup_steps):
        obs_np = np.stack([obs_dict[a] for a in env.agents]).astype(np.float32)
        action = session.run([output_name], {input_name: obs_np})[0]
        action = np.clip(action, -1.0, 1.0)
        obs_dict, _, _, truncs, _ = env.step(action)
        if all(truncs.values()):
            print(f"Environment truncated during warmup at step {step + 1}")
            break

    record_steps = min(record_steps, env.max_steps - env.step_count)
    if record_steps <= 0:
        raise RuntimeError(
            "No recording steps available after warmup. Reduce warmup_steps or increase env.max_steps."
        )

    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["tick", "bird_id", "x", "y", "z"])

        for tick in range(1, record_steps + 1):
            obs_np = np.stack([obs_dict[a] for a in env.agents]).astype(np.float32)
            action = session.run([output_name], {input_name: obs_np})[0]
            action = np.clip(action, -1.0, 1.0)
            obs_dict, _, _, truncs, _ = env.step(action)

            positions = env.pos_np
            for bird_index, pos in enumerate(positions):
                writer.writerow([tick, bird_index, float(pos[0]), float(pos[1]), float(pos[2])])

            if all(truncs.values()):
                print(f"Environment truncated during record phase at tick {tick}")
                break

    print(f"Recorded positions saved to {output_csv}")
    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or test a starling policy.")
    subparsers = parser.add_subparsers(dest="command", required=False)

    train_parser = subparsers.add_parser("train", help="Train a new policy and export ONNX.")
    train_parser.add_argument("--num_birds", type=int, default=150)
    train_parser.add_argument("--episodes", type=int, default=300)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--hidden", type=int, default=256)
    train_parser.add_argument("--onnx_path", type=str, default="starling_policy.onnx")
    train_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    test_parser = subparsers.add_parser("test", help="Run a saved ONNX policy and record bird positions.")
    test_parser.add_argument("--onnx_path", type=str, default="starling_policy.onnx")
    test_parser.add_argument("--num_birds", type=int, default=150)
    test_parser.add_argument("--warmup_steps", type=int, default=150)
    test_parser.add_argument("--record_steps", type=int, default=50)
    test_parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.command == "test":
        test(
            onnx_path=args.onnx_path,
            num_birds=args.num_birds,
            warmup_steps=args.warmup_steps,
            record_steps=args.record_steps,
            device_str=args.device,
        )
    else:
        train(
            num_birds=args.num_birds,
            episodes=args.episodes,
            lr=args.lr,
            hidden=args.hidden,
            onnx_path=args.onnx_path,
            device_str=args.device,
        )
