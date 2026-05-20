import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from starling_env import StarlingFlockEnv


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------

class SharedPolicy(nn.Module):
    """
    Shared policy used by every bird.

    Input  : observation vector (9-dim)
    Output : mean action (3-dim), passed through Tanh → [-1, 1]
    """

    def __init__(self, obs_dim: int = 9, action_dim: int = 3, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# REINFORCE training loop (GPU-accelerated)
# ---------------------------------------------------------------------------

def train(
    num_birds: int = 150,
    episodes: int = 200,
    gamma: float = 0.99,
    lr: float = 3e-4,
    action_std: float = 0.15,
    hidden: int = 128,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    onnx_path: str = "starling_policy.onnx",
):
    device = torch.device(device_str)
    print(f"Training on: {device}")

    env = StarlingFlockEnv(num_birds=num_birds, device=device)

    model = SharedPolicy(obs_dim=9, action_dim=3, hidden=hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Fixed std for exploration (could be learnable)
    log_std = torch.full((3,), np.log(action_std), device=device)

    for episode in range(episodes):

        obs_dict, _ = env.reset()
        N = num_birds

        log_probs_buf: list[torch.Tensor] = []
        rewards_buf:   list[float]         = []

        done = False

        while not done:

            # Stack all observations → (N, 9) on GPU in one transfer
            obs_tensor = torch.tensor(
                np.stack([obs_dict[a] for a in env.agents]),
                dtype=torch.float32,
                device=device,
            )  # (N, 9)

            mean_actions = model(obs_tensor)  # (N, 3)

            std = log_std.exp().unsqueeze(0).expand(N, -1)  # (N, 3)
            dist = torch.distributions.Normal(mean_actions, std)

            actions_tensor = dist.sample()                              # (N, 3)
            log_prob = dist.log_prob(actions_tensor).sum(dim=-1)       # (N,)

            # Convert to dict of numpy arrays for the env
            actions_np = actions_tensor.clamp(-1.0, 1.0).cpu().numpy()
            actions = {env.agents[i]: actions_np[i] for i in range(N)}

            next_obs, rewards, terms, truncs, _ = env.step(actions)

            avg_reward = float(np.mean(list(rewards.values())))
            rewards_buf.append(avg_reward)

            # Store mean log-prob across agents
            log_probs_buf.append(log_prob.mean())

            done = all(truncs.values())
            obs_dict = next_obs

        # --- compute discounted returns ---
        returns = []
        G = 0.0
        for r in reversed(rewards_buf):
            G = r + gamma * G
            returns.insert(0, G)

        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # --- REINFORCE loss ---
        log_probs_t = torch.stack(log_probs_buf)  # (T,)
        loss = -(log_probs_t * returns_t).sum()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        ep_reward = sum(rewards_buf)
        print(f"Episode {episode:4d} | Total reward: {ep_reward:8.2f} | Loss: {loss.item():8.4f}")

    # --- export to ONNX ---
    model.eval()
    dummy_input = torch.randn(1, 9, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=11,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch_size"}, "action": {0: "batch_size"}},
    )
    print(f"\nExported ONNX model → {onnx_path}")

    return model


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
