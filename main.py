import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from starling_env import StarlingFlockEnv

class SharedPolicy(nn.Module):

    def __init__(self, obs_dim=9, action_dim=3):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    device = torch.device("cpu")

    env = StarlingFlockEnv(
        num_birds=150
    )

    model = SharedPolicy(obs_dim=9, action_dim=3).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    gamma = 0.99

    # epsilon = 0.1

    episodes = 100

    for episode in range(episodes):

        obs, infos = env.reset()

        episode_reward = 0.0

        log_probs = []
        rewards_buffer = []

        done = False

        while not done:

            actions = {}

            step_log_probs = []

            for agent in env.agents:

                obs_tensor = torch.tensor(
                    obs[agent],
                    dtype=torch.float32
                ).unsqueeze(0)

                mean_action = model(obs_tensor)

                dist = torch.distributions.Normal(
                    mean_action,
                    0.15
                )

                action = dist.sample()

                log_prob = dist.log_prob(action).sum()

                actions[agent] = (
                    action.squeeze(0)
                    .detach()
                    .numpy()
                )

                step_log_probs.append(log_prob)

            next_obs, rewards, terms, truncs, infos = env.step(actions)

            avg_reward = np.mean(
                list(rewards.values())
            )

            rewards_buffer.append(avg_reward)

            log_probs.append(
                torch.stack(step_log_probs).mean()
            )

            episode_reward += avg_reward

            done = all(truncs.values())

            obs = next_obs

        returns = []

        G = 0

        for r in reversed(rewards_buffer):

            G = r + gamma * G

            returns.insert(0, G)

        returns = torch.tensor(
            returns,
            dtype=torch.float32
        )

        returns = (returns - returns.mean()) / returns.std()

        loss = 0

        for log_prob, G in zip(log_probs, returns):

            loss += -log_prob * G

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        print(
            f"Episode {episode} | "
            f"Reward: {episode_reward:.2f}"
        )

        dummy_input = torch.randn(1, 9)

    torch.onnx.export(
        model,
        dummy_input,
        "starling_policy.onnx",
        opset_version=11,
        input_names=["obs"],
        output_names=["action"],
    )

    print("\nExported: starling_policy.onnx")