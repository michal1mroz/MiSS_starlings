import numpy as np
import torch
import torch.nn.functional as F
from pettingzoo import ParallelEnv
from gymnasium.spaces import Box


class StarlingFlockEnv(ParallelEnv):
    """
    3D starling flock environment with GPU-accelerated physics and neighbor
    computations. All internal state is kept as torch tensors on `device`.

    Observation per bird (9-dim):
        [self_vel_x, self_vel_y, self_vel_z,
         avg_neighbor_rel_pos_x, y, z,
         avg_neighbor_vel_x, y, z]

    Action per bird (3-dim):
        delta-velocity [dx, dy, dz] clipped to [-1, 1]

    Reward components (each in [0, 1], equally weighted):
        - cohesion  : NND close to target_nnd
        - speed     : own speed close to target_speed
        - polarity  : cosine similarity with mean neighbor velocity
    """

    metadata = {"name": "starling_flock_v0"}

    def __init__(
        self,
        num_birds: int = 32,
        world_size: float = 20.0,
        max_speed: float = 1.0,
        num_neighbors: int = 7,
        max_steps: int = 500,
        # reward targets
        target_nnd: float = 1.0,
        target_speed: float = 0.6,
        # training device
        device: str | torch.device = "cpu",
    ):
        self.num_birds = num_birds
        self.world_size = world_size
        self.max_speed = max_speed
        self.num_neighbors = num_neighbors
        self.max_steps = max_steps
        self.target_nnd = target_nnd
        self.target_speed = target_speed
        self.device = torch.device(device)

        self.agents = [f"bird_{i}" for i in range(num_birds)]
        self.possible_agents = self.agents[:]

        obs_dim = 9
        act_dim = 3

        self.observation_spaces = {
            a: Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for a in self.agents
        }
        self.action_spaces = {
            a: Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
            for a in self.agents
        }

        # Runtime tensors – allocated in reset()
        self.pos: torch.Tensor = None   # (N, 3)
        self.vel: torch.Tensor = None   # (N, 3)
        self.neighbor_idx: torch.Tensor = None  # (N, K)
        self.step_count: int = 0

    # ------------------------------------------------------------------
    # PettingZoo API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.step_count = 0
        N = self.num_birds
        dev = self.device

        self.pos = torch.rand(N, 3, device=dev) * self.world_size          # (N,3)
        self.vel = (torch.rand(N, 3, device=dev) - 0.5)                    # (N,3)

        self.neighbor_idx = self._compute_neighbors_global()

        return self._all_obs(), {a: {} for a in self.agents}

    def step(self, actions: dict):
        self.step_count += 1
        N = self.num_birds
        dev = self.device

        # Stack actions → (N, 3) tensor
        act = torch.stack([
            torch.tensor(actions[a], dtype=torch.float32, device=dev)
            for a in self.agents
        ])  # (N, 3)

        act = act.clamp(-1.0, 1.0)

        # Update velocities
        self.vel = self.vel + act * 0.05

        # Clip to max speed
        speeds = self.vel.norm(dim=1, keepdim=True)          # (N,1)
        over = speeds > self.max_speed
        self.vel = torch.where(
            over.expand_as(self.vel),
            self.vel / speeds * self.max_speed,
            self.vel,
        )

        # Update positions with toroidal wrap
        self.pos = (self.pos + self.vel) % self.world_size

        # Refresh neighbor graph (local search)
        self.neighbor_idx = self._compute_neighbors_local()

        obs = self._all_obs()
        rewards = self._all_rewards()

        truncated = self.step_count >= self.max_steps
        terminations = {a: False for a in self.agents}
        truncations = {a: truncated for a in self.agents}
        infos = {a: {} for a in self.agents}

        return obs, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------
    # Neighbor computation (fully vectorised on GPU)
    # ------------------------------------------------------------------

    def _pairwise_distances(self) -> torch.Tensor:
        """Returns (N, N) squared-distance matrix with toroidal wrap."""
        diff = self.pos.unsqueeze(1) - self.pos.unsqueeze(0)   # (N,N,3)
        # Shortest path on torus
        diff = diff - self.world_size * torch.round(diff / self.world_size)
        return (diff ** 2).sum(dim=-1)                          # (N,N)

    def _compute_neighbors_global(self) -> torch.Tensor:
        """Full O(N²) search – used only at reset."""
        dist2 = self._pairwise_distances()                      # (N,N)
        dist2.fill_diagonal_(float("inf"))
        _, idx = dist2.topk(self.num_neighbors, dim=1, largest=False)
        return idx  # (N, K)

    def _compute_neighbors_local(self) -> torch.Tensor:
        """
        Approximate local search: consider each bird's current neighbours
        and their neighbours as candidates, then re-rank by distance.
        Falls back to global if candidate pool is too small.
        """
        N = self.num_birds
        K = self.num_neighbors

        # Build candidate set: self + neighbours + neighbours-of-neighbours
        # Shape trick: gather neighbor indices two hops out
        first_hop = self.neighbor_idx                           # (N, K)
        second_hop = self.neighbor_idx[first_hop].view(N, -1)  # (N, K²)

        self_idx = torch.arange(N, device=self.device).unsqueeze(1)  # (N,1)
        candidates = torch.cat([self_idx, first_hop, second_hop], dim=1)  # (N, 1+K+K²)

        # Compute distances to candidates only
        pos_self = self.pos.unsqueeze(1)                              # (N,1,3)
        pos_cand = self.pos[candidates]                               # (N,C,3)
        diff = pos_cand - pos_self
        diff = diff - self.world_size * torch.round(diff / self.world_size)
        dist2 = (diff ** 2).sum(dim=-1)                               # (N,C)

        # Mask self (distance = 0 at index 0 in candidates)
        dist2[:, 0] = float("inf")

        # Pick top-K closest
        if candidates.shape[1] <= K:
            return self._compute_neighbors_global()

        _, local_idx = dist2.topk(K, dim=1, largest=False)
        return candidates.gather(1, local_idx)  # (N, K)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _all_obs(self) -> dict:
        """Vectorised observation construction → dict of np arrays."""
        N = self.num_birds
        K = self.num_neighbors

        neighbor_pos = self.pos[self.neighbor_idx]              # (N,K,3)
        neighbor_vel = self.vel[self.neighbor_idx]              # (N,K,3)

        avg_rel_pos = (neighbor_pos - self.pos.unsqueeze(1)).mean(dim=1)  # (N,3)
        avg_vel = neighbor_vel.mean(dim=1)                                 # (N,3)

        obs_tensor = torch.cat([self.vel, avg_rel_pos, avg_vel], dim=1)   # (N,9)
        obs_np = obs_tensor.cpu().numpy().astype(np.float32)

        return {self.agents[i]: obs_np[i] for i in range(N)}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _all_rewards(self) -> dict:
        """
        Vectorised reward computation.

        Components:
          cohesion  – NND (nearest neighbour distance) close to target_nnd
          speed     – own speed close to target_speed
          polarity  – cosine similarity with mean neighbour velocity
        """
        N = self.num_birds

        # --- cohesion: nearest-neighbour distance ---
        neighbor_pos = self.pos[self.neighbor_idx]              # (N,K,3)
        diff = neighbor_pos - self.pos.unsqueeze(1)             # (N,K,3)
        diff = diff - self.world_size * torch.round(diff / self.world_size)
        dists = diff.norm(dim=-1)                               # (N,K)
        nnd = dists.min(dim=1).values                           # (N,)
        cohesion = 1.0 / (1.0 + (nnd - self.target_nnd).abs())

        # --- speed ---
        speeds = self.vel.norm(dim=1)                           # (N,)
        speed_r = 1.0 / (1.0 + (speeds - self.target_speed).abs())

        # --- polarity: cosine similarity with mean neighbour velocity ---
        neighbor_vel = self.vel[self.neighbor_idx]              # (N,K,3)
        avg_neighbor_vel = neighbor_vel.mean(dim=1)             # (N,3)

        # Cosine similarity; guard against zero vectors
        self_norm = F.normalize(self.vel, dim=1)
        neigh_norm = F.normalize(avg_neighbor_vel, dim=1)
        cos_sim = (self_norm * neigh_norm).sum(dim=1)           # (N,) in [-1,1]
        polarity = (cos_sim + 1.0) / 2.0                       # rescale to [0,1]

        reward = (cohesion + speed_r + polarity) / 3.0         # (N,)
        reward_np = reward.cpu().numpy()

        return {self.agents[i]: float(reward_np[i]) for i in range(N)}

    # ------------------------------------------------------------------
    # PettingZoo space accessors
    # ------------------------------------------------------------------

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
