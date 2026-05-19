import numpy as np

from pettingzoo import ParallelEnv
from gymnasium.spaces import Box

class StarlingFlockEnv(ParallelEnv):

    def __init__(
        self,
        num_birds=32,
        world_size=20.0,
        max_speed=1.0,
        neighbor_radius=3.0,
        num_neighbors=7,
        max_steps=500,
    ):

        self.num_birds = num_birds
        self.world_size = world_size
        self.max_speed = max_speed
        self.neighbor_radius = neighbor_radius
        self.num_neighbors = num_neighbors
        self.max_steps = max_steps

        self.agents = [f"bird_{i}" for i in range(num_birds)]

        # observation:
        # [self_vel_x,
        #  self_vel_y,
        #  self_vel_z,
        #  avg_neighbor_dx,
        #  avg_neighbor_dy,
        #  avg_neighbor_dz,
        #  avg_neighbor_vel_x,
        #  avg_neighbor_vel_y,
        #  avg_neighbor_vel_z]
        obs_dim = 9

        # action:
        # delta velocity x, y, z
        act_dim = 3

        self.observation_spaces = {
            a: Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32
            )
            for a in self.agents
        }

        self.action_spaces = {
            a: Box(
                low=-1.0,
                high=1.0,
                shape=(act_dim,),
                dtype=np.float32
            )
            for a in self.agents
        }

    def reset(self, seed=None, options=None): # reset env to start new epoch

        self.step_count = 0

        self.positions = {
            a: np.random.uniform(0, self.world_size, size=(3,)).astype(np.float32)
            for a in self.agents
        }

        self.velocities = {
            a: np.random.uniform(-0.5, 0.5, size=(3,)).astype(np.float32)
            for a in self.agents
        }

        self.neighbors = {
            a: self._compute_global_neighbors(a)
            for a in self.agents
        }

        observations = {
            a: self._get_obs(a)
            for a in self.agents
        }

        infos = {
            a: {}
            for a in self.agents
        }

        return observations, infos


    def step(self, actions):

        self.step_count += 1

        rewards = {}
        observations = {}
        terminations = {}
        truncations = {}
        infos = {}

        for agent, action in actions.items():

            action = np.clip(action, -1.0, 1.0)

            self.velocities[agent] += action * 0.05

            speed = np.linalg.norm(self.velocities[agent])

            if speed > self.max_speed:
                self.velocities[agent] /= speed
                self.velocities[agent] *= self.max_speed

        for agent in self.agents:

            self.positions[agent] += self.velocities[agent]

            # wrap around world
            self.positions[agent] %= self.world_size

        self._refresh_neighbors()

        for agent in self.agents:

            observations[agent] = self._get_obs(agent)

            rewards[agent] = self._compute_reward(agent)

            terminations[agent] = False

            truncations[agent] = (
                self.step_count >= self.max_steps
            )

            infos[agent] = {}

        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos
        )


    def _neighbors(self, agent):
        return self.neighbors.get(agent, [])

    def _compute_global_neighbors(self, agent):
        p = self.positions[agent]

        distances = []
        for other in self.agents:
            if other == agent:
                continue
            d = self.positions[other] - p
            distances.append((np.linalg.norm(d), other))

        distances.sort(key=lambda x: x[0])
        return [other for _, other in distances[: self.num_neighbors]]

    def _compute_local_neighbors(self, agent):
        previous = set(self.neighbors.get(agent, []))
        candidates = set(previous)

        for neighbor in previous:
            candidates.update(self.neighbors.get(neighbor, []))

        candidates.discard(agent)

        if len(candidates) < self.num_neighbors:
            remaining = [other for other in self.agents if other not in candidates and other != agent]
            candidates.update(remaining)

        distances = []
        p = self.positions[agent]
        for other in candidates:
            d = self.positions[other] - p
            distances.append((np.linalg.norm(d), other))

        distances.sort(key=lambda x: x[0])
        return [other for _, other in distances[: self.num_neighbors]]

    def _refresh_neighbors(self):
        new_neighbors = {}
        for agent in self.agents:
            new_neighbors[agent] = self._compute_local_neighbors(agent)
        self.neighbors = new_neighbors

    def _get_obs(self, agent):

        neighbors = self._neighbors(agent)

        self_vel = self.velocities[agent]

        if len(neighbors) == 0:

            return np.concatenate([
                self_vel,
                np.zeros(6, dtype=np.float32)
            ]).astype(np.float32)

        rel_positions = []
        rel_vels = []

        for n in neighbors:

            rel_positions.append(
                self.positions[n] - self.positions[agent]
            )

            rel_vels.append(
                self.velocities[n]
            )

        avg_rel_pos = np.mean(rel_positions, axis=0)
        avg_rel_vel = np.mean(rel_vels, axis=0)

        obs = np.concatenate([
            self_vel,
            avg_rel_pos,
            avg_rel_vel
        ])

        return obs.astype(np.float32)


    def _compute_reward(self, agent):

        neighbors = self._neighbors(agent)

        if len(neighbors) == 0:
            return -1.0

        # Nearest neighbor distance (NND)
        nnd = None
        for n in neighbors:
            d = np.linalg.norm(self.positions[n] - self.positions[agent])
            if nnd is None or d < nnd:
                nnd = d

        # Average neighbor velocity magnitude
        avg_neighbor_vel = np.mean(
            [np.linalg.norm(self.velocities[n]) for n in neighbors]
        )

        # Self velocity magnitude
        self_vel_mag = np.linalg.norm(self.velocities[agent])

        # Target values from empirical flock data
        # NND ranges from 0.68 to 1.51 m in real data, target ~1.0
        target_nnd = 1.0
        
        # Velocity ranges from 6.9 to 15.2 m/s in real data
        # Normalized to simulation max_speed of 1.0, target ~0.5-0.7
        target_vel = 0.6

        # Reward based on deviation from targets
        nnd_reward = 1.0 / (1.0 + np.abs(nnd - target_nnd))
        
        # Average velocity reward (own + neighbors)
        avg_vel = (self_vel_mag + avg_neighbor_vel) / 2.0
        vel_reward = 1.0 / (1.0 + np.abs(avg_vel - target_vel))

        # Combined reward
        return (nnd_reward + vel_reward) / 2.0