# Goals and scope of work

## Model

Parameters used for learning the model are:

 - Flocking event,
 - Number of birds,
 - Volume (m3),
 - Density r (m^-3),
 - NND r1 (m),
 - Velocity (m/s),
 - Concavity,
 - Balance shift,
 - Thickness I1 (m),
 - Aspect ratios (I2/I1,I3/I1),
 - Orientation parameters (**I1 * G**, **V * G**, **V * I1**)

### Reinforcement learning

Reinforcement learning (RL) is a type of machine learning process in which autonomous agents learn to make decisions by interacting with their environment.
Multi-agent reinforcement learning (MARL) is a sub-field of reinforcement learning. It focuses on studying the behavior of multiple learning agents that coexist in a shared environment. Each agent is motivated by its own rewards, and does actions to advance its own interests. Multi-agent reinforcement learning is modeled as some form of a Markov decision process (MDP).

### Agents set

Each agent represents a single bird. The number of agents in the model should satisfy:

$$N \in [400,2700]$$

This range is chosen based on the availability of empirical data. All agents share the same policy. They act independently, but their behaviors differ due to differences in local observations.

### State space

The state space consists of all variables required to describe the agents and the environment. For each agent:
- position $x_i^t$
- velocity $v_i^t$
- acceleration $a_i^t$ (optional)

The state of the environment would be described by: $$s_t = \{(x_i^t, v_i^t)\}^N_{i=1}$$, i.e., the set of states of all agents.

### Local observation

Each agent has access only to local information. The observation for agent $i$ is defined as: $$o_t^i = \{(x_j - x_i):j \in S_j\}$$, where $S_j$ is the set of the nearest neighbors.

### Action space

The action space defines the set of possible decisions available to each agent.

In this model, the action is defined as: $$\Delta v_i$$
i.e., the change in the velocity vector.

### Reward

The reward function should quantify the similarity between simulated flock behavior and real-world statistical properties.

## Goals

The goal is to learn a local interaction policy that generates realistic flocking structures.
Training set (TRAIN – 7 samples)

The following events were added to the training set:

 - 32-06
 - 28-10
 - 25-11
 - 25-10
 - 21-06
 - 29-03
 - 25-08

Validation set (VALIDATION – 3 samples)
The following were added to the validation set:
 - 16-05
 - 17-06
 - 31-01

Variable : Number of birds

Outputs :
Other variables describing the flock structure and geometry (NND (Nearest Neighbor Distance),
Velocity (m/s), Volume, Density, Concavity, Balance shift, Thickness I1, I2/I1, I3/I1, I1-G, V-G, V-I1)

In the context of reinforcement learning, these values ​​are treated as environmental observations based on which the reward is calculated.

## Scope of work

  1) Data preparation
  2) Building the simulation environment (implementing dynamic (3D) neighbor search (k-NN / radius), calculating global metrics)
  3) MARL definition (observation space, action space, reward function)
  4) Model implementation
  5) Training (hyperparameter tuning)
  6) Validation and analysis of results (comparison with real data)