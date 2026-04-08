# Empirical data

### [Scale-free correlations in starling flocks](https://pubmed.ncbi.nlm.nih.gov/20547832/)

This dataset (Cavagna et al., PNAS 2010) focuses on **collective motion and correlation structure**.  
The key result is that velocity fluctuations are **scale-free**, meaning the correlation length grows proportionally with flock size.


|Event| No. of birds| Polarization| Velocity, m/s| Flock’s size L, m|
|---|---|---|---|---|
|16-05 | 2,941 |0.962 |15.2| 79.2|
|17-06 | 552| 0.935 |9.4| 51.8|
|21-06 |717| 0.973| 11.8 |32.1|
|25-08 |1,571| 0.962| 12.1 |59.8|
|25-10 |1,047| 0.991| 12.5| 33.5|
|25-11| 1,176| 0.959| 10.2 |43.3
|28-10| 1,246| 0.982| 11.1 |36.5|
|29-03| 440| 0.963| 10.4| 37.1|
|...|...| ...| ...| ...|

**Notes:**
- **Polarization (Φ)** ≈ 1 indicates highly aligned motion (strong collective order)
- **Flock size L** is the maximum linear extension of the group
- Despite large variation in N, statistical properties remain consistent (scale invariance)

### [An empirical study of large, naturally occurring starling flocks](https://www.sciencedirect.com/science/article/pii/S0003347208001176)

This dataset (Cavagna et al., 2008) focuses on **geometrical and structural properties** of flocks reconstructed in 3D.



|Event|Number|Volume|Density|Nnd|Velocity|Concavity|Balance|Thickness|Aspect ratios (I_2/I_1)|Aspect ratios (I_3/I_1)|Orientation Parameter (I_1 * G)|Orientation Parameter (V * G)|Orientation Parameter (V * I_1)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|32-06|781|930|0.8|0.68|9.6|0.03|0.08|5.33|2.97|4.02|0.89|0.06|0.20|
|28-10|1246|1840|0.54|0.73|11.1|0.34|-0.06|5.29|3.44|6.93|0.80|0.09|0.41|
|25-11|1168|2340|0.38|0.79|8.8|0.37|-0.1|8.31|1.90|5.46|0.92|0.12|0.14|
|...|...|...|...|...|...|...|...|...|...|...|...|...|...|

**Notes:**
- **NND (nearest neighbor distance)** is typically ~0.7–0.8 m (≈ wingspan)
- **Density** varies significantly across flocks and within the flock (edge vs center)
- **Thickness** is much smaller than lateral dimensions → flocks are quasi-2D
- **Aspect ratios (I₂/I₁, I₃/I₁)** come from principal inertia axes:
  - quantify elongation and flatness of the flock
- **Concavity** measures deviation from convex shape
- **Balance** describes asymmetry in mass distribution
- **Orientation parameters** relate flock geometry to:
  - gravity vector (G)
  - velocity vector (V)


### [What underlies waves of agitation in starling flocks](https://link.springer.com/article/10.1007/s00265-015-1891-3)

This study analyzes **propagation of collective response (information transfer)** within flocks.

**Key empirical values:**
- Nearest neighbor distance:
  - ~1.1–1.3 m
- Reaction time between individuals:
  - ~0.07–0.1 s
- Propagation speed of agitation waves:
  - ~10–20 m/s

**Interpretation:**
- Information propagates as a **wave through the interaction network**
- Propagation speed is comparable to (or higher than) flock velocity
- This supports the hypothesis of **efficient, near-critical information transfer**


### [Interaction ruling animal collective behavior depends on topological rather than metric distance: Evidence from a field study](https://www.pnas.org/doi/10.1073/pnas.0711437105)

(6-7 neighbours)

# Models
### Boids
Boids were discussed in the previous stage. Implementing them would require a slight modifications regarding the controled parameters.
### Reinforcement learning
Article [1] shows 2 methods to model bird flocking with reinforcement learning. Both methods use multi agent approach, one learns with global knowledge for all agents,
the other utilizes local knowledge per agent.
Paper did not validate the results against empirical data, 
also cost function used for training was simple and wouldn't cover all required parameters and so it would need
a refinment.

## Validating results
For both approaches loss function can be defined to approximate parameters such as flock density, polarization or distance to the nearest neighbor.
Calculated values can be then compared to the data from the above mentioned sources.

## Tools
* [RLlib](https://docs.ray.io/en/latest/rllib/index.html) - Ray framework library for reinforcement learning
* [WarpDrive](https://github.com/salesforce/warp-drive) - python library for reinforcement learning using GPU acceleration.
* [PyGame](https://www.pygame.org/docs/) - python library for 2 dimmensional graphics.
* [Unity ml-agents](https://docs.unity3d.com/Packages/com.unity.ml-agents@3.0/manual/index.html) - Unity package enabling agent based simulation and learning with Unity engine.

# Bibliography
1. Martino Brambati and Antonio Celani and Marco Gherardi and Francesco Ginelli _Learning to flock in open space by avoiding collisions and staying together_, arXiv 2026, url: https://arxiv.org/abs/2506.15587 
