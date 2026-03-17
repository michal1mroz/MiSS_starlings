# Problem and domain analysis

### Starling murmuration

The European starling forms flocks consisting of massive numbers of birds. During migration, common starlings can fly at speeds of 60–80 km/h. Starling flocks move as a collective in seemingly random directions and without a clear leader.

A murmuration is a flock of 500 or more starlings (sometimes even thousands) that typically gather as they approach their roosting site. Smaller flocks travel from different directions and gradually merge with one another until they form one large, dynamic flock.

Within a murmuration, starlings swirl and swoop together, rapidly changing direction and creating constantly shifting shapes. The sharp pushing, pulling, diving, pulsating, and swooping of the flock — driven by individual movements — can confuse and discourage predators such as falcons, providing collective protection.

However, each bird does not track every other bird in the flock. Instead, it follows roughly its seven closest neighbors. By coordinating with these nearest neighbors, starlings can maintain the characteristic structure of the flock with minimal effort. Interestingly, this optimal number depends not on the total number of birds in the flock but on the flock’s geometry — particularly its thickness.

![Large starling flock](images/large-starling-murmuration.png)

Each starling also maintains a safe distance from its nearest neighbors. When a bird at the edge of the flock changes direction, that shift propagates through the flock as nearby birds adjust their movement in response.

In practice, individual birds respond to only three main aspects of flight — their own movement and that of nearby birds. These factors are often described as:

- **Attraction zone**: “In this area, you move toward the next bird.”
- **Repulsion zone**: “You avoid flying directly into another bird’s path, otherwise both would lose stability.”
- **Angular alignment**: “You adjust your direction to match that of your neighbor.”

### Boids

Boids is an artificial life program developed by Craig Reynolds in 1986 that simulates the flocking behavior of birds and other forms of group motion. Instead of controlling the behavior of an entire flock as a single entity, the Boids model defines a set of simple rules that govern the behavior of each individual bird.

Despite the simplicity of these rules, the simulation produces complex and realistic collective motion. Because of this, the model has been widely used in computer graphics, particularly for computer-generated behavioral animation in motion pictures.

In the Boids model, each bird (called a "boid") is represented as an object with its own position, velocity, and orientation. The behavior of each boid is governed by three basic rules:

- **Separation** – each bird attempts to maintain a reasonable distance from nearby birds in order to avoid crowding.
- **Alignment** – birds try to adjust their direction so that it matches the average direction of nearby birds.
- **Cohesion** – each bird attempts to move toward the average position of nearby birds.

![Large starling flock](images/Boid_3D.gif)


## Problem of simulating a flock of starlings

#### Mapping Biological Rules to the Boids Model

A key challenge is translating the biologically observed interaction rules — attraction zone, repulsion zone, and angular alignment — into the classical Boids framework defined by cohesion, separation, and alignment (introduced by Craig Reynolds).

#### The Neighborhood Definition Problem

A critical discrepancy between standard Boids and real starling behavior lies in how “neighbors” are defined:

 - In the classical Boids model:
     - Neighborhoods are metric — all agents within a fixed radius are considered neighbors.
 - In real starling flocks:
     - Neighborhoods are topological — each bird interacts with approximately 6–7 nearest neighbors, regardless of physical distance.

#### Generating Murmurations

In a basic Boids simulation, motion tends to stabilize into smooth, homogeneous flocking. However, real murmurations exhibit rapid directional changes and internal waves and ripples.

To achieve this, additional mechanisms must be introduced:

 - Delayed reactions

Birds do not respond instantaneously; introducing latency in updating velocity or direction can create wave-like propagation effects.

 - Stochastic perturbations (noise)

Small random variations in movement can prevent over-stabilization and help generate complex, dynamic patterns.

 - Local amplification of changes

Small directional changes at the edge of the flock should propagate through neighbor interactions, creating emergent waves across the entire system.

---

https://en.wikipedia.org/wiki/Starling

https://asknature.org/strategy/starlings-coordinate-movements-within-a-flock/

https://www.princeton.edu/news/2013/02/07/birds-feather-track-seven-neighbors-flock-together

https://www.birdfact.com/birds/starling/starling-murmuration

https://en.wikipedia.org/wiki/Boids


https://cs.stanford.edu/people/eroberts/courses/soco/projects/2008-09/modeling-natural-systems/boids.html

https://www.raymairlot.co.uk/blog/boids.html