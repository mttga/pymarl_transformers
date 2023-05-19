---
marp: true
theme: 

class:

#header: 'Header content'
#footer: 'Multi-Agent Reinforcment Learning for Acoustic Tracking'

#![height:500](renders/emergent_behaviours/graph_3v3_follow_with_big_error.gif)
---

<style>
section {
  font-size: 25px;
}

blockquote {
    border-top: 0em dashed #555;
    font-size: 45%;
    margin-top: auto;
}
</style>

<style>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
</style>

<style>
img[alt~="right"] {
    display: block;
    float: right;
}
</style>


<style>
div.twocols {
  margin-top: 35px;
  column-count: 2;
}
div.twocols p:first-child,
div.twocols h1:first-child,
div.twocols h2:first-child,
div.twocols ul:first-child,
div.twocols ul li:first-child,
div.twocols ul li p:first-child {
  margin-top: 0 !important;
}
div.twocols p.break {
  break-before: column;
  margin-top: 0;
}
</style>


<style>
.row {
  display: flex;
}
.column {
  flex: 33.33%;
  padding: 5px;
}

.row_centered {
  display: flex;
  justify-content: center;
}


.small_column {
  margin-top: auto;
  margin-bottom:  auto;
  padding: 10px;
}
</style>

<!-- class: invert -->

# TransfQMix: Transformers for Leveraging the Graph Structure of Multi-Agent Reinforcement Learning Problems

#### Matteo Gallici, Mario Martin, Ivan Masmitja

Universitat Polítecnica de Catalunia
AAMAS 2023

London, 1st June 2023


---
<!-- class: -->
## List of contents
1. Motivations
2. Related works
3. Proposed methods
4. Experiments
5. Summary and future work



--- 
<!-- class: invert -->
<!-- paginate: false -->
# 1. Motivations
---
<!-- class: -->
<!-- paginate: true -->

---


--- 
<!-- class: invert -->
<!-- paginate: false -->
# 2. Related Works
---
<!-- paginate: true -->
<!-- class: -->

# QMix [(Rashid et al., 2018)](https://arxiv.org/abs/1803.11485)

> Image from: [Rashid et al., 2018](https://arxiv.org/abs/1803.11485)


--- 
<!-- class: invert -->
<!-- paginate: false -->
# 3. Proposed Methods
---
<!-- class: -->
<!-- paginate: true -->


# Transformers Modules

Motivations:

- Transformers are graph neural networks

- Extend the graph reasoning to the agents

- Reduce the complexity and improve explainability of RNNs

- Increase transferability between tasks and scenarios


> Image from: [Chaitanya, 2020](https://graphdeeplearning.github.io/post/transformers-are-gnns)

<center>

| Level of Difficulty | Dropping Factor | Ocean Currents Velocity |
|---------------------|-----------------|-------------------------|
| control             | 0               | 0                       |
| easy                | 5\%             | $\frac{1}{10}v_a$       |
| medium              | 25\%            | $\frac{1}{3}v_a$        |
| hard                | 45\%            | $\frac{2}{3}v_a$        |

</center>

---
# UAT Team Reward

$$

r = \left\{r_i^d\right\}_{i=0}^m + 
\left\{r_i^e\right\}_{i=0}^m + 
\left\{r_{i,j}^{collision}\right\}_{i=0, j=0}^k 
\quad \text{for } i \neq j
$$

- Team reward based on the distance from landmarks $i \in \mathbf{L}$ and estimation error:

$$
\tiny
r_{i}^d=
\left\{\begin{array}{cl}
    -100 & \text { if } \quad d_i > 2d_{max} \\
    -0.1 & \text { if } \quad d_i >= d_{max} \\
    \lambda(0.5-d_i) & \text { if } \quad d_{t h} < d_i < d_{max} \\
    1 & \text { else }
\end{array}\right.
$$

$$
\tiny
r_{i}^e=\left\{
\begin{array}{cl}
    \lambda\left(0.004-e_{i}\right) & \text { if } \quad e_{i}>e_{th} \\
    0 & \text { else }
\end{array}
\right.
$$

- Penality for collisions:

$$
\tiny
r_{i,j}^{collision}\left\{
\begin{array}{cl}
    -10 & \text { if } \quad d_{i,j}<d_{min} \\
    0 & \text { else }
\end{array}
\right.
$$

- Each landmark tracked correctly by the **assigned agent** brings a reward **+1**
- Greedy algorithm to assign agents to landmarks **without repetitions**


--- 
<!-- class: invert -->
<!-- paginate: false -->
# 5. Experiments
---
<!-- class: -->
<!-- paginate: true -->






---
<!-- paginate: false -->
<!-- class: invert -->
# 6. Summary and Future Work

---
<!-- paginate: true -->
# Key Contributions: New MARL Environment 


- Fully designed and deployed with Ivan Masmitja
- Realistic dynamics: *ocean currents, information noise, sensors limitations, agents communication*
- 3 different tasks, 3 difficulties
- 2 Tracking models (LS and PF)
- Front-end animation tool
- **Release on GitHub** to allow MARL researches to test their model in a more realistic simulation


---
# Key Contributions: New MARL Methods 


### GraphQMix: 
  - combination of existing MARL methods
  - introduced self-attention-based graph reasoning in QMix
### TransfQMix: 
  - Designed and deployed from scratch
  - Replaced MLP and RNN networks with **transformers** in QMix
  - First off-policy method based on QMix **completely transferable**
  - Allow **zero-shot transfer, transfer learning and curriculum learning** for cooperation policies
  - Custom design to allow **short-memory** reasoning


---
# Key Contributions: Extensive Experiments 


- Extensive **exploration of MARL applicability** to the UAT scenario
- Proven the benefits of **integrating self-attention graph** reasoning in QMix:
  - Pushed the limits of state-of-the-art MARL algorithms from 5 to 7 agents
  - Improved agents coordination and distribution
- Demonstrated the **transferability** of TransfQMix in zero-shot transfer and transfer learning
- Proposed **curriculum learning** as a strategy to learn progressively large-team cooperation
- Several **ablation studies** to demonstrate the significance of the proposed methods
- Promising results in **long mission experiments** 


---
# Future Work:
- Test the proposed methods in **standard MARL environments**
- Optimize the **PF complexity** and include it in training
- Increase the **variability** of scenarios presented in training 
- Make the **reward function** invariant to the number of landmarks, so that zero-shot learning is theoretically valid
- Merge the UAT environment in a standard **MARL open-source library**
- **Publish a paper** extracted from the memory in a scientific journal or conference


---
<!-- paginate: false -->
<!-- class: -->



# Thank you!