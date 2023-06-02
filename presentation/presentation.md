---
marp: true
theme: 

title: TransfQMix Presentation

#header: 'Header content'
#footer: "AAMAS '23, 1st June 2023"

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


# TransfQMix: Transformers for Leveraging the Graph Structure of Multi-Agent Reinforcement Learning Problems

#### Matteo Gallici, Mario Martín, Ivan Masmitja

KEMLG research, Universitat Polítecnica de Catalunia, and ICM-CISCL.
AAMAS 2023

London, 2nd June 2023

<br>

<center>
<img src="../images/Logo_UPC.svg" style="width:55px; margin-right:30px;">
<img src="../images/BSC-blue.svg" style="width:200px; margin-right:20px;">
<img src="../images/logo-icm-color.svg" style="width:150px; margin-right:20px;">
<img src="../images/csic_logo.svg" style="width:170px; margin-right:20px;">
<img src="../images/logo-so-color.svg" style="width:120px;">

</center>

--- 
<!-- class: invert -->
<!-- paginate: false -->
# 1. Motivations
---
<!-- class: -->
<!-- paginate: true -->


## How are we using information in MARL?


- State-of-the-art approaches in MARL (MADDPG, VDN, QTran, QMix, QPlex, MAPPO, DICG) **focus on the learning model.** 

- Observations and State vectors are gathered from environments and input to function approximators. 

- How is this information structured? Are we using it effectively?

---

## Information Channels

- Information usually comes from multiple sources: *observed entities, different sensors, communication channels.*
- Agents can differentiate between information channels, because they occupy always the **same vector positions**. 

![bg right fit](../images/trad_obs_vector.svg)

<!-- if agents can differentiate between information sources, why not to process them coherently  -->
---

## Entities features

- Often, a subset of **identical** features describe the observed entities.
- **This is a graph structure!**
  - Vertices -> entities features
  - Edges -> ?
- Self-Attention can be used to learn the edges of the latent graph ([Li et al., 2021](https://arxiv.org/abs/2006.11438)).

![bg right fit](../images/entity_obs_vector.svg)

---

## Observations and State Matrices

- We reformulate the entire MARL problem in terms of graphs.
- The goal of the NNs is to learn the **latent coordination graph**.

<div class="row">

<div class="column">
<center>

$$
\begin{eqnarray}
\mathbf{O}_t^a = 
\begin{bmatrix}
ent_1
\\ \vdots \\ 
ent_k
\end{bmatrix}_t^a =
\begin{bmatrix}
f_{1,1} & \cdots & f_{1,z} \\
\vdots & \ddots & \vdots \\ 
f_{k,1} & \cdots & f_{k,z}
\end{bmatrix}
_t^a
\end{eqnarray}
$$

**Observation Matrix**

Includes vertices of the graph observed by the agent $a$ at time step $t$.

</center>
</div>

<div class="column">
<center>

$$
\begin{eqnarray}
\label{eq:state}
\mathbf{S}_t = 
\begin{bmatrix}
ent_1
\\ \vdots \\ 
ent_k
\end{bmatrix}_t = 
\begin{bmatrix}
f_{1,1} & \cdots & f_{1,z} \\
\vdots & \ddots & \vdots \\ 
f_{k,1} & \cdots & f_{k,z}
\end{bmatrix}_t
\end{eqnarray}
$$

**State Matrix**

Describe the complete graph of the environment at time step $t$.

</center>
</div>

</div>

---

<div class="row">

<div class="column">

## Graph-approach advantages

1. Better represents coordination problems.
2. Allow use of more appropriate NNs (GNN, Transformers).
3. Makes the NNs parameters **invariant to the number of agents**.
4. Makes transfer learning and curriculum learning easy to implement.


</div>

<div class="column">

## Disadvantages

1. Cannot differentiate entities a-priori in observations, f.i. raw images (although we can for states).
2. Cannot directly include last action and one hot enconding of agent id in the observation. 

</div>
</div>


--- 
<!-- class: invert -->
<!-- paginate: false -->
# 2. TransfQMix
---
<!-- paginate: true -->
<!-- class: -->

## Main features

- Extends QMix with the use graph matrices and transformers.
  - Independent q-learners.
  - Centralized mixer which respects the monotonicity constraint.
- Graph reasoning in **both the agents and mixer**.
  - UPDET [(Hu et al. 2022)](https://arxiv.org/abs/2101.08001): self-attention in agents.
  - QPlex [(Wang et al. 2021)](https://link.springer.com/article/10.1140/epjd/e2017-80024-y), DICG [(Li et al. 2021)](https://arxiv.org/abs/2006.11438): self-attention in mixer.

---
## Transformer Agent

![bg right fit](../images/transf_agent.svg)

Similar architecture to UPDET, but:
- Q-Values are derived from the agent hidden state, reinforcing recurrent passing of gradient.
- Separate output layer used for Policy Decoupling.

---
## Transformer Mixer

Transformer hypernetwork:
- Agents hidden states:
  - $W_1$: q-values embedding
- Recurrent mechanism:
  - $b1, W2, b2$: projection over $Q_{tot}$
- State embedder:
  - environment reasoning

![bg right fit](../images/transf_mixer.svg)

---

![bg fit](../images/transf_qmix.svg)


--- 
<!-- class: invert -->
<!-- paginate: false -->
# 3. Experiments
---
<!-- class: -->
<!-- paginate: true -->

## Spread

Setup:
- Observation entity features: $pos_x$, $pos_y$
- State entity features: position+velocity.
- Extended to 6 agents (normally only 3)
- Reported evaluation metric: **percentage of landmarks occupied** at the conclusion of an episode (**POL**)

![bg right fit](../videos/5v5.gif)

---
## Spread Results

<center>
<div class="row">

<div class="column">

<img src="../images/spread/spread_3v3.svg" style="width:300px">

Spread 3v3

</div>

<div class="column">

<img src="../images/spread/spread_4v4.svg" style="width:300px">

Spread 4v4

</div>

<div class="column">

<img src="../images/spread/spread_5v5.svg" style="width:300px">

Spread 5v5

</div>

<div class="column">

<img src="../images/spread/spread_6v6.svg" style="width:300px">

Spread 6v6

</div>
</div>
</center>

---
## StarCraft 2
Setup:
- Original state and observations already implicitly sed a graph-based observation and state
- Reported results for the hardest maps.

![bg right 80%](../images/sc2.gif)


---
## SC2 Results
<center>
<div class="row">

<div class="column">

<img src="../images/sc2/5m_vs_6m.svg" style="width:300px">

5m_vs_6m

</div>

<div class="column">

<img src="../images/sc2/8m_vs_9m.svg" style="width:300px">

8m_vs_9m

</div>

<div class="column">

<img src="../images/sc2/27m_vs_30m.svg" style="width:300px">

27m_vs_30m

</div>

<div class="column">

<img src="../images/sc2/6h_vs_8z.svg" style="width:300px">

6h_vs_8z

</div>
</div>

<div class="row">

<div class="column">

<img src="../images/sc2/5s10z.svg" style="width:300px">

5m_vs_6m

</div>

<div class="column">

<img src="../images/sc2/3s5z_vs_3s6z.svg" style="width:300px">

3s5z_vs_3s6z

</div>

<div class="column">

<img src="../images/sc2/MMM2.svg" style="width:300px">

MM2

</div>

<div class="column">

<img src="../images/sc2/corridor.svg" style="width:300px">

corridor

</div>
</div>
</center>

---
## Model Size

<center>
<div class="row">

<div class="column">

| **Model**   | **Agent** | **Mixer** |
|-------------|-----------|-----------|
| TransfQMix  | 50k       | 50k       |
| QMix        | 27k       | 18k       |
| QPlex       | 27k       | 251k      |
| O-CWQMix    | 27k       | 179k      |

Spread 3v3

</div>

<div class="column">

| **Model**   | **Agent** | **Mixer** |
|-------------|-----------|-----------|
| TransfQMix  | 50k       | 50k       |
| QMix        | 28k       | 56k       |
| QPlex       | 28k       | 597k      |
| O-CWQMix    | 28k       | 301k      |

Spread 6v6

</div>

<div class="column">

| **Model**   | **Agent** | **Mixer** |
|-------------|-----------|-----------|
| TransfQMix  | 50k       | 50k       |
| QMix        | 49k       | 283k      |
| QPlex       | 49k       | 3184k     |
| O-CWQMix    | 49k       | 1021k     |

SC2 27m_vs_30m

</div>

</div>
</center>

---
## Zero-Shot Transfer

Spread: **the learned policy is transferable between different team of agents**.

<center>

| **Model**        | **3v3** | **4v4** | **5v5** | **6v6** |
|------------------|---------|---------|---------|---------|
| TransfQMix (3v3) | **0.98**| 0.88    | 0.8     | 0.75    |
| TransfQMix (4v4) | 0.96    | **0.93**| **0.9** | 0.86    |
| TransfQMix (5v5) | 0.88    | 0.85    | 0.82    | 0.82    |
| TransfQMix (6v6) | 0.91    | 0.88    | 0.85    | 0.84    |
| TransfQMix (CL)  | 0.88    | 0.88    | 0.87    | **0.87**|
| State-of-the-art | 0.76    | 0.45    | 0.36    | 0.33    |

</center>

---
## Transfer Learning

SC2: **learning can be speeded up by transferring a learned policy.**

<center>
<div class="row">

<div class="column">

<img src="../images/sc2/8m_vs_9m to 5m_vs_6m.svg" style="width:300px">

8m_vs_9m to 5m_vs_6m

</div>

<div class="column">

<img src="../images/sc2/5s10z to 3s5z_vs_3s6z.svg" style="width:300px">

5s10z to 3s5z_vs_3s6z

</div>

</div>
</center>

---
<!-- paginate: false -->
<!-- class: invert -->
# 6. Conclusions

---
<!-- paginate: true -->
## Conclusions

- Take a look at how you prepare the inputs for your NNs!
- Take profit of the graph structure of MARL problems.
- Enable graph reasoning in NNs.
- Explore multi-agent transfer learning. 

---
<!-- paginate: false -->


# Thank you!

<div class="row">

<div class="column">

### Acknowledgments:
-  EU’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 893089.
- Severo Ochoa Centre of Excellence accreditation (CEX2019-000928-S). 
- David and Lucile Packard Foundation. 
- Barcelona Supercomputing Center in collaboration with the HPAI group.

</div>

<div class="column">

<center>

## Find out more!

<img src="../images/home_qr.svg" style="width: 60%">

### Poster: 22

</center>
</div>
</div>


---
<!-- paginate: false -->


## Additional features

- Some information get lost when dropping concatenation:
  - which of vertices represent "me"? 
  - which of the vertices are "collaborative agents"?
- Easier solution are flags into the vertex features:

<br>

<div class="row">

<div class="column">

$$
\begin{eqnarray}        
f_{i,\texttt{IS\_SELF}}^a =         \begin{cases}            1, & \text{if } i = a\\            0, & \text{otherwise.}        \end{cases}
\end{eqnarray}
$$

</div>

<div class="column">

$$
\begin{eqnarray}        
f_{i, \texttt{IS\_AGENT}}^a =         
\begin{cases}            1, & \text{if } i \in A\\            0, & \text{otherwise}        \end{cases}    
\end{eqnarray}
$$

</div>

---



## Ablation

<center>
<div class="row">

<div class="column">

<img src="../images/ablation_5m_vs_6m.svg" style="width:300px">

SC2: 5m_vs_6m

</div>

<div class="column">

<img src="../images/ablation_spread_6v6.svg" style="width:300px">

Spread: 6v6

</div>

</div>
</center>

---
## Future Work:

- Explore TransfQMix's generalization capabilities by integrating several tasks into a single learning pipeline (training same agents to solve all SC2 tasks).
- Examining the feasibility of transferring coordination policies between different MARL domains.
- Influence of multi-head self-attention on coordination reasoning will be investigated more thoroughly.

