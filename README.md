[![arXiv](https://img.shields.io/badge/arXiv-2301.05334-b31b1b.svg)](https://arxiv.org/abs/2301.05334)
[![Homepage](https://img.shields.io/badge/Project-Homepage-0077B5.svg)](https://mttga.github.io/pymarl_transformers/)
[![Poster](https://img.shields.io/badge/View-Poster-008037.svg)](https://mttga.github.io/pymarl_transformers/poster/index.html)
[![Presentation](https://img.shields.io/badge/View-Presentation-FFD700.svg)](https://mttga.github.io/pymarl_transformers/presentation/index.html)

# TransfQMix

Official repository of the [AAMAS 2023](https://aamas2023.soton.ac.uk/) paper: [TransfQMix: Transformers for Leveraging the Graph Structure of
Multi-Agent Reinforcement Learning Problems](https://arxiv.org/abs/2301.05334). The codebase is built on top of [Pymarl](https://github.com/oxwhirl/pymarl). 

## Usage 

### With docker
The repository makes available a Dockerfile to containerize the execution of the code with GPU support (recommended for transformer models). To build the image, run the standard:

```bash
sudo docker build . -t pymarl
```

You can then run any of the available models with the run.sh script: ```bash run.sh```. Change the last line of the script in order to choose your configuration files. For example:

```bash
# run StarCraft2 experiment
python3 src/main.py --config=transf_qmix_smac --env-config=sc2
# run Spread experiment
python3 src/main.py --config=transf_qmix --env-config=mpe/spread
```

Remember that you need [nvidia-container](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installation-guide) to use your GPU with Docker. 

### With python
If you want to run the codebase without docker, you can install the ```requirements.txt``` in a python 3.8 virtual environment (conda, pipenv). 

You will also need to install StarCraft2 in your computer: with linux, you can use the ```bash install_sc2.sh``` script. You will also need [SMAC](https://github.com/oxwhirl/smac.git) with ```pip install git+https://github.com/oxwhirl/smac.git```. 

Finally, install the pytorch version that is more suitable for your system. For example (for GPU support with CUDA 11.6): ```pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116```.

Then run the python commands as above. 

***

## Use transformers in new environments

In order to use TransfQMix (or just the transformer agent or mixer) in other environments, at every time-step you will need to return observation and state vectors that are reshapable as matrices. In particular, the shape of the matrices will be $(k, z)$, where $k$ is the number of entities and $z$ the entity features. Therefore, the observations and states are flattened vectors with a dimension of $k \times z$. For simplicity we can assume that $k$ and $z$ are the same for agents and mixer but this could not the case (use ```n_entities_obs``` and ```n_entities_state``` to differenciate $k$ for the agent and the mixer; to differenciate $z$, use ```obs_entity_feats``` and ```obs_state_feats```).

- The definition of the entity depends on the environment. In SC2, they are the allies+enemies. In Spread, the agents+landmarks. But an entity can be in principle any information channel (i.e. different sensors, communication channels, and so on). 

- The entity features $z$ are the features which describe every entity. Sparse matrices are not allowed, i.e. all the features should be used to described every entity. If it doesn't make sense to use some of them for some entities, pad them to 0. Check the paper for additional information about $k$ and $z$.

In the case of SC2 and Spread, we included two new environment parameters: ```obs_entity_mode``` and ```state_entity_mode``` that allow to chose how to return the observation and state vectors at every time step. If they are set to ```True```, the environment is expected to return flattened matrices that will be reshaped again as matrices internally by TransfQMix. If ```obs_entity_mode``` or ```state_entity_mode``` are set to ```True```, the original observation or state vectors are returned.

We encourage you to follow the same line, i.e. include a parameter in your environment that allows to chose if use the entity-mode or not. 

Take in mind:
1. In the __init__ method of your environment wrapper, you should define additional attributes in respect to the traditional pymarl (please check [MultiAgentEnv](src/envs/multiagentenv.py)):
    - ```self.obs_entity_feats```: number of features that define the entities in the obs matrix
    - ```self.state_entity_feat```: number of features that define the entities in the state matrix
    - ```self.n_entities```: number of fixed entities observed by agents and mixer
    - (optional) ```self.n_entities_obs```: number of entities observed by agents if different than n_entites
    - (optional) ```self.n_entities_state```: number of entities observed by mixer if different than n_entites
2. You can define different features for the entity observation matrix and for the entity observation state. Change ```obs_entity_feats``` and ```state_entity_feats``` accordingly.
3. The number of entities is assumed to be invariant during an episode. If an entity dies or is not observable, set all its features to 0s (this can be improved). 
4. The order of the entities in the flatten vectors is important:
    - For the agent only if your using policy decoupling: in this case you need to ensure that you're taking track of the positions of the entities which have some entity-based actions (for example, the ). This is because you will need to extract their specific embeddings in order to sample the entity-based actions from them. See the [SC2 agent](src/modules/agents/n_transf_agent_smac.py) for how this is done in StarCraft 2. 
    - For the mixer: the first entity features must be relative to the agents and must follow the same order of the agents q-values. The codebase puts the agents always in the same order, so this should not be a problem. 

***

## Extra

1. The repository supports parallelization with the [parallel_runner](src/runners/parallel_runner.py) with most of the models (i.e. a parallel environment running for each experiment), and also TransfQMix is ready to be used in parallel environments. **Performances are *not* comparable of models trained with different number of processes.** By default, an experiment with a single environment in single process is run. 
2. If you're using policy decoupling in a new environment, it is recommended that you add a new environment-specific transformer agent in order to menage the output layers. You can use [SC2 agent](src/modules/agents/n_transf_agent_smac.py) as a base. You would need only to change the output layers, the way in which the entity-based-action embeddings are extracted from the output of the transformer, and add your agent in the [agent registry](src/modules/agents/__init__.py). In future version this could parametrized, but for now this is the easiest way to go. If you're not using  policy decoupling, you can use the [standard transformer agent](src/modules/agents/n_transf_agent.py).
3. This codebase includes a [matplotlib](src/envs/mpe/animate/pyplot_animator.py)-based and a [plolty](src/envs/mpe/animate/plotly_animator.py)-based animation classes for the MPE environment, which allow to generate customized gifs at the end of an episode. You can get inspired from them to generate animations of your environment in a simpler way than using gym. Here is an example for 6v6 Spread:

![spread_5v5](https://s9.gifyu.com/images/animation_5v5.gif)

# Citation
If you use this codebase please cite:

```
@inproceedings{10.5555/3545946.3598825,
author = {Gallici, Matteo and Martin, Mario and Masmitja, Ivan},
title = {TransfQMix: Transformers for Leveraging the Graph Structure of Multi-Agent Reinforcement Learning Problems},
year = {2023},
publisher = {International Foundation for Autonomous Agents and Multiagent Systems},
address = {Richland, SC},
booktitle = {Proceedings of the 2023 International Conference on Autonomous Agents and Multiagent Systems},
pages = {1679–1687},
location = {London, United Kingdom},
series = {AAMAS '23}
}
```
