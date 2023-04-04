import importlib
from .environment import MultiAgentEnv
from gym.spaces import flatdim
import numpy as np
from envs.common_wrappers import TimeLimit, FlattenObservation

try:
    from .animate.plotly_animator import MPEAnimator
except:
    from .animate.pyplot_animator import MPEAnimator
    print('Using matplotlib to save the animation because plolty is not available')


class MPEWrapper(MultiAgentEnv):
    def __init__(self, key, time_limit, **kwargs):

        self.episode_limit = time_limit

        scenario = importlib.import_module("envs.mpe.scenarios."+key).Scenario()
        world = scenario.make_world(
            num_agents=kwargs["num_agents"],
            num_landmarks=kwargs["num_landmarks"]
        )

        # set the reward
        if kwargs.get("reward_discrete", False) and hasattr(scenario, "reward_discrete"):
            reward = scenario.reward_discrete
        elif kwargs.get("reward_rendundant", False) and hasattr(scenario, "reward_rendundant"):
            reward = scenario.reward_rendundant
        else:
            reward = scenario.reward

        env = MultiAgentEnv(
            world,
            reset_callback=scenario.reset_world,
            reward_callback=reward,
            observation_callback=(
                scenario.entity_observation if kwargs.get("obs_entity_mode", False) else 
                scenario.observation
            ),
            state_callback=(
                scenario.entity_state if kwargs.get("state_entity_mode", False) else
                None
            ),
            world_info_callback=getattr(scenario, "world_benchmark_data", None)
        )

        self._env = TimeLimit(env, max_episode_steps=time_limit)
        self._env = FlattenObservation(self._env)


        # basic env variables
        self.n_agents    = self._env.n_agents
        self.n_landmarks = getattr(scenario, "num_landmarks", len(self._env.world.landmarks)) 
        self.n_entities  = getattr(scenario, "num_entities", self.n_agents+self.n_landmarks)
        self.n_entity_types = getattr(scenario, "num_entity_types", 2)
        self._obs = None
        self._state = None

        self.obs_entity_feats = getattr(scenario, "entity_obs_feats", 0)
        self.state_entity_feats = getattr(scenario, "entity_state_feats", 0)

        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
            self._env.observation_space, key=lambda x: x.shape
        )

        # check if the scenario uses an entity state 
        self.custom_state = kwargs.get("state_entity_mode", False)
        self.custom_state_dim = self.state_entity_feats*self.n_entities

        self._seed = kwargs["seed"]
        self._env.seed(self._seed)

        # variables for animation
        self.t = 0

        self.agent_positions      = np.zeros((self.n_agents, self.episode_limit, 2))
        self.landmark_positions   = np.zeros((self.n_landmarks, self.episode_limit, 2))
        self.episode_rewards_all  = np.zeros(self.episode_limit)

    def step(self, actions, animate=False):
        """ Returns reward, terminated, info """
        actions = [int(a) for a in actions]
        self._obs, reward, done, info = self._env.step(actions)
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]

        if self.custom_state:
            self._state = self._env.get_state()

        # save step description if animation is required
        if animate:
            self.agent_positions[:,self.t,:] = self.get_agent_positions()
            self.landmark_positions[:,self.t,:] =  self.get_landmark_positions()
            self.episode_rewards_all[self.t] = float(sum(reward))
        self.t += 1

        return float(sum(reward)), all(done), info["world"]

    def get_agent_positions(self):
        """Returns the [x,y] positions of each agent in a list"""
        return [a.state.p_pos for a in self._env.world.agents]

    def get_landmark_positions(self):
        return [l.state.p_pos for l in self._env.world.landmarks]

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return flatdim(self.longest_observation_space)

    def get_state(self):
        if self.custom_state:
            return self._state
        else: 
            return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """ Returns the shape of the state"""
        if self.custom_state:
            return self.custom_state_dim
        else:
            return self.n_agents * flatdim(self.longest_observation_space)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        valid = flatdim(self._env.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return flatdim(self.longest_action_space)

    def reset(self):
        """ Returns initial observations and states"""
        self._obs = self._env.reset()
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]
        if self.custom_state:
            self._state = self._env.get_state()
        self.t = 0
        self.agent_positions.fill(0.)
        self.landmark_positions.fill(0.)
        self.episode_rewards_all.fill(0.)
        return self.get_obs(), self.get_state()

    def render(self):
        return self._env.render(mode='rgb_array')

    def close(self):
        self._env.close()

    def seed(self):
        return self._env.seed

    def save_replay(self):
        pass

    def save_animation(self, path):
        anim = MPEAnimator(self.agent_positions[:,:self.t,:],
                           self.landmark_positions[:,:self.t,:],
                           self.episode_rewards_all[:self.t])
        anim.save_animation(path)

    def get_stats(self):
        return {}
    
    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit
        }
        for x in [
            "obs_entity_feats",
            "state_entity_feats",
            "n_entities",
            "n_entities_obs",
            "n_entities_state"
        ]:
            self.check_add_attribute(env_info, x)
        return env_info
    
    def check_add_attribute(self, info, attribute):
        if hasattr(self, attribute):
            info[attribute] = getattr(self, attribute)
        elif attribute in {"n_entities_obs", "n_entities_state"}: 
            pass 
        else:
            logging.warning(f"To use transformers you should define the {attribute} attribute in your environment __init__")
