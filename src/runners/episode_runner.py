from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import imageio



class EpisodeRunner:

    def __init__(self, args, logger=None):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self, path):
        imageio.mimsave(path+'.gif', self.frames, duration=.04)

    def save_animation(self, path):
        self.env.save_animation(path)

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False, render=False, save_animation=False, benchmark_mode=False):
        self.reset()

        terminated = False

        # initialize render frames and agent-landmark positions array if render or test
        if benchmark_mode:
            self.infos = []
        if render:
            self.frames = [self.env.render()]
        
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {"state": [self.env.get_state()],
                                   "avail_actions": [self.env.get_avail_actions()],
                                   "obs": [self.env.get_obs()]}

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            # tell to the environment if the step must be animated
            if save_animation:
                reward, terminated, env_info = self.env.step(actions[0], animate=True)
            else:
                reward, terminated, env_info = self.env.step(actions[0])

            # append the frames if render
            if benchmark_mode:
                env_info['t'] = self.t
                env_info['reward'] = reward
                self.infos.append(env_info)
            if render:
                self.frames.append(self.env.render())

            episode_return += reward

            post_transition_data = {"actions": actions,
                                    "reward": [(reward,)],
                                    "terminated": [(terminated != env_info.get("episode_limit", False),)]}
            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {"state": [self.env.get_state()],
                     "avail_actions": [self.env.get_avail_actions()],
                     "obs": [self.env.get_obs()]}

        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        # log only if the logger is defined
        if self.logger is not None:
            if test_mode and (len(self.test_returns) == self.args.test_nepisode):
                self._log(cur_returns, cur_stats, log_prefix)
            elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
                self._log(cur_returns, cur_stats, log_prefix)
                if hasattr(self.mac.action_selector, "epsilon"):
                    self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
                self.log_train_stats_t = self.t_env

        if benchmark_mode:
            return self.infos
        else:
            return self.batch

    def _log(self, returns, stats, prefix):
        if self.logger is None:
            return
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
