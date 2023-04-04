import logging

class MultiAgentEnv(object):

    """
    Important: 
    In the __init__ function you should define some attributes:
    - self.obs_entity_feats: number of features that define the entities in the obs matrix
    - self.state_entity_feats: number of features that define the entities in the state matrix
    - self.n_entities: number of fixed entities observed by agent and mixer
    Optional:
    - self.n_entities_obs: number of entities observed by agent if different than n_entites
    - self.n_entities_state: number of entities observed by mixer if different than n_entites

    Notice:
    You should modify internally your environment in order to return the observations and states
    as observation and state matrices (if required) consistently to the attributes above.
    """

    
    def step(self, actions):
        """ Returns reward, terminated, info """
        raise NotImplementedError

    def get_obs(self):
        """ Returns all agent observations in a list """
        raise NotImplementedError

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise NotImplementedError

    def get_obs_size(self):
        """ Returns the shape of the observation """
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def get_state_size(self):
        """ Returns the shape of the state"""
        raise NotImplementedError

    def get_avail_actions(self):
        raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        raise NotImplementedError

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        raise NotImplementedError

    def reset(self):
        """ Returns initial observations and states"""
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self):
        raise NotImplementedError

    def save_replay(self):
        raise NotImplementedError


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