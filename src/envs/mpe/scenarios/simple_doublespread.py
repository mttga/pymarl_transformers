import numpy as np
from ..core import World, Agent, Landmark
from ..scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self, N):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = N
        num_landmarks = 2
        world.collaborative = True

        self.cooperative = True

        # generate colors:
        self.colors = [np.random.random(3) for _ in range(num_agents)]

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = "agent_{}".format(i)
            agent.collide = False
            agent.silent = True
            agent.size = 0.1
            agent.color = self.colors[i]

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        self.landmark_colors = [(0,0,1), (1,0,0)]
        for i, landmark in enumerate(world.landmarks):
            landmark.size = 0.1
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.color = self.landmark_colors[i]

        self.reset_world(world)
        return world

    def reset_world(self, world):
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = world.np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = world.np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        return self.global_reward(world)

    def global_reward(self, world):
        each = []
        for landmark in world.landmarks:
            each.append(sum([1.0 if self.is_collision(agent, landmark) else 0.0 for agent in world.agents]))
        return min(each)

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        # entity_color = []
        # for entity in world.landmarks:  # world.entities:
        #     entity_color.append(entity.color)
        # communication of all other agents
        # comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        # return np.concatenate(
        #     [agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm
        # )
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)