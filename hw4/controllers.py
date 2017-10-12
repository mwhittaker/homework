import logging
import time

import numpy as np

from cost_functions import trajectory_cost_fn

def d(s):
    logging.getLogger("mjw").debug(s)

class Controller():
    def __init__(self):
        pass

    # Get the appropriate action(s) for this state(s)
    def get_action(self, state):
        pass


class RandomController(Controller):
    def __init__(self, env):
        self.env = env

    def get_action(self, state):
        # Your code should randomly sample an action uniformly from the action
        # space
        return self.env.action_space.sample()


class MPCcontroller(Controller):
    """
    Controller built using the MPC method outlined in
    https://arxiv.org/abs/1708.02596
    """
    def __init__(self,
                 env,
                 dyn_model,
                 horizon=5,
                 cost_fn=None,
                 num_simulated_paths=10):
        self.env = env
        self.dyn_model = dyn_model
        self.horizon = horizon
        self.cost_fn = cost_fn
        self.num_simulated_paths = num_simulated_paths

    def get_action(self, state):
        # Note: be careful to batch your simulations through the model for
        # speed
        all_states = []
        all_actions = []
        all_next_states = []

        states = np.array([state] * self.num_simulated_paths)
        for _ in range(self.horizon):
            actions = np.array([self.env.action_space.sample() for _ in range(self.num_simulated_paths)])
            next_states = self.dyn_model.predict(states, actions)

            all_states.append(states)
            all_actions.append(actions)
            all_next_states.append(next_states)

            states = next_states

        costs = trajectory_cost_fn(self.cost_fn, all_states, all_actions, all_next_states)
        return all_actions[0][np.argmin(costs)]
