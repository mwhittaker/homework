import copy
import logging
import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from cheetah_env import HalfCheetahEnvNew
from controllers import MPCcontroller, RandomController
from cost_functions import cheetah_cost_fn, trajectory_cost_fn
from dynamics import NNDynamicsModel
import logz

def d(s):
    logging.getLogger("mjw").debug(s)

def paths_to_data(paths):
    return {
        "observations": np.concatenate([p["observations"] for p in paths]),
        "actions": np.concatenate([p["actions"] for p in paths]),
        "next_observations": np.concatenate([p["next_observations"] for p in paths]),
        "rewards": np.concatenate([p["rewards"] for p in paths]),
    }

def sample(env,
           controller,
           num_paths=10,
           horizon=1000,
           render=False,
           verbose=False):
    # Write a sampler function which takes in an environment, a controller
    # (either random or the MPC controller), and returns rollouts by running on
    # the env. Each path can have elements for observations, next_observations,
    # rewards, returns, actions, etc.
    paths = []
    for _ in range(num_paths):
        path = {
            "observations":[],
            "actions":[],
            "next_observations":[],
            "rewards":[],
        }
        obs = env.reset()
        for _ in range(horizon):
            action = controller.get_action(obs)
            next_obs, reward, done, _info = env.step(action)
            path["observations"].append(obs)
            path["actions"].append(action)
            path["next_observations"].append(next_obs)
            path["rewards"].append(reward)

            obs = next_obs
            if done:
                break
        paths.append(path)
    return paths

# Utility to compute cost a path for a given cost function
def path_cost(cost_fn, path):
    observations = path["observations"]
    actions = path["actions"]
    next_observations = path["next_observations"]
    return trajectory_cost_fn(cost_fn, observations, actions, next_observations)

def compute_normalization(data):
    # Write a function to take in a dataset and compute the means, and stds.
    # Return 6 elements: mean of s_t, std of s_t, mean of (s_t+1 - s_t), std of
    # (s_t+1 - s_t), mean of actions, std of actions
    observations = data["observations"]
    actions = data["actions"]
    next_observations = data["next_observations"]

    mean_obs = np.mean(observations, axis=0)
    std_obs = np.std(observations, axis=0)
    mean_deltas = np.mean(next_observations - observations, axis=0)
    std_deltas = np.std(next_observations - observations, axis=0)
    mean_action = np.mean(actions, axis=0)
    std_action = np.std(actions, axis=0)
    return mean_obs, std_obs, mean_deltas, std_deltas, mean_action, std_action

def plot_comparison(env, dyn_model):
    """
    Write a function to generate plots comparing the behavior of the model
    predictions for each element of the state to the actual ground truth, using
    randomly sampled actions.
    """
    """ YOUR CODE HERE """
    pass

def train(env,
         cost_fn,
         logdir=None,
         render=False,
         learning_rate=1e-3,
         onpol_iters=10,
         dynamics_iters=60,
         batch_size=512,
         num_paths_random=10,
         num_paths_onpol=10,
         num_simulated_paths=10000,
         env_horizon=1000,
         mpc_horizon=15,
         n_layers=2,
         size=500,
         activation=tf.nn.relu,
         output_activation=None):

    """

    Arguments:

    onpol_iters                 Number of iterations of onpolicy aggregation
    |_                          for the loop to run.

    dynamics_iters              Number of iterations of training for the
    |                           dynamics model which happen per iteration of
    |_                          the aggregation loop.

    batch_size                  Batch size for dynamics training.

    num_paths_random            Number of paths/trajectories/rollouts generated
    |                           by a random agent. We use these to train our
    |_                          initial dynamics model.

    num_paths_onpol             Number of paths to collect at each iteration of
    |                           aggregation, using the Model Predictive Control
    |_                          policy.

    num_simulated_paths         How many fictitious rollouts the MPC policy
    |                           should generate each time it is asked for an
    |_                          action.

    env_horizon                 Number of timesteps in each path.

    mpc_horizon                 The MPC policy generates actions by imagining
    |                           fictitious rollouts, and picking the first
    |                           action of the best fictitious rollout. This
    |                           argument is how many timesteps should be in
    |_                          each fictitious rollout.

    n_layers/size/activations   Neural network architecture arguments.

    """

    d("env                         = {}".format(env))
    d("env.observation_space       = {}".format(env.observation_space))
    d("env.action_space            = {}".format(env.action_space))
    d("env.observation_space.shape = {}".format(env.observation_space.shape))
    d("env.action_space.shape      = {}".format(env.action_space.shape))
    d("logdir                      = {}".format(logdir))
    d("render                      = {}".format(render))
    d("learning_rate               = {}".format(learning_rate))
    d("onpol_iters                 = {}".format(onpol_iters))
    d("dynamics_iters              = {}".format(dynamics_iters))
    d("batch_size                  = {}".format(batch_size))
    d("num_paths_random            = {}".format(num_paths_random))
    d("num_paths_onpol             = {}".format(num_paths_onpol))
    d("num_simulated_paths         = {}".format(num_simulated_paths))
    d("env_horizon                 = {}".format(env_horizon))
    d("mpc_horizon                 = {}".format(mpc_horizon))
    d("n_layers                    = {}".format(n_layers))
    d("size                        = {}".format(size))

    logz.configure_output_dir(logdir)

    #===========================================================================
    # First, we need a lot of data generated by a random agent, with which
    # we'll begin to train our dynamics model.
    random_controller = RandomController(env)
    random_paths = sample(
        env=env,
        controller=random_controller,
        num_paths=num_paths_random,
        horizon=env_horizon,
        render=render)

    #===========================================================================
    # The random data will be used to get statistics (mean and std) for the
    # observations, actions, and deltas (where deltas are o_{t+1} - o_t). These
    # will be used for normalizing inputs and denormalizing outputs from the
    # dynamics network.
    data = paths_to_data(random_paths)
    normalization = compute_normalization(data)

    #===========================================================================
    # Build dynamics model and MPC controllers.
    sess = tf.Session()

    dyn_model = NNDynamicsModel(
        env=env,
        n_layers=n_layers,
        size=size,
        activation=activation,
        output_activation=output_activation,
        normalization=normalization,
        batch_size=batch_size,
        iterations=dynamics_iters,
        learning_rate=learning_rate,
        sess=sess)

    mpc_controller = MPCcontroller(
        env=env,
        dyn_model=dyn_model,
        horizon=mpc_horizon,
        cost_fn=cost_fn,
        num_simulated_paths=num_simulated_paths)

    #===========================================================================
    # Tensorflow session building.
    sess.__enter__()
    tf.global_variables_initializer().run()

    #===========================================================================
    # Take multiple iterations of onpolicy aggregation at each iteration
    # refitting the dynamics model to current dataset and then taking onpolicy
    # samples and aggregating to the dataset.
    #
    # Note: You don't need to use a mixing ratio in this assignment for new and
    # old data as described in https://arxiv.org/abs/1708.02596
    for itr in range(onpol_iters):
        # Fit the dynamics.
        dyn_model.fit(data)

        # Generate on-policy rollouts.
        rl_paths = sample(
            env=env,
            controller=mpc_controller,
            num_paths=num_paths_onpol,
            horizon=env_horizon,
            render=render)

        # Compute metrics.
        costs = np.array([path_cost(cost_fn, path) for path in rl_paths])
        returns = np.concatenate([path["rewards"] for path in rl_paths])

        # Update data.
        new_data = paths_to_data(rl_paths)
        data = {
            "observations": np.concatenate([data["observations"], new_data["observations"]]),
            "actions": np.concatenate([data["actions"], new_data["actions"]]),
            "next_observations": np.concatenate([data["next_observations"], new_data["next_observations"]]),
            "rewards": np.concatenate([data["rewards"], new_data["rewards"]]),
        }
        # TODO(mwhittaker): Shuffle if we need to.

        # LOGGING
        # Statistics for performance of MPC policy using our learned dynamics
        # model
        logz.log_tabular('Iteration', itr)
        # In terms of cost function which your MPC controller uses to plan
        logz.log_tabular('AverageCost', np.mean(costs))
        logz.log_tabular('StdCost', np.std(costs))
        logz.log_tabular('MinimumCost', np.min(costs))
        logz.log_tabular('MaximumCost', np.max(costs))
        # In terms of true environment reward of your rolled out trajectory
        # using the MPC controller
        logz.log_tabular('AverageReturn', np.mean(returns))
        logz.log_tabular('StdReturn', np.std(returns))
        logz.log_tabular('MinimumReturn', np.min(returns))
        logz.log_tabular('MaximumReturn', np.max(returns))

        logz.dump_tabular()

def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v1')
    # Experiment meta-params
    parser.add_argument('--exp_name', type=str, default='mb_mpc')
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--render', action='store_true')
    # Training args
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--onpol_iters', '-n', type=int, default=1)
    parser.add_argument('--dyn_iters', '-nd', type=int, default=60)
    parser.add_argument('--batch_size', '-b', type=int, default=512)
    # Data collection
    parser.add_argument('--random_paths', '-r', type=int, default=10)
    parser.add_argument('--onpol_paths', '-d', type=int, default=10)
    parser.add_argument('--simulated_paths', '-sp', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=int, default=1000)
    # Neural network architecture args
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=500)
    # MPC Controller
    parser.add_argument('--mpc_horizon', '-m', type=int, default=15)
    # Misc
    parser.add_argument('--verbose', '-v', action="store_true")
    return parser

def main():
    args = get_parser().parse_args()

    # Establish the logger.
    format = "[%(asctime)-15s %(pathname)s:%(lineno)-3s] %(message)s"
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(format))
    logger = logging.getLogger("mjw")
    logger.propagate = False
    logger.addHandler(handler)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Set seed
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # Make data directory if it does not already exist
    if not(os.path.exists('data')):
        os.makedirs('data')
    timestamp = time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = "{}_{}_{}".format(args.exp_name, args.env_name, timestamp)
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    # Make env
    if args.env_name == "HalfCheetah-v1":
        env = HalfCheetahEnvNew()
        cost_fn = cheetah_cost_fn
    train(
        env=env,
        cost_fn=cost_fn,
        logdir=logdir,
        render=args.render,
        learning_rate=args.learning_rate,
        onpol_iters=args.onpol_iters,
        dynamics_iters=args.dyn_iters,
        batch_size=args.batch_size,
        num_paths_random=args.random_paths,
        num_paths_onpol=args.onpol_paths,
        num_simulated_paths=args.simulated_paths,
        env_horizon=args.ep_len,
        mpc_horizon=args.mpc_horizon,
        n_layers = args.n_layers,
        size=args.size,
        activation=tf.nn.relu,
        output_activation=None,
    )

if __name__ == "__main__":
    main()
