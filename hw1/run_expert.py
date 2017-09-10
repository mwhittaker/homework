#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral
cloning. Example usage:

    python run_expert.py \
        experts/Humanoid-v1.pkl \
        Humanoid-v1 \
        --render \
        --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho
(hoj@openai.com)
"""

import argparse
import gym
import json
import load_policy
import numpy as np
import os.path
import pickle
import sys
import tensorflow as tf
import tf_util

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the rollouts in a GUI"
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=20,
        help="Number of expert roll outs"
    )
    parser.add_argument(
        "--max_timesteps",
        type=int,
        help="Maximum number of steps per rollout"
    )
    parser.add_argument(
        "--stats_file",
        type=str,
        required=True,
        help="Output file for rollout stats"
    )
    parser.add_argument(
        "--rollouts_file",
        type=str,
        required=True,
        help="Output file for rollout observations and actions"
    )
    parser.add_argument(
        "expert_policy_file",
        type=str
    )
    parser.add_argument(
        "envname",
        type=str
    )
    return parser

def main():
    args = get_parser().parse_args()

    # Make sure our files exist!
    assert(os.path.exists(os.path.dirname(os.path.abspath(args.stats_file))))
    assert(os.path.exists(os.path.dirname(os.path.abspath(args.rollouts_file))))

    print("Loading and building expert policy.")
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print("Expert policy loaded and built.")

    with tf.Session():
        tf_util.initialize()

        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print("Iteration", i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action[0])
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1

                if args.render:
                    env.render()

                if steps >= max_steps:
                    break
            returns.append(totalr)

        # Statistics about the experts.
        stats = {}
        stats["envname"] = args.envname
        stats["num_rollouts"] = args.num_rollouts
        stats["max_steps"] = max_steps
        stats["returns"] = returns
        stats["mean_return"] = np.mean(returns)
        stats["stddev_return"] = np.std(returns)
        stats["observation_shape"] = np.array(observations).shape
        stats["action_shape"] = np.array(actions).shape
        with open(args.stats_file, "w") as f:
            json.dump(stats, f, indent=4)

        rollouts = {
            "observations": np.array(observations),
            "actions": np.array(actions),
        }
        with open(args.rollouts_file, "wb") as f:
            pickle.dump(rollouts, f)

if __name__ == "__main__":
    main()
