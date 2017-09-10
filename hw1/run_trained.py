#!/usr/bin/env python

"""
Code to load a trained policy and generate roll-out data. Example usage:

    python run_trained.py \
        --num_rollouts 20 \
        bc_checkpoints/Reacher-v1/Reacher-v1-1000.meta \
        Reacher-v1 \
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
        "meta_file",
        type=str
    )
    parser.add_argument(
        "envname",
        type=str
    )
    return parser

def main():
    args = get_parser().parse_args()

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(args.meta_file)
        saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(args.meta_file)))

        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit
        graph = tf.get_default_graph()
        opl = graph.get_tensor_by_name("observations:0")
        logits = graph.get_tensor_by_name("linear/logits:0")

        returns = []
        for i in range(args.num_rollouts):
            print("Iteration", i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = sess.run(logits, feed_dict={opl: obs[None,:]})
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
        with open(args.stats_file, "w") as f:
            json.dump(stats, f, indent=4)

if __name__ == "__main__":
    main()
