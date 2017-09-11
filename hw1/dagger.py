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
import network

def load_data(rollouts_filename):
    with open(rollouts_filename, "rb") as f:
        rollouts = pickle.load(f)
        return rollouts["observations"], rollouts["actions"]

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rollouts_file",
        type=str,
        required=True,
        help="File with expert rollouts.",
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=100,
        help="Number of expert roll outs per iteration"
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=10,
        help="Number of dagger iterations"
    )
    parser.add_argument(
        "--stats_file",
        type=str,
        required=True,
        help="Output file for rollout stats"
    )
    parser.add_argument(
        "--hidden1",
        type=int,
        default=64,
        help="Number of hiiden layer 1 units.",
    )
    parser.add_argument(
        "--training_steps",
        type=int,
        default=1000,
        help="Number of training steps",
    )
    parser.add_argument(
        "--hidden2",
        type=int,
        default=64,
        help="Number of hiiden layer 2 units.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=500,
        help="Batch size for batch gradient descent",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory of checkpoint",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "expert_policy_file",
        type=str
    )
    return parser

def main():
    args = get_parser().parse_args()
    observation_length = 17
    action_length = 6

    # Read the expert rollouts from disk.
    observations, actions = load_data(args.rollouts_file)
    print("observations shape = " + str(observations.shape))
    print("actions shape = " + str(actions.shape))

    # Make sure our files exist!
    assert(os.path.exists(os.path.dirname(os.path.abspath(args.stats_file))))

    # Load the expert.
    print("Loading and building expert policy.")
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print("Expert policy loaded and built.")

    # Assemble the network.
    opl = tf.placeholder(tf.float32, shape=(None, observation_length),
                         name="observations")
    apl = tf.placeholder(tf.float32, shape=(None, action_length),
                         name="actions")
    logits = network.inference(opl, observation_length,
                               args.hidden1, args.hidden2, action_length)
    errors, loss = network.loss(logits, apl)
    global_step, train_op = network.training(loss, args.learning_rate)

    with tf.Session() as sess:
        # Initialize the network.
        tf_util.initialize()
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_dir))

        env = gym.make("Walker2d-v1")
        max_steps = env.spec.timestep_limit

        avg_returns = []
        stddev_returns = []
        observations = list(observations)
        actions = list(actions)

        for iteration in range(args.num_iterations):
            obs = np.array(observations)
            acts = np.array(actions)
            assert(obs.shape[0] == acts.shape[0])

            # Train the network.
            if iteration != 0:
                num_batches = int(obs.shape[0] / args.batch_size)
                for step in range(args.training_steps):
                    i = step % num_batches
                    if i == 0:
                        p = np.random.permutation(obs.shape[0])
                        obs = obs[p]
                        acts = acts[p]
                    start = int(i * args.batch_size)
                    stop = int((i + 1) * args.batch_size)
                    feed_dict = {opl: obs[start:stop], apl: acts[start:stop]}
                    _, loss_value, step_value = sess.run([train_op, loss, global_step],
                                                         feed_dict=feed_dict)
                    if step % 100 == 0:
                        loss_value = sess.run(loss, feed_dict={opl: obs, apl:acts})
                        msg = "Iteration {}; step {}; loss = {}".format(
                            iteration, step_value, loss_value)
                        print(msg)

            # Generate new rollouts.
            rewards = []
            for i in range(args.num_rollouts):
                print("Iteration {}; rollout {}".format(iteration, i))
                obs = env.reset()
                done = False
                steps = 0
                totalr = 0
                while not done:
                    expert_action = policy_fn(obs[None,:])
                    observations.append(obs)
                    actions.append(expert_action[0])

                    action = sess.run(logits, feed_dict={opl: obs[None,:]})
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1
                    if steps >= max_steps:
                        break
                rewards.append(totalr)

            print("Iteration {}; average return {}".format(
                iteration, np.mean(rewards)))
            print("Iteration {}; stddev return {}".format(
                iteration, np.std(rewards)))
            avg_returns.append(np.mean(rewards))
            stddev_returns.append(np.std(rewards))

            with open(args.stats_file, "w") as f:
                stats = {
                    "mean_return": avg_returns,
                    "stddev_returns": stddev_returns
                }
                json.dump(stats, f, indent=4)


if __name__ == "__main__":
    main()
