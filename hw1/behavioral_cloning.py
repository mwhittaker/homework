import sys
import argparse
import pickle

import tensorflow as tf
import numpy as np
import network

ARGS = None

def load_data(rollouts_filename):
    with open(rollouts_filename, "rb") as f:
        rollouts = pickle.load(f)
        return rollouts["observations"], rollouts["actions"]

def main(_):
    # Read the expert rollouts from disk.
    observations, actions = load_data(ARGS.rollouts_file)
    print("observations shape = " + str(observations.shape))
    print("actions shape = " + str(actions.shape))
    observation_length = observations.shape[1]
    action_length = actions.shape[1]

    # Assemble the network.
    opl = tf.placeholder(tf.float32, shape=(None, observation_length))
    apl = tf.placeholder(tf.float32, shape=(None, action_length))
    logits = network.inference(opl, observation_length,
                               ARGS.hidden1, ARGS.hidden2, action_length)
    errors, loss = network.loss(logits, apl)
    train_op = network.training(loss, ARGS.learning_rate)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for step in range(ARGS.training_steps):
        feed_dict = {opl: observations, apl: actions}
        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
        if step % 100 == 0:
            msg = "step {}/{}; loss = {}".format(step, ARGS.training_steps,
                                                 loss_value)
            print(msg)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hidden1",
        type=int,
        default=64,
        help="Number of hiiden layer 1 units.",
    )
    parser.add_argument(
        "--hidden2",
        type=int,
        default=64,
        help="Number of hiiden layer 2 units.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--training_steps",
        type=int,
        default=1000,
        help="Number of training steps",
    )
    parser.add_argument(
        "rollouts_file",
        type=str,
        help="File with expert rollouts.",
    )
    return parser

if __name__ == "__main__":
    ARGS, unparsed = get_parser().parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
