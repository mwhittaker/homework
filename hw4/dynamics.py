import logging

import tensorflow as tf
import numpy as np

def d(s):
    logging.getLogger("mjw").debug(s)

def batch_indexes(xs, batch_size):
    return [(low, low + batch_size) for low in range(0, len(xs), batch_size)]

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder,
              output_size,
              scope,
              n_layers=2,
              size=500,
              activation=tf.tanh,
              output_activation=None):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

class NNDynamicsModel():
    def __init__(self,
                 env,
                 n_layers,
                 size,
                 activation,
                 output_activation,
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """
        # Store arguments for later.
        self.env = env
        self.normalization = normalization
        self.batch_size = batch_size
        self.iterations = iterations
        self.sess = sess

        # Build NN placeholders.
        assert(len(env.observation_space.shape) == 1)
        assert(len(env.action_space.shape) == 1)
        obs_dim = env.observation_space.shape[0]
        acts_dim = env.action_space.shape[0]
        self.obs_ph = tf.placeholder(tf.float32, [None, obs_dim])
        self.acts_ph = tf.placeholder(tf.float32, [None, acts_dim])
        self.next_obs_ph = tf.placeholder(tf.float32, [None, obs_dim])

        # Build NN.
        mean_obs, std_obs, mean_deltas, std_deltas, mean_acts, std_acts = normalization
        normalized_obs = (self.obs_ph - mean_obs) / std_obs
        normalized_acts = (self.acts_ph - mean_acts) / std_acts
        normalized_obs_and_acts = tf.concat([normalized_obs, normalized_acts], 1)
        self.predicted_normalized_deltas = build_mlp(
            input_placeholder=normalized_obs_and_acts,
            output_size=obs_dim,
            scope="dynamics",
            n_layers=n_layers,
            size=size,
            activation=activation,
            output_activation=output_activation)

        # Build cost function and optimizer.
        deltas = self.next_obs_ph - self.obs_ph
        normalized_deltas = (deltas - mean_deltas) / std_deltas
        loss = tf.losses.mean_squared_error(
            labels=normalized_deltas,
            predictions=self.predicted_normalized_deltas)
        self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    def fit(self, data):
        # Write a function to take in a dataset of (unnormalized)states,
        # (unnormalized)actions, (unnormalized)next_states and fit the dynamics
        # model going from normalized states, normalized actions to normalized
        # state differences (s_t+1 - s_t)
        unnormalized_obs = data["observations"]
        unnormalized_acts = data["actions"]
        unnormalized_next_obs = data["next_observations"]
        assert(len(unnormalized_obs) == len(unnormalized_acts) == len(unnormalized_next_obs))

        for epoch in range(self.iterations):
            if epoch % 10 == 0:
                d("Epoch {}/{}.".format(epoch, self.iterations))

            indexes = batch_indexes(unnormalized_obs, self.batch_size)
            for (itr, (low, high)) in enumerate(indexes):
                feed_dict = {
                    self.obs_ph: unnormalized_obs[low:high],
                    self.acts_ph: unnormalized_acts[low:high],
                    self.next_obs_ph: unnormalized_next_obs[low:high],
                }
                self.sess.run([self.update_op], feed_dict=feed_dict)
        d("Epoch {}/{}.".format(self.iterations, self.iterations))

    def predict(self, states, actions):
        # Write a function to take in a batch of (unnormalized) states and
        # (unnormalized) actions and return the (unnormalized) next states as
        # predicted by using the model
        assert(len(states) == len(actions))
        feed_dict = {
            self.obs_ph: states,
            self.acts_ph: actions,
        }
        normalized_deltas = self.sess.run(
            self.predicted_normalized_deltas,
            feed_dict=feed_dict)

        _, _, mean_deltas, std_deltas, _, _ = self.normalization
        unnormalized_deltas = (normalized_deltas * std_deltas) + mean_deltas
        return states + unnormalized_deltas
