# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import math
import tensorflow as tf

def inference(observations, observation_length, hidden1_units, hidden2_units,
              action_length):
  """
  Args:
    observations: Observations placeholder.
    observation_length: Size of an observation.
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.
    action_length: Size of an action.
  """
  # Hidden 1
  with tf.name_scope('hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([observation_length, hidden1_units],
                            stddev=1.0 / math.sqrt(float(observation_length))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
    hidden1 = tf.nn.relu(tf.matmul(observations, weights) + biases)

  # Hidden 2
  with tf.name_scope('hidden2'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

  # Linear
  with tf.name_scope('linear'):
    weights = tf.Variable(
        tf.truncated_normal([hidden2_units, action_length],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([action_length]), name='biases')
    logits = tf.matmul(hidden2, weights) + biases
  return logits


def loss(logits, labels):
  """
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, float - [batch_size, NUM_CLASSES].
  """
  diffs = tf.subtract(logits, labels)
  squared_diffs = tf.pow(diffs, tf.constant(2.0))
  errors = tf.reduce_sum(squared_diffs, 1)
  loss = tf.reduce_mean(errors)
  return errors, loss


def training(loss, learning_rate):
  """
  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
  """
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return global_step, train_op
