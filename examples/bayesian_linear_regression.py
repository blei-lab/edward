"""Bayesian linear regression using stochastic gradient Hamiltonian
Monte Carlo.

This version visualizes additional fits of the model.

References
----------
http://edwardlib.org/tutorials/supervised-regression
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf

from edward.models import Normal, Empirical

tf.flags.DEFINE_integer("N", default=40, help="Number of data points.")
tf.flags.DEFINE_integer("D", default=1, help="Number of features.")
tf.flags.DEFINE_integer("T", default=5000, help="Number of posterior samples.")
tf.flags.DEFINE_integer("nburn", default=100,
                        help="Number of burn-in samples.")
tf.flags.DEFINE_integer("stride", default=10,
                        help="Frequency with which to plots samples.")

FLAGS = tf.flags.FLAGS


def get_input_fn():
  """Returns `input_fn` for train and eval."""
  def build_toy_dataset(N, noise_std=0.5):
    X = np.concatenate([np.linspace(0, 2, num=N / 2),
                        np.linspace(6, 8, num=N / 2)])
    y = 2.0 * X + 10 * np.random.normal(0, noise_std, size=N)
    X = X.reshape((N, 1))
    return X, y
  features, labels = build_toy_dataset(N)
  def input_fn(params):
    """A simple input_fn using the experimental input pipeline."""
    batch_size = params["batch_size"]
    # TODO
    dataset = tf.data.TFRecordDataset(filename, buffer_size=None)
    dataset = dataset.cache().repeat()
    features, labels = dataset.make_one_shot_iterator().get_next()
    return features, labels
  return input_fn


def model(X):
  w = Normal(loc=tf.zeros(FLAGS.D), scale=tf.ones(FLAGS.D))
  b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
  y = Normal(loc=tf.tensordot(X, w, [[1], [0]]) + b,
             scale=tf.ones(FLAGS.N))
  return y


def model_fn(features, labels, mode, params):
  """Model fn which runs on TPU.

  Args:
    features: [None, 784]
    labels: [None, 10]
    mode: tf.estimator.ModeKeys.*
    params: dict of hyperparams.
  """
  qw = tf.get_variable("qw", [FLAGS.D])
  qb = tf.get_variable("qb", [])
  counter = tf.get_variable("counter", initializer=0.)
  qw_mom = tf.get_variable("qw_mom", [FLAGS.D],
                           initializer=tf.zeros_initializer())
  qb_mom = tf.get_variable("qb_mom", [], initializer=tf.zeros_initializer())

  new_states, new_counter, _, new_momentums = ed.sghmc(
      model,
      current_state=[qw, qb],
      counter=counter,
      momentums=[qw_mom, qb_mom],
      learning_rate=1e-3,
      align_latent=lambda name: {"w": "qw", "b": "qb"}.get(name),
      align_data=lambda name: {"y": "y"}.get(name),
      X=features,
      y=labels)

  if mode == tf.estimator.ModeKeys.PREDICT:
    predicted_classes = tf.argmax(logits, 1)
    predictions = {
        "class_ids": predicted_classes[:, tf.newaxis],
        "probabilities": tf.nn.softmax(logits),
    }
    return tf.estimator.EstimatorSpec(mode, loss=None, predictions=predictions)

  predictions = tf.argmax(logits, 1)
  accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)

  n_accept = tf.get_variable("n_accept", initializer=0, trainable=False)
  n_accept_over_t = n_accept / t

  tf.summary.scalar("accuracy", accuracy[1])
  tf.summary.scalar("n_accept", n_accept)

  if mode == tf.estimator.ModeKeys.EVAL:
    return tpu_estimator.TPUEstimatorSpec(
        mode=mode,
        loss=None,
        eval_metrics={"accuracy": accuracy,
                      "n_accept": n_accept,})

  train_op = []
  train_op.append(qw.assign(new_states[0]))
  train_op.append(qb.assign(new_states[1]))
  train_op.append(counter.assign(new_counter))
  train_op.append(qw_mom.assign(new_momentums[0]))
  train_op.append(qb_mom.assign(new_momentums[1]))
  train_op = tf.group(*train_op)
  return tpu_estimator.TPUEstimatorSpec(mode=mode, loss=None, train_op=train_op)


def main(_):
  tf.set_random_seed(42)

  train_input_fn = get_input_fn()
  eval_input_fn = get_input_fn()

  estimator = tf.Estimator(model_fn=model_fn)
  estimator.train(input_fn=train_input_fn,
                  max_steps=FLAGS.train_steps)

  eval_result = estimator.evaluate(input_fn=eval_input_fn)
  print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

  # # Plot posterior samples.
  # sns.jointplot(qb.params.eval()[FLAGS.nburn:FLAGS.T:FLAGS.stride],
  #               qw.params.eval()[FLAGS.nburn:FLAGS.T:FLAGS.stride])
  # plt.show()

  # # Posterior predictive checks.
  # y_post = ed.copy(y, {w: qw, b: qb})
  # # This is equivalent to
  # # y_post = Normal(loc=tf.tensordot(X, qw, [[1], [0]]) + qb,
  #                   scale=tf.ones(FLAGS.N))

  # print("Mean squared error on test data:")
  # print(ed.evaluate('mean_squared_error', data={X: X_test, y_post: y_test}))

  # print("Displaying prior predictive samples.")
  # n_prior_samples = 10

  # w_prior = w.sample(n_prior_samples).eval()
  # b_prior = b.sample(n_prior_samples).eval()

  # plt.scatter(X_train, y_train)

  # inputs = np.linspace(-1, 10, num=400)
  # for ns in range(n_prior_samples):
  #     output = inputs * w_prior[ns] + b_prior[ns]
  #     plt.plot(inputs, output)

  # plt.show()

  # print("Displaying posterior predictive samples.")
  # n_posterior_samples = 10

  # w_post = qw.sample(n_posterior_samples).eval()
  # b_post = qb.sample(n_posterior_samples).eval()

  # plt.scatter(X_train, y_train)

  # inputs = np.linspace(-1, 10, num=400)
  # for ns in range(n_posterior_samples):
  #     output = inputs * w_post[ns] + b_post[ns]
  #     plt.plot(inputs, output)

  # plt.show()

if __name__ == "__main__":
  tf.app.run()
