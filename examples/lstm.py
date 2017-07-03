#!/usr/bin/env python
"""LSTM language model on text8.

Default hyperparameters achieve ~79.4 NLL at epoch 50, ~76.1423 NLL at
epoch 200; ~13s/epoch on Titan X (Pascal).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import os
import string
import tensorflow as tf

from datetime import datetime
from edward.models import Categorical
from edward.util import maybe_download_and_extract, Progbar

data_dir = "data/text8"
log_dir = "log"
out_dir = "out"
n_epoch = 100
n_iter_per_epoch = 250
batch_size = 128
hidden_size = 512
timesteps = 64
lr = 1e-3

timestamp = datetime.strftime(datetime.utcnow(), "%Y%m%d_%H%M%S")
hyperparam_str = '_'.join([
    var + '_' + str(eval(var)).replace('.', '_')
    for var in ['batch_size', 'hidden_size', 'timesteps', 'lr']])
log_dir = os.path.join(log_dir, timestamp + '_' + hyperparam_str)
out_dir = os.path.join(out_dir, timestamp + '_' + hyperparam_str)
if not os.path.exists(log_dir):
  os.makedirs(log_dir)
if not os.path.exists(out_dir):
  os.makedirs(out_dir)


def text8(path):
  """Load the text8 data set (Mahoney, 2006)."""
  path = os.path.expanduser(path)
  url = 'http://mattmahoney.net/dc/text8.zip'
  maybe_download_and_extract(path, url)
  with open(os.path.join(path, 'text8')) as f:
    text = f.read()
  x_train = text[:int(90e6)]
  x_valid = text[int(90e6):int(95e6)]
  x_test = text[int(95e6):int(100e6)]
  return x_train, x_valid, x_test


def lstm_cell(x, h, c, name=None, reuse=False):
  """LSTM returning hidden state and content cell at a specific timestep."""
  nin = x.shape[-1].value
  nout = h.shape[-1].value
  with tf.variable_scope(name, default_name="lstm",
                         values=[x, h, c], reuse=reuse):
    wx = tf.get_variable("kernel/input", [nin, nout * 4],
                         dtype=tf.float32,
                         initializer=tf.orthogonal_initializer(1.0))
    wh = tf.get_variable("kernel/hidden", [nout, nout * 4],
                         dtype=tf.float32,
                         initializer=tf.orthogonal_initializer(1.0))
    b = tf.get_variable("bias", [nout * 4],
                         dtype=tf.float32,
                         initializer=tf.constant_initializer(0.0))

  z = tf.matmul(x, wx) + tf.matmul(h, wh) + b
  i, f, o, u = tf.split(z, 4, axis=1)
  i = tf.sigmoid(i)
  f = tf.sigmoid(f + 1.0)
  o = tf.sigmoid(o)
  u = tf.tanh(u)
  c = f * c + i * u
  h = o * tf.tanh(c)
  return h, c


def generator(array, batch_size, encoder):
  """Generate batch with respect to array (list)'s first axis, and encode
  strings in array to integers, with shape [batch_size, timesteps + 1].

  Each data point has timesteps + 1 characters. We will
  condition for 0 <= t <= timesteps as input and predict for 1 <= t <=
  timesteps + 1 as output.
  """
  while True:
    imb = np.random.randint(0, len(array) - timesteps, batch_size)
    out = np.asarray(
        [[encoder[c] for c in array[i:(i + timesteps + 1)]] for i in imb],
        dtype=np.int32)
    yield out


def language_model(input):
  """Form p(x[0], ..., x[timesteps]),

  \prod_{t=1}^{timesteps} p(x[t] | x[:t]),

  where x = [x[0], ..., x[timesteps - 1]] is `input`. We do not
  include p(x[0]) which is a constant wrt parameters. The input also
  does not include the timesteps index. To calculate
  the probability, we will call log_prob on
  x = [x[1], ..., x[timesteps]].

  We implement this separately from the generative model so the
  forward pass, e.g., embedding/dense layers, can be parallelized.

  [batch_size, timesteps] -> [batch_size, timesteps]
  """
  x = tf.one_hot(input, depth=vocab_size, dtype=tf.float32)
  h = tf.zeros([batch_size, hidden_size])
  c = tf.zeros([batch_size, hidden_size])
  hs = []
  reuse = None
  for t in range(timesteps):
    if t > 0:
      reuse = True
    xt = x[:, t, :]
    h, c = lstm_cell(xt, h, c, name="lstm", reuse=reuse)
    hs.append(h)

  h = tf.stack(hs, 1)
  logits = tf.layers.dense(h, vocab_size, name="dense")
  output = Categorical(logits=logits)
  return output


def language_model_gen(batch_size):
  """Generate x ~ prod p(x_t | x_{<t}). Output [batch_size, timesteps].
  """
  # Initialize data input randomly.
  x = tf.random_uniform([batch_size], 0, vocab_size, dtype=tf.int32)
  h = tf.zeros([batch_size, hidden_size])
  c = tf.zeros([batch_size, hidden_size])
  xs = []
  for _ in range(timesteps):
    x = tf.one_hot(x, depth=vocab_size, dtype=tf.float32)
    h, c = lstm_cell(x, h, c, name="lstm")
    logits = tf.layers.dense(h, vocab_size, name="dense")
    x = Categorical(logits=logits).value()
    xs.append(x)

  xs = tf.cast(tf.stack(xs, 1), tf.int32)
  return xs


ed.set_seed(42)

# DATA
x_train, _, _ = text8(data_dir)
vocab = string.ascii_lowercase + ' '
vocab_size = len(vocab)
encoder = dict(zip(vocab, range(vocab_size)))
decoder = {v: k for k, v in encoder.items()}

data = generator(x_train, batch_size, encoder)

# MODEL
x_ph_input = tf.placeholder(tf.int32, [None, timesteps])
x_ph_target = tf.placeholder(tf.int32, [None, timesteps])
with tf.variable_scope("language_model"):
  x = language_model(x_ph_input)

with tf.variable_scope("language_model", reuse=True):
  x_gen = language_model_gen(64)

# INFERENCE
inference = ed.MAP({}, {x: x_ph_target})

optimizer = tf.train.AdamOptimizer(learning_rate=lr)
inference.initialize(optimizer=optimizer, logdir=log_dir, log_timestamp=False)

print("Number of sets of parameters: {}".format(len(tf.trainable_variables())))
for v in tf.trainable_variables():
  print(v)

sess = ed.get_session()
tf.global_variables_initializer().run()

for epoch in range(n_epoch):
  print("Epoch: {0}".format(epoch))
  avg_loss = 0.0

  pbar = Progbar(n_iter_per_epoch)
  for t in range(1, n_iter_per_epoch + 1):
    pbar.update(t)
    x_batch = next(data)
    x_batch_input = x_batch[:, :timesteps]
    x_batch_target = x_batch[:, 1:(timesteps + 1)]
    info_dict = inference.update(
        feed_dict={x_ph_input: x_batch_input, x_ph_target: x_batch_target})
    avg_loss += info_dict['loss']

  # Print average loss over epoch.
  avg_loss /= n_iter_per_epoch
  print("log p(x): {:0.8f}".format(avg_loss))

  # Generate and write samples from model.
  x_samples = sess.run(x_gen)
  samples = [''.join([decoder[xt] for xt in sample]) for sample in x_samples]
  print("Samples:")
  for sample in samples[:5]:
    print(sample)
  with open(os.path.join(out_dir, "samples.txt"), "a") as myfile:
    myfile.write("\nEpoch: {0}\n".format(epoch))
    for sample in samples[:5]:
      myfile.write(sample + "\n")
