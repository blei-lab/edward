"""LSTM language model on text8.

Default hyperparameters achieve ~78.4 NLL at epoch 50, ~76.1423 NLL at
epoch 200; ~13s/epoch on Titan X (Pascal).

Samples after 200 epochs:
```
e the classmaker was cut apart rome the charts sometimes known a
hemical place baining examples of equipment accepted manner clas
uetean meeting sought to exist as this waiting an excerpt for of
erally enjoyed a film writer of unto one two volunteer humphrey
y captured by the saughton river goodness where stones were nota
```
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
from edward.util import Progbar
from observations import text8


tf.flags.DEFINE_string("data_dir", default="/tmp/data", help="")
tf.flags.DEFINE_string("log_dir", default="/tmp/log", help="")
tf.flags.DEFINE_integer("n_epoch", default=200, help="")
tf.flags.DEFINE_integer("batch_size", default=128, help="")
tf.flags.DEFINE_integer("hidden_size", default=512, help="")
tf.flags.DEFINE_integer("timesteps", default=64, help="")
tf.flags.DEFINE_float("lr", default=5e-3, help="")

FLAGS = tf.flags.FLAGS

timestamp = datetime.strftime(datetime.utcnow(), "%Y%m%d_%H%M%S")
hyperparam_str = '_'.join([
    var + '_' + str(eval(var)).replace('.', '_')
    for var in ['FLAGS.batch_size', 'FLAGS.hidden_size',
                'FLAGS.timesteps', 'FLAGS.lr']])
FLAGS.log_dir = os.path.join(FLAGS.log_dir, timestamp + '_' + hyperparam_str)
if not os.path.exists(FLAGS.log_dir):
  os.makedirs(FLAGS.log_dir)


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


def generator(input, batch_size, timesteps, encoder):
  """Generate batch with respect to input (a list). Encode its
  strings to integers, returning an array of shape [batch_size, timesteps].
  """
  while True:
    imb = np.random.randint(0, len(input) - timesteps, batch_size)
    encoded = np.asarray(
        [[encoder[c] for c in input[i:(i + timesteps)]] for i in imb],
        dtype=np.int32)
    yield encoded


def language_model(input, vocab_size):
  """Form p(x[0], ..., x[timesteps - 1]),

  \prod_{t=0}^{timesteps - 1} p(x[t] | x[:t]),

  To calculate the probability, we call log_prob on
  x = [x[0], ..., x[timesteps - 1]] given
  `input` = [0, x[0], ..., x[timesteps - 2]].

  We implement this separately from the generative model so the
  forward pass, e.g., embedding/dense layers, can be parallelized.

  [batch_size, timesteps] -> [batch_size, timesteps]
  """
  x = tf.one_hot(input, depth=vocab_size, dtype=tf.float32)
  h = tf.fill(tf.stack([tf.shape(x)[0], FLAGS.hidden_size]), 0.0)
  c = tf.fill(tf.stack([tf.shape(x)[0], FLAGS.hidden_size]), 0.0)
  hs = []
  reuse = None
  for t in range(FLAGS.timesteps):
    if t > 0:
      reuse = True
    xt = x[:, t, :]
    h, c = lstm_cell(xt, h, c, name="lstm", reuse=reuse)
    hs.append(h)

  h = tf.stack(hs, 1)
  logits = tf.layers.dense(h, vocab_size, name="dense")
  output = Categorical(logits=logits)
  return output


def language_model_gen(batch_size, vocab_size):
  """Generate x ~ prod p(x_t | x_{<t}). Output [batch_size, timesteps].
  """
  # Initialize data input randomly.
  x = tf.random_uniform([FLAGS.batch_size], 0, vocab_size, dtype=tf.int32)
  h = tf.zeros([FLAGS.batch_size, FLAGS.hidden_size])
  c = tf.zeros([FLAGS.batch_size, FLAGS.hidden_size])
  xs = []
  for _ in range(FLAGS.timesteps):
    x = tf.one_hot(x, depth=vocab_size, dtype=tf.float32)
    h, c = lstm_cell(x, h, c, name="lstm")
    logits = tf.layers.dense(h, vocab_size, name="dense")
    x = Categorical(logits=logits).value()
    xs.append(x)

  xs = tf.cast(tf.stack(xs, 1), tf.int32)
  return xs


def main(_):
  ed.set_seed(42)

  # DATA
  x_train, _, x_test = text8(FLAGS.data_dir)
  vocab = string.ascii_lowercase + ' '
  vocab_size = len(vocab)
  encoder = dict(zip(vocab, range(vocab_size)))
  decoder = {v: k for k, v in encoder.items()}

  data = generator(x_train, FLAGS.batch_size, FLAGS.timesteps, encoder)

  # MODEL
  x_ph = tf.placeholder(tf.int32, [None, FLAGS.timesteps])
  with tf.variable_scope("language_model"):
    # Shift input sequence to right by 1, [0, x[0], ..., x[timesteps - 2]].
    x_ph_shift = tf.pad(x_ph, [[0, 0], [1, 0]])[:, :-1]
    x = language_model(x_ph_shift, vocab_size)

  with tf.variable_scope("language_model", reuse=True):
    x_gen = language_model_gen(5, vocab_size)

  imb = range(0, len(x_test) - FLAGS.timesteps, FLAGS.timesteps)
  encoded_x_test = np.asarray(
      [[encoder[c] for c in x_test[i:(i + FLAGS.timesteps)]] for i in imb],
      dtype=np.int32)
  test_size = encoded_x_test.shape[0]
  print("Test set shape: {}".format(encoded_x_test.shape))
  test_nll = -tf.reduce_sum(x.log_prob(x_ph))

  # INFERENCE
  inference = ed.MAP({}, {x: x_ph})

  optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
  inference.initialize(optimizer=optimizer,
                       logdir=FLAGS.log_dir,
                       log_timestamp=False)

  print("Number of sets of parameters: {}".format(
      len(tf.trainable_variables())))
  print("Number of parameters: {}".format(
      np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])))
  for v in tf.trainable_variables():
    print(v)

  sess = ed.get_session()
  tf.global_variables_initializer().run()

  # Double n_epoch and print progress every half an epoch.
  n_iter_per_epoch = len(x_train) // (FLAGS.batch_size * FLAGS.timesteps * 2)
  epoch = 0.0
  for _ in range(FLAGS.n_epoch * 2):
    epoch += 0.5
    print("Epoch: {0}".format(epoch))
    avg_nll = 0.0

    pbar = Progbar(n_iter_per_epoch)
    for t in range(1, n_iter_per_epoch + 1):
      pbar.update(t)
      x_batch = next(data)
      info_dict = inference.update({x_ph: x_batch})
      avg_nll += info_dict['loss']

    # Print average bits per character over epoch.
    avg_nll /= (n_iter_per_epoch * FLAGS.batch_size * FLAGS.timesteps *
                np.log(2))
    print("Train average bits/char: {:0.8f}".format(avg_nll))

    # Print per-data point log-likelihood on test set.
    avg_nll = 0.0
    for start in range(0, test_size, batch_size):
      end = min(test_size, start + batch_size)
      x_batch = encoded_x_test[start:end]
      avg_nll += sess.run(test_nll, {x_ph: x_batch})

    avg_nll /= test_size
    print("Test average NLL: {:0.8f}".format(avg_nll))

    # Generate samples from model.
    samples = sess.run(x_gen)
    samples = [''.join([decoder[c] for c in sample]) for sample in samples]
    print("Samples:")
    for sample in samples:
      print(sample)

if __name__ == "__main__":
  tf.app.run()
