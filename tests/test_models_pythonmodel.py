import numpy as np
import tensorflow as tf
import pystan

model_code = """
    data {
      int<lower=0> N;
      int<lower=0,upper=1> y[N];
    }
    parameters {
      real<lower=0,upper=1> theta;
    }
    model {
      theta ~ beta(0.5, 0.5);  // Jeffreys' prior
      for (n in 1:N)
        y[n] ~ bernoulli(theta);
    }
"""
data = dict(N=10, y=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1])

print("The following message exists as Stan initializes an empty model.")
model = pystan.stan(model_code=model_code, data=data, iter=10, chains=1)

sess = tf.InteractiveSession()

## Test np.float
def log_prob(z):
    # np.float as input
    return np.array([model.log_prob(z)], dtype=np.float32)

z = tf.placeholder(shape=(1), dtype=tf.float32)
tf_model_log_prob = tf.py_func(log_prob, [z], [tf.float32])
sess.run(tf_model_log_prob, {z: np.array([2.0])})

## Test np.array
def log_prob(zs):
    # np.array as input
    return np.array([model.log_prob(z) for z in zs], dtype=np.float32)

zs = tf.placeholder(shape=(1,1), dtype=tf.float32)
tf_model_log_prob = tf.py_func(log_prob, [zs], [tf.float32])
sess.run(tf_model_log_prob, {zs: np.array([[2.0]])})

## Test when integrated into multiple functions
## in Inference class
zs = tf.placeholder(shape=(1,1), dtype=tf.float32)
def loss():
    return log_prob(zs)

## in Model class
def _py_log_prob(zs):
    # np.array as input
    return np.array([model.log_prob(z) for z in zs], dtype=np.float32)

def log_prob(zs):
    return tf.py_func(_py_log_prob, [zs], [tf.float32])

sess.run(loss(), {zs: np.array([[2.0]])})
