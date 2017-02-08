from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal


class test_random_variable_operators_class(tf.test.TestCase):

  def test_add(self):
    with self.test_session() as sess:
      x = Normal(mu=0.0, sigma=1.0)
      y = 5.0
      z = x + y
      z_value = x.value() + y
      z_eval, z_value_eval = sess.run([z, z_value])
      assert np.allclose(z_eval, z_value_eval)

  def test_radd(self):
    with self.test_session() as sess:
      x = Normal(mu=0.0, sigma=1.0)
      y = 5.0
      z = y + x
      z_value = y + x.value()
      z_eval, z_value_eval = sess.run([z, z_value])
      assert np.allclose(z_eval, z_value_eval)

  def test_sub(self):
    with self.test_session() as sess:
      x = Normal(mu=0.0, sigma=1.0)
      y = 5.0
      z = x - y
      z_value = x.value() - y
      z_eval, z_value_eval = sess.run([z, z_value])
      assert np.allclose(z_eval, z_value_eval)

  def test_rsub(self):
    with self.test_session() as sess:
      x = Normal(mu=0.0, sigma=1.0)
      y = 5.0
      z = y - x
      z_value = y - x.value()
      z_eval, z_value_eval = sess.run([z, z_value])
      assert np.allclose(z_eval, z_value_eval)

  def test_mul(self):
    with self.test_session() as sess:
      x = Normal(mu=0.0, sigma=1.0)
      y = 5.0
      z = x * y
      z_value = x.value() * y
      z_eval, z_value_eval = sess.run([z, z_value])
      assert np.allclose(z_eval, z_value_eval)

  def test_rmul(self):
    with self.test_session() as sess:
      x = Normal(mu=0.0, sigma=1.0)
      y = 5.0
      z = y * x
      z_value = y * x.value()
      z_eval, z_value_eval = sess.run([z, z_value])
      assert np.allclose(z_eval, z_value_eval)

  def test_div(self):
    with self.test_session() as sess:
      x = Normal(mu=0.0, sigma=1.0)
      y = 5.0
      z = x / y
      z_value = x.value() / y
      z_eval, z_value_eval = sess.run([z, z_value])
      assert np.allclose(z_eval, z_value_eval)

  def test_rdiv(self):
    with self.test_session() as sess:
      x = Normal(mu=0.0, sigma=1.0)
      y = 5.0
      z = y / x
      z_value = y / x.value()
      z_eval, z_value_eval = sess.run([z, z_value])
      assert np.allclose(z_eval, z_value_eval)

  def test_floordiv(self):
    with self.test_session() as sess:
      x = Normal(mu=0.0, sigma=1.0)
      y = 5.0
      z = x // y
      z_value = x.value() // y
      z_eval, z_value_eval = sess.run([z, z_value])
      assert np.allclose(z_eval, z_value_eval)

  def test_rfloordiv(self):
    with self.test_session() as sess:
      x = Normal(mu=0.0, sigma=1.0)
      y = 5.0
      z = y // x
      z_value = y // x.value()
      z_eval, z_value_eval = sess.run([z, z_value])
      assert np.allclose(z_eval, z_value_eval)

  def test_mod(self):
    with self.test_session() as sess:
      x = Normal(mu=0.0, sigma=1.0)
      y = 5.0
      z = x % y
      z_value = x.value() % y
      z_eval, z_value_eval = sess.run([z, z_value])
      assert np.allclose(z_eval, z_value_eval)

  def test_rmod(self):
    with self.test_session() as sess:
      x = Normal(mu=0.0, sigma=1.0)
      y = 5.0
      z = y % x
      z_value = y % x.value()
      z_eval, z_value_eval = sess.run([z, z_value])
      assert np.allclose(z_eval, z_value_eval)

  def test_lt(self):
    with self.test_session() as sess:
      x = Normal(mu=0.0, sigma=1.0)
      y = 5.0
      z = x < y
      z_value = x.value() < y
      z_eval, z_value_eval = sess.run([z, z_value])
      assert np.allclose(z_eval, z_value_eval)

  def test_le(self):
    with self.test_session() as sess:
      x = Normal(mu=0.0, sigma=1.0)
      y = 5.0
      z = x <= y
      z_value = x.value() <= y
      z_eval, z_value_eval = sess.run([z, z_value])
      assert np.allclose(z_eval, z_value_eval)

  def test_gt(self):
    with self.test_session() as sess:
      x = Normal(mu=0.0, sigma=1.0)
      y = 5.0
      z = x > y
      z_value = x.value() > y
      z_eval, z_value_eval = sess.run([z, z_value])
      assert np.allclose(z_eval, z_value_eval)

  def test_ge(self):
    with self.test_session() as sess:
      x = Normal(mu=0.0, sigma=1.0)
      y = 5.0
      z = x >= y
      z_value = x.value() >= y
      z_eval, z_value_eval = sess.run([z, z_value])
      assert np.allclose(z_eval, z_value_eval)

    # need to test with a random variable of boolean
  # def test_and(self):
  #   with self.test_session() as sess:
      # x = tf.cast(Bernoulli(p=0.5), tf.bool)
      # y = True
      # z = x & y
      # z_value = x.value() & y
      # z_eval, z_value_eval = sess.run([z, z_value])
      # assert np.allclose(z_eval, z_value_eval)

  # def test_rand(self):
  # def test_or(self):
  # def test_ror(self):
  # def test_xor(self):
  # def test_rxor(self):

  def test_pow(self):
    with self.test_session() as sess:
      x = Normal(mu=0.0, sigma=1.0)
      y = 5.0
      z = x ** y
      z_value = x.value() ** y
      z_eval, z_value_eval = sess.run([z, z_value])
      assert np.allclose(z_eval, z_value_eval)

  def test_rpow(self):
    with self.test_session() as sess:
      x = Normal(mu=0.0, sigma=1.0)
      y = 5.0
      z = y ** x
      z_value = y ** x.value()
      z_eval, z_value_eval = sess.run([z, z_value])
      assert np.allclose(z_eval, z_value_eval)

  # def test_invert(self):

  def test_neg(self):
    with self.test_session() as sess:
      x = Normal(mu=0.0, sigma=1.0)
      z = -x
      z_value = -x.value()
      z_eval, z_value_eval = sess.run([z, z_value])
      assert np.allclose(z_eval, z_value_eval)

  def test_abs(self):
    with self.test_session() as sess:
      x = Normal(mu=0.0, sigma=1.0)
      z = abs(x)
      z_value = abs(x.value())
      z_eval, z_value_eval = sess.run([z, z_value])
      assert np.allclose(z_eval, z_value_eval)

  def test_hash(self):
    x = Normal(mu=0.0, sigma=1.0)
    y = 5.0
    assert not hash(x) == hash(y)
    assert hash(x) == id(x)

  def test_eq(self):
    x = Normal(mu=0.0, sigma=1.0)
    y = 5.0
    assert not x == y
    assert x == x

if __name__ == '__main__':
  tf.test.main()
