"""Persistent randomness.

Our language defines random variables. They enable memoization in the
sense that the generative process of any values which depend on the
same random variable will be generated conditioned on the same samples.
Simulating the world multiple times (i.e., fetching the value out of
session) results in new memoized values. To avoid persistent
randomness, simply define another random variable to work with.

References
----------
https://probmods.org/chapters/02-generative-models.html#persistent-randomness-mem
"""
import edward as ed
import tensorflow as tf

from edward.models import Categorical


def eye_color(person):
  random_variables = {x.name: x for x in ed.random_variables()}
  if person + '/' in random_variables:
    return random_variables[person + '/']
  else:
    return Categorical(probs=tf.ones(3) / 3, name=person)


def main(_):
  # Only two categorical random variables are created.
  eye_color('bob')
  eye_color('alice')
  eye_color('bob')

if __name__ == "__main__":
  tf.app.run()
