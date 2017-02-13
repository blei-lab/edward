#!/usr/bin/env python
"""Persistent randomness.

Our language defines random variables. They enable memoization in the
sense that the generative process of any values which depend on the
same random variable will be generated conditioned on the same samples.
Simulating the world multiple times (i.e., fetching the value out of
session) results in new memoized values. To avoid persistent
randomness, simply define another random variable to work with.

References
----------
https://probmods.org/generative-models.html#persistent-randomness-mem
"""
import edward as ed
import tensorflow as tf

from edward.models import Categorical


def eye_color(person):
  random_variables = {x.name: x for x in
                      tf.get_collection('_random_variable_collection_')}
  if person + '/' in random_variables:
    return random_variables[person + '/']
  else:
    return Categorical(logits=ed.logit(tf.constant([1.0 / 3] * 3)), name=person)


# Only two categorical random variables are created.
eye_color('bob')
eye_color('alice')
eye_color('bob')
