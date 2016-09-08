#!/usr/bin/env python
"""
Undirected model.

As a proof of concept, we show to implement an undirected model with
the structure
x <-> y

We perform this by doing
x_ph -> y -> x
"""

# TODO
# for RBM/MRF inference, we are inferring variational distributions;
# compare it to the conditionally specified case
# inference based on finding normalizing constant instead of a
# conditional density p(z|x)
#
# in general we train
# p(random_vars | x)
#
# + TODO how to deal with potentials?

inference = ed.MFVI({x: qx, y: qy})

# calculate on test data
data = {x: np.array(0.0), y: np.array(1.0)}
