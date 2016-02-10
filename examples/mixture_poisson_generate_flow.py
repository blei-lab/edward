#!/usr/bin/env python
# Generate latent variables from an HVM fit to Mixture
# Poisson model and plot matrix of fixed size.
# KL: 0.302217156095
# KL: TODO
import matplotlib.pyplot as plt
import numpy as np

from blackbox.models import PosteriorMixturePoisson
from blackbox.likelihoods import MFPoisson
from blackbox.priors import Flow
from blackbox.hvm import HVM
from blackbox.util import discrete_density

if __name__ == '__main__':
  np.random.seed(42)
  model = PosteriorMixturePoisson(np.zeros((2,2)), np.zeros(2))
  q_mf = MFPoisson(model.num_vars)
  q_flow_length = 8
  q_prior = Flow(q_flow_length, q_mf.num_params)
  q_prior.u = np.array(
[[ 0.51626429, 0.57467573],
 [ 0.31654812, 0.31844728],
 [-0.71947567,-0.56391114],
 [ 1.00529281, 0.30895033],
 [-0.11556015, 0.93132195],
 [ 0.2416264 , 0.04971573],
 [ 0.04213006,-1.76075142],
 [-1.63126117,-0.51717116],
 [-0.55590891, 0.84960921],
 [-1.64989469,-2.00170066],
 [ 1.86841472, 0.06088118],
 [-0.22050878,-1.51841493],
 [-0.40856101, 0.27426751],
 [-1.31354096, 0.35216035],
 [-1.34704645,-1.30321233]]
             )
  q_prior.w = np.array(
[[ -7.38195617e-02,  2.40028866e+00],
 [ -9.61856810e-02, -1.20877604e+00],
 [  6.30638470e-01, -1.28509564e+00],
 [  3.10803818e-01, -1.48007822e+00],
 [ -3.96763323e-01, -1.85539925e-03],
 [  1.17840199e+00, -5.54116102e-01],
 [ -1.32305679e-01, -2.24653325e-01],
 [ -9.09403354e-01, -2.85710516e-01],
 [ -6.01782733e-01,  8.81755899e-01],
 [ -1.00118460e-01, -1.03357096e+00],
 [ -3.55261766e-02, -3.04739802e-02],
 [ -1.26625699e+00,  4.57568011e-01],
 [  1.72704478e+00,  5.42284410e-01],
 [ -4.97330631e-01, -2.08704477e-01],
 [  3.56409292e-02,  1.09475778e-01]]
             )
  q_prior.b = np.array(
[-0.10618505, 0.03353949,-1.36871247,-1.66809752, 1.97253588, 0.51024257,
  0.19599067, 0.72849366,-0.09291119,-1.14052528, 1.31740781, 1.05319505,
  0.03395029, 2.09010694,-3.6566931 ]
             )

  num_test_samples = 5000
  z_samples = np.zeros((num_test_samples, 2))
  for r in range(num_test_samples):
    lamda_unconst = q_prior.sample()
    lamda = q_mf.transform(lamda_unconst)
    q_mf.set_lamda(lamda)
    z_samples[r, :] = q_mf.sample()

  prob_table = discrete_density(z_samples, support=[[0,20], [0,20]])
  plt.matshow(prob_table)
  plt.axis('off')
  plt.show()
