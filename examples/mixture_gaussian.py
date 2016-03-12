#!/usr/bin/env python
"""
This is the implementation of the Bayesian Mixture of K Gaussians. 
The model is written in Stan.
Data: X = {x1,...,xn} where each xi is in R^d..we choose d=2 in our example
Probability model
    Likelihood: 
         x_i|c_i ~ N(mu_{c_i}, sigma_{c_i})
         c_i ~ Discrete(theta)
    Prior:
         theta ~ Dirichlet(alpha_0) where alpha_0 is a vector of dimension K
         mu_j ~ N(0, cI) iid...we take c = 10 in our example
         sigma_j ~ inverse_gamma(a, b) iid..we take a = b = 1 in our example
Variational model
    q(pi) ~ Dirichlet(alpha') where alpha' is a vector of dimension K 
    q(mu_j) ~ N(mj', Sigmaj') iid
    q(sigma_j) ~ inverse_gamma(aj', Bj') iid
    q(c_i) ~ Multinomial(phi_i)  iid...integrated out
"""
import blackbox as bb
import numpy as np

model_code = """
data{
   int<lower=0> N;
   int<lower=0> K;
   int<lower=0> D;
   vector[D] x[N];
}
transformed data{
   vector<lower=0>[K] alpha_0;
   for (k in 1:K) {
      alpha_0[k] <- 1.0/K;
   }
}
parameters{
   simplex[K] theta;
   vector[D] mu[K];
   vector<lower=0>[D] sigma[K];
}
model{
   theta ~ dirichlet(alpha_0);
   for (k in 1:K){
      mu[k] ~ normal(0.0, 10);
      sigma[k] ~ inv_gamma(1.0, 1.0);
   }
   for (n in 1:N) {
      real ps[K];
      for (k in 1:K){
         ps[k] <- log(theta[k]) + normal_log(x[n], mu[k], sigma[k]);
      }
      increment_log_prob(log_sum_exp(ps));
   }
}
"""
bb.set_seed(42)
model = bb.StanModel(model_code=model_code)
K = 2
variational = bb.MFMixGaussian(1, K)
x = np.loadtxt('./mix_data/mix_mock_data.txt', dtype='float32', delimiter=',')
N = len(x)
D = 2
data = bb.Data(dict(N=N, K=K, D=D , x=x))

inference = bb.MFVI(model, variational, data)
inference.run(n_iter=1000)



